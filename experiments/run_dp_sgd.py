"""Step 3 (record-level): DP-SGD (Abadi et al., 2016) in FedAvg, with a *meaningful*
epsilon via subsampled-Gaussian RDP.

Usage:
    uv run python experiments/run_dp_sgd.py

Why this exists (vs run_dp.py):
    ``run_dp.py`` is *update-level* DP-FedAvg -- it clips the whole multi-step weight
    delta and adds noise once per round, with full participation and no subsampling.
    Its epsilon is therefore vacuous (always >= 59, see that script) and accuracy
    collapses the moment noise bites. This script is the *record-level* mechanism the
    assignment's first reference actually describes (Abadi DP-SGD): each client clips
    the **per-example** gradient inside every local step and adds Gaussian noise, and
    the budget is accounted with the **subsampled-Gaussian RDP** accountant
    (Mironov-Talwar-Zhang 2019), the same machinery TF-Privacy / Opacus use.

What it shows (the richer finding):
  1. Per-example clipping degrades *gracefully* -- accuracy slides 0.88 -> ~0.6 as
     noise grows, instead of the update-level mechanism's cliff to chance. So the
     DP *mechanism* matters, not just the noise level.
  2. It still confirms the curse of dimensionality: a *formally* small epsilon (<10)
     needs noise that has already cost most of the accuracy on this 38K-param model
     with only 320 training images. Subsampling (q<1) pushes epsilon down but the
     small lots destroy the per-coordinate SNR (SNR ~ lot_size / (z*sqrt(dim))), so
     it buys lower epsilon only by sacrificing accuracy -- the contrast points below
     make this explicit. Compare with HE (Step 3-1), which hides the gradient at
     near-zero accuracy cost.

Effectiveness measure: accuracy. Privacy level: the value of epsilon.

Outputs:
    results/figures/dp_sgd_tradeoff.png
    results/figures/dp_sgd_leakage_demo.png
    results/metrics/dp_sgd_tradeoff.csv
    results/metrics/dp_sgd_subsampling.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils import (
    denormalize,
    get_test_loader,
    load_orl_dataset,
    split_iid,
    train_test_split,
)
from src.dlg_attack import compute_real_gradients, dlg_attack, idlg_label_inference
from src.dp_utils import (
    clip_grad_list,
    compute_epsilon_subsampled,
    dp_fedavg_grad_list,
    dp_sgd_local_update,
)
from src.federated import evaluate, get_device
from src.metrics import compute_psnr, compute_ssim
from src.models import LeNet

NUM_ROUNDS = 50
NUM_CLIENTS = 4
NUM_CLASSES = 40
NUM_ITERS = 300
TARGET_INDEX = 5
DELTA = 1e-5
SEEDS = (0, 1, 2)

# DP-SGD hyperparameters (tuned: plain SGD trains this Sigmoid net at lr=0.5; C=10 is
# ~half the median per-sample grad norm so clipping is active; 8 full-shard steps per
# round x 50 rounds = 400 steps, accounted with sampling rate Q).
CLIP_NORM = 10.0
LR = 0.5
LOCAL_STEPS = 8
SAMPLE_RATE = 1.0  # lot = full client shard each step; best accuracy-vs-epsilon here
# Noise multipliers; z=0 is the non-private baseline (epsilon = inf).
Z_VALUES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
# Contrast: smaller sampling rates reach a lower epsilon but the small lots collapse
# accuracy (curse of dimensionality), shown as a separate small table.
SUBSAMPLING_CONTRAST = ((0.5, 0.5), (0.25, 0.5), (0.25, 1.0))  # (q, z)

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"


def _client_tensors(shard) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise a client shard into (images, labels) tensors for DP-SGD."""
    xs = torch.stack([shard[i][0] for i in range(len(shard))])
    ys = torch.tensor([int(shard[i][1]) for i in range(len(shard))])
    return xs, ys


def _fedavg_dp_sgd(cts, *, z: float, q: float, seed: int, device) -> float:
    """Run FedAvg where every client trains with DP-SGD; return final test accuracy."""
    torch.manual_seed(seed)
    full = load_orl_dataset()
    _, test_set = train_test_split(full, seed=seed)
    test_loader = get_test_loader(test_set, batch_size=32)

    global_state = {k: v.clone() for k, v in LeNet(NUM_CLASSES).state_dict().items()}
    for rnd in range(NUM_ROUNDS):
        deltas, counts = [], []
        for ci, (xs, ys) in enumerate(cts):
            model = LeNet(NUM_CLASSES)
            model.load_state_dict(global_state)
            gen = torch.Generator().manual_seed(100_003 * seed + 7919 * rnd + 31 * ci + 1)
            delta = dp_sgd_local_update(
                model, xs, ys, clip_norm=CLIP_NORM, noise_multiplier=z,
                sample_rate=q, local_steps=LOCAL_STEPS, lr=LR, generator=gen,
            )
            deltas.append(delta)
            counts.append(xs.shape[0])
        total = sum(counts)
        global_state = {
            k: global_state[k] + sum((n / total) * d[k] for d, n in zip(deltas, counts))
            for k in global_state
        }
    model = LeNet(NUM_CLASSES)
    model.load_state_dict(global_state)
    return evaluate(model.to(device), test_loader, device)[0]


def utility_sweep(cts, device) -> dict[float, tuple[float, float]]:
    """DP-SGD FedAvg accuracy (mean, std over seeds) for each noise multiplier."""
    steps = NUM_ROUNDS * LOCAL_STEPS
    out = {}
    for z in Z_VALUES:
        accs = [_fedavg_dp_sgd(cts, z=z, q=SAMPLE_RATE, seed=s, device=device) for s in SEEDS]
        out[z] = (float(np.mean(accs)), float(np.std(accs)))
        eps = compute_epsilon_subsampled(SAMPLE_RATE, z, steps, DELTA)
        es = "inf" if eps == float("inf") else f"{eps:.1f}"
        print(f"[dp-sgd] utility z={z:<4} eps={es:>10}: acc={out[z][0]:.4f} +/- {out[z][1]:.4f}")
    return out


def privacy_sweep(imgs, lbls, mean, std) -> tuple[dict, dict, torch.Tensor]:
    """DLG reconstruction quality + image for each z (round-0 model).

    The leakage axis isolates the effect of the *noise*: the demo gradient is
    clipped to its **own** L2 norm (a no-op at z=0) and then perturbed with std
    ``z * norm``, so the curve descends purely with z -- directly comparable to the
    update-level DP-FedAvg leakage curve (``run_dp.py``). Clipping instead to a
    fixed ``C`` below the gradient's own norm would by itself break naive
    single-gradient DLG before any noise is added, flattening this axis to the
    floor (~6 dB) and hiding the trade-off. The actual DP-SGD mechanism still clips
    per-example *inside* training -- that is what the utility / epsilon axis
    accounts; this axis answers the separate question "how much noise stops DLG".
    """
    torch.manual_seed(0)
    model = LeNet(NUM_CLASSES, dlg_init=True).eval()
    image = imgs[TARGET_INDEX : TARGET_INDEX + 1]
    label = lbls[TARGET_INDEX : TARGET_INDEX + 1]
    orig01 = denormalize(image, mean, std)
    clean = compute_real_gradients(model, image, label)
    inferred = idlg_label_inference(clean, NUM_CLASSES)
    clip_g = clip_grad_list(clean, 1e12)[1]  # the gradient's own norm -> z is the only varied factor

    quality, recon = {}, {}
    for z in Z_VALUES:
        gen = torch.Generator().manual_seed(0)
        noisy = dp_fedavg_grad_list(clean, clip_g, z, generator=gen)  # clip to own norm, add std z*norm
        rec, _, _ = dlg_attack(
            model, noisy, tuple(image.shape), (1, NUM_CLASSES),
            num_iterations=NUM_ITERS, device="cpu", known_label=inferred, seed=0,
        )
        rec01 = denormalize(rec, mean, std)
        quality[z] = {"psnr": compute_psnr(orig01, rec01), "ssim": compute_ssim(orig01, rec01)}
        recon[z] = rec01
        print(f"[dp-sgd] privacy z={z:<4}: DLG psnr={quality[z]['psnr']:5.1f}dB "
              f"ssim={quality[z]['ssim']:.3f}")
    return quality, recon, orig01


def subsampling_contrast(cts, device) -> pd.DataFrame:
    """A few (q, z) points showing subsampling lowers epsilon but collapses accuracy."""
    steps = NUM_ROUNDS * LOCAL_STEPS
    rows = []
    for q, z in SUBSAMPLING_CONTRAST:
        accs = [_fedavg_dp_sgd(cts, z=z, q=q, seed=s, device=device) for s in SEEDS]
        eps = compute_epsilon_subsampled(q, z, steps, DELTA)
        rows.append({"sample_rate": q, "noise_multiplier": z, "epsilon": eps,
                     "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs))})
        print(f"[dp-sgd] subsampling q={q:<4} z={z:<4} eps={eps:8.1f}: "
              f"acc={rows[-1]['acc_mean']:.4f} +/- {rows[-1]['acc_std']:.4f}")
    return pd.DataFrame(rows)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    device = get_device()
    steps = NUM_ROUNDS * LOCAL_STEPS
    print(f"[dp-sgd] device: {device} | clip C={CLIP_NORM} lr={LR} local_steps={LOCAL_STEPS} "
          f"q={SAMPLE_RATE} | steps={steps} | delta={DELTA} | seeds={SEEDS}")

    full = load_orl_dataset()
    imgs, lbls = full.tensors
    train_set, _ = train_test_split(full, seed=0)
    cts = [_client_tensors(s) for s in split_iid(train_set, NUM_CLIENTS, seed=0)]

    print("[dp-sgd] === utility: DP-SGD FedAvg accuracy vs noise (=> epsilon) ===")
    util = utility_sweep(cts, device)
    print("[dp-sgd] === privacy: DLG leakage vs noise (per-example clip) ===")
    quality, recon, orig01 = privacy_sweep(imgs, lbls, full.mean, full.std)
    print("[dp-sgd] === contrast: subsampling lowers epsilon but collapses accuracy ===")
    sub = subsampling_contrast(cts, device)
    sub.to_csv(METRICS / "dp_sgd_subsampling.csv", index=False)

    rows = []
    for z in Z_VALUES:
        eps = compute_epsilon_subsampled(SAMPLE_RATE, z, steps, DELTA)
        rows.append({
            "noise_multiplier": z, "epsilon": eps, "clip_norm": CLIP_NORM,
            "sample_rate": SAMPLE_RATE, "acc_mean": util[z][0], "acc_std": util[z][1],
            "dlg_psnr": quality[z]["psnr"], "dlg_ssim": quality[z]["ssim"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "dp_sgd_tradeoff.csv", index=False)

    # --- Trade-off figure: accuracy (graceful) + DLG leakage vs z / epsilon ---
    def _eps_label(z):
        e = compute_epsilon_subsampled(SAMPLE_RATE, z, steps, DELTA)
        return "inf" if e == float("inf") else (f"{e:.0f}" if e < 1000 else f"{e:.0e}")

    x = list(range(len(Z_VALUES)))
    acc_m = df["acc_mean"].to_numpy(); acc_s = df["acc_std"].to_numpy()
    psnr = df["dlg_psnr"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(10.5, 4.8))
    ax1.plot(x, acc_m, "-o", color="tab:green", label="DP-SGD FedAvg accuracy (mean +/- std, 3 seeds)")
    ax1.fill_between(x, acc_m - acc_s, acc_m + acc_s, color="tab:green", alpha=0.2)
    ax1.set_ylabel("Final test accuracy", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"z={z}\nε={_eps_label(z)}" for z in Z_VALUES], fontsize=7)
    ax1.set_xlabel("DP-SGD noise multiplier z  (and subsampled-RDP privacy budget ε; δ=1e-5)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, psnr, "-s", color="tab:purple", label="DLG PSNR (leakage)")
    ax2.axhline(20.0, ls="--", color="gray", lw=1)
    ax2.set_ylabel("DLG reconstruction PSNR (dB)", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    lines = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=8)
    fig.suptitle(
        "Record-level DP-SGD: per-example clipping degrades accuracy GRACEFULLY\n"
        "(contrast the update-level DP-FedAvg cliff) -- but a small ε still costs most of it",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "dp_sgd_tradeoff.png", dpi=150)
    plt.close(fig)

    # --- Visual: victim reconstructed under growing per-example DP-SGD noise ---
    demo_z = [z for z in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0) if z in recon]
    fig, axes = plt.subplots(1, len(demo_z) + 1, figsize=(1.55 * (len(demo_z) + 1), 2.7))
    axes[0].imshow(orig01.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for ax, z in zip(axes[1:], demo_z):
        eps = compute_epsilon_subsampled(SAMPLE_RATE, z, steps, DELTA)
        es = "inf" if eps == float("inf") else f"{eps:.0f}"
        ax.imshow(recon[z].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"z={z}\neps={es}\n{quality[z]['psnr']:.0f}dB", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"DLG reconstruction of image #{TARGET_INDEX} vs DP-SGD noise multiplier z "
                 f"(Gaussian on the clipped gradient)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "dp_sgd_leakage_demo.png", dpi=150)
    plt.close(fig)

    print(f"[dp-sgd] wrote trade-off figures to {FIGURES} and metrics to {METRICS}")


if __name__ == "__main__":
    main()
