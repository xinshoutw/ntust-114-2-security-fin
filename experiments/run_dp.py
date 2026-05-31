"""Step 3: differential-privacy (DP-FedAvg) defence and its privacy-utility trade-off.

Usage:
    uv run python experiments/run_dp.py

The defence is the Gaussian mechanism of DP-FedAvg (McMahan et al., 2018) / DP-SGD
(Abadi et al., 2016): each client clips its update to an L2 norm bound ``C`` and
adds Gaussian noise with std ``z * C`` (``z`` = noise multiplier) before upload.
Clipping bounds the per-client sensitivity, which is what makes a finite privacy
budget ``epsilon`` possible; ``epsilon`` is accounted with RDP composition over the
communication rounds (see :mod:`src.dp_utils`). ``epsilon`` depends only on ``z``
and the number of rounds, so sweeping ``z`` sweeps ``epsilon``.

We report the trade-off the assignment asks for, with the **value of epsilon** as
the privacy axis:

  * Utility  - final FedAvg test accuracy (mean +/- std over seeds) vs epsilon.
  * Privacy  - DLG reconstruction quality (PSNR/SSIM) of one victim image whose
               gradient is protected by the *same* mechanism (multiplier z) vs
               epsilon.

Observation this surfaces: empirical privacy is cheap here but formal privacy is
not. A trace of noise -- z=0.01 -- already drops DLG reconstruction from ~84 dB to
~6 dB while accuracy is untouched (~0.89): the Gaussian noise is scaled to the
gradient's global L2 norm yet added per coordinate, so in ~38K dimensions the
per-coordinate signal-to-noise ratio falls below 1 even at tiny z. But that same z
buys only epsilon ~ 3e5 -- a vacuous guarantee. Pushing epsilon into a meaningful
range (<~10) needs z ~ O(1), which has long since collapsed accuracy to chance
(that happens by z ~ 0.05). So on this high-dimensional model formal DP and usable
accuracy are irreconcilable -- the curse of dimensionality -- in contrast to the HE
defence (Step 3-1), which hides the gradient entirely at near-zero accuracy cost.

Outputs:
    results/figures/dp_tradeoff.png
    results/figures/dp_leakage_demo.png
    results/metrics/dp_tradeoff.csv
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

from src.data_utils import denormalize, load_orl_dataset
from src.dlg_attack import compute_real_gradients, dlg_attack, idlg_label_inference
from src.dp_utils import clip_grad_list, compute_epsilon, dp_fedavg_grad_list
from src.federated import get_device, run_federated_learning
from src.metrics import compute_psnr, compute_ssim
from src.models import LeNet

NUM_ROUNDS = 50
NUM_CLASSES = 40
NUM_ITERS = 300
TARGET_INDEX = 5
CLIP_NORM = 7.0  # ~ median client-update L2 norm (active clipping; standard DP-SGD choice)
DELTA = 1e-5
SEEDS = (0, 1, 2)
# Noise multipliers; z=0 is the clip-only baseline (no noise -> epsilon = inf).
Z_VALUES = (0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0)

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"


def baseline_accuracy(device) -> float:
    """Final accuracy with no DP at all (no clipping, no noise) for reference."""
    accs = [
        run_federated_learning(num_rounds=NUM_ROUNDS, device=device, seed=s, verbose=False)
        ["history"][-1]["accuracy"]
        for s in SEEDS
    ]
    return float(np.mean(accs))


def utility_sweep(device) -> dict[float, tuple[float, float]]:
    """Final FedAvg accuracy (mean, std over seeds) for each noise multiplier."""
    out = {}
    for z in Z_VALUES:
        accs = []
        for s in SEEDS:
            res = run_federated_learning(
                num_rounds=NUM_ROUNDS, device=device, seed=s,
                dp_clip=CLIP_NORM, dp_noise_multiplier=z, verbose=False,
            )
            accs.append(res["history"][-1]["accuracy"])
        out[z] = (float(np.mean(accs)), float(np.std(accs)))
        eps = compute_epsilon(z, NUM_ROUNDS, DELTA)
        es = "inf" if eps == float("inf") else f"{eps:.1f}"
        print(f"[dp] utility  z={z:<5} eps={es:>10}: acc={out[z][0]:.4f} +/- {out[z][1]:.4f}")
    return out


def privacy_sweep(imgs, lbls, mean, std) -> tuple[dict, dict, torch.Tensor]:
    """DLG reconstruction quality + image for each noise multiplier (round-0 model).

    The single-sample gradient the attacker observes is protected by the same
    Gaussian mechanism (clip to its own norm, add std ``z * norm``), so the
    privacy axis shares the noise multiplier -- and therefore epsilon -- with the
    utility axis.
    """
    torch.manual_seed(0)
    model = LeNet(NUM_CLASSES, dlg_init=True).eval()
    image = imgs[TARGET_INDEX : TARGET_INDEX + 1]
    label = lbls[TARGET_INDEX : TARGET_INDEX + 1]
    orig01 = denormalize(image, mean, std)
    clean_grads = compute_real_gradients(model, image, label)
    inferred = idlg_label_inference(clean_grads, NUM_CLASSES)
    clip_g = clip_grad_list(clean_grads, 1e12)[1]  # the gradient's own L2 norm

    quality, recon = {}, {}
    for z in Z_VALUES:
        gen = torch.Generator().manual_seed(0)
        noisy = dp_fedavg_grad_list(clean_grads, clip_g, z, generator=gen)
        rec, _, _ = dlg_attack(
            model, noisy, tuple(image.shape), (1, NUM_CLASSES),
            num_iterations=NUM_ITERS, device="cpu", known_label=inferred, seed=0,
        )
        rec01 = denormalize(rec, mean, std)
        quality[z] = {"psnr": compute_psnr(orig01, rec01), "ssim": compute_ssim(orig01, rec01)}
        recon[z] = rec01
        print(f"[dp] privacy  z={z:<5}: DLG psnr={quality[z]['psnr']:5.1f}dB "
              f"ssim={quality[z]['ssim']:.3f}")
    return quality, recon, orig01


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"[dp] device: {device} | clip C={CLIP_NORM} | delta={DELTA} | seeds={SEEDS}")

    print("[dp] === reference: accuracy with no DP (no clip, no noise) ===")
    base_acc = baseline_accuracy(device)
    print(f"[dp] no-DP baseline accuracy = {base_acc:.4f}")

    print("[dp] === utility: FedAvg accuracy vs noise (=> epsilon) ===")
    util = utility_sweep(device)

    print("[dp] === privacy: DLG leakage vs noise (=> epsilon) ===")
    full = load_orl_dataset()
    imgs, lbls = full.tensors
    quality, recon, orig01 = privacy_sweep(imgs, lbls, full.mean, full.std)

    rows = []
    for z in Z_VALUES:
        eps = compute_epsilon(z, NUM_ROUNDS, DELTA)
        rows.append({
            "noise_multiplier": z,
            "epsilon": eps,
            "clip_norm": CLIP_NORM,
            "acc_mean": util[z][0],
            "acc_std": util[z][1],
            "dlg_psnr": quality[z]["psnr"],
            "dlg_ssim": quality[z]["ssim"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "dp_tradeoff.csv", index=False)

    # --- Trade-off figure: accuracy and leakage vs the privacy budget epsilon ---
    # epsilon spans ~5 orders of magnitude and z=0 is epsilon=inf, so we lay the
    # noise multipliers out as evenly-spaced categories and label each tick with
    # both z and its epsilon (the privacy level the assignment asks for).
    def _eps_label(z: float) -> str:
        e = compute_epsilon(z, NUM_ROUNDS, DELTA)
        return "inf" if e == float("inf") else (f"{e:.0f}" if e < 1000 else f"{e:.0e}")

    x = list(range(len(Z_VALUES)))
    acc_m = df["acc_mean"].to_numpy()
    acc_s = df["acc_std"].to_numpy()
    psnr = df["dlg_psnr"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(x, acc_m, "-o", color="tab:green", label="FedAvg accuracy (mean +/- std, 3 seeds)")
    ax1.fill_between(x, acc_m - acc_s, acc_m + acc_s, color="tab:green", alpha=0.2)
    ax1.axhline(base_acc, ls=":", color="tab:green", lw=1, alpha=0.7,
                label=f"no-DP baseline ({base_acc:.2f})")
    ax1.set_ylabel("Final test accuracy", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"z={z}\nε={_eps_label(z)}" for z in Z_VALUES], fontsize=8)
    ax1.set_xlabel("DP noise multiplier z  (and resulting privacy budget ε; δ=1e-5, 50 rounds)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, psnr, "-s", color="tab:purple", label="DLG PSNR (leakage)")
    ax2.axhline(20.0, ls="--", color="gray", lw=1)
    ax2.set_ylabel("DLG reconstruction PSNR (dB)", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    lines = ax1.get_lines()[:2] + ax2.get_lines()[:1]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=8)
    fig.suptitle(
        "DP-FedAvg trade-off: a trace of noise (z=0.01) already defeats DLG, but a\n"
        "formally meaningful ε needs noise that has collapsed accuracy to chance", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "dp_tradeoff.png", dpi=150)
    plt.close(fig)

    # --- Visual: the victim image reconstructed under growing noise / shrinking epsilon ---
    fig, axes = plt.subplots(1, len(Z_VALUES) + 1, figsize=(1.55 * (len(Z_VALUES) + 1), 2.7))
    axes[0].imshow(orig01.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for ax, z in zip(axes[1:], Z_VALUES):
        eps = compute_epsilon(z, NUM_ROUNDS, DELTA)
        es = "inf" if eps == float("inf") else f"{eps:.0f}"
        ax.imshow(recon[z].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"z={z}\neps={es}\n{quality[z]['psnr']:.0f}dB", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"DLG reconstruction of image #{TARGET_INDEX} vs DP noise (clip+Gaussian)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "dp_leakage_demo.png", dpi=150)
    plt.close(fig)

    print(f"[dp] wrote trade-off figures to {FIGURES} and metrics to {METRICS}")


if __name__ == "__main__":
    main()
