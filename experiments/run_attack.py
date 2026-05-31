"""Run the DLG/iDLG attack and produce the leakage figures + metrics.

Usage:
    uv run python experiments/run_attack.py

Threat-model note (important for honest reporting):
    DLG here inverts a *single-sample loss gradient* -- the canonical Zhu et al.
    setting. The real FedAvg client in this project uploads a multi-step *weight
    delta* (batch_size=8, one local epoch of SGD/Adam steps), which is much
    harder to invert. So these reconstructions show the upper bound of leakage
    from one clean gradient, not an attack on the exact bytes FedAvg transmits.
    The batch-size sweep below makes that gap concrete: leakage collapses as the
    gradient aggregates more samples.

Experiments:
  * Demo        - an untrained (round-0) model, several victim images: how
                  perfectly a single gradient leaks one image.
  * Batch sweep - round-0 model, batch_size in {1,2,4,8}: leakage vs how many
                  samples the gradient averages over.
  * DLG vs iDLG - same image/model, joint-label DLG vs analytic-label iDLG:
                  iDLG converges faster and more stably.
  * Rounds      - the FedAvg model across many snapshot rounds, attacking one
                  fixed image, to resolve how leakage decays as training proceeds.

Outputs:
    results/figures/dlg_demo_comparison.png
    results/figures/dlg_batchsize_sweep.png
    results/figures/dlg_vs_idlg.png
    results/figures/dlg_rounds_comparison.png
    results/figures/dlg_quality_vs_round.png
    results/figures/dlg_loss_curve.png
    results/metrics/dlg_attack_results.csv
    results/metrics/dlg_batchsize.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils import denormalize, load_orl_dataset
from src.dlg_attack import compute_real_gradients, dlg_attack, idlg_label_inference
from src.metrics import compute_mse, compute_psnr, compute_ssim
from src.models import LeNet

DEVICE = "cpu"  # LBFGS-based DLG is most stable on CPU
NUM_CLASSES = 40
NUM_ITERS = 300
DEMO_INDICES = [0, 10, 50, 90, 130, 200, 310, 399]  # one image from several subjects
ROUNDS_TARGET_INDEX = 5
PANEL_ROUNDS = (1, 6, 12, 20, 50)  # subset of SNAPSHOT_ROUNDS shown as an image strip
BATCH_SIZES = (1, 2, 4, 8)
BATCH_INDICES = [0, 40, 80, 120, 160, 200, 240, 280]  # distinct subjects
SUCCESS_PSNR = 20.0

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"


def attack_one(model, image, label, mean, std):
    """Reconstruct ``image`` (normalised) with iDLG and return metrics + history."""
    grads = compute_real_gradients(model, image, label)
    inferred = idlg_label_inference(grads, NUM_CLASSES)
    rec, _, history = dlg_attack(
        model, grads, tuple(image.shape), (1, NUM_CLASSES),
        num_iterations=NUM_ITERS, device=DEVICE, known_label=inferred,
    )
    orig01 = denormalize(image, mean, std)
    rec01 = denormalize(rec, mean, std)
    metrics = {
        "psnr": compute_psnr(orig01, rec01),
        "ssim": compute_ssim(orig01, rec01),
        "mse": compute_mse(orig01, rec01),
        "label": int(label.item()),
        "inferred_label": inferred,
    }
    return orig01, rec01, metrics, history


def run_demo(imgs, lbls, mean, std):
    torch.manual_seed(0)
    model = LeNet(NUM_CLASSES, dlg_init=True).to(DEVICE).eval()
    rows, panels = [], []
    for img_id in DEMO_INDICES:
        image, label = imgs[img_id : img_id + 1], lbls[img_id : img_id + 1]
        orig01, rec01, m, _ = attack_one(model, image, label, mean, std)
        rows.append({"image_id": img_id, "round": 0, **m})
        panels.append((img_id, orig01, rec01, m))
        print(f"[attack] demo image {img_id:3d}: psnr={m['psnr']:5.1f}dB ssim={m['ssim']:.3f}")

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 3.8))
    for col, (img_id, orig01, rec01, m) in enumerate(panels):
        axes[0, col].imshow(orig01.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"#{img_id}", fontsize=9)
        axes[1, col].imshow(rec01.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"{m['psnr']:.0f}dB", fontsize=9)
        for r in (0, 1):
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=10)
    axes[1, 0].set_ylabel("recovered", fontsize=10)
    fig.suptitle("DLG single-image leakage on an untrained model", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "dlg_demo_comparison.png", dpi=150)
    plt.close(fig)
    return rows


def run_batch_sweep(imgs, lbls, mean, std):
    """Attack a batch gradient for several batch sizes; leakage drops with size.

    The gradient is the mean over the batch, so individual images become harder
    to disentangle. We use plain DLG (joint label optimisation) because analytic
    label inference is single-sample. Quality is the permutation-invariant
    best-match PSNR: each original scored against its closest reconstruction.
    """
    torch.manual_seed(0)
    model = LeNet(NUM_CLASSES, dlg_init=True).to(DEVICE).eval()
    rows, panels = [], []
    for bs in BATCH_SIZES:
        ids = BATCH_INDICES[:bs]
        images = imgs[ids]
        labels = lbls[ids]
        grads = compute_real_gradients(model, images, labels)
        rec, _, _ = dlg_attack(
            model, grads, tuple(images.shape), (bs, NUM_CLASSES),
            num_iterations=NUM_ITERS, device=DEVICE, known_label=None,
        )
        orig01 = denormalize(images, mean, std)
        rec01 = denormalize(rec, mean, std)
        # Best-match PSNR per original (reconstruction order is not guaranteed).
        best = []
        for i in range(bs):
            best.append(max(compute_psnr(orig01[i], rec01[j]) for j in range(bs)))
        mean_psnr = sum(best) / bs
        rows.append({"batch_size": bs, "mean_best_psnr": mean_psnr})
        panels.append((bs, orig01, rec01))
        print(f"[attack] batch size {bs}: mean best-match psnr={mean_psnr:5.1f}dB")

    plt.figure(figsize=(6, 4.2))
    plt.plot([r["batch_size"] for r in rows], [r["mean_best_psnr"] for r in rows], "-o")
    plt.axhline(SUCCESS_PSNR, ls="--", color="gray", label=f"success threshold {SUCCESS_PSNR:.0f} dB")
    plt.xscale("log", base=2)
    plt.xticks(BATCH_SIZES, [str(b) for b in BATCH_SIZES])
    plt.xlabel("Batch size (samples averaged into one gradient)")
    plt.ylabel("Mean best-match PSNR (dB)")
    plt.title("DLG leakage collapses as the gradient averages more samples")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "dlg_batchsize_sweep.png", dpi=150)
    plt.close()

    # Visual strip: originals vs best-effort reconstructions for the largest batch.
    bs, orig01, rec01 = panels[-1]
    fig, axes = plt.subplots(2, bs, figsize=(1.5 * bs, 3.4))
    for i in range(bs):
        axes[0, i].imshow(orig01[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(rec01[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        for r in (0, 1):
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=10)
    axes[1, 0].set_ylabel("recovered", fontsize=10)
    fig.suptitle(f"DLG on a batch of {bs}: individual faces no longer separable", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "dlg_batchsize_demo.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(METRICS / "dlg_batchsize.csv", index=False)
    return rows


def run_dlg_vs_idlg(imgs, lbls, mean, std):
    """Compare joint-label DLG against analytic-label iDLG on the same target."""
    torch.manual_seed(0)
    model = LeNet(NUM_CLASSES, dlg_init=True).to(DEVICE).eval()
    image, label = imgs[0:1], lbls[0:1]
    orig01 = denormalize(image, mean, std)
    grads = compute_real_gradients(model, image, label)
    inferred = idlg_label_inference(grads, NUM_CLASSES)

    results = {}
    for name, known in (("DLG (joint label)", None), ("iDLG (analytic label)", inferred)):
        rec, _, history = dlg_attack(
            model, grads, tuple(image.shape), (1, NUM_CLASSES),
            num_iterations=NUM_ITERS, device=DEVICE, known_label=known,
        )
        psnr = compute_psnr(orig01, denormalize(rec, mean, std))
        results[name] = (history, psnr)
        print(f"[attack] {name:24s}: final psnr={psnr:5.1f}dB")

    plt.figure(figsize=(7, 4.5))
    for name, (history, psnr) in results.items():
        plt.semilogy(range(1, len(history) + 1), history, label=f"{name}  ({psnr:.0f} dB)")
    plt.xlabel("LBFGS iteration")
    plt.ylabel("Gradient-matching loss (log scale)")
    plt.title("iDLG vs DLG: analytic label inference converges faster")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "dlg_vs_idlg.png", dpi=150)
    plt.close()


def run_rounds(imgs, lbls, mean, std):
    snapshot_path = RESULTS / "fl_snapshots.pt"
    if not snapshot_path.exists():
        print(f"[attack] {snapshot_path} not found - run experiments/run_fl.py first; skipping rounds")
        return []
    snapshots = torch.load(snapshot_path, weights_only=True)
    rounds = sorted(snapshots)
    image = imgs[ROUNDS_TARGET_INDEX : ROUNDS_TARGET_INDEX + 1]
    label = lbls[ROUNDS_TARGET_INDEX : ROUNDS_TARGET_INDEX + 1]

    rows, recs, histories = [], {}, []
    for rnd in rounds:
        model = LeNet(NUM_CLASSES, dlg_init=False).to(DEVICE)
        model.load_state_dict(snapshots[rnd])
        model.eval()
        orig01, rec01, m, history = attack_one(model, image, label, mean, std)
        rows.append({"image_id": ROUNDS_TARGET_INDEX, "round": rnd, **m})
        recs[rnd] = rec01
        if rnd in PANEL_ROUNDS:
            histories.append((rnd, history))
        print(f"[attack] round {rnd:2d} image {ROUNDS_TARGET_INDEX}: psnr={m['psnr']:5.1f}dB ssim={m['ssim']:.3f}")

    orig01 = denormalize(image, mean, std)

    # (1) Curated image strip at a few representative rounds.
    panel_rounds = [r for r in PANEL_ROUNDS if r in recs]
    fig, axes = plt.subplots(1, len(panel_rounds) + 1, figsize=(2.0 * (len(panel_rounds) + 1), 2.6))
    axes[0].imshow(orig01.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for ax, rnd in zip(axes[1:], panel_rounds):
        psnr = next(r["psnr"] for r in rows if r["round"] == rnd)
        ax.imshow(recs[rnd].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"round {rnd}\n{psnr:.0f}dB", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"DLG reconstruction of image #{ROUNDS_TARGET_INDEX} across FL rounds", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "dlg_rounds_comparison.png", dpi=150)
    plt.close(fig)

    # (2) Dense quality-vs-round curve: this resolves the privacy cliff.
    df = pd.DataFrame(rows)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(df["round"], df["psnr"], "-o", color="tab:blue", label="PSNR")
    ax1.axhline(SUCCESS_PSNR, ls="--", color="gray", lw=1)
    ax1.set_xlabel("FL communication round (model training progress)")
    ax1.set_ylabel("PSNR (dB)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df["round"], df["ssim"], "-s", color="tab:red", label="SSIM")
    ax2.set_ylabel("SSIM", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(-0.05, 1.05)
    fig.suptitle("DLG leakage decays as the FedAvg model trains", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "dlg_quality_vs_round.png", dpi=150)
    plt.close(fig)

    # (3) Optimisation convergence for the representative panel rounds.
    plt.figure(figsize=(7, 4.5))
    for rnd, history in histories:
        plt.semilogy(range(1, len(history) + 1), history, label=f"round {rnd}")
    plt.xlabel("LBFGS iteration")
    plt.ylabel("Gradient-matching loss (log scale)")
    plt.title("DLG optimisation convergence by FL round")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "dlg_loss_curve.png", dpi=150)
    plt.close()
    return rows


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    full = load_orl_dataset()
    imgs, lbls = full.tensors
    mean, std = full.mean, full.std

    print("[attack] === demo setting (untrained model) ===")
    demo_rows = run_demo(imgs, lbls, mean, std)
    print("[attack] === batch-size sweep (untrained model) ===")
    run_batch_sweep(imgs, lbls, mean, std)
    print("[attack] === DLG vs iDLG (untrained model) ===")
    run_dlg_vs_idlg(imgs, lbls, mean, std)
    print("[attack] === rounds setting (trained models) ===")
    round_rows = run_rounds(imgs, lbls, mean, std)

    df = pd.DataFrame(demo_rows + round_rows)
    csv_path = METRICS / "dlg_attack_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[attack] wrote {csv_path} and figures to {FIGURES}")

    demo_psnr = [r["psnr"] for r in demo_rows]
    success = sum(p > SUCCESS_PSNR for p in demo_psnr)
    print(
        f"[attack] demo success rate (PSNR > {SUCCESS_PSNR:.0f} dB): "
        f"{success}/{len(demo_psnr)} = {success / len(demo_psnr):.0%}  "
        f"(mean {sum(demo_psnr) / len(demo_psnr):.1f} dB)"
    )


if __name__ == "__main__":
    main()
