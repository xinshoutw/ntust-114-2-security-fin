"""Run the DLG/iDLG attack and produce the leakage figures + metrics.

Usage:
    uv run python experiments/run_attack.py

Two settings:
  * Demo   - an untrained (round-0) model, several different victim images, to
             show how perfectly a single gradient leaks an image.
  * Rounds - the FedAvg model at rounds 1/10/25/50 (from run_fl.py snapshots),
             attacking one fixed image to see how reconstruction quality shifts
             as training progresses.

Outputs:
    results/figures/dlg_demo_comparison.png
    results/figures/dlg_rounds_comparison.png
    results/figures/dlg_loss_curve.png
    results/metrics/dlg_attack_results.csv
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
SUCCESS_PSNR = 20.0

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"


def attack_one(model, image, label, mean, std):
    """Reconstruct ``image`` (normalised) and return ``(rec01, metrics, history)``."""
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
    rows = []
    panels = []
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


def run_rounds(imgs, lbls, mean, std):
    snapshot_path = RESULTS / "fl_snapshots.pt"
    if not snapshot_path.exists():
        print(f"[attack] {snapshot_path} not found - run experiments/run_fl.py first; skipping rounds")
        return []
    snapshots = torch.load(snapshot_path, weights_only=True)
    rounds = sorted(snapshots)
    image = imgs[ROUNDS_TARGET_INDEX : ROUNDS_TARGET_INDEX + 1]
    label = lbls[ROUNDS_TARGET_INDEX : ROUNDS_TARGET_INDEX + 1]

    rows, panels, histories = [], [], []
    for rnd in rounds:
        model = LeNet(NUM_CLASSES, dlg_init=False).to(DEVICE)
        model.load_state_dict(snapshots[rnd])
        model.eval()
        orig01, rec01, m, history = attack_one(model, image, label, mean, std)
        rows.append({"image_id": ROUNDS_TARGET_INDEX, "round": rnd, **m})
        panels.append((rnd, rec01, m))
        histories.append((rnd, history))
        print(f"[attack] round {rnd:2d} image {ROUNDS_TARGET_INDEX}: psnr={m['psnr']:5.1f}dB ssim={m['ssim']:.3f}")

    orig01 = denormalize(image, mean, std)
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(2.0 * (len(panels) + 1), 2.6))
    axes[0].imshow(orig01.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for ax, (rnd, rec01, m) in zip(axes[1:], panels):
        ax.imshow(rec01.squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"round {rnd}\n{m['psnr']:.0f}dB", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"DLG reconstruction of image #{ROUNDS_TARGET_INDEX} across FL rounds", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "dlg_rounds_comparison.png", dpi=150)
    plt.close(fig)

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
