"""Step 3: differential-privacy (Gaussian-noise) defence and its privacy-utility trade-off.

Usage:
    uv run python experiments/run_dp.py

A lighter alternative to homomorphic encryption: the client adds calibrated
Gaussian noise to its update before upload (DP-FedAvg). The server still sees a
plaintext gradient, but a noisy one. Sweeping the noise level ``sigma`` (noise
std as a fraction of the update's RMS magnitude; see :mod:`src.dp_utils`) traces
the classic trade-off:

  * Utility  - final FedAvg test accuracy drops as sigma grows.
  * Privacy  - DLG reconstruction quality (PSNR/SSIM) drops as the *same*
               relative noise is added to the gradient the attacker inverts.

Both curves share one comparable x-axis, so the figure shows where noise starts
to buy privacy and where it starts to cost accuracy.

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
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils import denormalize, load_orl_dataset
from src.dlg_attack import compute_real_gradients, dlg_attack, idlg_label_inference
from src.dp_utils import add_relative_gaussian_noise_list
from src.federated import get_device, run_federated_learning
from src.metrics import compute_psnr, compute_ssim
from src.models import LeNet

NUM_ROUNDS = 50
NUM_CLASSES = 40
NUM_ITERS = 300
TARGET_INDEX = 5
SIGMAS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5)

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"


def utility_sweep(device) -> dict[float, float]:
    """Final FedAvg test accuracy for each noise level."""
    acc = {}
    for sigma in SIGMAS:
        out = run_federated_learning(
            num_rounds=NUM_ROUNDS, device=device, dp_sigma=sigma, verbose=False
        )
        acc[sigma] = out["history"][-1]["accuracy"]
        print(f"[dp] utility  sigma={sigma:<5}: final accuracy={acc[sigma]:.4f}")
    return acc


def privacy_sweep(imgs, lbls, mean, std) -> tuple[dict, dict]:
    """DLG reconstruction quality and image for each noise level (round-0 model)."""
    torch.manual_seed(0)
    model = LeNet(NUM_CLASSES, dlg_init=True).eval()
    image = imgs[TARGET_INDEX : TARGET_INDEX + 1]
    label = lbls[TARGET_INDEX : TARGET_INDEX + 1]
    orig01 = denormalize(image, mean, std)
    clean_grads = compute_real_gradients(model, image, label)
    inferred = idlg_label_inference(clean_grads, NUM_CLASSES)

    quality, recon = {}, {}
    for sigma in SIGMAS:
        gen = torch.Generator().manual_seed(0)
        noisy = add_relative_gaussian_noise_list(clean_grads, sigma, generator=gen)
        rec, _, _ = dlg_attack(
            model, noisy, tuple(image.shape), (1, NUM_CLASSES),
            num_iterations=NUM_ITERS, device="cpu", known_label=inferred,
        )
        rec01 = denormalize(rec, mean, std)
        quality[sigma] = {
            "psnr": compute_psnr(orig01, rec01),
            "ssim": compute_ssim(orig01, rec01),
        }
        recon[sigma] = rec01
        print(f"[dp] privacy  sigma={sigma:<5}: DLG psnr={quality[sigma]['psnr']:5.1f}dB "
              f"ssim={quality[sigma]['ssim']:.3f}")
    return quality, recon, orig01


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"[dp] device: {device}")

    print("[dp] === utility: FedAvg accuracy vs noise ===")
    acc = utility_sweep(device)
    print("[dp] === privacy: DLG leakage vs noise ===")
    full = load_orl_dataset()
    imgs, lbls = full.tensors
    quality, recon, orig01 = privacy_sweep(imgs, lbls, full.mean, full.std)

    rows = [
        {
            "sigma": s,
            "final_accuracy": acc[s],
            "dlg_psnr": quality[s]["psnr"],
            "dlg_ssim": quality[s]["ssim"],
        }
        for s in SIGMAS
    ]
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "dp_tradeoff.csv", index=False)

    # Trade-off figure: accuracy and leakage on one shared sigma axis.
    x = list(range(len(SIGMAS)))
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(x, df["final_accuracy"], "-o", color="tab:green", label="FedAvg accuracy")
    ax1.set_xlabel("DP noise level sigma (noise std / update RMS)")
    ax1.set_ylabel("Final test accuracy", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in SIGMAS])
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, df["dlg_psnr"], "-s", color="tab:purple", label="DLG PSNR (leakage)")
    ax2.axhline(20.0, ls="--", color="gray", lw=1)
    ax2.set_ylabel("DLG reconstruction PSNR (dB)", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    fig.suptitle("DP trade-off: more noise blunts DLG leakage but lowers accuracy", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "dp_tradeoff.png", dpi=150)
    plt.close(fig)

    # Visual: the victim image reconstructed under growing noise.
    fig, axes = plt.subplots(1, len(SIGMAS) + 1, figsize=(1.7 * (len(SIGMAS) + 1), 2.5))
    axes[0].imshow(orig01.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for ax, s in zip(axes[1:], SIGMAS):
        ax.imshow(recon[s].squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"sigma={s}\n{quality[s]['psnr']:.0f}dB", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"DLG reconstruction of image #{TARGET_INDEX} vs DP noise", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "dp_leakage_demo.png", dpi=150)
    plt.close(fig)

    print(f"[dp] wrote trade-off figures to {FIGURES} and metrics to {METRICS}")


if __name__ == "__main__":
    main()
