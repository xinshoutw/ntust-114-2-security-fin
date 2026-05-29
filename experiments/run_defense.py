"""Evaluate the CKKS homomorphic-encryption defence against DLG.

Usage:
    uv run python experiments/run_defense.py

Three parts:
  A. Convergence - encrypted FedAvg vs the identical plaintext trajectory, to
     show CKKS does not hurt accuracy.
  B. Defence     - the server holds only ciphertext, so DLG has no plaintext
     gradient to invert; feeding the raw ciphertext bytes to the attack yields
     pure noise, next to the successful plaintext-gradient reconstruction.
  C. Trade-off   - per-round encrypt/aggregate/decrypt timing and the
     plaintext-vs-ciphertext communication blow-up.

Outputs:
    results/figures/he_accuracy_comparison.png
    results/figures/he_time_breakdown.png
    results/figures/he_defense_demo.png
    results/metrics/he_training.csv
    results/metrics/he_communication.csv
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import he_utils
from src.data_utils import denormalize, load_orl_dataset
from src.dlg_attack import compute_real_gradients, dlg_attack, idlg_label_inference
from src.federated import get_device, run_federated_learning_he
from src.metrics import compute_psnr, compute_ssim
from src.models import LeNet

HE_ROUNDS = 30
NUM_CLASSES = 40
DEMO_INDEX = 5

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"


def part_a_convergence(device):
    print(f"[defense] running encrypted FedAvg for {HE_ROUNDS} rounds...")
    he = run_federated_learning_he(num_rounds=HE_ROUNDS, device=device, encrypt=True, seed=0)
    print(f"[defense] running the identical plaintext trajectory for {HE_ROUNDS} rounds...")
    plain = run_federated_learning_he(num_rounds=HE_ROUNDS, device=device, encrypt=False, seed=0)

    he_hist = pd.DataFrame(he["history"])
    plain_hist = pd.DataFrame(plain["history"])

    plt.figure(figsize=(7, 4.5))
    plt.plot(he_hist["round"], he_hist["accuracy"], "-o", ms=3, label="Encrypted FedAvg (CKKS)")
    plt.plot(plain_hist["round"], plain_hist["accuracy"], "--s", ms=3, label="Plaintext FedAvg")
    plt.xlabel("Communication round")
    plt.ylabel("Test accuracy")
    plt.title("HE-protected vs plaintext FedAvg accuracy")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "he_accuracy_comparison.png", dpi=150)
    plt.close()

    max_gap = float((he_hist["accuracy"] - plain_hist["accuracy"]).abs().max())
    print(
        f"[defense] final acc: HE={he_hist['accuracy'].iloc[-1]:.4f} "
        f"plaintext={plain_hist['accuracy'].iloc[-1]:.4f} | max gap over rounds={max_gap:.4f}"
    )
    return he


def part_c_tradeoff(he):
    timing = pd.DataFrame(he["timing"])
    history = pd.DataFrame(he["history"])
    merged = history.merge(timing, on="round", how="left").fillna(0.0)
    METRICS.mkdir(parents=True, exist_ok=True)
    merged.to_csv(METRICS / "he_training.csv", index=False)

    comm = he["comm"]
    ratio = comm["ciphertext_bytes"] / comm["plaintext_bytes"]
    pd.DataFrame([{
        "plaintext_size_bytes": comm["plaintext_bytes"],
        "ciphertext_size_bytes": comm["ciphertext_bytes"],
        "ratio": ratio,
    }]).to_csv(METRICS / "he_communication.csv", index=False)
    print(
        f"[defense] communication per client update: plaintext={comm['plaintext_bytes']/1024:.0f} KB "
        f"ciphertext={comm['ciphertext_bytes']/1024:.0f} KB ({ratio:.1f}x)"
    )

    plt.figure(figsize=(8, 4.5))
    rounds = timing["round"]
    enc, agg, dec = timing["encrypt"], timing["aggregate"], timing["decrypt"]
    plt.bar(rounds, enc, label="encrypt (client)")
    plt.bar(rounds, agg, bottom=enc, label="aggregate (server)")
    plt.bar(rounds, dec, bottom=enc + agg, label="decrypt (client)")
    plt.xlabel("Communication round")
    plt.ylabel("Time (s)")
    plt.title("Per-round CKKS time breakdown")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES / "he_time_breakdown.png", dpi=150)
    plt.close()
    print(
        f"[defense] mean per-round time: encrypt={enc.mean():.3f}s "
        f"aggregate={agg.mean():.3f}s decrypt={dec.mean():.3f}s"
    )


def _ciphertext_surrogate(serialized, shapes):
    """Turn a parameter's raw ciphertext bytes into a same-shaped float tensor.

    This is the best an attacker can do with only ciphertext (no secret key):
    treat the encrypted payload as if it were the gradient. It is statistical
    noise, so DLG cannot reconstruct anything.
    """
    surrogate = []
    for name, shape in shapes.items():
        need = int(np.prod(shape))
        raw = np.frombuffer(serialized[name], dtype=np.uint8).astype(np.float32)
        reps = math.ceil(need / len(raw))
        arr = np.tile(raw, reps)[:need]
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)  # gradient-like scale
        surrogate.append(torch.tensor(arr, dtype=torch.float32).reshape(shape))
    return surrogate


def part_b_defense_demo():
    torch.manual_seed(0)
    full = load_orl_dataset()
    imgs, lbls = full.tensors
    mean, std = full.mean, full.std
    model = LeNet(NUM_CLASSES, dlg_init=True).eval()

    image = imgs[DEMO_INDEX : DEMO_INDEX + 1]
    label = lbls[DEMO_INDEX : DEMO_INDEX + 1]
    orig01 = denormalize(image, mean, std)

    # Unprotected: the server has the plaintext gradient -> DLG succeeds.
    real_grads = compute_real_gradients(model, image, label)
    inferred = idlg_label_inference(real_grads, NUM_CLASSES)
    rec_plain, _, _ = dlg_attack(
        model, real_grads, tuple(image.shape), (1, NUM_CLASSES),
        num_iterations=300, device="cpu", known_label=inferred,
    )
    psnr_plain = compute_psnr(orig01, denormalize(rec_plain, mean, std))

    # Protected: encrypt the gradient; the server only sees ciphertext bytes.
    state = model.state_dict()
    grad_dict = {name: g for name, g in zip(state.keys(), real_grads)}
    ctx = he_utils.create_he_context()
    serialized = he_utils.serialize_encrypted(he_utils.encrypt_gradients(grad_dict, ctx))
    surrogate = _ciphertext_surrogate(serialized, he_utils.get_shapes(state))
    rec_he, _, _ = dlg_attack(
        model, surrogate, tuple(image.shape), (1, NUM_CLASSES),
        num_iterations=300, device="cpu", known_label=inferred,
    )
    psnr_he = compute_psnr(orig01, denormalize(rec_he, mean, std))
    print(f"[defense] DLG PSNR: plaintext-gradient={psnr_plain:.1f}dB  on-ciphertext={psnr_he:.1f}dB")

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    for ax, img, title in zip(
        axes,
        [orig01, denormalize(rec_plain, mean, std), denormalize(rec_he, mean, std)],
        ["original", f"no defence\nDLG on gradient\n{psnr_plain:.0f} dB",
         f"HE defence\nDLG on ciphertext\n{psnr_he:.0f} dB"],
    ):
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Homomorphic encryption defeats gradient leakage", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "he_defense_demo.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"[defense] device: {device} (HE always runs on CPU)")

    print("[defense] === A. convergence: encrypted vs plaintext ===")
    he = part_a_convergence(device)
    print("[defense] === C. trade-off: time + communication ===")
    part_c_tradeoff(he)
    print("[defense] === B. defence demo: DLG on plaintext vs ciphertext ===")
    part_b_defense_demo()
    print(f"[defense] wrote figures to {FIGURES} and metrics to {METRICS}")


if __name__ == "__main__":
    main()
