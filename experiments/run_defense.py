"""Evaluate the CKKS homomorphic-encryption defence against DLG.

Usage:
    uv run python experiments/run_defense.py

Three parts:
  A. Convergence - encrypted FedAvg vs the identical plaintext trajectory, to
     show CKKS does not hurt accuracy.
  B. Defence     - the *structural* argument: the server holds only ciphertext
     under a public (secret-key-free) context, so it cannot decrypt the gradient
     and therefore cannot even form the DLG gradient-matching objective. We show
     the successful plaintext-gradient reconstruction beside what the server
     actually holds under HE (a high-entropy ciphertext blob) and confirm its
     decrypt() raises.
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
from src.metrics import compute_psnr
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


def _ciphertext_preview(serialized: dict[str, bytes], side: int = 32) -> np.ndarray:
    """Render the server's actual HE view -- raw ciphertext bytes -- as an image.

    This is *all* an honest-but-curious server holds: a high-entropy byte blob.
    It is shown only to make "the server sees ciphertext, not a gradient"
    concrete; it is NOT fed to DLG (the server cannot decrypt it, so there is no
    gradient for DLG to invert in the first place).
    """
    blob = b"".join(serialized.values())
    arr = np.frombuffer(blob[: side * side], dtype=np.uint8).astype(np.float32) / 255.0
    return arr.reshape(side, side)


def _byte_entropy(serialized: dict[str, bytes]) -> float:
    """Shannon entropy of the ciphertext in bits/byte (8.0 = perfectly uniform)."""
    blob = np.frombuffer(b"".join(serialized.values()), dtype=np.uint8)
    counts = np.bincount(blob, minlength=256).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def part_b_defense_demo():
    """Show the HE defence as it actually works: the server never gets the gradient.

    No defence  -> the server holds the plaintext gradient and DLG reconstructs
                   the victim's face.
    HE defence  -> the client encrypts the gradient; the server's public context
                   has no secret key, so decrypt() raises and the server only
                   ever holds a high-entropy ciphertext blob. With no plaintext
                   gradient, the DLG objective grad_diff = ||g_dummy - g_real||^2
                   cannot even be formed. The defence is structural, not "DLG ran
                   on ciphertext and happened to fail".
    """
    torch.manual_seed(0)
    full = load_orl_dataset()
    imgs, lbls = full.tensors
    mean, std = full.mean, full.std
    model = LeNet(NUM_CLASSES, dlg_init=True).eval()

    image = imgs[DEMO_INDEX : DEMO_INDEX + 1]
    label = lbls[DEMO_INDEX : DEMO_INDEX + 1]
    orig01 = denormalize(image, mean, std)

    # --- No defence: the server has the plaintext gradient -> DLG succeeds. ---
    real_grads = compute_real_gradients(model, image, label)
    inferred = idlg_label_inference(real_grads, NUM_CLASSES)
    rec_plain, _, _ = dlg_attack(
        model, real_grads, tuple(image.shape), (1, NUM_CLASSES),
        num_iterations=300, device="cpu", known_label=inferred,
    )
    psnr_plain = compute_psnr(orig01, denormalize(rec_plain, mean, std))

    # --- HE defence: client encrypts; server gets a public (no-secret-key) ctx. ---
    state = model.state_dict()
    grad_dict = {name: g for name, g in zip(state.keys(), real_grads)}
    full_ctx = he_utils.create_he_context()
    public_ctx = he_utils.create_public_context(full_ctx)
    serialized = he_utils.serialize_encrypted(he_utils.encrypt_gradients(grad_dict, full_ctx))

    # The server links the ciphertext to its public context and tries to read it.
    on_server = he_utils.deserialize_encrypted(serialized, public_ctx)
    try:
        on_server[next(iter(on_server))].decrypt()
        server_can_decrypt, decrypt_error = True, "none"
    except Exception as exc:  # noqa: BLE001 - exact type is TenSEAL-internal
        server_can_decrypt, decrypt_error = False, type(exc).__name__

    entropy = _byte_entropy(serialized)
    print(
        f"[defense] no-defence DLG PSNR={psnr_plain:.1f}dB | "
        f"server can decrypt under HE? {server_can_decrypt} ({decrypt_error}) | "
        f"ciphertext entropy={entropy:.2f}/8.00 bits/byte"
    )
    print(
        "[defense] => structural defence: no secret key -> no plaintext gradient "
        "-> the DLG gradient-matching objective cannot be formed."
    )

    preview = _ciphertext_preview(serialized)
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.1))
    panels = [
        (orig01.squeeze(), "original\n(victim image)"),
        (
            denormalize(rec_plain, mean, std).squeeze(),
            f"NO defence\nserver holds plaintext grad\nDLG succeeds: {psnr_plain:.0f} dB",
        ),
        (
            preview,
            "HE defence\nserver's entire view = ciphertext\n"
            f"decrypt() raises ({decrypt_error})\nentropy {entropy:.1f}/8 bits",
        ),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        "HE defeats gradient leakage by withholding the plaintext gradient", fontsize=11
    )
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
