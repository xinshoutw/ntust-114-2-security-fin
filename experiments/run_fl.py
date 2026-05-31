"""Train the federated model and a centralized baseline, then save curves + metrics.

Usage:
    uv run python experiments/run_fl.py

Outputs:
    results/figures/fl_accuracy_curve.png
    results/figures/fl_loss_curve.png
    results/metrics/fl_training.csv
    results/fl_global_model.pt          final FedAvg weights
    results/fl_snapshots.pt             {round: weights} for the attack experiment
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

from src.federated import get_device, run_federated_learning, train_centralized

NUM_ROUNDS = 50
NUM_CLIENTS = 4
LOCAL_EPOCHS = 1
LR = 0.01
# Dense early sampling so the DLG leakage-vs-round curve resolves the "privacy
# cliff" (gradients stop leaking once the model leaves its high-curvature init).
SNAPSHOT_ROUNDS = (1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"


def main() -> None:
    device = get_device()
    print(f"[fl] device: {device}")

    print(f"[fl] running FedAvg for {NUM_ROUNDS} rounds across {NUM_CLIENTS} clients...")
    fl = run_federated_learning(
        num_rounds=NUM_ROUNDS,
        num_clients=NUM_CLIENTS,
        local_epochs=LOCAL_EPOCHS,
        lr=LR,
        device=device,
        snapshot_rounds=SNAPSHOT_ROUNDS,
    )

    print(f"[fl] running centralized baseline for {NUM_ROUNDS} epochs...")
    central = train_centralized(num_epochs=NUM_ROUNDS, lr=LR, device=device)

    fl_hist = pd.DataFrame(fl["history"]).rename(
        columns={"accuracy": "fl_accuracy", "loss": "fl_loss"}
    )
    central_hist = pd.DataFrame(central["history"]).rename(
        columns={"accuracy": "central_accuracy", "loss": "central_loss"}
    )
    merged = fl_hist.merge(central_hist, on="round")

    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS / "fl_training.csv"
    merged.to_csv(csv_path, index=False)
    print(f"[fl] wrote {csv_path}")

    # Accuracy curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(merged["round"], merged["fl_accuracy"], "-o", ms=3, label="Federated (FedAvg)")
    plt.plot(merged["round"], merged["central_accuracy"], "--s", ms=3, label="Centralized")
    plt.xlabel("Communication round / epoch")
    plt.ylabel("Test accuracy")
    plt.title("Federated vs centralized accuracy (ORL faces, 4 clients)")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fl_accuracy_curve.png", dpi=150)
    plt.close()

    # Loss curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(merged["round"], merged["fl_loss"], "-o", ms=3, label="Federated (FedAvg)")
    plt.plot(merged["round"], merged["central_loss"], "--s", ms=3, label="Centralized")
    plt.xlabel("Communication round / epoch")
    plt.ylabel("Test loss (cross-entropy)")
    plt.title("Federated vs centralized loss (ORL faces, 4 clients)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fl_loss_curve.png", dpi=150)
    plt.close()
    print(f"[fl] wrote figures to {FIGURES}")

    torch.save(fl["global_state"], RESULTS / "fl_global_model.pt")
    torch.save(fl["snapshots"], RESULTS / "fl_snapshots.pt")
    print(f"[fl] saved final model + {len(fl['snapshots'])} round snapshots")

    final = merged.iloc[-1]
    print(
        f"[fl] final accuracy: FL={final['fl_accuracy']:.4f}  "
        f"centralized={final['central_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
