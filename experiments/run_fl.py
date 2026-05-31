"""Train the federated model and a centralized baseline, then save curves + metrics.

Usage:
    uv run python experiments/run_fl.py

Three things are produced:
  * Convergence  - FedAvg (IID) vs centralized, averaged over several seeds so the
    curves carry a mean +/- std band (a single seed makes the two indistinguishable
    and can even put FedAvg above centralized by test-set noise).
  * Non-IID      - FedAvg under an IID split vs a pathological label-partition split
    (each client owns a disjoint block of subjects), swept over the number of local
    epochs. This surfaces client drift: at one local epoch the two splits are
    indistinguishable, and the IID-minus-non-IID gap only opens up with more local
    steps (~5 pp by E=20), since each client overfits its own subjects before the
    averages reconcile. The effect is modest here because there are only 4 clients
    with full participation; the dramatic non-IID collapse needs many clients.
  * Snapshots    - the seed-0 IID run's weights at the snapshot rounds, saved for
    the DLG leakage-vs-round experiment (run_attack.py).

Outputs:
    results/figures/fl_accuracy_curve.png
    results/figures/fl_loss_curve.png
    results/figures/fl_noniid_comparison.png
    results/metrics/fl_training.csv
    results/metrics/fl_noniid.csv
    results/fl_global_model.pt          final FedAvg weights (seed 0, IID)
    results/fl_snapshots.pt             {round: weights} for the attack experiment
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

from src.federated import get_device, run_federated_learning, train_centralized

NUM_ROUNDS = 50
NUM_CLIENTS = 4
LOCAL_EPOCHS = 1
LR = 0.01
SEEDS = (0, 1, 2)  # average the convergence curves over seeds for a mean +/- std band
SNAPSHOT_SEED = 0  # the attack reconstructs from this run's snapshots
# Dense early sampling so the DLG leakage-vs-round curve resolves the "privacy
# cliff" (gradients stop leaking once the model leaves its high-curvature init).
SNAPSHOT_ROUNDS = (1, 2, 4, 6, 8, 10, 12, 15, 18, 20, 22, 25, 30, 40, 50)
# Local-epoch settings for the non-IID drift sweep. Drift only bites once clients
# take many local steps between rounds: at E=1 the IID and non-IID splits are
# indistinguishable, and the gap opens up only by E=10-20 (each client overfits
# to its own disjoint subjects before the averages are reconciled).
NONIID_LOCAL_EPOCHS = (1, 5, 10, 20)

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"


def _stack(histories: list[list[dict]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Stack a per-seed history list into (mean, std) arrays over rounds."""
    mat = np.array([[h[key] for h in hist] for hist in histories])
    return mat.mean(axis=0), mat.std(axis=0)


def run_convergence(device) -> pd.DataFrame:
    """FedAvg (IID) vs centralized over several seeds; also save seed-0 snapshots."""
    fl_histories, central_histories = [], []
    snapshots, global_state = None, None
    for seed in SEEDS:
        snap_rounds = SNAPSHOT_ROUNDS if seed == SNAPSHOT_SEED else ()
        fl = run_federated_learning(
            num_rounds=NUM_ROUNDS, num_clients=NUM_CLIENTS, local_epochs=LOCAL_EPOCHS,
            lr=LR, device=device, seed=seed, snapshot_rounds=snap_rounds, verbose=False,
        )
        central = train_centralized(num_epochs=NUM_ROUNDS, lr=LR, device=device, seed=seed, verbose=False)
        fl_histories.append(fl["history"])
        central_histories.append(central["history"])
        if seed == SNAPSHOT_SEED:
            snapshots, global_state = fl["snapshots"], fl["global_state"]
        print(f"[fl] seed {seed}: FL={fl['history'][-1]['accuracy']:.4f}  "
              f"centralized={central['history'][-1]['accuracy']:.4f}")

    rounds = [h["round"] for h in fl_histories[0]]
    fl_acc_m, fl_acc_s = _stack(fl_histories, "accuracy")
    fl_loss_m, fl_loss_s = _stack(fl_histories, "loss")
    c_acc_m, c_acc_s = _stack(central_histories, "accuracy")
    c_loss_m, c_loss_s = _stack(central_histories, "loss")

    df = pd.DataFrame({
        "round": rounds,
        "fl_accuracy_mean": fl_acc_m, "fl_accuracy_std": fl_acc_s,
        "fl_loss_mean": fl_loss_m, "fl_loss_std": fl_loss_s,
        "central_accuracy_mean": c_acc_m, "central_accuracy_std": c_acc_s,
        "central_loss_mean": c_loss_m, "central_loss_std": c_loss_s,
    })
    df.to_csv(METRICS / "fl_training.csv", index=False)
    print(f"[fl] wrote {METRICS / 'fl_training.csv'}")

    # Accuracy curve with mean +/- std bands.
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, fl_acc_m, "-o", ms=3, color="tab:blue", label="Federated (FedAvg, IID)")
    plt.fill_between(rounds, fl_acc_m - fl_acc_s, fl_acc_m + fl_acc_s, color="tab:blue", alpha=0.2)
    plt.plot(rounds, c_acc_m, "--s", ms=3, color="tab:orange", label="Centralized")
    plt.fill_between(rounds, c_acc_m - c_acc_s, c_acc_m + c_acc_s, color="tab:orange", alpha=0.2)
    plt.xlabel("Communication round / epoch")
    plt.ylabel("Test accuracy")
    plt.title(f"Federated vs centralized accuracy (ORL faces, 4 clients, {len(SEEDS)} seeds)")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fl_accuracy_curve.png", dpi=150)
    plt.close()

    # Loss curve with mean +/- std bands.
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, fl_loss_m, "-o", ms=3, color="tab:blue", label="Federated (FedAvg, IID)")
    plt.fill_between(rounds, fl_loss_m - fl_loss_s, fl_loss_m + fl_loss_s, color="tab:blue", alpha=0.2)
    plt.plot(rounds, c_loss_m, "--s", ms=3, color="tab:orange", label="Centralized")
    plt.fill_between(rounds, c_loss_m - c_loss_s, c_loss_m + c_loss_s, color="tab:orange", alpha=0.2)
    plt.xlabel("Communication round / epoch")
    plt.ylabel("Test loss (cross-entropy)")
    plt.title(f"Federated vs centralized loss (ORL faces, 4 clients, {len(SEEDS)} seeds)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fl_loss_curve.png", dpi=150)
    plt.close()
    print(f"[fl] wrote convergence figures to {FIGURES}")

    torch.save(global_state, RESULTS / "fl_global_model.pt")
    torch.save(snapshots, RESULTS / "fl_snapshots.pt")
    print(f"[fl] saved seed-{SNAPSHOT_SEED} model + {len(snapshots)} round snapshots")
    print(f"[fl] convergence: FL={fl_acc_m[-1]:.4f}+/-{fl_acc_s[-1]:.4f}  "
          f"centralized={c_acc_m[-1]:.4f}+/-{c_acc_s[-1]:.4f}")
    return df


def run_noniid(device) -> pd.DataFrame:
    """IID vs non-IID FedAvg final accuracy across local-epoch counts (client drift)."""
    rows = []
    for le in NONIID_LOCAL_EPOCHS:
        for split in ("iid", "noniid"):
            accs = [
                run_federated_learning(
                    num_rounds=NUM_ROUNDS, num_clients=NUM_CLIENTS, local_epochs=le,
                    lr=LR, device=device, seed=s, split=split, verbose=False,
                )["history"][-1]["accuracy"]
                for s in SEEDS
            ]
            rows.append({
                "local_epochs": le, "split": split,
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            })
            print(f"[fl] non-IID sweep: E={le} {split:6s} "
                  f"acc={rows[-1]['acc_mean']:.4f}+/-{rows[-1]['acc_std']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "fl_noniid.csv", index=False)
    print(f"[fl] wrote {METRICS / 'fl_noniid.csv'}")

    iid = df[df["split"] == "iid"].set_index("local_epochs")
    non = df[df["split"] == "noniid"].set_index("local_epochs")
    epochs = list(NONIID_LOCAL_EPOCHS)
    x = np.arange(len(epochs))
    w = 0.38
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - w / 2, iid.loc[epochs, "acc_mean"], w, yerr=iid.loc[epochs, "acc_std"],
            capsize=4, color="tab:blue", label="IID split")
    plt.bar(x + w / 2, non.loc[epochs, "acc_mean"], w, yerr=non.loc[epochs, "acc_std"],
            capsize=4, color="tab:red", label="Non-IID split (disjoint subjects)")
    plt.xticks(x, [f"E={e}" for e in epochs])
    plt.xlabel("Local epochs per round (more local steps => more client drift)")
    plt.ylabel("Final test accuracy (mean +/- std, 3 seeds)")
    plt.title("IID vs non-IID FedAvg: the gap widens with local epochs (client drift)")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fl_noniid_comparison.png", dpi=150)
    plt.close()
    print(f"[fl] wrote {FIGURES / 'fl_noniid_comparison.png'}")
    return df


def main() -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"[fl] device: {device}")

    print(f"[fl] === convergence: FedAvg(IID) vs centralized, {len(SEEDS)} seeds ===")
    run_convergence(device)
    print("[fl] === non-IID: client drift vs local epochs ===")
    run_noniid(device)


if __name__ == "__main__":
    main()
