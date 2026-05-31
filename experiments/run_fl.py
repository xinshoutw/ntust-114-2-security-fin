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
# The attack reconstructs from these runs' snapshots; we now keep ALL seeds (not
# just seed 0) so the leakage-vs-round success rate can be averaged over seeds and
# its early-round jitter (a single-seed artefact) smoothed out.
SNAPSHOT_SEEDS = SEEDS
# Dense early sampling so the DLG leakage-vs-round curve resolves the "privacy
# cliff" (gradients stop leaking once the model leaves its high-curvature init).
SNAPSHOT_ROUNDS = (1, 2, 4, 6, 8, 10, 12, 15, 18, 20, 22, 25, 30, 40, 50)

# --- Non-IID client-drift experiment (Dirichlet label skew + partial participation) ---
# The old 4-client, full-participation block split barely moved (drift needs scale).
# We now use more clients with partial participation and a *tunable* Dirichlet(alpha)
# label skew, and plot per-round convergence curves (not just final accuracy), which
# is where drift actually shows up: smaller alpha => slower, noisier convergence and a
# real final-accuracy gap.
NONIID_CLIENTS = 10
NONIID_SAMPLE_RATE = 0.5  # 5 of 10 clients per round
NONIID_LOCAL_EPOCHS = 5
# Each entry: (label, split, dirichlet_alpha). alpha is ignored unless split=dirichlet.
NONIID_SETTINGS = (
    ("IID", "iid", None),
    ("Dirichlet alpha=1.0 (mild)", "dirichlet", 1.0),
    ("Dirichlet alpha=0.1 (severe)", "dirichlet", 0.1),
)

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
    snapshots_by_seed: dict[int, dict] = {}
    global_state = None
    for seed in SEEDS:
        snap_rounds = SNAPSHOT_ROUNDS if seed in SNAPSHOT_SEEDS else ()
        fl = run_federated_learning(
            num_rounds=NUM_ROUNDS, num_clients=NUM_CLIENTS, local_epochs=LOCAL_EPOCHS,
            lr=LR, device=device, seed=seed, snapshot_rounds=snap_rounds, verbose=False,
        )
        central = train_centralized(num_epochs=NUM_ROUNDS, lr=LR, device=device, seed=seed, verbose=False)
        fl_histories.append(fl["history"])
        central_histories.append(central["history"])
        if seed in SNAPSHOT_SEEDS:
            snapshots_by_seed[seed] = fl["snapshots"]
        if seed == SEEDS[0]:
            global_state = fl["global_state"]
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
    torch.save(snapshots_by_seed, RESULTS / "fl_snapshots.pt")
    n_snap = sum(len(s) for s in snapshots_by_seed.values())
    print(f"[fl] saved seed-{SEEDS[0]} model + {n_snap} round snapshots "
          f"across {len(snapshots_by_seed)} seeds (for the multi-seed leakage curve)")
    print(f"[fl] convergence: FL={fl_acc_m[-1]:.4f}+/-{fl_acc_s[-1]:.4f}  "
          f"centralized={c_acc_m[-1]:.4f}+/-{c_acc_s[-1]:.4f}")
    return df


def run_noniid(device) -> pd.DataFrame:
    """IID vs Dirichlet non-IID FedAvg *convergence curves* under partial participation.

    With many clients (NONIID_CLIENTS) and partial participation (NONIID_SAMPLE_RATE),
    a tunable Dirichlet(alpha) label skew makes client drift visible: smaller alpha
    gives a slower, noisier convergence and a lower final accuracy. We plot the
    per-round curves (3-seed mean +/- std) -- the gap and the instability are the
    point, not just the endpoint.
    """
    colors = ["tab:blue", "tab:orange", "tab:red"]
    curves = []  # (label, rounds, acc_mean, acc_std, final_mean, final_std)
    for (label, split, alpha) in NONIID_SETTINGS:
        histories = []
        for s in SEEDS:
            res = run_federated_learning(
                num_rounds=NUM_ROUNDS, num_clients=NONIID_CLIENTS,
                local_epochs=NONIID_LOCAL_EPOCHS, lr=LR, device=device, seed=s,
                split=split, dirichlet_alpha=(alpha or 0.5),
                client_sample_rate=NONIID_SAMPLE_RATE, verbose=False,
            )
            histories.append(res["history"])
        rounds = [h["round"] for h in histories[0]]
        acc_m, acc_s = _stack(histories, "accuracy")
        curves.append((label, rounds, acc_m, acc_s, acc_m[-1], acc_s[-1]))
        print(f"[fl] non-IID: {label:28s} final acc={acc_m[-1]:.4f}+/-{acc_s[-1]:.4f}")

    # Per-round CSV: one accuracy mean/std column per setting.
    data = {"round": curves[0][1]}
    for (label, _, acc_m, acc_s, _, _) in curves:
        key = label.split()[0].lower() if label != "IID" else "iid"
        if "0.1" in label:
            key = "dirichlet_a0.1"
        elif "1.0" in label:
            key = "dirichlet_a1.0"
        data[f"{key}_acc_mean"] = acc_m
        data[f"{key}_acc_std"] = acc_s
    df = pd.DataFrame(data)
    df.to_csv(METRICS / "fl_noniid.csv", index=False)
    print(f"[fl] wrote {METRICS / 'fl_noniid.csv'}")

    plt.figure(figsize=(7.5, 4.8))
    for (label, rounds, acc_m, acc_s, fm, fs), c in zip(curves, colors):
        plt.plot(rounds, acc_m, "-o", ms=2.5, color=c, label=f"{label}  (final {fm:.2f})")
        plt.fill_between(rounds, acc_m - acc_s, acc_m + acc_s, color=c, alpha=0.15)
    plt.xlabel("Communication round")
    plt.ylabel("Test accuracy (mean +/- std, 3 seeds)")
    plt.title(f"Client drift: IID vs Dirichlet non-IID\n"
              f"({NONIID_CLIENTS} clients, {NONIID_SAMPLE_RATE:.0%} sampled/round, "
              f"E={NONIID_LOCAL_EPOCHS} local epochs)")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
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
