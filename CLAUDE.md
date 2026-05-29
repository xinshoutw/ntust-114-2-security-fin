# Federated Learning: Attacks & Defenses

## Project

NTUST 114-2 資訊安全期末專題。手刻 Federated Learning + DLG 攻擊 + Homomorphic Encryption 防禦。

## Tech Stack

- Python 3.12, managed by `uv`
- PyTorch (MPS backend on Apple Silicon)
- TenSEAL (CKKS homomorphic encryption)
- matplotlib for figures

## Structure

```
src/           # all source modules
experiments/   # runnable experiment scripts
results/       # generated figures + metrics + model checkpoints (committed, shared with the team)
data/          # datasets (gitignored; auto-downloaded at runtime)
```

## Conventions

- Device: auto-detect MPS > CPU. TenSEAL ops always on CPU.
- All experiment scripts in `experiments/` are standalone entrypoints: `uv run python experiments/run_*.py`
- Figures saved to `results/figures/`, metrics CSV to `results/metrics/`
- Code comments in English, print/log output in English

## Key Constraints

- ORL Faces dataset, resized 32×32 grayscale
- CNN: 3-layer LeNet variant (~40K params) matching DLG paper
- FL: 4 clients, FedAvg, IID split
- DLG attack: LBFGS optimizer, 300 iterations
- HE: TenSEAL CKKS, poly_modulus_degree=8192

## Execution Order

Follow `PLAN.md` for detailed implementation steps.
