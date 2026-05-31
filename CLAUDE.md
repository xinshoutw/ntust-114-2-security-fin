# Federated Learning: Attacks & Defenses

## Project

NTUST 114-2 資訊安全期末專題。手刻 Federated Learning + DLG/iDLG 梯度反演攻擊 + 兩種防禦：Differential Privacy（DP-FedAvg，Step 3）與 Homomorphic Encryption（CKKS，Step 3-1 bonus）。四個 Step 全數實作。

## Tech Stack

- Python 3.12, managed by `uv`
- PyTorch (MPS backend on Apple Silicon)
- TenSEAL (CKKS homomorphic encryption)
- scikit-image (PSNR/SSIM), pandas (metrics CSVs), matplotlib for figures
- Self-contained RDP accountant for the DP privacy budget (no opacus dependency)

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

- ORL Faces dataset, resized 32×32 grayscale, **z-score normalized** (required — raw [0,1] pixels saturate the Sigmoid net onto a loss plateau that never trains)
- CNN: 3-conv LeNet variant (~38K params) matching the DLG paper — **Sigmoid** activations, **strided** convs (no pooling), raw logits, **DLG uniform(-0.5,0.5) init by default** (also needed for convergence)
- FL: 4 clients, FedAvg (sample-weighted), IID split, 50 rounds, Adam local optimizer
- DLG attack: LBFGS optimizer, 300 iterations; iDLG label inference; attacks the single-sample loss gradient (not the multi-step FedAvg delta). Attack success = PSNR > 20 dB; leakage-vs-round is reported as **attack success rate over several victims** (per-victim PSNR is bimodal — recovered ~50 dB or failed ~5 dB — so a mean PSNR misleads)
- Step 3 DP: DP-FedAvg — per-client L2 clip + Gaussian noise (std = z·C); privacy budget ε via RDP composition. No subsampling amplification, so ε stays large: the finding is that empirical privacy is cheap (z=0.01 defeats DLG at ~no accuracy cost) but a *formally* meaningful ε needs noise that has already collapsed accuracy to chance
- Step 3-1 HE: TenSEAL CKKS, poly_modulus_degree=8192; sample-weighted aggregation on ciphertext

## Execution Order

Follow `PLAN.md` for detailed implementation steps.
