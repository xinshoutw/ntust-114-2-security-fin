# Federated Learning: Attacks & Defenses

## Project

NTUST 114-2 資訊安全期末專題。手刻 Federated Learning + DLG/iDLG 梯度反演攻擊 + 兩種防禦：Differential Privacy（DP-FedAvg，Step 3）與 Homomorphic Encryption（CKKS，Step 3-1 bonus）。四個 Step 全數實作。

## Tech Stack

- Python 3.12, managed by `uv`
- PyTorch (MPS backend on Apple Silicon)
- TenSEAL (CKKS homomorphic encryption)
- scikit-image (PSNR/SSIM), pandas (metrics CSVs), matplotlib for figures
- Self-contained RDP accountant for the DP privacy budget — plain Gaussian RDP and the subsampled-Gaussian (DP-SGD) accountant (Mironov–Talwar–Zhang), no opacus dependency (uses scipy.special)
- `torch.func` (vmap+grad) for per-example gradients in DP-SGD

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
- FL: 4 clients, FedAvg (sample-weighted), IID split, 50 rounds, Adam local optimizer. Also supports `split="dirichlet"` (tunable α label skew) and partial participation (`client_sample_rate`) for the non-IID drift experiment, which uses 10 clients @ 50% participation: IID ~0.91 vs Dirichlet α=0.1 ~0.80 with 3× the variance (the old 4-client/full-participation block split barely moved, so it was redesigned)
- DLG attack: LBFGS optimizer, 300 iterations; iDLG label inference; the headline attacks invert the single-sample loss gradient (the leakage upper bound, not the multi-step FedAvg delta). `run_real_delta` closes that gap: a 1-sample/1-SGD-step upload is `delta = -lr·g`, inverted near-perfectly (~real proof the upload leaks); the multi-step Adam delta a real client sends resists naive inversion. Attack success = PSNR > 20 dB; leakage-vs-round is **attack success rate pooled over victims × 3 snapshot seeds** (per-victim PSNR is bimodal — ~50 dB or ~5 dB — so a mean misleads; seed-averaging removes the early-round jitter)
- Step 3 DP, two complementary mechanisms:
  - **Update-level DP-FedAvg** (`run_dp.py`): per-client L2 clip + Gaussian noise on the aggregate delta; plain-RDP ε. No subsampling amplification → ε vacuous (≥59) and accuracy cliffs to chance. Finding: empirical privacy is cheap (z≈0.002 defeats DLG at ~no accuracy cost) but a *formally* meaningful ε is unreachable at usable accuracy.
  - **Record-level DP-SGD** (`run_dp_sgd.py`, the Abadi mechanism): per-example gradient clipping + Gaussian noise + subsampled-Gaussian RDP. Degrades *gracefully* (acc ~0.88→~0.60 as ε falls ~2800→~300) — the mechanism matters — but a small ε (<10) still costs most of the accuracy; subsampling lowers ε only by trading away accuracy (curse of dimensionality on a 38K-param model with 320 images). Sharper, properly-accounted confirmation of the same conclusion, contrasted with HE.
- Step 3-1 HE: TenSEAL CKKS, poly_modulus_degree=8192; sample-weighted aggregation on ciphertext; CKKS rounding error vs plaintext aggregation ~1e-7 (why accuracy is untouched)

## Execution Order

Follow `PLAN.md` for detailed implementation steps.
