# Federated Learning: Attacks & Defenses

NTUST 114-2 資訊安全期末專題。從頭手刻 Federated Learning，示範 Deep Leakage
from Gradients (DLG) 梯度反演攻擊，並以 TenSEAL CKKS 同態加密做防禦。

Threat model: **honest-but-curious server** — 伺服器忠實執行聚合，但會嘗試從
client 上傳的梯度還原其私有訓練影像。

## Pipeline

1. **Federated learning** — 4 個 client、IID 切分、FedAvg，在 ORL 人臉資料集上
   訓練一個 ~38K 參數的 LeNet（DLG paper 版本：Sigmoid + strided conv）。
2. **Gradient-leakage attack** — iDLG 以 LBFGS 匹配梯度，從單一梯度還原人臉。
3. **HE defense** — client 先用 CKKS 加密 weight update 再上傳，伺服器只在密文上
   做同態平均，全程接觸不到明文梯度。

## Setup

```bash
uv sync                       # Python 3.12 + torch / tenseal / scikit-image ...
```

ORL/AT&T/Olivetti 人臉資料集會在第一次執行時自動下載（原始鏡像若失效，改從 GitHub
PNG 鏡像取得同一份資料），整理成 `data/orl_faces/s1..s40`。

## Run

```bash
uv run python experiments/run_fl.py        # FedAvg + centralized baseline
uv run python experiments/run_attack.py    # DLG/iDLG leakage (needs run_fl first)
uv run python experiments/run_defense.py    # CKKS defense + trade-off analysis
uv run pytest                              # FedAvg / metrics / iDLG / HE roundtrip
```

## Results

| 階段 | 指標 | 結果 |
|------|------|------|
| FL 收斂 | 測試準確率 (50 rounds) | FedAvg **0.89**，centralized baseline 0.88 |
| DLG 攻擊（未訓練模型） | PSNR / 成功率 (>20 dB) | 平均 **83.8 dB**，**8/8** 張完美還原 |
| DLG vs 訓練進度 | image #5 PSNR | round 1: 60 dB → round 25: 5 dB（收斂後梯度幾乎不洩漏）|
| HE 收斂性 | 加密 vs 明文準確率差 | 全程 ≤ **0.0125**（CKKS 精度損失可忽略）|
| HE 防禦 | DLG on 明文 vs 密文 | 84 dB（成功）vs **5 dB**（純噪音）|
| HE 成本 | 通訊量 / 每輪耗時 | 密文 **32.7×** 明文；encrypt 0.17s / aggregate 0.03s / decrypt 0.01s |

圖表輸出於 `results/figures/`，數據於 `results/metrics/`（皆 gitignored）。

## Key implementation notes

- **架構選擇**：採用 DLG paper 的 LeNet（Sigmoid、strided conv、~38K 參數）。
  Sigmoid 的二階導數平滑，是 LBFGS 反演能成功的關鍵；ReLU 會讓重建失敗。
- **訓練可收斂性**：純 Sigmoid 網路餵原始 `[0,1]` 像素會卡在損失平台，需把輸入
  z-score 正規化並用 DLG uniform 初始化，才能順利訓練到 ~88%。
- **CKKS 切片**：`poly_modulus_degree=8192` 只有 4096 slots，但 TenSEAL 會自動把
  超長向量拆成多個密文，故每個參數直接加密成單一 `ckks_vector` 即可。
- **裝置**：FL 訓練用 MPS，DLG（LBFGS）與所有 TenSEAL 運算固定在 CPU。

詳見 `PLAN.md`。
