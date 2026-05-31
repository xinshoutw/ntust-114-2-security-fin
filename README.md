# 聯邦學習：梯度洩漏攻擊與防禦

NTUST 114-2 資訊安全期末專題。從頭手刻 Federated Learning，示範梯度反演攻擊（DLG / iDLG），再用差分隱私與同態加密兩種防禦把攻擊擋回去。

Threat model：**honest-but-curious server**——伺服器忠實執行 FedAvg 聚合，但會試圖從 client 上傳的梯度還原其私有訓練影像。

## 總覽

「不上傳資料」不等於「不洩漏資料」。聯邦學習讓每個 client 只交出梯度、把原始人臉留在本地，看似就保護了隱私；但**單一梯度就足以反推出原始人臉**。本專案分四步把攻擊做出來、再擋回去：

- **Step 1 — 聯邦學習**：手刻 FedAvg，4 個 client、IID 切分，在 ORL 人臉上訓練一個 LeNet（DLG 論文版：Sigmoid + strided conv，約 38K 參數），收斂到 ~0.89 測試準確率，與 centralized baseline 相當。
- **Step 2 — 梯度反演攻擊**：以 **DLG / iDLG**（LBFGS 匹配梯度）從單一梯度還原人臉。用兩條軸觀察攻擊何時失效——**batch size**（`1 → 8`）與**訓練進度**（`round 1 → 50`）。
- **Step 3 — 差分隱私**：client 上傳前對 weight update 加高斯噪音（**DP-FedAvg**），掃描噪音強度 `σ ∈ {0, 0.01, 0.05, 0.1, 0.25, 0.5}`，量測隱私（DLG PSNR）與效用（準確率）之間的取捨。
- **Step 3-1 — 同態加密（Bonus）**：client 用 **TenSEAL CKKS** 加密 update，server 只在密文上做加法與 `× 1/N`，全程拿不到明文梯度，DLG 連目標函數都湊不出來。

四組實驗共用 `seed = 0` 的 train/test 切分與 client 分割，明文、DP、HE 的結果才能彼此對照。

---

## 示例圖

**Step 1 — 聯邦學習收斂**

| 準確率（FedAvg vs Centralized） | 損失 |
|:---:|:---:|
| ![fl accuracy](results/figures/fl_accuracy_curve.png) | ![fl loss](results/figures/fl_loss_curve.png) |

**Step 2 — 梯度反演攻擊**

從隨機雜訊開始，LBFGS 逐步把 dummy 影像逼近成原始人臉（iter 10 已認得出人、iter 30 後幾乎完美）：

![dlg progression](results/figures/dlg_progression.png)

| 未訓練模型：8 人各一張（皆完美還原，故上下兩列幾乎一模一樣） | 還原品質隨 batch size 崩潰 |
|:---:|:---:|
| ![dlg demo](results/figures/dlg_demo_comparison.png) | ![dlg batch](results/figures/dlg_batchsize_sweep.png) |

| 還原品質隨訓練進度衰退（PSNR / SSIM） | iDLG vs DLG 的收斂速度 |
|:---:|:---:|
| ![dlg rounds](results/figures/dlg_quality_vs_round.png) | ![dlg vs idlg](results/figures/dlg_vs_idlg.png) |

**Step 3 — 差分隱私**

| 隱私–效用權衡 | 不同 `σ` 下的還原 |
|:---:|:---:|
| ![dp tradeoff](results/figures/dp_tradeoff.png) | ![dp leakage](results/figures/dp_leakage_demo.png) |

**Step 3-1 — 同態加密**

| HE 防禦：server 只看得到密文 | 加密 vs 明文準確率 |
|:---:|:---:|
| ![he defense](results/figures/he_defense_demo.png) | ![he accuracy](results/figures/he_accuracy_comparison.png) |

每輪 CKKS 各階段耗時（encrypt / aggregate / decrypt）：

![he time](results/figures/he_time_breakdown.png)

> ORL / AT&T / Olivetti 是同一份人臉資料庫（40 人 × 10 張，64×64 灰階），首次執行時自動從 GitHub PNG 鏡像下載，整理成 `data/orl_faces/s1..s40`；載入時 resize 到 32×32 並做 z-score 正規化。

---

## 環境與執行

```bash
uv sync                                    # Python 3.12 + torch / tenseal / scikit-image ...

uv run python experiments/run_fl.py        # Step 1：FedAvg + centralized baseline
uv run python experiments/run_attack.py    # Step 2：DLG / iDLG 攻擊（需先跑 run_fl）
uv run python experiments/run_dp.py        # Step 3：差分隱私防禦
uv run python experiments/run_defense.py   # Step 3-1：CKKS 同態加密防禦
uv run pytest                              # FedAvg / metrics / DLG / DP / HE roundtrip
```

圖表輸出於 `results/figures/`、數據於 `results/metrics/`，皆已 commit 與團隊共享；資料集會自動下載、與虛擬環境一併排除在版控外。

## 重點數據

| 階段 | 結果 |
|------|------|
| FL 收斂 | FedAvg 0.89、centralized 0.88（相當）|
| DLG（未訓練模型） | 8/8 完美還原，平均 83.8 dB |
| DLG vs batch size | batch `1 → 8`：80 → 18 dB |
| DLG vs 訓練進度 | round 20 仍有 38 dB，round 25 起斷崖至 ~5 dB |
| DP 防禦 | `σ 0 → 0.5`：DLG 84 → 9 dB，準確率僅 0.89 → 0.86 |
| HE 收斂 | 加密 vs 明文準確率差 ≤ 0.0125 |
| HE 防禦 | server 無 secret key，`decrypt()` 直接拋例外，密文熵 7.97/8 bits/byte |
| HE 成本 | 密文 32.7× 明文；每輪 encrypt 0.17s / aggregate 0.03s / decrypt 0.01s |
