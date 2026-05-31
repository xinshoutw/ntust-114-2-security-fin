# PLAN.md — Execution Plan

## Overview

四個 Step，每個 Step 產出程式碼 + 實驗結果（圖表與 CSV）：

- **Step 1**: Federated Learning 系統（FedAvg, 人臉辨識）
- **Step 2**: Gradient Leakage Attack（DLG / iDLG）
- **Step 3**: Differential Privacy 防禦（DP-FedAvg：裁剪 + 高斯噪音，以 ε 量化）
- **Step 3-1**: Homomorphic Encryption 防禦（TenSEAL CKKS, Bonus）

Threat Model：**Honest-but-curious Server** — Server 忠實執行 FedAvg 聚合，但會嘗試從 client 上傳的更新還原其私有訓練影像。Step 3 / 3-1 的目標是讓這個還原失效（DP：擾動到無法反演；HE：根本拿不到明文）。

---

## Phase 0: Project Setup

```bash
uv init && uv python pin 3.12
uv add torch torchvision matplotlib scikit-image tenseal pandas numpy
```

目錄：`src/`（模組）、`experiments/`（standalone 入口）、`results/figures`、`results/metrics`、`data/`（gitignored，執行時自動下載）。

ORL/AT&T/Olivetti（同一份，40 人 × 10 張）首次執行時自動從 `lloydmeta/Olivetti-PNG` GitHub 鏡像下載並整理成 `data/orl_faces/s1..s40`。

---

## Phase 1: Step 1 — Federated Learning System

### 1.1 `src/data_utils.py`
- `load_orl_dataset(img_size=32, normalize=True)`：讀全部影像 → resize 32×32 灰階 → **z-score 正規化**（零均值單位變異；Sigmoid 網路餵原始 [0,1] 會卡在 loss plateau，這步是收斂關鍵），mean/std 附在 dataset 上供攻擊端反正規化。
- `train_test_split(test_per_subject=2)`：每人留 2 張 = 80 張 global test，其餘 320 張訓練。
- `split_iid(num_clients=4)`：IID 等分。
- `split_noniid(num_clients=4)`：病態 label-block 切分（每 client 持有不重疊的 subject 區塊）。
- `split_dirichlet(num_clients, alpha, ...)`：可調 Dirichlet(α) label skew（Hsu et al. 2019）；α 越小越極端 non-IID，α 大趨近 IID。配合多 client + 部分參與才看得出 client drift。
- `denormalize(images, mean, std)`：把重建影像映回 [0,1] 顯示。

### 1.2 `src/models.py`
DLG 原論文版 LeNet（**Sigmoid**、**strided conv**、輸出 raw logits）：
```
Input (1×32×32)
  → Conv2d(1,12,5,stride=2,padding=2) → Sigmoid
  → Conv2d(12,12,5,stride=2,padding=2) → Sigmoid
  → Conv2d(12,12,5,stride=1,padding=2) → Sigmoid
  → Flatten(12×8×8=768) → FC(768, 40)
```
約 38K 參數。**預設套用 DLG uniform(-0.5,0.5) 初始化**（`dlg_init=True`）：不只是攻擊 demo 需要，深層 Sigmoid 在預設/Xavier 初始化下會卡在平台、SGD 逃不出來，uniform 初始化打破對稱才訓得動。ReLU 的二階導數幾乎為零、會破壞 DLG 的二階梯度匹配，故用 Sigmoid。

### 1.3 `src/fl_client.py`
- `update_model(global_state_dict)`：載入最新 global 權重。
- `train_one_round(local_epochs=1, lr=0.01, dp_clip=None, dp_noise_multiplier=0.0)`：本地訓練（Adam，每輪重建 optimizer），回傳 `(weight_delta = w_local - w_global, num_samples)`。`dp_clip` 設定時套用 DP-FedAvg（見 Phase 3）。
- 注意：這裡的「update」是 weight delta，不是 autograd loss gradient；DLG 需要的是真正的單樣本 loss gradient，攻擊時另外算。

### 1.4 `src/fl_server.py`
- `aggregate(client_updates)`：樣本加權 FedAvg `w ← w + Σ (n_i/N)·delta_i`。

### 1.5 `src/federated.py`
- `run_federated_learning(num_rounds=50, num_clients=4, ..., dp_clip=None, dp_noise_multiplier=0.0, split="iid", dirichlet_alpha=0.5, client_sample_rate=1.0, snapshot_rounds=())`：主迴圈，回傳 history + 最終權重 + 指定 round 的 snapshot（供攻擊實驗）。`split` 選 `iid`/`noniid`/`dirichlet`；`client_sample_rate<1` 啟用部分參與（每輪隨機抽一部分 client 訓練，FedAvg 標準作法，讓 client drift 浮現）。
- `train_centralized(...)`：把資料集中起來訓練同一個 LeNet（資料集中、上界）。
- `train_local_only(...)`：每個 client 只用自己的分片訓練、**永不聚合**（資料留本地但也不協作、下界），回傳逐輪「各 client 在 global test 上準確率的平均」。`centralized ≥ FedAvg ≫ local-only` 的差距就是 FedAvg 的價值。
- `run_federated_learning_he(...)`：HE 模式（見 Phase 3-1）。

### 1.6 `experiments/run_fl.py`
三部分：
- **收斂**：centralized vs FedAvg(IID) vs local-only，跑 `seed ∈ {0,1,2}` 取 mean±std。三條 baseline 把聯邦的價值夾出來（centralized 上界、FedAvg、local-only 下界）。密集 snapshot（round 1,2,4,6,8,10,12,15,18,20,22,25,30,40,50，**三個 seed 全存**）供 Step 2 退化曲線跨 seed 平均。
- **non-IID（client drift）**：10 clients、部分參與（每輪抽 50%）、5 local epochs，比較 IID vs Dirichlet(α=1.0) vs Dirichlet(α=0.1) 的**逐輪收斂曲線**（3 seed mean±std）——α 越小，收斂越慢越震盪、最終準確率越低。畫收斂曲線（非僅最終 bar），drift 才看得到。
產出：`fl_accuracy_curve.png`、`fl_loss_curve.png`、`fl_noniid_comparison.png`、`fl_training.csv`、`fl_noniid.csv`、`fl_global_model.pt`、`fl_snapshots.pt`（`{seed: {round: state}}`）。
預期：centralized 0.91±0.03 ≥ FedAvg 0.89±0.03 ≫ local-only 0.53±0.01（聚合補回約 36 pp 即聯邦的價值）；non-IID 收斂曲線隨 α 下降——IID ~0.91、α=1.0 ~0.90、α=0.1 ~0.80 且變異約 3×（drift 明顯）。

---

## Phase 2: Step 2 — DLG / iDLG Attack

### 2.1 `src/dlg_attack.py`
- `compute_real_gradients(model, image, label)`：Server 觀察到的單樣本 loss gradient。
- `idlg_label_inference(gradients, num_classes)`：從最後一層 FC 梯度解析推 label（單樣本 CE 下只有真類別的 row-sum 為負）。
- `dlg_attack(model, real_gradients, ..., num_iterations=300, known_label=None, image_log=None, log_iters=())`：LBFGS 最小化 dummy 與 real gradient 的 L2 距離（`create_graph=True` 走二階）。`known_label=None` 為 plain DLG（同時優化 label）；給 `known_label` 為 iDLG（固定 label，更穩更快）。`image_log`/`log_iters` 可擷取中間影像做 noise→face 進程圖。

### 2.2 `src/metrics.py`
`compute_psnr`（exact match 短路回 +inf）、`compute_ssim`、`compute_mse`（皆在 [0,1] 上算）。

### 2.3 `experiments/run_attack.py`
- **Demo**（未訓練 dlg_init 模型）：對多張圖各跑一次 iDLG，展示近乎完美還原（8/8、平均 ~84 dB）。
- **Progression**：擷取單張圖從隨機雜訊 → 人臉的重建過程（`dlg_progression.png`）。
- **Batch-size sweep**：固定 round 0，batch ∈ {1,2,4,8} 跑 plain DLG，best-match PSNR 隨 batch 單調下降（80→18 dB）。
- **DLG vs iDLG**：同圖對比收斂速度。
- **Rounds**：對 `fl_snapshots.pt` 各 round，每輪攻擊 8 名受害者 **× 3 snapshot seeds = 24 次攻擊**並 pool（跨 seed 平均，消除單 seed 的前段抖動）。主曲線畫**攻擊成功率（PSNR > 20 dB 的比例）**、輔以 mean PSNR——per-victim PSNR 是雙峰（重建成功 ~50 dB 或失敗 ~5 dB，中間幾乎沒有），用平均會報出沒有任何受害者落在的數值，成功率才單調可讀。隱私臨界（跨 seed 平均）：round 1 全成功、round 2–12 約 54–88% 震盪、round 15–30 由 58% 降到 8%、round 40 起 0%（比單 seed 的「20–25 急崖」更平緩、也更誠實）。
- **Real delta（補 threat-model 缺口）**：`run_real_delta` 攻擊**真正上傳的 weight delta**而非另算的梯度。1 樣本 + 1 步 SGD 時 `delta = -lr·g`，反演近乎完美（~84 dB，證明真實上傳可被反演）；改成真實 FedAvg 的多步 Adam delta 後，naive 反演失效（~6 dB）。產出 `dlg_real_delta.png`、`dlg_real_delta.csv`。

攻擊成功判定：PSNR > 20 dB（SSIM > 0.5）。
產出：`dlg_demo_comparison.png`、`dlg_progression.png`、`dlg_batchsize_sweep.png`、`dlg_vs_idlg.png`、`dlg_quality_vs_round.png`、`dlg_rounds_comparison.png`、`dlg_loss_curve.png`、`dlg_real_delta.png`、`dlg_attack_results.csv`、`dlg_batchsize.csv`、`dlg_quality_vs_round.csv`、`dlg_real_delta.csv`。

> Threat-model 注記（誠實報告）：主要實驗攻擊的是「單樣本、單次 backward 的乾淨梯度」（洩漏上界）；真實 FedAvg client 上傳的是 batch、多步優化後的 weight delta，反演困難得多。`run_real_delta` 直接量化此差距，batch-size sweep 與 rounds 曲線進一步具體化。

---

## Phase 3: Step 3 — Differential Privacy Defense

### 3.1 `src/dp_utils.py`
DP-FedAvg / DP-SGD 的高斯機制：
- `clip_update(update, C)` / `clip_grad_list(grads, C)`：把更新裁剪到 L2 範數 `C`（界定 sensitivity，沒有它就沒有合法 ε）。
- `dp_fedavg_update(update, clip_norm, noise_multiplier)` / `dp_fedavg_grad_list(...)`：先裁剪再加 `N(0,(z·C)²)` 高斯噪音（`z` = noise multiplier）。
- `gaussian_rdp` / `rdp_to_epsilon` / `compute_epsilon`：以 **RDP**（Mironov 2017）把 `z` 與 round 數換算成 **(ε, δ)** 預算（單次高斯機制為 `(α, α/(2z²))`-RDP，跨輪相加後轉 (ε,δ)）。ε 只取決於 z 與 round 數。
- `compute_rdp_subsampled_gaussian` / `compute_epsilon_subsampled`：**子取樣高斯機制**的 RDP（Mironov–Talwar–Zhang 2019，log 空間數值穩定，與 TF-Privacy/Opacus 同），給 DP-SGD 的隱私放大。q=1 退化為 plain Gaussian（已測）。
- `per_sample_gradients`（`torch.func` vmap+grad）/ `dp_sgd_local_update`：DP-SGD 本地步——Poisson 取樣 lot、**逐樣本**梯度裁剪到 C、加 `N(0,(zC)²)`、除以期望 lot 大小後做 SGD step，回傳 weight delta。

### 3.2 整合
`FLClient.train_one_round(dp_clip=C, dp_noise_multiplier=z)` 在更新離開 client 前套用；`run_federated_learning(dp_clip, dp_noise_multiplier)` 串接。

### 3.3 `experiments/run_dp.py`
clip `C=7`（≈ 更新範數中位數），掃 noise multiplier `z ∈ {0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, …, 1.0}`（含 z<0.01 的細粒度點以解析隱私 knee），δ=1e-5、50 rounds：
- **效用**：FedAvg 最終準確率（3 seed mean±std）vs ε。
- **隱私**：對單樣本梯度套同一機制後跑 DLG，PSNR/SSIM vs ε。
產出：`dp_tradeoff.png`（x 軸標 z 與對應 ε）、`dp_leakage_demo.png`、`dp_tradeoff.csv`。

> 觀察：高維小模型下，足以打垮 DLG 的噪音（z≈0.002 即把 84→12 dB、且準確率仍 ~0.90）對應的 ε 仍是天文數字（無實質保證）；要拿到有意義的 ε（≲60）所需的噪音已把準確率打到隨機水準。這是 **update-level** 機制的代價（無放大、聚合裁剪），與 HE 形成對比（維度詛咒）。

### 3.4 `experiments/run_dp_sgd.py`（record-level DP-SGD，Abadi 機制）

對照 `run_dp.py` 的補強：clip **逐樣本**梯度（C=10）、Poisson 取樣、subsampled-RDP 計 ε。固定 q（lot=整個 client shard）、lr=0.5、8 steps/round、50 rounds，掃 `z ∈ {0,0.1,…,3.0}`：
- **效用**：DP-SGD FedAvg 最終準確率（3 seed mean±std）vs ε。
- **隱私**：對單樣本梯度裁到**自身範數**後加噪（隔離噪音效應，z 為唯一變因）跑 DLG，PSNR vs ε。
- **子取樣對照**：幾個 q<1 點，展示放大能把 ε 壓低但小 lot 摧毀 per-coordinate SNR、準確率崩。
產出：`dp_sgd_tradeoff.png`、`dp_sgd_leakage_demo.png`、`dp_sgd_tradeoff.csv`、`dp_sgd_subsampling.csv`。

> 觀察（比 update-level 更細緻）：逐樣本裁剪讓準確率**優雅下降**（ε 2824/1046/296/156 ↔ acc 0.94/0.89/0.76/0.56，ε≈98 才崩到隨機），證明 DP「機制」選擇有差（同 ε 下 DP-SGD 準確率遠高於 update-level）；但要 ε<10 仍得犧牲大半準確率，子取樣（q=0.25 把 ε 壓到 49）只是用準確率換更低的 ε——38K 參數、320 張影像上的維度詛咒，以正確放大計帳再次確認，並與 HE（零準確率代價）對比。DLG 洩漏軸（裁到自身範數以隔離噪音）：z=0 乾淨梯度 84 dB、一加噪即崩到 ~5 dB，與 DP-FedAvg 同樣「經驗隱私便宜」——兩機制的差別在**準確率軸**（優雅下降 vs 急崖），不在洩漏軸。

---

## Phase 3-1: Step 3-1 — Homomorphic Encryption Defense (Bonus)

### 3-1.1 `src/he_utils.py`
TenSEAL CKKS：`create_he_context`（含 secret key，client 持有）、`create_public_context`（去 secret key，server 持有）、`encrypt/decrypt/serialize/deserialize_gradients`、`aggregate_encrypted(encrypted_list, weights=...)`（密文加法 + 明文標量乘，做**樣本加權** FedAvg；與 FLServer 同語意，但必須跑在密文上故另寫）。
參數：`poly_modulus_degree=8192`、`coeff_mod_bit_sizes=[60,40,40,60]`、`global_scale=2**40`。

### 3-1.2 金鑰分發 / 每輪流程
1. 初始化：產生 CKKS context；server 收 `make_context_public()` 後的 public context，clients 收完整 context。
2. 每輪：client 本地訓練 → encrypt(delta) → 上傳密文；server（只有 public key）密文聚合後回傳；client decrypt 並套用。

### 3-1.3 `experiments/run_defense.py`
- **A. 收斂性**：加密 FedAvg vs 同 seed 的明文軌跡（逐輪最大差 ≤ 0.025，終局 0.875 vs 0.8625）。
- **B. 防禦展示（結構論證）**：server 持 public context → `decrypt()` 直接拋 `ValueError`（無 secret key）→ 連 DLG 的目標函數 `‖g_dummy − g_real‖²` 都湊不出來。圖中第三格展示 server 實際只持有高熵密文（entropy ≈ 8/8 bits/byte）。**這不是「DLG 跑了但失敗」，而是「攻擊者拿不到攻擊所需的明文梯度」。**
- **C. Trade-off**：每輪 encrypt/aggregate/decrypt 耗時、密文 vs 明文通訊量（~32.7×）。
產出：`he_accuracy_comparison.png`、`he_time_breakdown.png`、`he_defense_demo.png`、`he_training.csv`、`he_communication.csv`。

---

## Execution Summary

```bash
uv sync
uv run python experiments/run_fl.py        # Step 1：FedAvg + centralized baseline（需先跑以產生 snapshots）
uv run python experiments/run_attack.py    # Step 2：DLG / iDLG 攻擊
uv run python experiments/run_dp.py        # Step 3：update-level DP-FedAvg（accuracy / DLG vs ε）
uv run python experiments/run_dp_sgd.py    # Step 3：record-level DP-SGD（subsampled-RDP，graceful trade-off）
uv run python experiments/run_defense.py   # Step 3-1：CKKS 同態加密防禦
uv run pytest                              # FedAvg / metrics / DLG / DP / DP-SGD / dirichlet / HE roundtrip
```

每個 experiment script 都是 self-contained：載入資料、建模、跑實驗、存圖存 CSV。

---

## Reference Repos

- DLG 原始碼: https://github.com/mit-han-lab/dlg
- iDLG (paper): https://arxiv.org/pdf/2001.02610
- Inverting Gradients: https://github.com/JonasGeiping/invertinggradients
- DP-SGD (Abadi 2016): https://github.com/tensorflow/privacy
- TenSEAL: https://github.com/OpenMined/TenSEAL
- FedBoosting (HE+FL): https://github.com/Rand2AI/FedBoosting

---

## Notes

- FL training 可用 MPS 加速；DLG（LBFGS）與所有 TenSEAL/HE 操作在 CPU（TenSEAL 不支援 MPS，LBFGS 在 CPU 較穩）。
- z-score 正規化 + DLG 初始化 + Adam 三者缺一，Sigmoid LeNet 會卡在 2.5% chance。
- DP 的 ε 只取決於 noise multiplier z（與子取樣率 q）與步數；clip 界 C 只影響效用（裁剪偏差 vs 噪音尺度），不進 ε。
- DP-SGD 的 per-coordinate SNR ~ lot_size / (z·√dim)：38K 維、每 client 僅 80 張，小 lot 會讓 SNR<1，這就是「有意義 ε 必崩準確率」的根因；用 `torch.func` vmap 算逐樣本梯度。
- HE/TenSEAL 在 CPU；fc.weight (40×768) 超過 8192 slot，TenSEAL 自動跨多密文打包（會印 matmul/conv2d disabled 警告，本專案只用加法與標量乘，無影響）。
