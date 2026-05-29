# PLAN.md — Execution Plan

## Overview

完成 3 個 Step，每個 Step 產出程式碼 + 實驗結果（圖表與 CSV）。

- **Step 1**: Federated Learning 系統（FedAvg, 人臉辨識）
- **Step 2**: Gradient Leakage Attack（DLG / iDLG）
- **Step 3-1**: Homomorphic Encryption 防禦（TenSEAL CKKS）

Threat Model: **Honest-but-curious Server** — Server 忠實執行聚合但嘗試從 gradient 還原 client 資料。

---

## Phase 0: Project Setup

```
uv init
uv python pin 3.12
uv add torch torchvision matplotlib scikit-image tenseal pandas
```

建立目錄結構：

```
src/__init__.py
src/models.py
src/data_utils.py
src/fl_client.py
src/fl_server.py
src/federated.py
src/dlg_attack.py
src/he_utils.py
src/metrics.py
experiments/
results/figures/
results/metrics/
data/
```

下載 ORL Faces dataset：
- URL: https://git-disl.github.io/GTDLBench/datasets/orl_faces/ 或 Kaggle
- 若無法直接下載，用 torchvision 的 LFW 或手動放入 `data/orl_faces/`
- 備案：用 torchvision.datasets.LFWPeople 或類似小型人臉資料集

---

## Phase 1: Step 1 — Federated Learning System

### 1.1 `src/data_utils.py`

- `load_orl_dataset(data_dir, img_size=32)` → 讀取所有圖片, resize 32×32 grayscale, normalize [0,1], 回傳 (images_tensor, labels_tensor)
- `split_iid(dataset, num_clients=4)` → IID 切分, 回傳 list of Subset
- `get_test_loader(dataset, batch_size=32)` → global test DataLoader（每人保留 2 張 = 80 張）

### 1.2 `src/models.py`

```python
class LeNet(nn.Module):
    # Input: (1, 32, 32)
    # Conv2d(1, 12, 5, padding=2) → ReLU
    # Conv2d(12, 12, 5, padding=2) → ReLU
    # Conv2d(12, 12, 5, padding=2) → ReLU → MaxPool2d(2)
    # Flatten → FC(12*16*16, 40) → LogSoftmax
```

~40K 參數。用 NLLLoss（配 LogSoftmax）或 CrossEntropyLoss（去掉 LogSoftmax）。

### 1.3 `src/fl_client.py`

```python
class FLClient:
    def __init__(self, client_id, dataset, model_cls, device):
        ...
    
    def update_model(self, global_state_dict):
        """載入 global model weights"""
    
    def train_one_round(self, local_epochs=1, lr=0.01):
        """
        local training，回傳:
        - gradients: list of (param - original_param) for each parameter
        - num_samples: 本 client 的樣本數（用於加權平均）
        """
    
    def get_gradients(self):
        """取得 model updates (差值), 不是 torch.autograd 的 grad"""
```

注意：這裡的 "gradients" 實際上是 weight updates (Δw = w_local - w_global)。DLG 攻擊需要的是真正的 loss gradient，所以攻擊時要另外算。

### 1.4 `src/fl_server.py`

```python
class FLServer:
    def __init__(self, model_cls, device):
        self.global_model = model_cls().to(device)
    
    def aggregate(self, client_updates):
        """
        FedAvg: weighted average of client weight updates
        client_updates: list of (state_dict_delta, num_samples)
        """
    
    def get_global_state_dict(self):
        ...
```

### 1.5 `src/federated.py`

```python
def run_federated_learning(
    num_rounds=50, num_clients=4, local_epochs=1, lr=0.01,
    device="mps"
) -> dict:
    """
    完整 FL 訓練迴圈。
    回傳 history dict: {round: {accuracy, loss}}
    每 5 rounds print accuracy。
    """
```

### 1.6 `experiments/run_fl.py`

- 執行 FL 訓練
- 產出：
  - `results/figures/fl_accuracy_curve.png` — accuracy vs round
  - `results/figures/fl_loss_curve.png` — loss vs round
  - `results/metrics/fl_training.csv` — round, accuracy, loss
  - 儲存最終 model: `results/fl_global_model.pt`

### 1.7 Centralized Baseline（同一個 script 或額外跑）

- 不做 FL，直接用全部 training data 訓練同一個 LeNet
- 記錄 baseline accuracy 作為對照
- 輸出到同一個 figure 做對比

---

## Phase 2: Step 2 — DLG Attack

### 2.1 `src/dlg_attack.py`

```python
def dlg_attack(
    model,              # 當前 global model (攻擊時的模型狀態)
    real_gradients,     # 目標 client 的真實 gradient (list of tensors)
    image_shape,        # (1, 1, 32, 32)
    label_shape,        # (1, num_classes)
    num_iterations=300,
    device="cpu"        # DLG 用 CPU 比較穩定
) -> (recovered_image, recovered_label, loss_history):
    """
    DLG: 隨機初始化 dummy image + dummy label，
    用 LBFGS 最小化 dummy gradient 與 real gradient 的 L2 距離。
    """

def compute_real_gradients(model, image, label, criterion):
    """
    給定單張圖片，計算 model 對這張圖的 loss gradient。
    這是攻擊者視角：Server 收到某 client 對單張圖片的 gradient。
    """

def idlg_label_inference(gradients, num_classes):
    """
    iDLG: 從最後一層 FC 的 gradient 推斷 label。
    cross-entropy 的性質: grad of last layer 最大的 index = true label。
    """
```

### 2.2 `src/metrics.py`

```python
def compute_psnr(original, recovered) -> float:
def compute_ssim(original, recovered) -> float:
def compute_mse(original, recovered) -> float:
```

使用 skimage.metrics.structural_similarity 和 skimage.metrics.peak_signal_noise_ratio。

### 2.3 `experiments/run_attack.py`

兩組實驗設定：

**Demo 設定（展示最佳還原效果）：**
- 用 Round 0 的 model（隨機初始化或訓練 1 round 後）
- 對 5~10 張不同圖片各做一次 DLG
- batch_size=1（單張攻擊）

**真實 FL 設定（觀察品質衰退）：**
- 用 Round 1, 10, 25, 50 的 model
- 對同一張圖片做 DLG，觀察不同訓練階段的還原品質

產出：
- `results/figures/dlg_demo_comparison.png` — grid: 每列 = original | recovered, 多張圖
- `results/figures/dlg_rounds_comparison.png` — 同一張圖在不同 round 的還原效果
- `results/figures/dlg_loss_curve.png` — DLG 優化過程的 loss 下降
- `results/metrics/dlg_attack_results.csv` — image_id, round, psnr, ssim, mse
- Print 攻擊成功率：PSNR > 20dB 的比例

攻擊成功判定：

| 指標 | 成功 | 部分成功 | 失敗 |
|------|------|---------|------|
| PSNR | > 20 dB | 15-20 dB | < 15 dB |
| SSIM | > 0.5 | 0.3-0.5 | < 0.3 |

---

## Phase 3: Step 3-1 — HE Defense

### 3.1 `src/he_utils.py`

```python
def create_he_context():
    """
    建立 TenSEAL CKKS context。
    poly_modulus_degree=8192, coeff_mod_bit_sizes=[60,40,40,60]
    global_scale=2**40
    回傳 full_context (含 secret key)
    """

def create_public_context(full_context):
    """
    複製 context 並移除 secret key。
    給 Server 用。
    """

def encrypt_gradients(gradients_dict, context):
    """
    將 model 的 gradient state dict 加密。
    每個 parameter tensor → flatten → ts.ckks_vector(context, flat_list)
    回傳 dict of encrypted vectors
    """

def decrypt_gradients(encrypted_dict, context, shapes_dict):
    """
    解密 encrypted gradient dict，reshape 回原始 tensor shapes。
    """

def serialize_encrypted(encrypted_dict) -> bytes:
    """ts.ckks_vector.serialize() for each, pack into dict"""

def deserialize_encrypted(data_bytes, context) -> dict:
    """反序列化"""

def aggregate_encrypted(encrypted_list, num_clients, public_context)[118;1:3u:
    """
    密文聚合：
    1. 逐 parameter 做密文加法 (sum)
    2. 乘以明文 1/num_clients (FedAvg 平均)
    回傳 aggregated encrypted dict
    """
```

### 3.2 修改 FL Pipeline

在 `src/federated.py` 加入 HE 模式：

```python
def run_federated_learning(
    ...,
    use_he=False,       # 開關
    he_rounds=5,        # HE 模式只跑幾 rounds
):
```

HE 模式的每輪流程：
1. Client: 拿到 global weights → local train → 算 gradient
2. Client: encrypt(gradient, full_context) → serialize → 上傳
3. Server: deserialize(各 client 密文, public_context) → aggregate_encrypted → serialize → 回傳
4. Client: deserialize → decrypt → 更新 local model

Server 全程只接觸密文，無法做 DLG。

### 3.3 `experiments/run_defense.py`

實驗內容：

**A. HE-FL 收斂性驗證**
- 跑 HE-FL 5 rounds（或更多，看時間）
- 對比同 rounds 數的 plaintext FL accuracy
- 理論上 accuracy 應該一致（CKKS 精度損失極小）

**B. 防禦效果驗證**
- HE on 時，Server 只有密文 → 直接 print 說明 DLG 無法執行（沒有明文 gradient）
- 額外驗證：把密文 decrypt 前的 raw bytes 餵給 DLG → 產出純噪音

**C. Trade-off 分析**
- 時間：量測每 round 的 encrypt / aggregate / decrypt 各階段耗時
- 通訊量：量測明文 gradient 大小 vs 密文大小
- Accuracy：HE on vs off 的收斂對比

產出：
- `results/figures/he_accuracy_comparison.png` — HE on vs off accuracy curve
- `results/figures/he_time_breakdown.png` — 各階段耗時 stacked bar chart
- `results/figures/he_defense_demo.png` — 防禦前(Step2 還原成功) vs 防禦後(攻擊失效)
- `results/metrics/he_training.csv` — round, accuracy, loss, encrypt_time, decrypt_time, aggregate_time
- `results/metrics/he_communication.csv` — plaintext_size_bytes, ciphertext_size_bytes, ratio

---

## Execution Summary

```bash
# Phase 0
uv init && uv python pin 3.12
uv add torch torchvision matplotlib scikit-image tenseal pandas

# Phase 1 — FL 系統
# 實作 src/ 模組 → 跑 experiments/run_fl.py
uv run python experiments/run_fl.py

# Phase 2 — DLG 攻擊
# 實作攻擊模組 → 跑 experiments/run_attack.py
uv run python experiments/run_attack.py

# Phase 3 — HE 防禦
# 實作 HE 模組 → 跑 experiments/run_defense.py
uv run python experiments/run_defense.py
```

每個 experiment script 應該是 self-contained 的：載入資料、建立模型、跑實驗、存圖存 CSV。

---

## Reference Repos (for implementation reference)

- DLG 原始碼: https://github.com/mit-han-lab/dlg
- Inverting Gradients: https://github.com/JonasGeiping/invertinggradients
- TenSEAL examples: https://github.com/OpenMined/TenSEAL/tree/main/tutorials
- FedBoosting (HE+FL): https://github.com/Rand2AI/FedBoosting

---

## Notes

- DLG 攻擊在 MPS 上可能有數值問題，LBFGS 建議在 CPU 跑
- TenSEAL 不支援 MPS，所有 HE 操作在 CPU
- FL training 可以用 MPS 加速，攻擊/防禦實驗在 CPU
- ORL dataset 若自動下載失敗，需手動下載放入 data/orl_faces/，結構為 data/orl_faces/s1/ ~ s40/，每個資料夾 10 張 .pgm
