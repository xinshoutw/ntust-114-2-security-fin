# Final Project：Federated Learning & Its Attacks and Defenses

## 專案規劃書 v2

> 課程：NTUST 114-2 資訊安全（碩士班）
> 繳交期限：2026/06/11 23:00
> 完成範圍：Step 1 + Step 2 + Step 3（Differential Privacy）+ Step 3-1（Bonus, Homomorphic Encryption）— 四個 Step 全數實作

---

## 一、策略總覽

### 核心原則

- **確保能跑出東西**：選最成熟的方法、最小的資料集，先求有再求好
- **手刻 FL**：教授建議不要用 TFF/Flower 等框架，自己寫 FL loop，後續攻擊和防禦才知道要改哪裡
- **技術棧統一**：Python + PyTorch + uv（延續 HW1/HW3 的慣例）

### Threat Model（威脅模型）

在進入技術細節前，先定義整個專案的安全假設：

- **攻擊者身分**：Honest-but-curious Server（誠實但好奇的伺服器）
- **攻擊者能力**：Server 忠實執行 FedAvg 聚合，但會嘗試從收到的 gradient 還原 client 的原始資料
- **攻擊方法**：Gradient Leakage Attack（DLG / iDLG）
- **防禦目標**：讓 Server 無法從上傳的更新還原私有影像，使 DLG 攻擊失效
- **防禦方法**：
  - **Differential Privacy（Step 3）**：Client 上傳前先把更新 L2 裁剪、再加高斯噪音（DP-FedAvg），以 ε 量化隱私
  - **Homomorphic Encryption（Step 3-1, CKKS）**：Client 加密更新後上傳，Server 只能操作密文、拿不到明文梯度

### 完成目標

| Step | 內容 | 成果 |
|------|------|------|
| Step 1 | 建立 FL 系統做人臉辨識 | FL 訓練收斂、準確率合理（FedAvg ~0.89） |
| Step 2 | Gradient Leakage Attack（DLG / iDLG） | 從梯度還原出原始人臉圖片（未訓練模型 ~84 dB） |
| Step 3 | Differential Privacy 防禦（DP-FedAvg） | 裁剪 + 高斯噪音，accuracy vs **ε** 的 privacy-utility trade-off |
| Step 3-1 | Homomorphic Encryption 防禦（Bonus） | HE 加密梯度後攻擊失效 + accuracy/time/通訊量 trade-off |

---

## 二、技術架構

### 2.1 資料集

**AT&T/ORL Faces**，resize 到 **32×32 灰階**。

| 屬性 | 值 |
|------|-----|
| 總量 | 400 張（40 人 × 10 張） |
| 圖片尺寸 | resize 至 32×32 灰階 |
| FL 分割 | IID，4 clients 各 100 張 |
| Test set | 每人保留 2 張 = 80 張 global test set |

選擇理由：DLG 在小圖 + 淺層 CNN 上還原效果最好。32×32 是 DLG 原論文使用的尺寸範圍。ORL 在 HW3 已用過，熟悉度高。

### 2.2 系統架構圖

```
                    ┌──────────────────────────────┐
                    │     FL Server (Aggregator)    │
                    │                              │
                    │  持有：public key only        │
                    │  執行：密文加法 + 明文乘 1/N   │
                    │                              │
                    │  Step 2 攻擊點：               │
                    │  Server 嘗試從 gradient 做 DLG │
                    └──────┬───────────┬───────────┘
                      ▲ encrypted  │ encrypted
                      │ gradients  ▼ aggregated
               ┌──────┴──────┬─────┴──────┬──────────────┐
               │  Client 1   │  Client 2  │  Client 3 …  │
               │             │            │              │
               │  持有：      │            │              │
               │  full context│            │              │
               │  (public +   │            │              │
               │   secret key)│            │              │
               │             │            │              │
               │  local data │ local data │  local data  │
               │  local train│ local train│  local train │
               │  encrypt ↑  │            │              │
               │  decrypt ↓  │            │              │
               └─────────────┴────────────┴──────────────┘
```

### 2.3 模型設計

DLG 原論文版 LeNet 變體：

```
Input (1×32×32)
  → Conv2d(1, 12, 5, stride=2, padding=2) → Sigmoid
  → Conv2d(12, 12, 5, stride=2, padding=2) → Sigmoid
  → Conv2d(12, 12, 5, stride=1, padding=2) → Sigmoid
  → Flatten(12×8×8 = 768) → FC(768, num_classes=40)   # 輸出 raw logits（配 CrossEntropyLoss）
```

參數量約 38K。三個關鍵設計：(1) **Sigmoid**（非 ReLU）——ReLU 二階導數幾乎為零會破壞 DLG 的二階梯度匹配；(2) **strided conv**（非 pooling）下採樣；(3) 預設套用 **DLG uniform(-0.5,0.5) 初始化**。注意：輸入須做 **z-score 正規化**，否則 Sigmoid 在原始 [0,1] 像素下會卡在 loss plateau 訓不動（z-score + DLG-init + Adam 三者缺一即停在 2.5% chance）。

### 2.4 FL 訓練流程（FedAvg）

```python
for round_idx in range(num_rounds):
    global_weights = global_model.state_dict()
    
    client_updates = []
    for client in clients:
        client.model.load_state_dict(global_weights)
        gradients = client.train_one_epoch()  # local_epochs=1
        client_updates.append(gradients)
    
    # FedAvg: 樣本加權平均  w ← w + Σ (n_i/N) · delta_i
    aggregated = fedavg(client_updates)
    global_model.load_state_dict(aggregated)
    
    evaluate(global_model, test_data)
```

參數設定：num_clients=4, num_rounds=50, local_epochs=1, batch_size=8

### 2.5 DLG 攻擊（Step 2）

#### 兩種實驗設定

| 設定 | batch_size | 攻擊時機 | 用途 |
|------|-----------|---------|------|
| **Demo 設定** | 1 | Round 0（未訓練 dlg_init 模型） | 展示最佳還原效果（含 noise→face 進程圖）|
| **Batch-size sweep** | 1,2,4,8 | Round 0 | 還原品質隨 batch 單調下降（80→18 dB）|
| **真實 FL 設定** | 1 | 密集 snapshot（round 1…50），每輪 8 名受害者取 mean±std | 觀察攻擊品質隨訓練衰退、定位隱私臨界（round 20–25 斷崖）|

#### 攻擊流程

```python
# 攻擊者 = Server，已知某 client 傳上來的 real_gradients
dummy_data = torch.randn(image_shape, requires_grad=True)
dummy_label = torch.randn(label_shape, requires_grad=True)
optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

for step in range(300):
    def closure():
        optimizer.zero_grad()
        dummy_pred = model(dummy_data)
        dummy_loss = criterion(dummy_pred, F.softmax(dummy_label, dim=-1))
        dummy_gradients = torch.autograd.grad(
            dummy_loss, model.parameters(), create_graph=True
        )
        # 最小化 dummy gradients 與 real gradients 的 L2 距離
        grad_diff = sum(
            (dg - rg).pow(2).sum()
            for dg, rg in zip(dummy_gradients, real_gradients)
        )
        grad_diff.backward()
        return grad_diff
    optimizer.step(closure)
```

#### 攻擊成功判定

| 指標 | 成功 | 部分成功 | 失敗 |
|------|------|---------|------|
| PSNR | > 20 dB | 15-20 dB | < 15 dB |
| SSIM | > 0.5 | 0.3-0.5 | < 0.3 |

#### iDLG 改良（可選加分項）

先從 gradient 的最後一層推斷 label（cross-entropy loss 的性質），再固定 label 只優化 image，收斂更快更穩。

### 2.6 DP 防禦（Step 3）

DP-FedAvg / DP-SGD 的高斯機制：每個 client 在更新離開前，先把 weight delta **L2 裁剪**到範數界 `C`（界定 sensitivity，沒有它就沒有合法 ε），再加 `N(0, (z·C)²)` 高斯噪音（`z` = noise multiplier）。隱私預算 **ε** 以 **RDP**（Mironov 2017）合成 T 輪後換算（單次高斯機制為 `(α, α/(2z²))`-RDP）；ε 只取決於 z 與輪數，clip 界 C 只影響效用。

實驗（`run_dp.py`）：clip `C=7`（≈ 更新範數中位數），掃 `z ∈ {0, 0.01, …, 1.0}`，δ=1e-5、50 rounds，量 (a) 準確率（3 seed mean±std）vs ε、(b) 對單樣本梯度套同一機制後的 DLG PSNR vs ε。產出 `dp_tradeoff.png`（x 軸標 z 與對應 ε）、`dp_leakage_demo.png`、`dp_tradeoff.csv`。

觀察：高維小模型下，足以打垮 DLG 的噪音（z≈0.01 即把 84→6 dB、準確率仍 ~0.89）對應的 ε 仍是天文數字（無實質保證）；要拿到有意義的 ε（≲60）所需的噪音已把準確率打到隨機水準。DP 在此代價極高（維度詛咒），與 HE 形成對比。

### 2.7 HE 防禦（Step 3-1）

#### 金鑰分發流程

```
1. 初始化階段：
   - 產生一組 TenSEAL CKKS context（含 public key + secret key）
   - 對 Server：context.make_context_public() → 只給 public key
   - 對所有 Clients：給完整 context（含 secret key）

2. 每輪 FL：
   Client:
     a. local training → 得到 plaintext gradients
     b. encrypt(gradients, context) → encrypted_gradients
     c. serialize(encrypted_gradients) → bytes
     d. 上傳 bytes 到 Server

   Server（只有 public key，無法解密）:
     a. deserialize(bytes) → encrypted_gradients
     b. encrypted_sum = Σ encrypted_gradients    ← 密文加法
     c. encrypted_avg = encrypted_sum × (1/N)    ← 明文標量乘法
     d. serialize(encrypted_avg) → bytes
     e. 回傳 bytes 給所有 Clients

   Client:
     a. deserialize(bytes) → encrypted_avg
     b. decrypt(encrypted_avg, secret_key) → plaintext global update
     c. 更新 local model
```

#### 防禦驗證（結構論證，非「跑 DLG 但失敗」）

- HE 開啟後，Server 持 public context、**無 secret key**，`decrypt()` 直接拋 `ValueError` → 拿不到明文梯度 → 連 DLG 的目標函數 `‖g_dummy − g_real‖²` 都湊不出來 → 攻擊在前提上就被否決。
- 視覺化：三格圖 = original | HE off（Server 持明文梯度，DLG 還原成功 ~84 dB）| HE on（Server 全部所見 = 高熵密文，entropy ≈ 8/8 bits/byte，`decrypt()` raises）。**密文不是餵給 DLG 的輸入**，只是用來展示「Server 手上只有密文」。

#### TenSEAL 參數設定

```python
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

# Server 的 public-only context
public_context = context.copy()
public_context.make_context_public()
```

#### HE 實驗規模

由於 CKKS 加解密速度慢，HE 實驗策略：
- **完整跑 3~5 rounds**（證明 FL + HE 能收斂）
- **比較 accuracy**：HE on vs HE off（前 5 rounds 的收斂趨勢對比）
- **時間拆解**：分別量測 encrypt / serialize / aggregate / deserialize / decrypt 各階段耗時
- 不需要跑滿 50 rounds — 報告中說明原因即可

---

## 三、評估指標完整清單

| 指標類別 | 指標 | 用於 |
|----------|------|------|
| **FL 效能** | Accuracy（global test set） | Step 1, 3, 3-1 |
| **FL 效能** | Loss curve per round | Step 1, 3-1 |
| **攻擊品質** | PSNR (dB) | Step 2, 3 |
| **攻擊品質** | SSIM | Step 2, 3 |
| **攻擊品質** | MSE | Step 2 |
| **攻擊品質** | 視覺對比圖（original vs recovered） | Step 2, 3 |
| **防禦效果** | 攻擊成功率（PSNR > 20dB 的比例） | Step 2 vs 3 vs 3-1 |
| **DP 隱私** | 隱私預算 **ε**（RDP 合成計算） | Step 3 |
| **DP trade-off** | Accuracy vs ε、DLG PSNR vs ε | Step 3 |
| **HE 開銷** | 每 round 耗時（拆分各階段） | Step 3-1 |
| **HE 開銷** | 密文大小 vs 明文 gradient 大小 | Step 3-1 |
| **HE 開銷** | Accuracy 對比（HE on vs off） | Step 3-1 |

---

## 四、參考資源

### Step 1 + Step 2

| 資源 | 說明 |
|------|------|
| [mit-han-lab/dlg](https://github.com/mit-han-lab/dlg) | DLG 原始碼，含 FL setup + 攻擊，**最核心的參考** |
| [JonasGeiping/invertinggradients](https://github.com/JonasGeiping/invertinggradients) | Inverting Gradients，攻擊效果更好（備選） |
| [Koukyosyumei/AIJack](https://github.com/Koukyosyumei/AIJack) | FL 攻防一體 library，參考架構設計 |

### Step 3（DP）

| 資源 | 說明 |
|------|------|
| [Abadi et al., DP-SGD (2016)](https://arxiv.org/abs/1607.00133) | Deep Learning with Differential Privacy，clip + 高斯噪音 |
| [Mironov, Rényi DP (2017)](https://arxiv.org/abs/1702.07476) | RDP 合成，本專案 ε accountant 的依據 |
| [pytorch/opacus](https://github.com/pytorch/opacus) | PyTorch DP 函式庫（本專案自寫精簡 RDP，未依賴）|

### Step 3-1

| 資源 | 說明 |
|------|------|
| [OpenMined/TenSEAL](https://github.com/OpenMined/TenSEAL) | Python HE library，CKKS scheme |
| [Rand2AI/FedBoosting](https://github.com/Rand2AI/FedBoosting) | 教授指定的 HE+FL 參考 |

---

## 五、專案目錄結構

```
FinalProject/
├── pyproject.toml
├── README.md
├── data/
│   └── orl_faces/
├── src/
│   ├── models.py           # LeNet（Sigmoid + strided conv + DLG init）
│   ├── data_utils.py       # 資料載入（z-score 正規化）、切分 client
│   ├── fl_server.py        # FL Server（樣本加權 FedAvg）
│   ├── fl_client.py        # FL Client（local training、回傳 delta、可選 DP 裁剪+噪音）
│   ├── federated.py        # FL 主訓練迴圈（明文 / DP / HE 三種模式）
│   ├── dlg_attack.py       # DLG / iDLG 攻擊
│   ├── dp_utils.py         # DP-FedAvg：裁剪 + 高斯噪音 + RDP ε accountant
│   ├── he_utils.py         # TenSEAL CKKS context、加解密、密文聚合
│   └── metrics.py          # PSNR, SSIM, MSE 計算
├── experiments/
│   ├── run_fl.py           # Step 1 實驗
│   ├── run_attack.py       # Step 2 實驗
│   ├── run_dp.py           # Step 3 實驗（DP）
│   └── run_defense.py      # Step 3-1 實驗（HE）
├── results/
│   ├── figures/
│   └── metrics/
└── docs/
    ├── division-of-labor.md
    └── report/
```

---

## 六、工作模組拆解

### 模組 A：資料集與模型基礎

- 下載 ORL faces、resize 32×32、normalize、轉 tensor
- 切分成 4 client IID datasets + global test set
- 實作 LeNet CNN
- 跑 centralized training baseline（不做 FL，確認模型本身能收斂）

**產出**：`data_utils.py`、`models.py`、centralized baseline accuracy 數字

### 模組 B：FL 系統（Step 1）

- 實作 `fl_client.py`：local training、回傳 gradients
- 實作 `fl_server.py`：FedAvg 聚合
- 實作 `federated.py`：完整 FL 迴圈
- 產出 FL 訓練曲線

**產出**：可運作的 FL pipeline、accuracy vs round 圖

### 模組 C：DLG 攻擊（Step 2）

- 實作 DLG 攻擊（LBFGS optimizer）
- Demo 設定 + 真實 FL 設定 兩組實驗
- 視覺化 original vs recovered
- 計算 PSNR / SSIM / MSE
- （可選）實作 iDLG label 推斷

**產出**：`dlg_attack.py`、`metrics.py`、還原對比圖、指標表

### 模組 C2：DP 防禦（Step 3）

- 實作 `dp_utils.py`：L2 裁剪、高斯噪音、RDP ε accountant
- 在 `fl_client` / `federated` 串接 DP（`dp_clip` / `dp_noise_multiplier`）
- 掃 noise multiplier z，畫 accuracy vs ε、DLG PSNR vs ε

**產出**：`dp_utils.py`、`run_dp.py`、privacy-utility trade-off 圖與 CSV

> 註：下方分工表尚未把此模組指派給成員（分工表為未完成草稿）。

### 模組 D：HE 防禦（Step 3-1）

- 安裝 TenSEAL、學習 CKKS API
- 實作 `he_utils.py`：context 管理、encrypt/decrypt/serialize/deserialize
- 修改 FL pipeline 支援 HE on/off 開關
- 跑 3~5 rounds HE-FL
- 防禦驗證：HE on 時跑 DLG → 展示攻擊失效
- Trade-off：accuracy 對比、各階段耗時、密文大小

**產出**：`he_utils.py`、修改後的 `fl_server.py` / `fl_client.py`、防禦對比圖、trade-off 表格

### 模組 E：報告與簡報

- Word 報告（Threat Model → Step 1 → 2 → 3-1 → 觀察與結論）
- PPT 簡報
- 程式碼加中文註解
- 分工表
- 打包 zip

**從 D1 開始建報告骨架 outline**，每天收各模組的圖表與數據。

---

## 七、時程規劃

| 日期 | 天數 | 里程碑 | 模組 | 備註 |
|------|------|--------|------|------|
| 5/29-5/31 | D1-D3 | 環境建置 + 資料 + 模型 baseline | A | 同步建報告骨架 |
| 6/01-6/03 | D4-D6 | **FL 系統完成** (Step 1 ✓) | B | |
| 6/03-6/06 | D6-D9 | **DLG 攻擊完成** (Step 2 ✓) | C | |
| 6/06-6/08 | D9-D11 | **DP 防禦完成** (Step 3 ✓) | C2 | clip + 高斯噪音 + RDP ε |
| 6/03-6/05 | D6-D8 | HE 加密管線先行開發 | D 前半 | 不依賴模組 C |
| 6/06-6/08 | D9-D11 | **HE 防禦驗證完成** (Step 3-1 ✓) | D 後半 | 需要 C 的攻擊碼 |
| 6/09-6/11 | D12-D14 | 報告 + PPT + 打包 | E | |

**硬性 deadline**：6/08 前所有程式碼完成，最後三天只做文件。

---

## 八、五人分工角色定義

| 角色 | 負責模組 | 核心任務 | 工作量 |
|------|----------|----------|--------|
| **P1：FL 架構** | B | 手刻 FedAvg、client/server、FL 迴圈 | ★★★★ |
| **P2：模型與資料** | A | 資料集、CNN、centralized baseline | ★★★ |
| **P3：攻擊** | C | DLG 攻擊、兩組實驗設定、視覺化 | ★★★★ |
| **P4：防禦** | D | TenSEAL 整合、HE 管線、防禦驗證 | ★★★★★ |
| **P5：報告整合** | E + 支援 | 報告、PPT、程式碼註解、分工表 | ★★★ |

### 角色依賴

```
P2 (D1-D3) → P1 (D4-D6) → P3 (D6-D9) → 驗證
                          ↘              ↗
                    P4 前半 (D6-D8) → P4 後半 (D9-D11)
                    
P5：D1 起持續收集 → D12-D14 集中產出
```

---

## 九、風險與備案

| 風險 | 機率 | 備案 |
|------|------|------|
| DLG 還原效果差 | 中 | 用 Demo 設定（batch=1, early round）保底；真實設定的低品質結果當作觀察心得 |
| TenSEAL 安裝/相容問題 | 中 | **5/30 前全員安裝測試**；備選 Pyfhel |
| HE 加解密太慢 | 高 | 只跑 3~5 rounds；報告中說明原因；時間拆解用小規模量測 |
| HE 整個跑不通 | 低 | DP（Step 3，裁剪+高斯噪音+ε）已是**獨立完成**的防禦；HE 僅為 bonus，即使失敗仍交得出完整防禦 |
| FL 不收斂 | 低 | centralized baseline 先驗證模型沒問題 |
| 時間不夠 | 中 | Step 3-1 從簡（只證明攻擊失效即可）；報告可以分段寫不需要等全部完成 |

---

## 十、繳交清單

- [ ] PPT 簡報
- [ ] 程式碼（含中文註解）
- [ ] Word 報告
  - [ ] 報告開頭列出完成了哪些 Step
  - [ ] Threat Model 說明
  - [ ] 每個 Step 的方法、結果、觀察
  - [ ] 分工表
- [ ] 打包為 `TeamName_FinalProject.zip`
