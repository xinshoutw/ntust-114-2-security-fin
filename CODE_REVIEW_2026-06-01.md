# 期末專題實作審查報告

> 對照文件：`Final Project_v1.pdf`（Federated Learning & Its Attacks and Defenses）
> 審查日期：2026-06-01　|　審查範圍：架構與邏輯正確性、實驗結果是否符合預期、參數調整建議
> 註：依使用者要求，**分工表（division-of-labor）暫不納入本次審查**。

> **更新（2026-06-01，同日）**：**P0-1 與 P1-1 已修復並重跑**。
> - P0-1：`run_dp_sgd.py` privacy demo 改裁到梯度自身範數，DLG 洩漏曲線現為 z=0 **84 dB** → 加噪即 ~5 dB（不再是貼地平線）。
> - P1-1：新增 `train_local_only`，Step 1 收斂圖呈現三方對照 **centralized 0.91 ≥ FedAvg 0.89 ≫ local-only 0.53**。
> 其餘項目（P1-2/P1-3/P2/P3）仍為建議，未實作。

---

## 0. 結論摘要（TL;DR）

**完成度：4/4 步驟全數實作（含 Bonus），對照 PDF 評分標準（"grades depend on how many steps you completed"）可拿滿步驟數。** 核心實作正確、誠實、超出作業要求；**沒有發現會導致結論錯誤的硬性 bug**。本報告的價值集中在「讓實驗結果更有區分性」與「補齊交付格式」。

最該優先處理的三件事：

| # | 項目 | 類型 | 為什麼重要 |
|---|------|------|-----------|
| **1** | **DP-SGD 圖的 DLG-洩漏軸是退化平線**（恆 ~5–6 dB，與 `z` 無關） | 結果區分性 | 圖上那條「DLG PSNR」紫線一開始就貼地，讀者會誤判「加噪沒作用」。根因：privacy demo 把梯度裁到固定 `C=10`（真實 norm=24），**裁剪本身就擊垮了 DLG**。 |
| **2** | **Step 1 缺「為何要聯邦」的下界 baseline**（local-only） | 結果價值 | 目前只證明 FedAvg ≈ centralized（「沒變差」），沒證明 FedAvg ≫ 單機孤立（「有好處」）。補一條 local-only 曲線即可三方對照。 |
| **3** | **交付格式**：PDF 要求 **PPT + Word 報告 + zip（`TeamName_FinalProject.zip`）** | 合規 | 目前只有 Markdown。報告開頭須**列出完成步驟**（PDF 明文要求）。 |

整體評語：**A 級實作**。程式模組化清楚、註解充分（符合 PDF "add some comments"）、threat-model 誠實標註、30 個測試覆蓋 FedAvg/DP/HE/metrics。下面逐項說明。

---

## 1. PDF 需求對照表

| PDF Step | 要求重點 | 實作位置 | 狀態 | 證據 |
|----------|---------|---------|------|------|
| **Step 1** 建立 FL 系統 | 任意框架/資料皆可（建議人臉） | 手刻 FedAvg：`federated.py` / `fl_client.py` / `fl_server.py`，ORL 人臉 | ✅ 完整 | FedAvg 0.89±0.03，centralized 0.91±0.03 |
| **Step 2** 梯度洩漏攻擊 | 還原 client 原始影像、展示成功案例、分享觀察 | DLG+iDLG：`dlg_attack.py` / `run_attack.py` | ✅ 超出要求 | demo 8/8 還原 ~84 dB；batch/round/real-delta 三軸觀察 |
| **Step 3** DP 防禦 | 擾動方式、privacy(ε)–accuracy 取捨 | 兩種機制：`run_dp.py`（update-level）+ `run_dp_sgd.py`（record-level Abadi） | ✅ 超出要求 | 兩條 accuracy–ε 前緣 + DLG 洩漏對照 |
| **Step 3-1** HE 防禦（Bonus） | 擾動方式、accuracy–time 取捨 | TenSEAL CKKS：`he_utils.py` / `run_defense.py` | ✅ 完整 | 準確率 ≈ 明文、CKKS 誤差 1.6e-7、50 輪 11.6s、密文 32.7× |

> **PDF 對 Step 3 的兩項量測（"Effectiveness measure: Accuracy" / "Privacy level: the value of ε"）皆有對應**，且額外把 DLG 重建品質當作「經驗隱私」軸，比單看 ε 更有說服力。Step 3-1 的兩項（Accuracy / Time consuming）也都有量測。

---

## 2. 架構與邏輯正確性分析

### 2.1 Step 1 — FedAvg（✅ 正確）

- **聚合公式正確**：`fl_server.py` 做 `w ← w + Σ_i (n_i/N)·delta_i`，client 回傳 `delta = w_end − w_start`。由於每輪 client 先 `update_model(global_state)` 再訓練，`w + mean(delta_i) = mean(w_i)`，與「直接平均最終權重」的標準 FedAvg 等價。✔
- **設計選擇有據**：Sigmoid + strided conv + `dlg_init`（uniform(−0.5,0.5)）+ z-score 正規化，這組合對「DLG 可反演」與「Sigmoid 網路可收斂」兩件事都是必要的（見 `models.py` / `data_utils.py` 註解），且有測試與收斂結果背書。✔
- **non-IID 支援正確**：`split_dirichlet`（可調 α）與 partial participation（`client_sample_rate`，seeded）實作標準、可重現。✔

### 2.2 Step 2 — DLG / iDLG（✅ 正確，且 threat-model 誠實）

- **iDLG 標籤推斷正確**：`gradients[-2]` 確為 `fc.weight` 梯度（參數順序 conv×3→fc.weight→fc.bias），單樣本 CE 下 `dL/dW_i = (softmax_i − onehot_i)·feature`，feature 為 Sigmoid 輸出（≥0），故真類那列列和最負，`argmin` 命中。✔
- **DLG 目標正確**：LBFGS 最小化 `Σ‖g_dummy − g_real‖²`，`create_graph=True` 走二階。✔
- **誠實度加分**：`run_attack.py` 與 `run_real_delta` 明確區分「單樣本梯度上界洩漏」與「真實多步 Adam delta 難反演」，並用 1-step SGD delta（`delta=−lr·g`，84 dB）證明**真實上傳的訊息確實可反演**。這正是 PDF "recover original data of each client" 的嚴謹回應。✔

### 2.3 Step 3 — DP（✅ 機制與會計正確；兩個觀念需在報告中講清楚）

- **兩種機制都對**：
  - update-level（`dp_fedavg_update`）：對整個 delta 做 L2 clip + 高斯噪音；plain-RDP 會計。
  - record-level（`dp_sgd_local_update`）：用 `torch.func`（vmap+grad）逐樣本梯度、逐樣本裁剪、加噪、除以期望 lot size，再走 SGD。✔ 這是 Abadi DP-SGD 的正規形。
- **RDP 會計數值正確**（我手算驗證）：plain Gaussian，`z=1`、50 輪、δ=1e-5 → ε≈59（在 α≈1.75 取最小），與 CSV `59.1` 吻合；subsampled-Gaussian（MTZ）log-space 實作標準，`q=1` 退化為 plain，400 步 `z=1` → ε≈296，與 CSV 吻合。✔
- **需在報告中明講的兩個觀念（非 bug，但會被老師追問）**：
  1. **主 DP-SGD 掃描用 `q=1`（SAMPLE_RATE=1.0），等於沒有 subsampling 放大**。所以 ε 偏大（296@z=1）的原因是「全參與 + 400 步合成」，而「優雅下降」來自**逐樣本裁剪**而非放大——這點 docstring 已寫對，報告務必照搬，否則「subsampled-Gaussian 會計」的標題會誤導。
  2. **privacy 軸與 utility 軸不是同一個釋出物**：utility 是「50 輪 delta clip+noise」的實際準確率；privacy 是「單一梯度 clip+noise」的 DLG 重建。兩者共用 `z`/`ε` 標籤，但 ε 是訓練預算、DLG 是單次釋出洩漏——這是這類圖的慣例，但要在 caption 註一句。

### 2.4 Step 3-1 — HE / CKKS（✅ 正確）

- **金鑰處理正確**：client 持 full context（含 secret key），server 只拿 `make_context_public()` 後的公開 context；`decrypt()` 在 server 端拋例外（demo 已驗證）。✔ 「結構性防禦」論述成立——server 連 DLG 目標函數都湊不出。
- **密文聚合正確**：`aggregate_encrypted` 用明文權重 `n_i/N` 乘密文再相加（CKKS 支援 ct+ct、ct×pt），解密後等於加權平均 delta，相對 L2 誤差 1.6e-7。✔ 這解釋了為何準確率幾乎不掉。

### 2.5 測試覆蓋

30 個測試（`test_fl_aggregation` 6、`test_dp` 18、`test_he` 3、`test_metrics_and_attack` 3），涵蓋 FedAvg 等價性、RDP 會計、DP-SGD、Dirichlet、HE round-trip、iDLG 標籤、PSNR/SSIM。覆蓋面足夠支撐結論。✔

---

## 3. 實驗結果是否符合預期？

逐步用實際數據檢視（數據來自 `results/metrics/*.csv`）：

| 實驗 | 數據 | 是否符合預期 | 區分性 |
|------|------|:---:|:---:|
| **FL 收斂** | FedAvg 0.888、centralized 0.913（round 50） | ✅ 符合（FL≈上界） | 🟡 兩線重疊（這是「沒變差」的正面訊號，但缺下界對照） |
| **non-IID drift** | IID 0.908 / α=1.0 0.896 / α=0.1 0.804；α=0.1 變異 ~3.3× | ✅ 符合 | 🟢 α=0.1 清楚拉開；🟡 α=1.0 與 IID 幾乎重合 |
| **DLG demo** | 8/8、73–87 dB | ✅ 完美 | 🟢 極清楚 |
| **batch sweep** | 80→70→57→18 dB（bs 1/2/4/8） | ✅ 單調崩潰 | 🟢 在 bs 4→8 跨過 20 dB 門檻，漂亮 |
| **leakage vs round** | 成功率 r1=100% → r40=0% | ✅ 有臨界 | 🟡 中段非單調（r2=62%→r4=54%→r8=88%），cliff 不夠乾淨 |
| **real delta** | 1-step SGD 84 dB / 10-step Adam 5.6 dB | ✅ 符合 | 🟢 對比強烈 |
| **DP-FedAvg** | z=0.002：acc 0.90 / DLG 11.6 dB / ε 7.8e6；z=0.05：acc 0.19；z=1.0：ε59 acc 0.03 | ✅ 符合（經驗便宜、形式不可用） | 🟢 取捨清楚；🟡 ε 量級荒謬（需註解）、x 軸後半段全是 chance（冗點） |
| **DP-SGD** | acc 0.95→0.76(z=1,ε296)→0.56(z=1.5)→chance(z=2)；**DLG 恆 5–6 dB** | ✅ accuracy 軸符合（優雅下降） | 🔴 **DLG 軸退化成平線**（見問題 #1） |
| **subsampling** | q=0.5,z=0.5→ε454,acc0.78；q=0.25→崩 | ✅ 符合 | 🟢 對比清楚 |
| **HE** | acc≈明文（gap≤0.025）、誤差1.6e-7、11.6s、32.7× | ✅ 符合 | 🟢 清楚 |

**總體：結果方向全部符合預期。** 區分性最弱的兩處：DP-SGD 的洩漏軸（🔴 退化）與 leakage-vs-round 的中段噪聲（🟡），其餘只是錦上添花。

---

## 4. 問題清單（依優先級）

### 🔴 P0 — 影響圖表正確解讀

**P0-1　DP-SGD 圖的 DLG-洩漏軸退化為平線**
`run_dp_sgd.py::privacy_sweep` 用 `dp_fedavg_grad_list(clean, CLIP_NORM=10, z)`，而單樣本梯度 norm=**24**。`z=0`（無噪音、僅裁剪到 10）DLG 就只有 **6.39 dB**，之後各 `z` 都維持 ~5 dB。圖上紫線是條貼地直線，**無法呈現「噪音越大、洩漏越少」的取捨**，且與 README 既有說明（「DLG 已被裁剪本身擋住、與 z 無關」）一致——代表這是已知的呈現弱點。
**修法（二擇一）**：
- (A)【建議】把 privacy demo 的 clip 改成**梯度自身 norm**（與 `run_dp.py` 一致，`z=0` 為 no-op→84 dB），讓洩漏曲線**隨 z 下降**，便能直接對比「DP-SGD 要多少 `z` 才擋住 DLG」vs DP-FedAvg。
- (B) 直接從 DP-SGD 圖**移除洩漏軸**，只留 accuracy–ε（DP-SGD 真正的賣點是優雅下降），把「裁剪本身即擋住單梯度 DLG」改用一句話 + 一張小圖交代。

### 🟠 P1 — 提升結果價值與區分性

**P1-1　Step 1 補 local-only 下界 baseline**
目前 centralized ≈ FedAvg 只說明「聯邦沒有變差」。再加一條「每個 client 只用自己分片獨立訓練、不聚合」的曲線（取各 client 測試準確率的平均），就能呈現 **centralized ≥ FedAvg ≫ local-only**，把「為什麼要聯邦」量化出來。實作極小（迴圈裡少做 aggregate 即可），加分明顯。

**P1-2　leakage-vs-round 的 cliff 不夠乾淨**
中段非單調（r2 跌到 62% 又在 r8 回升到 88%）會引發「為何洩漏先降又升」的疑問。建議把每輪受害者數從 8 增到 **16**（成功率分母由 24→48），曲線會更平滑、cliff 更可辨識；snapshot round 也可加密 r3/r5/r35 讓臨界區更細。

**P1-3　DP-SGD / DP-FedAvg 的 ε「可用點」缺一個對照錨**
兩條曲線都沒有任何「ε<10 且準確率非 chance」的點（這正是維度詛咒的結論）。建議在 DP-SGD 表格**明確補一列 ε≈10 的點**（即使 acc=chance），讓「想要單位數 ε 就得放棄準確率」這句結論有**具體數字**支撐，而非僅靠文字。

### 🟡 P2 — 打磨

- **P2-1　DP-FedAvg 的 `Z_VALUES` 後半冗餘**：z=0.1/0.2/0.5/1.0 都已 chance，x 軸 14 點有一半在死區。把點重新分配到 knee（z∈[0,0.05]）能讓曲線更細緻、ε 標籤不擁擠。
- **P2-2　non-IID 的「mild」設定意義不大**：α=1.0（0.896）與 IID（0.908）幾乎重合。把中間檔改成 **α=0.3 或 0.5**，可得 IID > α=0.5 > α=0.1 的單調三分，故事更乾淨。
- **P2-3　跨實驗 baseline 不一致**：FL（Adam lr=0.01→0.89）、DP-SGD（SGD lr=0.5→0.95）、HE（Adam→0.875–0.91）的非私有準確率不同。不影響各自結論，但報告若要橫向比「DP vs HE 的準確率代價」，需點明各自基準。
- **P2-4　ε 量級在圖上易誤讀**：DP-FedAvg 出現 ε≈3e7。建議在 caption 直接寫「無 subsampling 放大，故 ε 形式上無意義」，避免老師誤以為算錯。

### 🟢 P3 — 交付合規（PDF 明文要求，非程式問題）

- **P3-1**　PDF 要 **PPT + Word 報告 + zip（`TeamName_FinalProject.zip`）**；目前只有 Markdown。需把 `README.md` + 本報告整理成 **Word**，並做一份 **PPT**。
- **P3-2**　報告**開頭須列出完成步驟**（"Please list it in the beginning of your report"）——確保 Word 版第一段就寫「Step 1 / 2 / 3 / 3-1 全數完成」。
- **P3-3**　分工表（依使用者指示本次略過，但提交前需補）。

---

## 5. 參數調整建議（一覽表）

| 實驗 | 參數 | 現值 | 建議 | 預期效果 |
|------|------|------|------|---------|
| DP-SGD privacy | privacy-demo clip | 固定 `C=10` | 改為**梯度自身 norm**（或拿掉此軸） | 洩漏曲線隨 z 下降，呈現真正取捨（P0-1） |
| Step 1 | baseline 組數 | centralized + FedAvg | **+ local-only** | 三方對照、量化聯邦價值（P1-1） |
| leakage vs round | 每輪受害者數 | 8（×3 seed=24） | **16**（×3=48） | cliff 更平滑可辨（P1-2） |
| leakage vs round | `SNAPSHOT_ROUNDS` | …8,10,12,15,18,20… | 加 **3,5,35** | 臨界區解析更細 |
| DP-FedAvg | `Z_VALUES` | 14 點（半在死區） | 密化 z∈[0,0.05]、刪 z≥0.2 | 曲線更細、ε 標籤不擠（P2-1） |
| DP-SGD | 補 ε 錨點 | 最小有訊號 ε=156 | 補一列 **ε≈10**（acc=chance） | 「ε<10 不可用」有數字（P1-3） |
| non-IID | 中間 α | 1.0（≈IID） | **0.3 或 0.5** | IID>α=0.5>α=0.1 單調三分（P2-2） |
| FL 收斂 | `lr`/`local_epochs` | 0.01 / 1 | （可選）lr 0.02 或 E=2 | 拉近 FL→centralized，gap 更小 |

> 上述都是**不改架構、只調參數/加一條 baseline** 的修改，風險低、對「結果可辨識度」邊際效益高。

---

## 6. 改進策略與優先順序（建議執行順序）

1. **【P0-1，~30 分鐘】** 修 DP-SGD privacy demo 的 clip（改成梯度自身 norm），重跑 `run_dp_sgd.py`，確認紫線變成隨 z 下降的曲線。← **最先做，CP 值最高**
2. **【P1-1，~30 分鐘】** Step 1 加 local-only baseline，重跑 `run_fl.py`，更新收斂圖與「重點數據」表。
3. **【P1-2 / P1-3，~1 小時】** 受害者數 8→16、補 snapshot rounds、DP-SGD 補 ε≈10 列；重跑 `run_attack.py`、`run_dp_sgd.py`。
4. **【P2 批次，~30 分鐘】** non-IID 中間檔改 α=0.5、DP-FedAvg `Z_VALUES` 重分配、各圖 caption 補 ε 註解；重跑對應 script。
5. **【P3 交付，~2 小時】** 把 `README.md` + 本報告匯出成 **Word**、做 **PPT**（開頭列完成步驟）、補分工表、打包 `TeamName_FinalProject.zip`。

**驗證**：每步重跑後 `uv run pytest`（30 tests 應全綠），並核對 `results/metrics/*.csv` 與圖一致。

---

## 7. 一句話總評

> 這是一份**正確、誠實、超出作業要求**的實作——四步全做、threat-model 不浮誇、會計數值經得起手算。要再上一層，重點不在「修 bug」（幾乎沒有），而在**讓三張取捨圖的對比更銳利**（修 DP-SGD 洩漏軸、補 local-only 下界、平滑 round cliff）與**補齊 Word/PPT/zip 交付**。把第 6 節前三步做完，報告的「可辨識性與區分性」會明顯提升。
