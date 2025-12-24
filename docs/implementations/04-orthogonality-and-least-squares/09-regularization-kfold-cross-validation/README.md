# 實作說明：09-regularization-kfold-cross-validation（04-orthogonality-and-least-squares）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/09-regularization-kfold-cross-validation/`
- 概念說明：`04-orthogonality-and-least-squares/09-regularization-kfold-cross-validation/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/09-regularization-kfold-cross-validation/python/regularization_kfold_cv_numpy.py`

## 目標與背景

上一個單元用 hold-out（一次切分）選 `λ/k`，這很直覺但可能不穩：如果「剛好」驗證集抽到比較難/比較簡單的樣本，你選到的參數會晃動。

因此本單元改用 **k-fold cross-validation**：

- 把資料切成 `k` 份（folds）
- 輪流用其中 1 份當驗證集、另外 `k-1` 份當訓練集
- 對同一個超參數（例如某個 `λ`）會得到 `k` 個驗證誤差
- 用 **平均驗證誤差**（以及標準差）來做更穩定的選擇

並額外做一個實用小技巧：不引入繪圖套件，直接用 **ASCII bar** 輸出「超參數 vs 驗證 RMSE」的簡易曲線。

## 如何執行

> 需要 `numpy`：請先依 `requirements.txt` 安裝（見 repo 根目錄 `README.md` 的 venv 範例）。

```bash
cd 04-orthogonality-and-least-squares/09-regularization-kfold-cross-validation/python
python3 regularization_kfold_cv_numpy.py
```

## 核心做法（重點）

1. 仍使用 Vandermonde（多項式）設計矩陣，刻意讓問題病態以凸顯正則化與參數選擇的重要性。
2. 用 `k_fold_splits()` 產生 `k` 組 `(train_idx, val_idx)`。
3. 針對每個候選 `λ`（Ridge）或 `k`（TSVD）：
   - 在每個 fold 上 fit 一次、算一次 train/val RMSE
   - 匯總成 `mean ± std`（比 hold-out 更不受單次切分影響）
4. 以 `val_rmse_mean` 最小者為 best，並印出 ASCII bar（越長代表越好/越低的驗證 RMSE）。

## 詳細說明

### 1) k-fold cross-validation 的核心概念

假設有 `k=5`：

- 你會做 5 次訓練/驗證
- 每次驗證集都是不同的 1/5 資料
- 最後對每個超參數得到 5 個驗證 RMSE

你不再只看一次切分的結果，而是看「平均表現」與「波動程度」：

- 平均驗證 RMSE（越小越好）
- 驗證 RMSE 的標準差（越小通常代表更穩定）

### 2) 為什麼要同時看 train 與 val？

典型現象：

- 正則化越強（`λ` 越大、或 TSVD `k` 越小），模型越受限：
  - train RMSE 往往變差（偏差上升）
  - val RMSE 先變好再變差（U 形）

因此看 train/val 兩者能更直觀地理解 bias–variance tradeoff。

### 3) ASCII 曲線怎麼做？

本單元把每個超參數的 `val_rmse_mean` 映射成一條 bar：

- 先取整個序列的 `vmin` 與 `vmax`
- 令 `score = (vmax - value)/(vmax - vmin)`（越小 RMSE → 分數越高）
- bar 長度 `≈ score * width`

節錄自 `regularization_kfold_cv_numpy.py`：

```python
score = (vmax - value) / (vmax - vmin)  # EN: Map lower RMSE -> higher score in [0,1].
n = int(round(score * width))  # EN: Convert score to a character count.
return "#" * n  # EN: Return the bar string.
```

## 驗證方式（你應該看到的現象）

- Ridge 的 `λ` 表格/曲線應呈現「中間某段最好」的趨勢（val RMSE 最小點）。
- TSVD 的 `k` 表格/曲線常會在「太小欠擬合」與「太大過擬合/不穩」之間有最佳點。
- 在輸出中你會看到 `mean±std`：若某些參數的 `std` 明顯較大，代表該設定在不同 folds 上表現不穩。

