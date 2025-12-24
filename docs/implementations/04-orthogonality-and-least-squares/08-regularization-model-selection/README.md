# 實作說明：08-regularization-model-selection（04-orthogonality-and-least-squares）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/08-regularization-model-selection/`
- 概念說明：`04-orthogonality-and-least-squares/08-regularization-model-selection/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/08-regularization-model-selection/python/regularization_model_selection_numpy.py`

## 目標與背景

延續上一個單元的「Ridge/TSVD 正則化」，本單元把重點放在 **怎麼選參數**：

- Ridge：選 `λ`（越大越強的 shrinkage）
- TSVD：選 `k`（保留前幾個奇異值方向，像是「只用前 k 個可靠方向回歸」）

用 ML 的語言就是：**用驗證集挑超參數（hyperparameters）**，避免只看訓練誤差造成過擬合。

## 如何執行

> 需要 `numpy`：請先依 `requirements.txt` 安裝（見 repo 根目錄 `README.md` 的 venv 範例）。

```bash
cd 04-orthogonality-and-least-squares/08-regularization-model-selection/python
python3 regularization_model_selection_numpy.py
```

## 核心做法（重點）

1. 建立「多項式設計矩陣」（Vandermonde），刻意讓問題更病態、特徵更相關。
2. 做 hold-out split（訓練/驗證）：
   - 對一串 `λ` 依序解 Ridge，計算 train/val RMSE，選 `val RMSE` 最小的 `λ`
   - 對一串 `k` 依序解 TSVD，計算 train/val RMSE，選 `val RMSE` 最小的 `k`
3. 做 bias–variance 觀察：
   - 固定 `A` 與 `x_true`，重複產生不同噪聲的 `b`
   - 看 `x̂` 的平均（bias）與波動（variance）如何隨正則化變化

## 詳細說明

### 1) 為什麼驗證誤差常是 U 形？

- `λ` 太小 / `k` 太大：模型太自由，會把噪聲也「學」進去 → 高變異（variance）、驗證誤差上升
- `λ` 太大 / `k` 太小：模型太受限，連真實結構也表達不了 → 高偏差（bias）、驗證誤差上升

因此常看到：**訓練誤差單調變差，但驗證誤差先降後升**。

### 2) hold-out 怎麼選 `λ`？

節錄自 `regularization_model_selection_numpy.py`：對每個 `λ` 做一次 fit + score，並以 `val_rmse` 最小者為 best：

```python
ridge_lambdas = np.concatenate(([0.0], np.logspace(-12, 2, num=15)))  # EN: Candidate λ values including λ=0 baseline.
for lam in ridge_lambdas:  # EN: Loop over λ candidates.
    metrics = fit_and_score(
        A_train=A_train,
        b_train=b_train,
        A_val=A_val,
        b_val=b_val,
        solver=lambda A_in, b_in, lam=lam: solve_ridge_svd_filter(A_in, b_in, lam),
        solver_key=f"λ={lam:.3e}",
    )
```

### 3) bias–variance 怎麼在程式裡「量化」？

本單元用係數層級的 proxy：

- `bias ≈ ‖E[x̂] - x_true‖`
- `variance ≈ RMS(‖x̂ - E[x̂]‖)`

並同時統計驗證集 RMSE 在多次噪聲下的平均與標準差（越小代表越穩定）。

## 驗證方式（你應該看到的現象）

- Ridge：`λ` 變大時 `‖x̂‖` 會下降、穩定性變好，但 train RMSE 會變差；val RMSE 會有最佳點。
- TSVD：`k` 太大時會把不可靠方向也用進去；`k` 太小會欠擬合；val RMSE 通常也有最佳點。

