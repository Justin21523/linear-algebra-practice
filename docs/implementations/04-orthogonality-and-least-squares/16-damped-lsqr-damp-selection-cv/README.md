# 實作說明：用 k-fold CV 選 damp（04-orthogonality-and-least-squares/16-damped-lsqr-damp-selection-cv）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/16-damped-lsqr-damp-selection-cv/`
- 單元概念：`04-orthogonality-and-least-squares/16-damped-lsqr-damp-selection-cv/README.md`
- 實作檔案：
  - `04-orthogonality-and-least-squares/16-damped-lsqr-damp-selection-cv/python/damped_lsqr_damp_selection_cv_numpy.py`
- 文件路徑：`docs/implementations/04-orthogonality-and-least-squares/16-damped-lsqr-damp-selection-cv/README.md`

## 目標與背景

- 本實作示範：用 **k-fold cross-validation** 選擇 Damped LSQR 的 `damp`（等價 Ridge 的 `λ=damp²`）。
- 你會看到典型的 U-shape：`damp` 太小 → 解不穩/過擬合；`damp` 太大 → 解被過度縮小/欠擬合。
- 這個流程非常貼近 ML：把 `damp` 當作超參數，用驗證誤差選擇，而不是手動猜。

## 如何執行

```bash
cd 04-orthogonality-and-least-squares/16-damped-lsqr-damp-selection-cv/python
python3 damped_lsqr_damp_selection_cv_numpy.py
```

## 核心做法（重點步驟）

1. 建資料：用 Vandermonde（高次多項式）製造病態設計矩陣 `A`，再加噪得到 `b`。
2. 針對每個候選 `damp`：
   - 對每個 fold：用 Damped LSQR 在訓練集解 `min(‖Ax-b‖² + damp²‖x‖²)` 得到 `x_hat`
   - 在驗證集算 RMSE，收集 fold 的分數
3. 把每個 `damp` 的 val RMSE 做 mean±std，挑 mean 最小者
4. 用最佳 `damp` 在全資料重訓，並驗證最佳化條件 `Aᵀ(Ax-b)+damp²x≈0`

## 詳細說明（繁中）

### `damp` 與 Ridge `λ` 的關係

本單元的目標函數是：

- `min_x ‖Ax-b‖² + damp²‖x‖²`

若你習慣 Ridge 的寫法 `min ‖Ax-b‖² + λ‖x‖²`，那就是 `λ = damp²`。

### 為什麼 Damped LSQR 不用形成 `AᵀA`？

用擴增系統改寫：

- `min_x ‖Ax-b‖² + damp²‖x‖²  ==  min_x ‖[A; damp I]x - [b; 0]‖²`

這樣 LSQR 只需要 `(A v)` 與 `(Aᵀ u)`（以及 `damp*v`、`damp*u_bottom`）就能跑，不必形成 `AᵀA`，在大型/稀疏問題更重要。

### CV 為什麼能選出合理的 `damp`？

- `damp` 太小：模型自由度高，容易把噪聲也擬合進去（validation RMSE 上升）。
- `damp` 太大：係數被壓太小，模型表達能力不足（validation RMSE 上升）。
- 中間區域通常最好：validation RMSE 最低，就是你要的 `damp`。

## 程式碼區段（節錄 + 解釋）

> 節錄「擴增系統」的 matvec，這是 Damped LSQR 的核心。

```text
def matvec_A_aug(v):
    top = A @ v
    bottom = damp * v
    return np.concatenate([top, bottom])

def matvec_AT_aug(u_aug):
    u_top = u_aug[:m]
    u_bottom = u_aug[m:]
    return (A.T @ u_top) + (damp * u_bottom)
```

- 這段在做什麼：用 `[A; damp I]` 的算子形式提供 `A_aug v` 與 `A_augᵀ u`，讓 LSQR 能直接解「帶正則化」的問題。
- 為什麼這樣寫：避免形成 `AᵀA`，也避免真的把 `A_aug` 拼出來（大型問題會很大）。

## 驗證方式與預期輸出

- `k-fold CV` 表格會印：
  - 每個 `damp` 的 train/val RMSE mean±std
  - `||x||_2(mean)`：`damp` 越大通常越小（正則化效果）
  - ASCII 曲線：val 越小條越長（更容易看出最佳區域）
- 選出最佳 `damp` 後，程式會在全資料重訓並印：
  - `||A^T r + damp^2 x||_2` 應該很小（代表接近最優）

