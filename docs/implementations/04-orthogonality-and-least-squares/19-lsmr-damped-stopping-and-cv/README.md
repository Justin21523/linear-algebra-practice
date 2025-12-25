# 實作說明：Damped LSMR + stopping + CV（04-orthogonality-and-least-squares/19-lsmr-damped-stopping-and-cv）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/19-lsmr-damped-stopping-and-cv/`
- 單元概念：`04-orthogonality-and-least-squares/19-lsmr-damped-stopping-and-cv/README.md`
- 實作檔案：
  - `04-orthogonality-and-least-squares/19-lsmr-damped-stopping-and-cv/python/lsmr_damped_stopping_and_cv_numpy.py`
- 文件路徑：`docs/implementations/04-orthogonality-and-least-squares/19-lsmr-damped-stopping-and-cv/README.md`

## 目標與背景

- 本實作示範：用 **Damped LSMR**（可視為 normal equations 上的 MINRES 等價）解 Ridge：
  `min ‖Ax-b‖² + damp²‖x‖²`，並用實務 stopping criteria 收斂驗收。
- 接著用 **k-fold CV** 以 validation RMSE 選 `damp`（等價 Ridge `λ=damp²`），讓流程更貼近 ML 超參數調整。

## 如何執行

```bash
cd 04-orthogonality-and-least-squares/19-lsmr-damped-stopping-and-cv/python
python3 lsmr_damped_stopping_and_cv_numpy.py
```

## 核心做法（重點步驟）

1. 透過擴增系統改寫 Ridge：
   - `min ‖Ax-b‖² + damp²‖x‖² == min ‖[A; damp I]x - [b; 0]‖²`
2. 用 LSMR 的觀點（normal equations）解：
   - `(AᵀA + damp²I)x = Aᵀb`（但不形成 `AᵀA`，只做 `A v` 與 `Aᵀ u`）
3. stopping criteria（LSQR 風格）同時驗：
   - augmented residual：`‖[Ax-b; damp x]‖`
   - gradient/optimality：`‖Aᵀ(Ax-b)+damp²x‖`
4. k-fold CV 掃 `damp`，印出 mean±std + ASCII 曲線，挑 validation RMSE 最小者。

## 詳細說明（繁中）

### `damp` 是什麼？為什麼等價 Ridge `λ`？

本單元用的目標函數是：

- `min_x ‖Ax-b‖² + damp²‖x‖²`

若你習慣 Ridge 的寫法 `min ‖Ax-b‖² + λ‖x‖²`，那就有：

- `λ = damp²`

### 為什麼 stopping 用 `‖Aᵀr + damp²x‖` 很自然？

令 `r = Ax-b`，則 Ridge 目標的梯度（stationarity/optimality）為：

- `∇f(x) = Aᵀ(Ax-b) + damp²x = Aᵀr + damp²x`

因此 `‖Aᵀr + damp²x‖` 越小，代表你越接近最優解。

### 為什麼還看 augmented residual `‖[Ax-b; damp x]‖`？

它直接對應擴增系統的殘差：

- `r_aug = [Ax-b; damp x]`

這個量同時反映「資料擬合」與「正則化項」的折衷，並可搭配 `atol/btol` 做混合（absolute/relative）停止條件。

## 程式碼區段（節錄 + 解釋）

> 節錄 ridge 梯度與兩個主要 stopping test（簡化表示）。

```text
r = A @ x - b
grad = (A.T @ r) + (damp * damp) * x
rnorm_aug = sqrt(||r||^2 + (damp||x||)^2)

if rnorm_aug <= btol*||b|| + atol*||A_aug||*||x||: stop
if ||grad|| <= atol*||A_aug||*rnorm_aug: stop
```

- `grad`：就是 `Aᵀ(Ax-b)+damp²x`，對應 Ridge 最優條件。
- 兩個 stopping test 分別檢查「殘差是否足夠小」與「最優條件是否足夠滿足」。

## 驗證方式與預期輸出

- CV 表格會印：
  - 每個 `damp` 的 train/val RMSE mean±std
  - `||x||_2(mean)`：`damp` 越大通常越小（更強的 shrinkage）
  - ASCII 曲線：val 越小條越長（更容易看最佳區域）
- 最終會印：
  - `stop reason`、迭代數
  - `||Ax-b||`、`||[Ax-b;damp x]||`、`||A^T r + damp^2 x||`

