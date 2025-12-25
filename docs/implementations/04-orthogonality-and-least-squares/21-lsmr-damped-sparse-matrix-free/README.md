# 實作說明：Sparse / Matrix-Free Ridge（CSR）+ Damped LSMR + 預條件化（04-orthogonality-and-least-squares/21-lsmr-damped-sparse-matrix-free）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/21-lsmr-damped-sparse-matrix-free/`
- 單元概念：`04-orthogonality-and-least-squares/21-lsmr-damped-sparse-matrix-free/README.md`
- 實作檔案：
  - `04-orthogonality-and-least-squares/21-lsmr-damped-sparse-matrix-free/python/lsmr_damped_sparse_matrix_free_numpy.py`
- 文件路徑：`docs/implementations/04-orthogonality-and-least-squares/21-lsmr-damped-sparse-matrix-free/README.md`

## 目標與背景

這個單元把資料換成「大規模回歸/推薦系統常見」的形式：**稀疏設計矩陣**。重點不是 CSR 本身，而是把思維轉成：

- 你通常不會也不該形成 `AᵀA`
- solver 只需要 `A @ x` 與 `Aᵀ @ y`（matrix-free）
- 仍然可以用 Ridge 最優條件驗收：`‖Aᵀ(Ax-b)+damp²x‖` 是否接近 0

## 如何執行

```bash
cd 04-orthogonality-and-least-squares/21-lsmr-damped-sparse-matrix-free/python
python3 lsmr_damped_sparse_matrix_free_numpy.py
```

## 詳細說明（繁中）

### CSR 與 matrix-free 的意義

CSR（Compressed Sparse Row）用三個陣列描述稀疏矩陣：

- `data`：所有非零值
- `indices`：每個非零值對應的 column index
- `indptr`：每一列在 `data/indices` 的切片範圍

在大規模設定下，你要養成的習慣是：把 `A` 當成「能做 matvec 的黑盒」，而不是「能隨便轉置/平方/求逆的 dense 陣列」。

本單元的 solver 只用兩個函數：

- `matvec_A(x) = A x`
- `matvec_AT(y) = Aᵀ y`

### Ridge 的驗收量（為什麼看 `‖Aᵀr + damp²x‖`）

令 `r = Ax-b`，Ridge 目標：

- `f(x)=‖Ax-b‖² + damp²‖x‖²`

其梯度（忽略常數 2）：

- `∇f(x)=Aᵀ(Ax-b)+damp²x = Aᵀr + damp²x`

所以 `‖Aᵀr + damp²x‖` 越小，代表越接近最優解；這在稀疏/矩陣乘子算子下依然成立。

### 預條件化（column scaling）在 sparse 場景為什麼常用？

在 sparse regression 裡，欄位尺度差異通常很大（尤其是 one-hot / count feature / hashing feature）。
column scaling 的右預條件化：

- `x = D⁻¹ y`
- `D_j = sqrt(‖A[:,j]‖² + damp²)`

等同先把特徵「重新縮放」到比較均衡的尺度，常能顯著降低 Krylov 法的迭代數。

### continuation / warm-start（沿 damp 路徑接續）

本單元示範：把 damp 從大到小掃過去，並把前一個 `damp` 的解當作下一個點的 `x_init`。
這通常會比每個點都從 `x=0` 開始省很多迭代（尤其是你在做超參數路徑或 CV 時）。

## 程式碼區段（節錄 + 解釋）

> CSR 的 matvec（只依賴 `data/indices/indptr`，不需要 dense `A`）。

```text
for each row i:
  y[i] = dot(data[row], x[indices[row]])
```

> warm-start 的等價 RHS（解增量 `delta`，最後合成 `x = x0 + delta`）。

```text
b_top    = b - A @ x0
b_bottom = -damp * x0
delta    = solver(A, b_top, b_bottom)
x_hat    = x0 + delta
```

## 預期輸出

- **Cold-start comparison**：在同一個 sparse 資料上，比較 `none` vs `col-scaling` 的迭代數、`‖Ax-b‖`、`‖Aᵀr+damp²x‖`
- **Continuation / warm-start**：列出同一路徑下 cold vs warm 的 `total iters`、`solve_s` 與 `speedup`
