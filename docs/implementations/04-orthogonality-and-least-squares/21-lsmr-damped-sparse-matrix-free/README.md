# 實作說明：Sparse / Matrix-Free Ridge（CSR）+ Damped LSMR + k-fold CV 曲線總成本 + CountSketch rand-QR（04-orthogonality-and-least-squares/21-lsmr-damped-sparse-matrix-free）

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
- 在 sparse 場景中比較三種右預條件化：`none / col-scaling / rand-QR(CountSketch)`
- 把「超參數選擇」改成更貼近 ML 的做法：**k-fold CV 掃 `damp` 曲線**，同時比較整條曲線的總成本（時間/迭代數）
- 延續單元 20 的觀念：除了挑 best `damp`，也要看「掃完整條 curve」要花多少（尤其 `rand-QR` 類 build 成本很敏感）

## 如何執行

```bash
cd 04-orthogonality-and-least-squares/21-lsmr-damped-sparse-matrix-free/python
python3 lsmr_damped_sparse_matrix_free_numpy.py
```

## 核心做法（重點步驟）

1. 題目：Ridge / damped least squares
   - `min_x ‖Ax-b‖² + damp²‖x‖²`（`λ = damp²`）
2. Matrix-free solver：Damped LSMR（教學版）
   - `A` 用 CSR 保存，迭代中只呼叫 `A@x` 與 `Aᵀ@y`
   - 用 `‖Aᵀ(Ax-b)+damp²x‖` 驗收是否接近最優
3. 三種右預條件化（x = M⁻¹y）
   - none：`M=I`
   - col-scaling：`M=diag(sqrt(‖A[:,j]‖² + damp²))`
   - rand-QR（CountSketch）：用 sketch `S` 做 `S[A;damp I] = Q R`，取 `M=R`
4. k-fold CV 掃 `damp` 曲線
   - 每個 `damp`：在 train fold fit，算 train/val RMSE（mean±std）
   - 同時累積整條曲線：`total_build_s / total_solve_s / total_iters`
5. 掃 curve 做快（speedups）
   - continuation/warm-start：`damp` 由大到小，沿路把上一點的解當下一點起點
   - rand-QR reuse：每個 fold 共用同一份 CountSketch（`shared sketch`），或更激進地共用同一個 `R`（`fixed-R`）

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

### k-fold CV 掃 damp 曲線（為什麼要看「整條曲線」總成本？）

在真實 ML 管線中，你很少只訓練一次；你會：

- 針對多個 `damp` 候選點，掃一整條曲線
- 而且每個候選點通常要配合 k-fold CV（更穩的泛化估計）

如果你只看「最佳驗證誤差」，你很容易忽略一件更實務的事：**同樣跑完一條曲線，不同方法的總成本可能差非常多**。

因此本單元在每次 CV sweep 會同時報告：

- `total_build_s`：整條曲線（所有 folds × 所有 damps）預條件器建置時間總和
- `total_solve_s`：整條曲線求解時間總和
- `total_iters`：整條曲線總迭代數（在 matrix-free 場景常可視為 matvec 成本 proxy）

CV 的品質部分，則是用：

- `train_rmse(mean±std)`：訓練集 RMSE（跨 folds 的平均與標準差）
- `val_rmse(mean±std)`：驗證集 RMSE（跨 folds 的平均與標準差）

### sparse rand-QR（CountSketch）為什麼特別適合？

在 dense 場景（單元 20），rand-QR 類預條件器常用「密集的隨機矩陣」做 sketch；但在 sparse 場景，你更想要：

- 建置成本接近 `O(nnz)`（跟非零元素數量成正比）
- 不把稀疏矩陣 `A` 轉成 dense

本單元用的 CountSketch（對「列」做 hashing + 隨機正負號）可以做到：

- `SA_top = S_top A`：用一次掃過 CSR（`O(nnz)`）就能建好
- 針對 Ridge 的擴增矩陣 `A_aug = [A; damp I]`，identity 部分可以「不用掃 A」就更新：
  - 每個特徵 `j` 在 sketch 的 bucket `h_bottom[j]` 上，加上 `damp * scale * sign_bottom[j]`

然後對 `S A_aug` 做 QR：

- `S A_aug = Q R`
- 取 `M = R` 當右預條件器（`x = M⁻¹ y`）

### 掃 curve 做快：warm-start + shared sketch + fixed-R（取捨是什麼？）

本單元做了三種「掃曲線」策略（同一個 CV splits、同一組 `damp` grid）：

1. baseline
   - 每個 `damp` 都 cold-start（`x=0`）
   - rand-QR 每個 `damp` 都重建 CountSketch + QR（最慢，但最直覺）
2. speedups（最推薦先看）
   - continuation/warm-start：`damp` 由大到小，沿路用上一點的 `x` 當起點
   - rand-QR 使用 `shared sketch`：每 fold 只建一次 CountSketch，之後每個 `damp` 只需要更新 identity 影響並做 QR
3. 進階：fixed-R
   - 每 fold 只在某個參考 `damp_ref` 做一次 QR 取 `R_ref`，整條曲線都重用
   - build 成本最低，但因為 `M` 不再跟 `damp` 同步，迭代數/最佳 `damp`/品質可能會跟 shared-sketch 不同（要用 `total_iters` 與 `best_val_rmse` 判斷值不值得）

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

> CountSketch rand-QR（sparse 版）的核心更新：先建 `SA_top = S_top A`，每個 `damp` 只補上 identity 部分再 QR 取 `R`。

```text
# one-time per fold:
SA_top = CountSketch(A)            # O(nnz)

# per damp:
A_sketch = SA_top.copy()
for j in 0..n-1:
  A_sketch[h_bottom[j], j] += damp * scale * sign_bottom[j]
R = qr(A_sketch).R
```

> k-fold CV sweep（每個 preconditioner）會輸出「曲線表格 + 整條曲線總成本」，並在最後印 baseline vs speedups 對照表。

```text
for fold in folds:
  build A_tr, A_va
  maybe build shared_sketch (rand-QR only)
  for damp in damps:
    x_init = prev_x if warm_start else None
    fit on A_tr, evaluate on A_va
    accumulate total_build_s / total_solve_s / total_iters
```

## 預期輸出

- **Per-damp solver comparison（full data）**：比較 `none / col-scaling / rand-QR(countsketch)` 的 iters、`‖Ax-b‖`、`‖Aᵀr+damp²x‖`、build/solve time。
- **k-fold CV sweep（baseline）**：對每個 preconditioner 印 CV 表（train/val RMSE mean±std + ASCII 曲線），並印 `total_build_s / total_solve_s / total_iters`。
- **Baseline vs speedups**：把 baseline 與 speedups（warm-start + rand-QR shared sketch）放到同一張表比「時間/迭代數/品質」。
- **rand-QR reuse variants**：額外印一行 fixed-R 的總成本與 best `damp`/best val，方便你對照 shared-sketch 的取捨。
