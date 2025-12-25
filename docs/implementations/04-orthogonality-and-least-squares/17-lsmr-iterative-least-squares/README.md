# 實作說明：LSMR（04-orthogonality-and-least-squares/17-lsmr-iterative-least-squares）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/17-lsmr-iterative-least-squares/`
- 單元概念：`04-orthogonality-and-least-squares/17-lsmr-iterative-least-squares/README.md`
- 實作檔案：
  - `04-orthogonality-and-least-squares/17-lsmr-iterative-least-squares/python/lsmr_manual.py`
- 文件路徑：`docs/implementations/04-orthogonality-and-least-squares/17-lsmr-iterative-least-squares/README.md`

## 目標與背景

- 本實作示範：用 **LSMR 的核心觀念**（等價於 normal equations 上的 MINRES）來解最小平方 `min‖Ax-b‖`，但全程只用 `A v` 與 `Aᵀ u`。
- 你會同時看到 LSQR（baseline）與 LSMR 的輸出，並比較：
  - 迭代數
  - `‖Ax-b‖`（資料殘差）
  - `‖Aᵀ(Ax-b)‖`（正常方程殘差 / optimality）

## 如何執行

```bash
cd 04-orthogonality-and-least-squares/17-lsmr-iterative-least-squares/python
python3 lsmr_manual.py
```

## 核心做法（重點步驟）

1. 建一個病態最小平方題：用指定奇異值譜合成 `A`，讓 `cond(A)` 很大。
2. 跑 LSQR：用 Golub–Kahan bidiagonalization + recurrences 解 `min‖Ax-b‖`（停止條件用 `‖Aᵀr‖`）。
3. 跑 LSMR（教學版）：
   - 用 Golub–Kahan 產生 bidiagonal `B_k`
   - 形成 `T_k = B_kᵀ B_k`（k×k 三對角）
   - 在每個 k 解小型 least-squares：`y_k = argmin‖‖Aᵀb‖e1 - T_k y‖`
   - 回推：`x_k = V_k y_k`（`V_k` 由 bidiagonalization 產生）
4. 對照 NumPy `lstsq`（小尺寸才做）確認結果合理。

## 詳細說明（繁中）

### LSMR 跟「normal equations」的關係

最小平方的最優條件是：

- `Aᵀ(Ax-b)=0`

當 `A` 具有 full column rank 時，等價於解：

- `(AᵀA)x = Aᵀb`

LSMR 的核心觀念是：在不形成 `AᵀA` 的情況下，仍然能用 Krylov 方法（MINRES）在 normal equations 上迭代逼近解。

### 為什麼用 `‖Aᵀr‖` 當收斂指標？

令 `r = Ax-b`，則 `Aᵀr` 就是目標函數 `‖Ax-b‖²/2` 的梯度。當 `‖Aᵀr‖` 很小時，代表你已經接近最優點（stationary point）。

因此本單元把 `‖Aᵀr‖` 視為核心驗收量，並用它做停止條件（與大型最小平方實作習慣一致）。

## 程式碼區段（節錄 + 解釋）

> 節錄「用 bidiagonalization 建 T_k，再解小型 least-squares」的關鍵段落。

```text
T_k = build_tridiagonal_T_from_golub_kahan(alphas=alpha_arr, betas=beta_arr, k=k)
rhs = np.zeros((k,), dtype=float)
rhs[0] = g_norm
y_k, *_ = np.linalg.lstsq(T_k, rhs, rcond=None)
x = V_basis[:, :k] @ y_k
arnorm = l2_norm(rhs - (T_k @ y_k))
```

- 這段在做什麼：
  - `T_k` 是 normal equations 在 Krylov 子空間上的投影（三對角矩陣）
  - `y_k` 是「讓 normal residual 最小」的子空間解（MINRES 定義）
  - `x_k` 是把子空間解 lift 回原空間
  - `arnorm`（`‖Aᵀr‖`）可直接由小系統殘差得到

## 驗證方式與預期輸出

- 你可以檢查：
  - `||A^T r||` 是否持續下降並在停止時很小
  - 與 `numpy.linalg.lstsq` 的相對誤差是否合理（浮點下通常不會是 0）
- 注意：
  - 本實作是「教學版」，會存 `V_basis` 並每步解小系統，不是最省記憶體/最快的工業實作。

