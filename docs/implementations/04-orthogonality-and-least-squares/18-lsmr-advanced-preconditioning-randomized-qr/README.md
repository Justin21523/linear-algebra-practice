# 實作說明：Randomized QR 預條件化（04-orthogonality-and-least-squares/18-lsmr-advanced-preconditioning-randomized-qr）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/18-lsmr-advanced-preconditioning-randomized-qr/`
- 單元概念：`04-orthogonality-and-least-squares/18-lsmr-advanced-preconditioning-randomized-qr/README.md`
- 實作檔案：
  - `04-orthogonality-and-least-squares/18-lsmr-advanced-preconditioning-randomized-qr/python/lsmr_randomized_qr_preconditioning_numpy.py`
- 文件路徑：`docs/implementations/04-orthogonality-and-least-squares/18-lsmr-advanced-preconditioning-randomized-qr/README.md`

## 目標與背景

- 本實作示範：用 **Randomized sketch + QR** 做右預條件化（right preconditioning），讓 LSMR/LSQR 類 Krylov 迭代法大幅加速。
- 這個方向很貼近大規模稀疏最小平方（例如大規模回歸/系統辨識）：你通常不想形成 `AᵀA`，而是靠 matvec + 預條件器把迭代數壓下來。

## 如何執行

```bash
cd 04-orthogonality-and-least-squares/18-lsmr-advanced-preconditioning-randomized-qr/python
python3 lsmr_randomized_qr_preconditioning_numpy.py
```

## 核心做法（重點步驟）

1. 建一個病態的 `A`（奇異值跨很多個數量級）。
2. 比較三種策略（都用同一個 solver 框架）：
   - 無預條件：`M=I`
   - column scaling：`M=D=diag(‖A[:,j]‖₂)`
   - randomized QR：取 sketch `S`，做 `SA = QR`，用 `M=R`
3. 都用右預條件化改寫：
   - `x = M⁻¹y`
   - 解 `min‖A M⁻¹ y - b‖`
   - 回推 `x = M⁻¹y`
4. 在原座標（x-space）驗證：
   - `‖Ax-b‖`
   - `‖Aᵀ(Ax-b)‖`
   - 迭代數與時間（build vs solve）

## 詳細說明（繁中）

### 為什麼 Randomized QR 可以當預條件器？

在 Blendenpik/LSRN 風格的方法中，你先對 `A` 做一個 oversampled 的隨機 sketch（例如 `S` 有 `s≈4n` 列），得到 `SA`（尺寸 `s×n`），再做：

- `SA = QR`

把 `R` 當作右預條件器，等價於把未知數改成 `x = R⁻¹y`，使得 `A R⁻¹` 往往具有更好的條件數，因此 Krylov 法收斂更快。

### 右預條件化的算子包裝（matvec-only）

大型/稀疏問題中，你通常只做得出 `A v` 與 `Aᵀ u`；右預條件化的關鍵就是把它包成新的算子 `B = A M⁻¹`：

- `B y = A(M⁻¹y)`
- `Bᵀ u = M⁻ᵀ(Aᵀu)`

因此你可以不形成 `B`，只要提供兩個函數（matvec / transpose matvec），就能直接把預條件器接到 LSMR/LSQR 上。

## 程式碼區段（節錄 + 解釋）

> 節錄右預條件化的 matvec 包裝（`B=A M⁻¹`）。

```text
def matvec_B(y):
    return A @ apply_Minv(y)

def matvec_BT(u):
    return apply_Minv_T(A.T @ u)
```

- 這段在做什麼：把原本的 `A` 換成「隱式的」`A M⁻¹`，並確保 transpose-matvec 也一致。
- `apply_Minv`：
  - column scaling：`y / D`
  - randomized QR：解三角系統 `R x = y`（`x=R⁻¹y`）

## 驗證方式與預期輸出

- 程式會印一張表：
  - `iters`：迭代數（rand-QR 常常顯著更少）
  - `build_s` vs `solve_s`：建預條件器 vs 迭代求解時間
  - `||Ax-b||`、`||A^T r||`、`||x||`：原座標下的解品質/最優性
- 預期現象：
  - column scaling 幫助有限但幾乎不花成本
  - randomized QR 會花 build 成本，但能把 iterations 壓到很低（整體常更快）

