# 實作說明：Damped LSMR + 預條件化 + CV 成本對照（04-orthogonality-and-least-squares/20-lsmr-damped-preconditioning-and-cv）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/20-lsmr-damped-preconditioning-and-cv/`
- 單元概念：`04-orthogonality-and-least-squares/20-lsmr-damped-preconditioning-and-cv/README.md`
- 實作檔案：
  - `04-orthogonality-and-least-squares/20-lsmr-damped-preconditioning-and-cv/python/lsmr_damped_preconditioning_and_cv_numpy.py`
- 文件路徑：`docs/implementations/04-orthogonality-and-least-squares/20-lsmr-damped-preconditioning-and-cv/README.md`

## 目標與背景

- 本實作示範：把 **Ridge（damp）** 與 **預條件化（right preconditioning）** 放在同一題裡，直接比較：
  - `none`（無預條件）
  - `column scaling`（便宜、常見）
  - `randomized QR`（Blendenpik/LSRN 風格：建置較貴，但迭代數常大幅下降）
- 再用 **k-fold CV** 掃一整條 `damp` 曲線，除了挑最佳驗證誤差，也比較「跑完整條曲線」的總成本（build/solve time、總迭代數）。

## 如何執行

```bash
cd 04-orthogonality-and-least-squares/20-lsmr-damped-preconditioning-and-cv/python
python3 lsmr_damped_preconditioning_and_cv_numpy.py
```

## 核心做法（重點步驟）

1. 題目：解 Ridge
   - `min_x ‖Ax-b‖² + damp²‖x‖²`（`λ = damp²`）
2. Solver：Damped LSMR（教學版）
   - 用 Golub–Kahan bidiagonalization 建立子空間
   - 以「normal equations 上 MINRES 等價」的定義式，在每個 k 解小型三對角系統取得當前解
3. Right preconditioning：令 `x = M⁻¹ y`，改解 `min ‖[A;damp I] M⁻¹ y - [b;0]‖`
4. 三種 `M`：
   - none：`M=I`
   - column scaling：`M=diag(‖[A;damp I](:,j)‖₂) = sqrt(‖A[:,j]‖² + damp²)`
   - randomized QR：用 sketch `S` 做 `S[A;damp I]=QR`，取 `M=R`
5. 以一致指標驗收（原座標 x-space）：
   - `‖Ax-b‖`
   - `‖Aᵀ(Ax-b)+damp²x‖`（Ridge 最優條件）

## 詳細說明（繁中）

### `damp` 與預條件化的分工

- `damp`（Ridge）會讓 `(AᵀA + damp²I)` 變得更好解、數值更穩定，但不代表迭代一定夠快。
- 預條件化是在「不改問題答案」的前提下，改變迭代的座標系，讓 Krylov 法的收斂加速。

你要看的不是只有「最後答案好不好」，而是「在達到同等最優條件時，迭代數與總時間差多少」。

### 為什麼 stopping 要看 `‖Aᵀr + damp²x‖`？

令 `r=Ax-b`，Ridge 目標的梯度是：

- `∇f(x)=Aᵀ(Ax-b)+damp²x = Aᵀr + damp²x`

因此 `‖Aᵀr + damp²x‖` 越小，代表越接近最優點；這也是本單元跨預條件化設定的共同驗收量。

### CV 的「總成本」為什麼要比？

在真實 ML 中，你通常要掃一整條超參數曲線（例如多個 `damp`），而不是只訓練一次。
所以就算某個預條件器單次解很快，如果 build 成本很高、而且每個 fold/每個 `damp` 都要重建，總成本可能反而不划算。

本單元因此同時報告：

- `total_build_s`：整條曲線所有 fits 的預條件器建置時間總和
- `total_solve_s`：整條曲線所有 fits 的求解時間總和
- `total_iters`：整條曲線所有 fits 的迭代數總和（與 matvec 成本高度相關）

### 掃 `damp` 曲線怎麼做快？（continuation / warm-start）

直覺：相鄰的 `damp` 解通常很接近（尤其是 log-grid），所以你不需要每個點都「從 x=0 重新開始」。

本單元用一個實務常見的小技巧來做到 warm-start：**把「初始猜測 x0」轉成一次「解增量 delta」的等價最小平方**。

- 原題（Ridge）：`min_x ‖Ax-b‖² + damp²‖x‖²`
- 令 `x = x0 + delta`，可等價成解：
  - `min_delta ‖A delta - (b - A x0)‖² + ‖damp delta - (-damp x0)‖²`

你可以把它理解成：

- 不去改 solver（不用在 LSMR 裡加「初始解」功能）
- 而是把「初始解」吸收到 RHS 裡，讓 solver 解一個「修正量 delta」
- 最後再把 `x = x0 + delta` 合回去

### rand-QR 的 reuse：shared sketch / fixed-R

rand-QR 類預條件器（Blendenpik/LSRN 風格）的痛點是：**build 很貴**。如果你在 CV 要掃 15 個 `damp`、5 folds，就是 75 次 build。

本單元示範兩種「不改問題答案，但把曲線掃描做快」的方式：

1. **shared sketch**（每個 fold 共用同一個 sketch）
   - 把 sketch `S` 分成：`S=[S_top, S_bottom]`，對應到 `[A; damp I]` 的上/下兩塊
   - 因為 `S[A;damp I] = S_top A + damp S_bottom`，所以你可以：
     - 先算一次 `S_top A`（最貴）
     - 每個 `damp` 只要更新 `+ damp S_bottom`，再做一次 QR 取 `R`
2. **fixed-R**（每個 fold 只在某個參考 `damp_ref` build 一次 `R`，整條曲線都重用）
   - build 幾乎砍到「每 fold 一次」
   - 代價是：因為 `M` 不再跟著 `damp` 變，收斂速度/品質可能會跟 shared-sketch 不同（所以要看 `best_val_rmse` 與 `total_iters`）

## 程式碼區段（節錄 + 解釋）

> 節錄 right preconditioning 的算子包裝（把 `B = [A;damp I] M⁻¹` 當成新的線性算子）。

```text
def matvec_B(y):
    x = apply_Minv(y)
    return concat([A @ x, damp * x])

def matvec_BT(u_aug):
    z = (A.T @ u_top) + (damp * u_bottom)
    return apply_Minv_T(z)
```

- `matvec_B`：讓 solver 只看到「預條件化後」的算子 `B`，不需要真的形成 `B`。
- `matvec_BT`：transpose-matvec 必須一致（這點做錯，迭代法會壞掉或收斂異常）。

> 節錄 warm-start（continuation）用的「修正量 delta」等價 RHS（讓每個 damp 不用從零開始）。

```text
if x_init is None:
    b_top = b
    b_bottom = None
    x0 = zeros(n)
else:
    b_top = b - A @ x0
    b_bottom = -damp * x0

x_delta = solver(A, b_top, b_bottom)
x_hat = x0 + x_delta
```

- `b_top=b-Ax0`：把「目前解 x0」的資料殘差變成新的 RHS。
- `b_bottom=-damp*x0`：讓擴增系統的下半部殘差對應到 `damp*(x0+delta)`。
- solver 回傳的是 `delta`，最後再合成 `x_hat`。

> 節錄 shared-sketch 的核心等式：`S[A;damp I] = S_top A + damp S_bottom`（避免每個 damp 都重建完整 SA_aug）。

```text
S = rademacher(s, m+n)
S_top = S[:, :m]
S_bottom = S[:, m:]
SA_top = S_top @ A

A_sketch = SA_top + damp * S_bottom
R = qr(A_sketch).R
```

## 驗證方式與預期輸出

- **Per-damp solver comparison** 表格會列出：
  - `iters`、`build_s`、`solve_s`
  - `||Ax-b||`、`||grad||=||Aᵀ(Ax-b)+damp²x||`、`||x||`
- **k-fold CV（baseline）** 會對每個 preconditioner 印出：
  - damp vs train/val RMSE（mean±std）
  - `iters(mean)`（成本 proxy）
  - `Total curve cost`（整條曲線的 build/solve/iters 總成本）
- **CV sweep speedups** 會額外印出：
  - baseline vs speedups 的總時間/總迭代數/最佳 val RMSE 對照表
  - rand-QR 的 `shared-sketch` vs `fixed-R`（兩者都搭配 warm-start）的摘要行
