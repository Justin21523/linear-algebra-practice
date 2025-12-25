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

## 驗證方式與預期輸出

- **Per-damp solver comparison** 表格會列出：
  - `iters`、`build_s`、`solve_s`
  - `||Ax-b||`、`||grad||=||Aᵀ(Ax-b)+damp²x||`、`||x||`
- **k-fold CV** 會對每個 preconditioner 印出：
  - damp vs train/val RMSE（mean±std）
  - `iters(mean)`（成本 proxy）
  - `Total curve cost`（整條曲線的 build/solve/iters 總成本）

