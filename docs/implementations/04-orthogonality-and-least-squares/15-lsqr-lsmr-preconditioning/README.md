# 實作說明：LSQR/LSMR 預條件化（04-orthogonality-and-least-squares/15-lsqr-lsmr-preconditioning）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/15-lsqr-lsmr-preconditioning/`
- 單元概念：`04-orthogonality-and-least-squares/15-lsqr-lsmr-preconditioning/README.md`
- 實作檔案：
  - `04-orthogonality-and-least-squares/15-lsqr-lsmr-preconditioning/python/lsqr_lsmr_preconditioning_numpy.py`
- 文件路徑：`docs/implementations/04-orthogonality-and-least-squares/15-lsqr-lsmr-preconditioning/README.md`

## 目標與背景

- 本實作示範：對大型最小平方迭代法（LSQR/LSMR 類）做 **右預條件化（right preconditioning）**，讓收斂變快。
- 你要記住的直覺：當 `A` 的各欄尺度差很多（或條件數很大），Krylov 迭代會「走得很慢」；先把欄尺度拉齊，通常就能顯著改善。
- 預條件化的重點不是改變答案，而是改變「迭代走的座標系」來加速收斂。

## 如何執行

```bash
cd 04-orthogonality-and-least-squares/15-lsqr-lsmr-preconditioning/python
python3 lsqr_lsmr_preconditioning_numpy.py
```

## 核心做法（重點步驟）

1. 建一個病態最小平方題：用指定奇異值譜做 `A`，讓 `cond(A)` 很大。
2. 基準：直接用 LSQR 解 `min‖Ax-b‖`，用 `‖Aᵀr‖` 當停止條件。
3. 右預條件化（column scaling）：
   - 設 `D = diag(‖A[:,j]‖₂)`（每欄的 2-norm）
   - 令 `x = D⁻¹ y`，改解 `min‖(A D⁻¹) y - b‖`
   - 解出 `y` 後回推 `x = D⁻¹ y`
4. 對照（normal equation baseline）：用 CG 在 `AᵀA x = Aᵀb` 上跑（並示範 Jacobi/diagonal 預條件器）。

## 詳細說明（繁中）

### 右預條件化為什麼合理？

你想解的是：

- 最小平方：`min_x ‖Ax-b‖₂`

若 `D` 是可逆對角矩陣，令 `x = D⁻¹y`，則：

- `min_x ‖Ax-b‖₂  =  min_y ‖A(D⁻¹y) - b‖₂  =  min_y ‖(A D⁻¹) y - b‖₂`

所以「在 y-space 解」再換回 x-space，最佳解是一樣的（只要 `D` 可逆且你完整收斂）。

### Column scaling 的直覺

若某些欄的尺度非常大/小，迭代法在不同方向上的步幅會差很多；把每一欄除以 `‖A[:,j]‖₂`，相當於讓各方向的尺度先被標準化。

在實務上這非常常見（尤其是 ML 的特徵尺度不一致時），也對應到資料前處理的 feature scaling 概念。

### 為什麼還要提 LSMR？

LSMR 可以理解為「在 normal equations 上做 MINRES 的等價形式」，與 LSQR 同屬 matvec-only 的大型最小平方迭代法。
**重點是：右預條件化的包裝方式完全相同**——把 `A` 替換成 `A D⁻¹`，最後再把解轉回 `x=D⁻¹y`。

本單元程式以 CGNR/PCGNR 當作 normal-equation baseline 來對照「預條件化能讓迭代更快」這件事（不把焦點放在完整實作 LSMR 細節）。

## 程式碼區段（節錄 + 解釋）

> 節錄 right preconditioning 的 matvec 包裝（把 `A D⁻¹` 當成新的線性算子）。

```text
def matvec_B(y):
    return A @ (y / D_safe)

def matvec_BT(u):
    return (A.T @ u) / D_safe
```

- 這段在做什麼：讓 LSQR 只看到 `(A D⁻¹)` 的 matvec / transpose-matvec，不需要真的形成 `A D⁻¹`。
- 為什麼重要：大型/稀疏問題中，你常常只有 matvec，不能隨便形成新矩陣；這種「算子包裝」是標準技巧。

## 驗證方式與預期輸出

- 你可以檢查：
  - `iters`：預條件化後通常會減少（尤其病態矩陣很明顯）。
  - `||Ax-b||` 與 `||Aᵀr||`：應該與未預條件化的解在同量級（差異主要在收斂速度）。
- 常見陷阱：
  - `D` 若有 0（零欄），需要做保護（程式用 `max(D, EPS)`）。
  - 若你用不同的停止條件，迭代數比較可能不公平；建議至少統一用 `||Aᵀr||` 門檻。

