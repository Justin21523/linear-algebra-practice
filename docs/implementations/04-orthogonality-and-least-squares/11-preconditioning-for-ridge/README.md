# 實作說明：11-preconditioning-for-ridge（04-orthogonality-and-least-squares）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/11-preconditioning-for-ridge/`
- 概念說明：`04-orthogonality-and-least-squares/11-preconditioning-for-ridge/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/11-preconditioning-for-ridge/python/preconditioned_cg_ridge_numpy.py`

## 目標與背景

上一個單元你已經看到：病態矩陣下 GD 會變慢，而 CG 通常更快。

但在更極端的病態情況下（或維度更大），CG 也可能開始變慢，因為它的收斂仍然受系統矩陣 `H` 的條件數影響：

- `H = AᵀA + λI`

這時就會引入 **預條件化（preconditioning）**：

- 找一個容易反解/容易乘的 `M ≈ H`
- 把問題「等價變形」成更好解的形式（概念上）
- 讓 Krylov 子空間方法（如 CG）更快收斂

本單元示範最簡單也最常見的 Jacobi（對角）預條件器。

## 如何執行

> 需要 `numpy`：請先依 `requirements.txt` 安裝（見 repo 根目錄 `README.md` 的 venv 範例）。

```bash
cd 04-orthogonality-and-least-squares/11-preconditioning-for-ridge/python
python3 preconditioned_cg_ridge_numpy.py
```

## 核心做法（重點）

1. 先建立一個「非常病態」的 `A`（奇異值衰減到 `1e-8` 級別），再加上 `λ>0` 形成 SPD 系統：
   - `H = AᵀA + λI`
2. 比較兩種方法在同一個 `Hx=g` 上的殘差下降：
   - CG：`r_k = g - Hx_k`
   - PCG（Jacobi）：先用 `z_k = M^{-1} r_k`（其中 `M = diag(H)`）

## 詳細說明

### 1) Jacobi 預條件器是什麼？

Jacobi（對角）預條件器取：

`M = diag(H)`

因為 Ridge 的 `H = AᵀA + λI`，所以：

`M_jj = (AᵀA)_jj + λ = ‖A[:,j]‖₂² + λ`

這個 `M` 的優點是：

- 便宜：只要算每個欄向量的平方和
- 好反解：`M^{-1}` 就是逐元素除法

程式節錄（計算 `diag(H)` 與 `M^{-1}`）：

```python
diag_H = np.sum(A * A, axis=0) + lam  # EN: Compute diagonal entries of H: diag(A^T A) + λ.
inv_diag = 1.0 / np.maximum(diag_H, EPS)  # EN: Build inverse diagonal for Jacobi preconditioner.
```

### 2) PCG 跟 CG 的差別在哪？

CG 只用 `r_k` 更新方向；PCG 會先把殘差「縮放」成：

`z_k = M^{-1} r_k`

再用 `r_kᵀ z_k` 這個內積來計算步長/更新係數。直覺上：

- 如果 `M` 抓到了 `H` 的尺度差異（不同維度的量級差），就能減少病態造成的迭代卡住

### 3) 你應該看到的現象

在非常病態的案例裡：

- PCG 的 `||r||` 通常能更快下降（迭代數較少）
- 但 Jacobi 不一定是最好預條件器；它只是最便宜、最好理解的入門選擇

## 驗證方式

- 看輸出的 `cond(A)` 與 `cond(H)`：應該都很大（尤其 `cond(H)`）
- 比較 CG vs PCG 的：
  - `iters`
  - `final ||r||_2`
  - checkpoint（例如 iter=10/20/40）時的 `||r||` 下降速度

