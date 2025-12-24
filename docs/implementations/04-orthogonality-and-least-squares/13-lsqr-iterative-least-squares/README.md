# 實作說明：13-lsqr-iterative-least-squares（04-orthogonality-and-least-squares）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/13-lsqr-iterative-least-squares/`
- 概念說明：`04-orthogonality-and-least-squares/13-lsqr-iterative-least-squares/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/13-lsqr-iterative-least-squares/python/lsqr_manual.py`

## 目標與背景

你已經看過：

- Normal equation（`AᵀA x = Aᵀb`）在病態矩陣下會把條件數平方化，數值變差
- 大型問題更適合「只做矩陣向量乘法」的迭代法（GD/CG/PCG）

LSQR 就是最小平方問題的經典迭代法之一，特色是：

- 只需要 `A v` 與 `Aᵀ u`（matvec-only）
- 不需要形成 `AᵀA`
- 在實務上常用於大型/稀疏 least squares（SciPy 也有同名實作）

## 如何執行

> 需要 `numpy`：請先依 `requirements.txt` 安裝（見 repo 根目錄 `README.md` 的 venv 範例）。

```bash
cd 04-orthogonality-and-least-squares/13-lsqr-iterative-least-squares/python
python3 lsqr_manual.py
```

## 核心做法（重點）

1. 用 `lsqr()` 只透過 `matvec_A(v)=A@v` 與 `matvec_AT(u)=A.T@u` 迭代求解。
2. 在每次迭代（示範用）額外計算：
   - 殘差：`r = Ax-b`、`‖r‖₂`
   - 正規方程殘差：`‖Aᵀr‖₂`（最小平方的一階必要條件）
3. 用 `np.linalg.lstsq(A,b)` 當作小尺寸的參考解，對照：
   - `‖Ax-b‖₂` 是否接近
   - `x_lsqr` 與 `x_lstsq` 的相對差異

## 詳細說明

### 1) 最小平方的一階條件（為什麼要看 `‖Aᵀr‖`？）

對 `min_x 1/2‖Ax-b‖²`，令 `r=Ax-b`，梯度為：

`∇f(x) = Aᵀ(Ax-b) = Aᵀr`

最佳解 `x̂` 需滿足 `Aᵀr = 0`（允許浮點誤差），因此 `‖Aᵀr‖` 是非常直接的驗算指標。

### 2) LSQR 在做什麼？

LSQR 背後是 Golub–Kahan bidiagonalization：它逐步把問題投影到一個小維度的雙對角系統，並透過一連串的正交旋轉更新解。

你不需要先理解完整推導；先掌握兩個實務重點即可：

- **只用 `A v` 與 `Aᵀ u`**：這讓它能套用到巨大/稀疏矩陣
- **避免形成 `AᵀA`**：通常比直接解 normal equation 更穩

### 3) 本單元的「病態案例」怎麼做？

程式用共線性設計矩陣：

- `A = [1, x1, x2]`
- `x2 ≈ x1`（`col_eps` 非常小）

這會讓 `cond(A)` 很大，凸顯「迭代法/穩定解法」的必要性。

## 程式碼區段（節錄）

以下節錄自 `lsqr_manual.py`：用 matvec 的方式傳入 LSQR（不形成 `AᵀA`）：

```python
def matvec_A(v: np.ndarray) -> np.ndarray:
    return A @ v

def matvec_AT(u: np.ndarray) -> np.ndarray:
    return A.T @ u

lsqr_res = lsqr(matvec_A=matvec_A, matvec_AT=matvec_AT, b=b, n=n)
```

## 驗證方式（你應該看到的現象）

- 在較好條件的 case：LSQR 與 `lstsq` 解很接近，`‖Ax-b‖` 與 `‖Aᵀr‖` 都很小
- 在較差條件的 case：仍應看到 LSQR 的 `‖Aᵀr‖` 逐步下降並接近參考解（可能需要較多迭代）

