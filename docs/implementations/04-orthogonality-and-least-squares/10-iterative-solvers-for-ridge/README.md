# 實作說明：10-iterative-solvers-for-ridge（04-orthogonality-and-least-squares）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/10-iterative-solvers-for-ridge/`
- 概念說明：`04-orthogonality-and-least-squares/10-iterative-solvers-for-ridge/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/10-iterative-solvers-for-ridge/python/iterative_solvers_ridge_numpy.py`

## 目標與背景

當問題規模變大（樣本很多、特徵很多）時，你通常不會：

- 建 `AᵀA` 再解 `(AᵀA+λI)x=Aᵀb`
- 更不會真的算反矩陣

而是改用「只需要矩陣乘向量」的 **迭代法** 來求解 Ridge Regression。

本單元用兩個對照案例（well-conditioned vs ill-conditioned）展示：

- GD（Gradient Descent）在病態問題下會明顯變慢（受條件數影響大）
- CG（Conjugate Gradient）對 SPD 系統通常更快、更適合大型問題

## 如何執行

> 需要 `numpy`：請先依 `requirements.txt` 安裝（見 repo 根目錄 `README.md` 的 venv 範例）。

```bash
cd 04-orthogonality-and-least-squares/10-iterative-solvers-for-ridge/python
python3 iterative_solvers_ridge_numpy.py
```

## 核心做法（重點）

1. 先造兩個 `A`（同樣大小，但奇異值光譜不同）：
   - `s_good = [1,1,1,...]` → `cond(A)` 小（好收斂）
   - `s_bad = logspace(0,-6)` → `cond(A)` 大（病態，難收斂）
2. 設定 Ridge：`f(x)=1/2‖Ax-b‖² + 1/2 λ‖x‖²`
3. 用 SVD 得到「參考解」`x*`（closed-form），用來量測相對誤差
4. 跑兩種迭代法：
   - GD：`x ← x - α∇f(x)`，其中 `α=1/L`，`L=‖A‖₂²+λ`
   - CG：解 SPD 系統 `Hx=g`，`H=AᵀA+λI`、`g=Aᵀb`，並用 `matvec(v)=Aᵀ(Av)+λv` 避免顯式形成 `AᵀA`

## 詳細說明

### 1) GD 的步長為什麼用 `1/L`？

Ridge 是凸二次函數，梯度為：

`∇f(x)=Aᵀ(Ax-b)+λx`

它的 Lipschitz 常數是：

`L = ‖A‖₂² + λ = σ_max(A)² + λ`

選 `α ≤ 1/L` 可以保證 GD 不會「爆掉」，通常也會保證目標值下降（至少在理論上）。

節錄自 `iterative_solvers_ridge_numpy.py`：

```python
L = sigma_max**2 + lam  # EN: Lipschitz constant for ridge gradient (||A||_2^2 + λ).
step_size = 1.0 / L  # EN: Safe GD step size for convergence on a convex quadratic.
```

### 2) CG 在解什麼？為什麼比較快？

Ridge 的一階條件可改寫成 SPD 線性系統：

`(AᵀA+λI)x = Aᵀb`

CG 是專門解 SPD 系統的迭代法（不需要矩陣反矩陣），在理想狀況（精確算術）下最多 `n` 次就能收斂；實務上也常比 GD 少很多步。

本單元刻意不顯式建 `AᵀA`，而是用：

`matvec(v)=Aᵀ(Av)+λv`

這是大型資料時更常見的寫法（只做矩陣乘向量）。

### 3) 你應該觀察到的現象

- well-conditioned：GD/CG 都會很快接近 `x*`
- ill-conditioned：GD 的迭代數會顯著上升，且 `f(x)`/`‖∇f‖` 下降很慢；CG 通常仍能在較少步數把系統殘差壓下來

## 驗證方式

跑完後請特別看：

- `cond(A)`：兩個 case 差很多（第二個會非常大）
- GD 的 checkpoint：`iter 1000` 時在病態案例通常仍不夠好
- CG 的 `||r||`：在病態案例仍會比 GD 更快下降（通常）
- `rel_err_to_x*`：越小代表越接近 closed-form 參考解

