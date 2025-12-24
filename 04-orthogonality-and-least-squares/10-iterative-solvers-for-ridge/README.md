# 迭代法解 Ridge：Gradient Descent vs Conjugate Gradient（看懂條件數與收斂）

> 實務 ML 常遇到「資料很多、矩陣很大」，你不會真的去算 `(AᵀA+λI)⁻¹`；你會用迭代法解

## 學習目標

完成本單元後，你應該能：

1. 把 Ridge Regression 寫成最小化問題：
   - `min_x f(x) = 1/2‖Ax-b‖₂² + 1/2 λ‖x‖₂²`
2. 把 Ridge 寫成 SPD 線性系統（方便用 CG）：
   - `(AᵀA + λI)x = Aᵀb`
3. 實作並比較兩種常見迭代法：
   - Gradient Descent（GD）：簡單、但收斂率受條件數影響很大
   - Conjugate Gradient（CG）：對 SPD 系統通常更快、更適合大型問題
4. 用數值指標驗算與比較：
   - 目標函數 `f(x)` 是否下降
   - `‖∇f(x)‖` 或 `‖(AᵀA+λI)x - Aᵀb‖` 是否趨近 0
   - 在 well-conditioned vs ill-conditioned 的案例下，迭代次數差異

## 本單元實作（Python）

```
python/
└── iterative_solvers_ridge_numpy.py  # GD vs CG 解 Ridge，並對比條件數影響
```

