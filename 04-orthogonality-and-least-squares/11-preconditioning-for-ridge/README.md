# 預條件化（Preconditioning）：用 PCG 加速 Ridge 的迭代求解

> 病態矩陣下，即使是 CG 也可能變慢；預條件化的目標是「把系統變得更好解」

## 學習目標

完成本單元後，你應該能：

1. 理解 Ridge 正規方程是 SPD 系統：`(AᵀA+λI)x = Aᵀb`
2. 知道「預條件化」的概念：用一個容易反解的 `M≈H`（`H=AᵀA+λI`）改變收斂行為
3. 實作並比較：
   - CG：解 `Hx=g`
   - PCG（Preconditioned CG）：解 `M⁻¹Hx=M⁻¹g`（概念上）
4. 觀察 Jacobi（對角）預條件器的效果：
   - `M = diag(H)`，也就是 `M_jj = ‖A[:,j]‖² + λ`
   - 在共線性/病態案例下，PCG 通常比 CG 需要更少迭代

## 本單元實作（Python）

```
python/
└── preconditioned_cg_ridge_numpy.py  # CG vs PCG（Jacobi）：殘差下降與迭代數比較
```

