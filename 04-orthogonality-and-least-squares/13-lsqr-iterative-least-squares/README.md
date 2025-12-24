# LSQR：不形成 AᵀA 的最小平方迭代法（大型問題常用）

> `min_x ‖Ax-b‖` 的經典迭代解法：只需要 `A v` 與 `Aᵀ u`，不用建 `AᵀA`

## 學習目標

完成本單元後，你應該能：

1. 理解 LSQR 解的是最小平方：
   - `x̂ = argmin_x ‖Ax-b‖₂`
2. 了解為什麼「不要形成 AᵀA」：
   - 會把條件數平方化、放大數值誤差（`cond(AᵀA)≈cond(A)^2`）
3. 實作 LSQR 的核心迭代（Golub–Kahan bidiagonalization 的思路）
4. 用數值方式驗算：
   - 殘差 `‖Ax̂-b‖₂`
   - 正規方程殘差 `‖Aᵀ(Ax̂-b)‖₂`
   - 與 `np.linalg.lstsq` 的結果對照（小尺寸）

## 本單元實作（Python）

```
python/
└── lsqr_manual.py  # 手刻 LSQR（只用 matvec：A@v 與 A.T@u）+ 與 lstsq 對照
```

