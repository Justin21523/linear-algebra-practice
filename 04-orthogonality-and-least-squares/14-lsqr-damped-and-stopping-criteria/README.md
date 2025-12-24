# Damped LSQR（Ridge）+ 更完整 stopping criteria：大型最小平方的實務版本

> 在大型問題裡，LSQR 常用來解 least squares；而 damped LSQR 等價於 Ridge（Tikhonov）正則化

## 學習目標

完成本單元後，你應該能：

1. 理解 damped LSQR 解的是：
   - `min_x (‖Ax-b‖₂² + damp²‖x‖₂²)`（其中 `damp ≥ 0`）
2. 知道它等價於把系統擴增成：
   - `min_x ‖[A; damp I]x - [b; 0]‖₂`
3. 實作「只用 matvec」的 damped LSQR：
   - 只需要 `A v` 與 `Aᵀ u`（以及簡單的 `damp` 乘法）
4. 使用更完整的停止條件（stopping criteria），同時看：
   - `‖r‖ = ‖Ax-b‖`
   - `‖Aᵀr‖`（不正則化最小平方的一階條件）
   - `‖Aᵀ(Ax-b) + damp² x‖`（Ridge 的一階條件）
   - 相對/絕對容忍度（`atol`, `btol`）的意義

## 本單元實作（Python）

```
python/
└── lsqr_damped_stopping_manual.py  # LSQR + damp（Ridge）+ 更完整 stopping criteria + 與 lstsq / ridge(SVD) 對照
```

