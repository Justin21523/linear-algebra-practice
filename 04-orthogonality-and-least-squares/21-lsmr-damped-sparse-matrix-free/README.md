# Sparse / Matrix-Free Ridge：Damped LSMR（CSR matvec）+ 預條件化

> 目標：把「大規模回歸常見的稀疏設計矩陣」搬進來，練習 **matrix-free** 的思維：
>
> - 矩陣 `A` 以 CSR（Compressed Sparse Row）儲存
> - solver 只用 `A @ x` / `Aᵀ @ y`（不形成 `AᵀA`）
> - 仍然驗收 Ridge 最優條件：`‖Aᵀ(Ax-b)+damp²x‖`
> - 另外示範：沿 `damp` 路徑做 continuation/warm-start 減少迭代數

## 本單元實作（Python）

```
python/
└── lsmr_damped_sparse_matrix_free_numpy.py   # CSR matvec + damped LSMR (teaching) + col-scaling preconditioning + warm-start path
```

