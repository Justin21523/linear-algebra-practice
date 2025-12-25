# Sparse / Matrix-Free Ridge：Damped LSMR（CSR matvec）+ k-fold CV + sparse rand-QR（CountSketch）

> 目標：把「大規模回歸常見的稀疏設計矩陣」搬進來，練習 **matrix-free** 的思維，並把「CV 掃超參數曲線的總成本」也一起納入：
>
> - 矩陣 `A` 以 CSR（Compressed Sparse Row）儲存
> - solver 只用 `A @ x` / `Aᵀ @ y`（不形成 `AᵀA`）
> - 仍然驗收 Ridge 最優條件：`‖Aᵀ(Ax-b)+damp²x‖`（或等價形式）
> - 比較 `none / col-scaling / rand-QR(CountSketch)` 三種右預條件化
> - 做 **k-fold CV** 掃 `damp` 曲線，報告 `total_build_s / total_solve_s / total_iters`（整條曲線總成本）
> - 示範「跑整條曲線做快」：continuation/warm-start + rand-QR 的 `shared sketch` / `fixed-R`

## 本單元實作（Python）

```
python/
└── lsmr_damped_sparse_matrix_free_numpy.py   # CSR matvec + damped LSMR (teaching) + (none/col/rand-QR) + k-fold CV + curve cost + warm-start
```

## 如何執行

> 需要 `numpy`（見 repo 根目錄的 `requirements.txt`）。

```bash
cd 04-orthogonality-and-least-squares/21-lsmr-damped-sparse-matrix-free/python
python3 lsmr_damped_sparse_matrix_free_numpy.py
```

## 你應該會看到什麼？

- **Per-damp solver comparison**：在 full data 上比較 `none/col/rand-QR` 的 iters、`‖Ax-b‖`、`‖Aᵀr+damp²x‖`、build/solve time。
- **k-fold CV sweep（baseline）**：每個 preconditioner 印出一張 CV 表（train/val RMSE mean±std + ASCII 曲線），並印出整條曲線總成本。
- **Baseline vs speedups**：把「baseline（每點 cold-start、rand-QR 每點重建）」與「speedups（warm-start + rand-QR 共用 sketch）」放到同一張表比時間/迭代數/品質。
