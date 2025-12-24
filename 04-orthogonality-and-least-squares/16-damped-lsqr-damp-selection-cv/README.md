# 用 k-fold CV 選 damp：Damped LSQR 的正則化強度（Ridge）

> 目標：用「驗證集表現」挑選 `damp`（等價於 Ridge 的 `λ=damp²`），讓解既穩定又不過度偏差

## 學習目標

1. 連結兩種寫法：`min(‖Ax-b‖² + damp²‖x‖²)` ⇔ `min‖[A; damp I]x-[b;0]‖`
2. 用 k-fold cross-validation 系統化選 `damp`（而不是憑直覺）
3. 會讀 val RMSE 的 U-shape：`damp` 太小→過擬合/不穩定；太大→欠擬合/偏差變大

## 本單元實作（Python）

```
python/
└── damped_lsqr_damp_selection_cv_numpy.py   # Damped LSQR + k-fold CV + ASCII 曲線
```

