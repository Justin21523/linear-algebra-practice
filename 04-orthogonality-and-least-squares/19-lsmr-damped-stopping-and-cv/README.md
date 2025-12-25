# Damped LSMR（Ridge）+ stopping criteria + k-fold CV 選 damp

> 目標：用 LSMR（normal equations 上的 MINRES 等價）解 Ridge：
>
> `min_x ‖Ax-b‖² + damp²‖x‖²`，並用更實務的 stopping criteria 與 k-fold CV 選擇 `damp`

## 學習目標

1. 理解 `damp` 與 Ridge `λ` 的關係：`λ = damp²`
2. 用 `‖Aᵀ(Ax-b)+damp²x‖`（梯度/optimality）當作主要收斂指標，並加入相對/絕對混合門檻（atol/btol）
3. 以 ML 的方式選超參數：k-fold CV 掃 `damp`，看 validation RMSE 的 U-shape，挑最佳值

## 本單元實作（Python）

```
python/
└── lsmr_damped_stopping_and_cv_numpy.py   # Damped LSMR + stopping + k-fold CV（ASCII 曲線）
```

