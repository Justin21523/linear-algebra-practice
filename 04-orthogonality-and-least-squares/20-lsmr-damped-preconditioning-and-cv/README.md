# Damped LSMR + 預條件化 + CV：none / column-scaling / randomized-QR 對照

> 目標：把「Ridge（damp）」與「預條件化」放在同一題裡做完整比較：
>
> - solver：Damped LSMR（normal equations 上的 MINRES 等價）
> - preconditioning：none / column scaling / randomized QR（Blendenpik/LSRN 風格）
> - selection：k-fold CV 掃 `damp`，同時比較「整條曲線」的總成本（時間/迭代數）

## 學習目標

1. 看到 `damp` 與預條件化的互補：`damp` 改善條件數但不一定夠；好的 preconditioner 能把迭代數大幅壓下來
2. 用一致的收斂驗收：`‖Ax-b‖` 與 `‖Aᵀ(Ax-b)+damp²x‖`
3. 以 ML 角度做決策：不只看最佳 val RMSE，也要看「CV 掃完整條曲線」的總成本

## 本單元實作（Python）

```
python/
└── lsmr_damped_preconditioning_and_cv_numpy.py   # three preconditioners × damp sweep + k-fold CV cost comparison
```

