# 進階預條件化：Randomized QR（Blendenpik/LSRN 風格）加速 LSMR/LSQR

> 目標：用「隨機 sketch + QR」做右預條件化（right preconditioning），顯著降低 LSMR/LSQR 的迭代數

## 學習目標

1. 理解右預條件化：`x = M⁻¹y`，改解 `min‖A M⁻¹ y - b‖`
2. 了解 Randomized QR 預條件器的直覺：用 oversampled sketch 讓 `A M⁻¹` 更接近「良好條件數」
3. 對照三種策略的收斂差異：
   - 無預條件
   - column scaling（簡單、便宜）
   - randomized QR（較貴，但通常迭代數大幅下降）

## 本單元實作（Python）

```
python/
└── lsmr_randomized_qr_preconditioning_numpy.py   # LSMR + (none / col-scaling / rand-QR) 對照
```

