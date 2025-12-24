# Least Squares：Normal Equation vs QR vs SVD（數值穩定性比較）

> 同一個最小平方問題，用不同解法會出現「看起來答案差不多、但其實穩定性差很多」的現象

## 學習目標

完成本單元後，你應該能：

1. 把最小平方寫成：`x̂ = argmin_x ‖Ax-b‖₂`
2. 理解並實作三種常見解法：
   - 正規方程（Normal Equation）：`(AᵀA)x = Aᵀb`
   - QR：`A = QR`，解 `Rx = Qᵀb`
   - SVD：`A = UΣVᵀ`，用 pseudo-inverse 得到穩定解
3. 用「可驗算」的診斷指標比較方法差異：
   - 殘差大小：`‖Ax̂-b‖₂`
   - 最佳化必要條件：`Aᵀr ≈ 0`（其中 `r=b-Ax̂`）
   - 穩定性：`b` 很小的擾動 `δ` 會讓 `x̂` 改變多少
4. 了解為什麼 **Normal Equation 會把條件數平方化**：`cond(AᵀA) ≈ cond(A)^2`

## 為什麼跟機器學習有關？

- 線性回歸（Linear Regression）本質就是最小平方；共線性（multicollinearity）會讓解很不穩定。
- 訓練流程常在解近似最小平方或做分解；選錯方法會讓誤差放大，導致訓練不穩或結果漂移。
- SVD/QR 是數值線代常見的「穩定解法」；Normal Equation 只適合條件數小、問題規模小的教學/快速估算。

## 本單元實作（Python）

```
python/
└── least_squares_compare_numpy.py  # Normal / QR / SVD：殘差、Aᵀr、擾動穩定性比較
```

