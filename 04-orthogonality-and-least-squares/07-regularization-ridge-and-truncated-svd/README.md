# 正則化最小平方：Ridge（Tikhonov）與 Truncated SVD

> 共線性（multicollinearity）會讓最小平方解非常不穩定；正則化的核心就是「用一點偏差換取穩定」

## 學習目標

完成本單元後，你應該能：

1. 寫出 Ridge Regression（L2 正則化）的目標函數：
   - `x̂_λ = argmin_x (‖Ax-b‖₂² + λ‖x‖₂²)`
2. 知道 Ridge 的一階條件（最小化必要條件）不再是 `Aᵀr=0`：
   - 若 `r=b-Ax̂_λ`，則 `Aᵀr ≈ λx̂_λ`
3. 理解 Truncated SVD（TSVD）是「頻譜截斷」的正則化：
   - 把很小的奇異值方向丟掉，避免除以接近 0 的數字放大噪聲
4. 用數值診斷比較「不正則化 vs 正則化」在病態矩陣下的差異：
   - 殘差：`‖Ax̂-b‖₂`
   - LS 最佳性（不正則化）：`‖Aᵀr‖₂`
   - Ridge 最佳性：`‖Aᵀr - λx̂‖₂`
   - 穩定性：`b` 加微小擾動後，`x̂` 的變化量

## 為什麼跟機器學習有關？

- 線性回歸 + L2 正則化就是 **Ridge Regression**；共線性時可顯著減少係數爆炸。
- Truncated SVD 很像「只用前 k 個主成分做回歸」（PCA regression）：丟掉不可靠的方向。
- 這些技巧是你理解「數值穩定 / bias–variance tradeoff」的最快入口。

## 本單元實作（Python）

```
python/
└── regularization_ridge_tsvd_numpy.py  # Ridge / TSVD：殘差、最小化條件、擾動穩定性比較
```

