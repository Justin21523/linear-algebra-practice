# 正則化參數選擇：用 Hold-out 找 Ridge 的 λ 與 TSVD 的 k

> 正則化不是「越大越好」：λ/k 會改變 bias–variance tradeoff；本單元用簡單驗證集來選參數

## 學習目標

完成本單元後，你應該能：

1. 用 hold-out（訓練/驗證切分）為 Ridge 選擇 `λ`，為 TSVD 選擇截斷秩 `k`
2. 觀察典型曲線：
   - 訓練誤差通常隨正則化變強而上升
   - 驗證誤差常呈現 U 形（太小會過擬合/高變異，太大會欠擬合/高偏差）
3. 用重複加噪實驗，定量觀察 bias–variance：
   - `mean(x̂)` 距離 `x_true`（bias）
   - `x̂` 在不同噪聲下的波動（variance）

## 本單元實作（Python）

```
python/
└── regularization_model_selection_numpy.py  # 以多項式設計矩陣示範：選 λ/k + bias–variance
```

