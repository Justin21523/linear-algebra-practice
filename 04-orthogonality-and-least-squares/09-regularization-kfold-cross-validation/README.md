# 正則化參數選擇（更穩）：k-fold Cross-Validation + ASCII 曲線輸出

> hold-out 只有一次切分，結果容易受「剛好抽到哪一批資料」影響；k-fold cross-validation 通常更穩定

## 學習目標

完成本單元後，你應該能：

1. 理解 k-fold cross-validation 的流程（把資料分成 k 份，輪流當驗證集）
2. 用 k-fold 來選 Ridge 的 `λ` 與 TSVD 的 `k`
3. 輸出「超參數 vs 驗證誤差」的簡單 ASCII 曲線，直觀看到最佳點與 U 形趨勢

## 本單元實作（Python）

```
python/
└── regularization_kfold_cv_numpy.py  # k-fold CV：掃 λ/k，印 table + ASCII curve
```

