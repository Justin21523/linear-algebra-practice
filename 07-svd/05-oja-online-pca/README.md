# Oja’s Online PCA：線上／串流主成分分析（Online PCA）

> 在資料非常大或「一筆一筆流入」時，用 Oja’s rule 近似前 `k` 個主成分，不必形成完整協方差或做完整 SVD

## 學習目標

1. 理解 Oja’s rule 對「最大化投影變異」的直覺：逐步學出主方向
2. 熟悉 top-1 與 top-k（block Oja / subspace Oja）的差異：向量 vs 子空間
3. 會用子空間主角度（principal angles）/ explained variance ratio 驗證收斂品質

## 本單元實作（Python）

```
python/
└── oja_online_pca_numpy.py   # block Oja（mini-batch）學 top-k + 與 covariance eigendecomp 對照
```

## 你可以觀察的重點

- Learning rate（含衰減）對收斂速度/穩定性的影響
- 特徵值間距（eigengap）大/小時，學第一主成分會快/慢很多
- 資料噪聲越大，能達到的最佳相似度也會下降（不要期待 1.0）

