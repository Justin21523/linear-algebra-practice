# Randomized SVD：隨機化奇異值分解（Randomized SVD / Randomized Range Finder）

> 用「隨機投影 + 小矩陣 SVD」近似大型矩陣的前 `k` 個奇異值/奇異向量（常用於大規模 PCA、低秩近似）

## 學習目標

1. 理解 Randomized SVD 的核心流程：`AΩ → QR → B=QᵀA → SVD(B) → U≈QŪ`
2. 了解 oversampling `p` 與 power iteration `q` 對精度/穩定性的影響
3. 用可量化的指標驗證近似品質：奇異值誤差、重建誤差、子空間相似度（principal angles）

## 本單元實作（Python）

```
python/
└── randomized_svd_pca_numpy.py   # Randomized SVD（含 p/q）+ 與 full SVD 對照（誤差/子空間）
```

## 延伸到 PCA（你會在 ML 常遇到）

若 `X` 是中心化後的資料矩陣（`n_samples × n_features`），其 PCA 主成分方向是 `X` 的 **右奇異向量** `V` 的前幾個欄向量；
Randomized SVD 就能在不做完整分解的情況下近似前 `k` 個主成分（常用在超大資料/稀疏資料）。

