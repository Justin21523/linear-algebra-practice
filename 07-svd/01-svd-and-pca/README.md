# SVD + PCA：奇異值分解與主成分分析 (Singular Value Decomposition & PCA)

> 用同一個框架理解：分解、最佳低秩近似、降維與重建誤差

## 學習目標

完成本單元後，你應該能：

1. 寫出並辨認 SVD：`A = UΣVᵀ`（理解 U/V/Σ 的形狀與意義）
2. 用數值方法驗證正交性：`UᵀU ≈ I`、`VᵀV ≈ I`
3. 做低秩近似：只取前 `k` 個奇異值/向量，並比較重建誤差
4. 用 SVD 做 PCA：計算主成分方向、解釋變異比例（explained variance ratio）、投影與重建

## 為什麼跟機器學習有關？

- PCA：最常用的線性降維方法之一（特徵降維、去噪、視覺化）
- 低秩近似：矩陣壓縮、Latent factor models、推薦系統（matrix factorization）
- 數值穩定：很多 ML 訓練流程都在解最小平方/做分解，選錯方法會放大誤差或造成訓練不穩定

## 本單元實作（Python）

```
python/
├── svd_pca_manual.py   # 不呼叫 np.linalg.svd：用 AᵀA 的特徵分解（eigh）組出 thin SVD + PCA
└── svd_pca_numpy.py    # 使用 np.linalg.svd：做 SVD/PCA 並驗證性質
```

