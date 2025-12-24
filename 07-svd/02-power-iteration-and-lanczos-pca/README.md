# 大型 PCA：Power Iteration / Lanczos（只求第一主成分，不做完整 SVD）

> 當特徵數很大時，你通常只想要「前幾個主成分」；這時用迭代法找最大特徵值/特徵向量會更實用

## 學習目標

完成本單元後，你應該能：

1. 把 PCA 的第一主成分寫成「協方差矩陣的最大特徵向量」：
   - `C = (Xcᵀ Xc)/(n-1)`，第一主成分 `v1 = argmax_{‖v‖=1} vᵀCv`
2. 理解「不形成 C」也能做矩陣向量乘法：
   - `Cv = Xcᵀ(Xc v)/(n-1)`（只需要 `Xc v` 與 `Xcᵀ u`）
3. 實作兩種常見的最大特徵對（largest eigenpair）迭代法：
   - Power Iteration：概念最簡單，但收斂率受特徵值間距影響
   - Lanczos：對稱矩陣的 Krylov 方法，通常收斂更快
4. 用數值方式驗證結果：
   - `‖Cv - λv‖`（特徵殘差）
   - 與完整分解（小尺寸）對照 `cosine similarity`

## 本單元實作（Python）

```
python/
└── pca_power_lanczos_numpy.py  # Power iteration / Lanczos 求第一主成分 + 對照完整 SVD（小尺寸驗證）
```

