# Top-k PCA：Block Power / Deflation / Lanczos（只求前 k 個主成分）

> 真實資料常只需要前幾個主成分：降維、壓縮、去噪；本單元示範三種「不用完整 SVD」的做法

## 學習目標

完成本單元後，你應該能：

1. 把 PCA 前 k 個主成分視為協方差矩陣 `C` 的前 k 個特徵向量（最大特徵值方向）
2. 用 matvec 避免顯式形成 `C`：
   - `Cv = Xcᵀ(Xc v)/(n-1)`
3. 實作三種 top-k 方法並理解差異：
   - **Block power / orthogonal iteration**：一次迭代一整個 k 維子空間
   - **Deflation**：先找第一主成分，再把它「移除」後重複找下一個
   - **Lanczos（Ritz top-k）**：建立 Krylov 子空間後一次取出多個 Ritz eigenpairs
4. 用數值方式驗算：
   - 每個向量的特徵殘差 `‖Cv_i-λ_i v_i‖`
   - 子空間一致性（與參考 SVD 的 `V_k` 比較）

## 本單元實作（Python）

```
python/
└── pca_topk_methods_numpy.py  # block power / deflation / Lanczos：求前 k 個主成分並對照 full SVD（小尺寸驗證）
```

