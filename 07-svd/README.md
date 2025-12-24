# 第七章：奇異值分解 (Singular Value Decomposition, SVD)

> 對機器學習與數值線性代數非常核心的分解：降維（PCA）、低秩近似、去噪、推薦系統、嵌入表示等

## 本章概述

奇異值分解將任意矩陣分解為 `A = UΣVᵀ`，其中：

- `U`、`V` 為正交矩陣（orthogonal matrices）
- `Σ` 為非負奇異值（singular values）構成的對角矩陣

SVD 的重點不只是「算出來」，而是理解：

- 低秩近似（low-rank approximation）為什麼能做壓縮/去噪
- PCA 與 SVD 的關係（資料中心化後，主成分方向就是 `V` 的前幾個向量）
- 數值上為什麼要偏好 SVD/QR，而不是自己硬寫不穩定的公式

## 本章單元

```
01-svd-and-pca/   SVD 基礎 + 低秩近似 + PCA（以 NumPy 與「不用 np.linalg.svd」的方式對照）
```

