# 實作說明：01-svd-and-pca（07-svd）

## 對應原始碼

- 單元路徑：`07-svd/01-svd-and-pca/`
- 概念說明：`07-svd/01-svd-and-pca/README.md`
- 程式實作：
  - `07-svd/01-svd-and-pca/python/svd_pca_manual.py`
  - `07-svd/01-svd-and-pca/python/svd_pca_numpy.py`

## 目標與背景

本單元針對機器學習常見需求，把 SVD 與 PCA 用「可驗算」的方式連起來：

- SVD：`A = UΣVᵀ`（重建、正交性驗證、低秩近似）
- PCA：資料中心化後 `Xc` 的 SVD 直接給出主成分方向（`V`），並可計算 explained variance ratio

## 如何執行

### Python

```bash
cd 07-svd/01-svd-and-pca/python
python3 svd_pca_manual.py
python3 svd_pca_numpy.py
```

## 核心做法（重點）

- 先用小矩陣 `A` 做 SVD，確認：
  - `UᵀU ≈ I`、`VᵀV ≈ I`
  - `A_hat = UΣVᵀ` 重建誤差很小
- 再做低秩近似：只取前 `k` 個奇異值/向量，觀察 `||A - A_k||_F` 如何變大
- 最後把 PCA 寫成：中心化後 `Xc` 做 SVD，取 `V` 的前 `k` 欄投影並重建

## 詳細說明

### 1) SVD 的形狀（shape）一定要先釐清

若 `A` 是 `m×n`：

- `U`：`m×m`（full）或 `m×r`（thin）
- `Σ`：`m×n`（full）或 `r×r`（thin）
- `V`：`n×n`（full）或 `n×r`（thin）
- `r = rank(A)`（或 `min(m,n)` 的 thin/economy 版本）

本 repo 的 `svd_pca_manual.py` 使用 **thin SVD**（只保留非零奇異值的部分），更貼近 ML 的「降維/低秩」用途。

### 2) 為什麼可以用 `AᵀA` 的特徵分解得到 SVD？

因為 `AᵀA` 對稱且半正定（PSD），可以寫成：

`AᵀA v_i = λ_i v_i`，其中 `λ_i = σ_i^2`  
所以奇異值：`σ_i = sqrt(λ_i)`，右奇異向量：`v_i`

接著用：

`u_i = (A v_i) / σ_i`

就能得到左奇異向量（對應非零奇異值時成立）。

> 注意：`AᵀA` 會把條件數平方化（cond 變差），因此這種做法適合教學與小矩陣；實務上更常直接用 `svd()`。

### 3) 低秩近似（Low-rank approximation）

把 SVD 只保留前 `k` 個：

`A_k = U_k Σ_k V_kᵀ`

在 Frobenius norm 下，`A_k` 是最佳 rank-k 近似（Eckart–Young theorem）。本單元用 `||A - A_k||_F` 做直觀比較。

### 4) PCA 與 SVD 的關係（ML 最常用）

對資料矩陣 `X (n_samples×n_features)`：

1. 先中心化：`Xc = X - mean(X)`
2. 做 thin SVD：`Xc = U Σ Vᵀ`
3. 主成分方向（principal directions）就是 `V` 的 columns
4. 協方差矩陣特徵值：
   - `eigenvalues = (σ_i^2)/(n_samples-1)`
   - explained variance ratio：`eigenvalues / sum(eigenvalues)`
5. 投影與重建：
   - `Z = Xc @ V_k`
   - `X_hat = Z @ V_kᵀ + mean`

## 程式碼區段（節錄）

以下節錄自 `07-svd/01-svd-and-pca/python/svd_pca_manual.py`（核心：用 `eigh(AᵀA)` 組 thin SVD）：

```python
def thin_svd_via_eigh(A: np.ndarray, eps: float = EPS) -> ThinSVD:
    m, n = A.shape
    AtA = A.T @ A
    eigenvalues, V = np.linalg.eigh(AtA)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[order]
    V_sorted = V[:, order]
    singular_values = np.sqrt(np.clip(eigenvalues_sorted, 0.0, None))
    keep = singular_values > eps
    s = singular_values[keep]
    V_r = V_sorted[:, keep]
    U = (A @ V_r) / s
    return ThinSVD(U=U, s=s, Vt=V_r.T)
```

## 驗證方式

- SVD：
  - `UᵀU ≈ I`、`VᵀV ≈ I`
  - `||A - UΣVᵀ||_F` 應接近 0（或非常小）
  - `k=1` 的 `||A - A_1||_F` 應明顯大於完整重建
- PCA：
  - explained variance ratio 總和應接近 1
  - `k=1` 重建後 `||X - X_hat||_F` 會大於 `k=2`（若 2D 資料）

