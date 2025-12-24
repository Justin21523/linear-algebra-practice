# 實作說明：03-top-k-pca-block-power-deflation-lanczos（07-svd）

## 對應原始碼

- 單元路徑：`07-svd/03-top-k-pca-block-power-deflation-lanczos/`
- 概念說明：`07-svd/03-top-k-pca-block-power-deflation-lanczos/README.md`
- 程式實作：
  - `07-svd/03-top-k-pca-block-power-deflation-lanczos/python/pca_topk_methods_numpy.py`

## 目標與背景

上一個單元你已經能用 power iteration / Lanczos 找「第一主成分」。

但實務更常需要 top-k（例如 k=10、k=50）：

- 降維（特徵壓縮）
- 去噪（保留主要訊號方向）
- 視覺化（k=2/3）

本單元把「只找第一個」擴展成「一次找前 k 個」，並對照三條典型路線：

1. **Block power / orthogonal iteration**：一次迭代一個 k 維子空間
2. **Deflation**：逐個找，再把已找到的方向從運算子中移除
3. **Lanczos（top-k Ritz）**：一次建立 Krylov 子空間，再從小矩陣 `T` 同時取多個 Ritz eigenpairs

## 如何執行

> 需要 `numpy`：請先依 `requirements.txt` 安裝（見 repo 根目錄 `README.md` 的 venv 範例）。

```bash
cd 07-svd/03-top-k-pca-block-power-deflation-lanczos/python
python3 pca_topk_methods_numpy.py
```

## 核心做法（重點）

### 1) PCA 仍然是特徵分解問題

對中心化資料 `Xc`：

`C = (XcᵀXc)/(n-1)`

top-k PCA 就是求 `C` 的前 k 個特徵向量。

### 2) 不形成 C：只寫 matvec

`Cv = Xcᵀ(Xc v)/(n-1)`

這讓你能把 `C` 當作「黑盒子線性算子」，只要能做 `matvec(v)`，就能跑迭代法。

### 3) 三種方法的差異

- Block power：
  - 迭代：`Q ← orth(CQ)`（Q 是 `n×k` 子空間基底）
  - 收斂後用 Rayleigh–Ritz 得到 `(λ_i, v_i)`
- Deflation：
  - 找到 `(λ1,v1)` 後改用 `C' = C - λ1 v1 v1ᵀ` 再找下一個
  - 實作上只需要在 matvec 裡減掉 rank-1 項
- Lanczos（top-k Ritz）：
  - 先做 n_steps 次 Lanczos 得到 `Q_m` 與三對角 `T_m`
  - 對 `T_m` 做特徵分解，取最大 k 個 Ritz pairs，再回推到原空間：`V = Q_m Y_k`

## 詳細說明

### 1) 子空間一致性怎麼驗？

top-k 的重點常常不是「每個向量完全一樣」，而是「span(V_k) 是否一致」。

程式用：

`S = svd(V_refᵀ V_est)`

其奇異值就是 principal angles 的 cosines（越接近 1 越好），因此可用來量化子空間相似度。

### 2) 什麼是 eigen-residual？

對每個 `(λ_i, v_i)`，用：

`‖Cv_i - λ_i v_i‖`

檢查它到底是不是一個好特徵向量（殘差越小越好）。

### 3) 你應該看到的現象

在同一份資料上：

- block power 通常穩定、好理解，但可能需要較多迭代
- deflation 直覺但會累積誤差（尤其當近似 eigenvector 不夠準時）
- Lanczos 往往用較少 steps 就能得到很好的 top-k 子空間近似（但需要注意正交性漂移，實務常做 reorthogonalization）

## 程式碼區段（節錄）

以下節錄自 `pca_topk_methods_numpy.py`：以 matvec 方式表示 covariance：

```python
scale = 1.0 / max(n_samples - 1, 1)
def cov_matvec(v: np.ndarray) -> np.ndarray:
    return scale * (Xc.T @ (Xc @ v))
```

## 驗證方式

- 看 `subspace overlap singular values`：應該接近全 1
- 看 `residual norms`：應該很小（相對於特徵值尺度）
- 比較三個方法的 `iters`（block power）/ `total power iters`（deflation）/ `krylov dim`（Lanczos）

