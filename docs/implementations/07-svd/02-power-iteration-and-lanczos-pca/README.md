# 實作說明：02-power-iteration-and-lanczos-pca（07-svd）

## 對應原始碼

- 單元路徑：`07-svd/02-power-iteration-and-lanczos-pca/`
- 概念說明：`07-svd/02-power-iteration-and-lanczos-pca/README.md`
- 程式實作：
  - `07-svd/02-power-iteration-and-lanczos-pca/python/pca_power_lanczos_numpy.py`

## 目標與背景

在真實機器學習場景中，你常常只需要：

- 第一主成分（或前幾個主成分）
- 最大特徵值/特徵向量（dominant eigenpair）

而不是把整個矩陣做完整 SVD / 完整特徵分解（太慢、太大、太耗記憶體）。

因此本單元示範兩個「只用矩陣向量乘法」的經典迭代法：

1. **Power iteration**：最簡單直覺
2. **Lanczos**：對稱矩陣的 Krylov 方法，通常更快

## 如何執行

> 需要 `numpy`：請先依 `requirements.txt` 安裝（見 repo 根目錄 `README.md` 的 venv 範例）。

```bash
cd 07-svd/02-power-iteration-and-lanczos-pca/python
python3 pca_power_lanczos_numpy.py
```

## 核心做法（重點）

### 1) PCA ⇔ 協方差矩陣最大特徵向量

令中心化資料 `Xc`（`n_samples×n_features`），協方差矩陣：

`C = (Xcᵀ Xc)/(n-1)`

第一主成分方向 `v1` 就是 `C` 的最大特徵向量。

### 2) 不形成 C：只寫 matvec

若你要算 `Cv`，可寫成：

`Cv = Xcᵀ (Xc v)/(n-1)`

這讓演算法只需要 `Xc v` 與 `Xcᵀ u` 兩次乘法，避免建 `C (n_features×n_features)`。

### 3) Power iteration 的驗算

每次更新：

- `v ← C v`
- `v ← v/‖v‖`

收斂後用殘差驗算：

`‖Cv-λv‖`（其中 `λ = vᵀCv`）

### 4) Lanczos 的驗算（Ritz pair）

Lanczos 會在 Krylov 子空間中建立一個小的三對角矩陣 `T_k`，其最大特徵值/向量會給出 `C` 的近似最大特徵對（Ritz pair）。

同樣用：

`‖Cv-λv‖`

驗算精度。

## 詳細說明

### 1) 為什麼 Lanczos 通常比 Power iteration 快？

Power iteration 只靠「反覆乘上 C」放大最大特徵向量的成分；若 `λ1` 與 `λ2` 很接近（譬如資料主方向不明顯），收斂會變慢。

Lanczos 則一次把更多資訊打包到 Krylov 子空間，等於在更豐富的子空間內尋找最佳近似，因此常能更快逼近極端特徵值。

### 2) 這份程式做了哪些驗證？

因為本單元在小尺寸下也能跑，所以程式會額外用 `np.linalg.svd(Xc)` 做「參考解」：

- 參考第一主成分：`v_ref = Vt[0]`
- 參考特徵值：`λ_ref = σ1²/(n-1)`

然後對比：

- `cosine similarity = |v_refᵀ v|`（方向一致性，忽略正負號）
- `‖Cv-λv‖`（特徵殘差）
- 用 `v` 做 rank-1 重建 `X_hat` 的 `‖X-X_hat‖_F`

## 程式碼區段（節錄）

以下節錄自 `pca_power_lanczos_numpy.py`（核心：用 `Xc` 寫 covariance 的 matvec）：

```python
scale = 1.0 / max(n_samples - 1, 1)
def cov_matvec(v: np.ndarray) -> np.ndarray:
    return scale * (Xc.T @ (Xc @ v))
```

## 驗證方式（你應該看到的現象）

- Lanczos 通常在較少 steps 下就能得到很高的 `cosine similarity` 與很小的 `‖Cv-λv‖`
- Power iteration 也能收斂，但可能需要更多 iters（尤其當主方向不明顯時）

