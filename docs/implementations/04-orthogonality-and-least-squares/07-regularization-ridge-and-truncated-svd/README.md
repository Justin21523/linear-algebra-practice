# 實作說明：07-regularization-ridge-and-truncated-svd（04-orthogonality-and-least-squares）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/07-regularization-ridge-and-truncated-svd/`
- 概念說明：`04-orthogonality-and-least-squares/07-regularization-ridge-and-truncated-svd/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/07-regularization-ridge-and-truncated-svd/python/regularization_ridge_tsvd_numpy.py`

## 目標與背景

延續上一個單元的「病態/共線性最小平方」，本單元要回答：

- 為什麼最小平方解在共線性下會 **係數爆炸**、對小擾動很敏感？
- 透過 **Ridge（Tikhonov）** 與 **Truncated SVD（TSVD）**，如何用正則化降低不穩定？

## 如何執行

> 需要 `numpy`：請先依 `requirements.txt` 安裝（見 repo 根目錄 `README.md` 的 venv 範例）。

```bash
cd 04-orthogonality-and-least-squares/07-regularization-ridge-and-truncated-svd/python
python3 regularization_ridge_tsvd_numpy.py
```

## 核心做法（重點）

程式會建立兩種「刻意做壞」的資料：

1. **Ill-conditioned（幾乎共線）**：第二個特徵 `x2 ≈ x1`
2. **Rank-deficient（完全共線）**：`x2 == x1`（rank 下降，最小平方解不唯一）

並比較：

- `LS-QR`、`LS-SVD(pinv)`（不正則化的最小平方基準）
- `Ridge-SVD`（用 SVD 的 filter factors）
- `Ridge-Normal`（解 `(AᵀA+λI)x=Aᵀb`）
- `TSVD k=2`（丟掉最小奇異值方向）

輸出會包含：

- 殘差：`‖Ax̂-b‖`
- LS 最佳性：`‖Aᵀr‖`（只對「不正則化」理論上應接近 0）
- Ridge 最佳性：`‖Aᵀr-λx̂‖`（理論上應接近 0）
- TSVD 子空間最佳性：`‖U_kᵀ r‖`（理論上應接近 0）
- 穩定性：`‖Δx‖/‖Δb‖`（越小越穩）

## 詳細說明

### 1) Ridge 的目標與一階條件

Ridge Regression（L2 正則化）是：

`x̂_λ = argmin_x (‖Ax-b‖₂² + λ‖x‖₂²)`

令 `r=b-Ax̂_λ`，對 `x` 取導數並令其為 0，可得：

`Aᵀr = λx̂_λ`

所以你在 Ridge 下 **不應該再用** `Aᵀr≈0` 當驗算；要改驗 `Aᵀr-λx̂≈0`。

### 2) Ridge 的 SVD 視角（filter factors）

若 `A=UΣVᵀ`，Ridge 解可以寫成：

`x̂_λ = V diag( σ / (σ²+λ) ) Uᵀ b`

其中 `σ/(σ²+λ)` 會把小奇異值方向「壓下去」，避免除以很小的 `σ` 放大噪聲。

節錄自 `regularization_ridge_tsvd_numpy.py`：

```python
factors = s / (s**2 + lam)  # EN: Ridge filter factors: σ/(σ^2+λ) damp small σ directions.
return Vt.T @ (factors * (U.T @ b))  # EN: x = V diag(factors) U^T b.
```

### 3) Truncated SVD（TSVD）在做什麼？

TSVD 的想法更直接：把最小的奇異值方向直接丟掉，只用前 `k` 個：

`x̂_k = V_k diag(1/σ_k) U_kᵀ b`

這等價於「只在可靠的子空間做回歸」，所以程式用 `‖U_kᵀ r‖` 來驗算殘差是否正交於該子空間。

### 4) 你應該觀察到的現象

- 在 ill-conditioned case：不正則化時 `‖x̂‖` 往往很大、穩定性指標 `‖Δx‖/‖Δb‖` 也大；Ridge/TSVD 通常會顯著下降。
- 在 rank-deficient case：`LS-QR` 可能失敗；SVD(pinv)/Ridge 仍可回傳合理解（Ridge 對 `λ>0` 會給唯一解）。

