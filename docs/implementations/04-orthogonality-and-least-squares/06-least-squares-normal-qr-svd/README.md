# 實作說明：06-least-squares-normal-qr-svd（04-orthogonality-and-least-squares）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/06-least-squares-normal-qr-svd/`
- 概念說明：`04-orthogonality-and-least-squares/06-least-squares-normal-qr-svd/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/06-least-squares-normal-qr-svd/python/least_squares_compare_numpy.py`

## 目標與背景

本單元用同一個最小平方問題 `min_x ‖Ax-b‖₂`，比較三種解法在「病態矩陣/共線性」下的差異：

1. **Normal Equation**：`(AᵀA)x = Aᵀb`（容易不穩，因為條件數被平方化）
2. **QR**：`A=QR`，解 `Rx = Qᵀb`（通常比 Normal 更穩）
3. **SVD**：`A=UΣVᵀ`，用 pseudo-inverse 求解（最通用，能處理 rank-deficient）

同時用三個指標驗算：

- 殘差：`‖Ax̂-b‖₂`
- 最小平方必要條件：`Aᵀr≈0`（`r=b-Ax̂`）
- 擾動穩定性：`b` 加很小的 `Δb` 後，`x̂` 的變化 `‖Δx‖` 有多大

## 如何執行

### Python

> 需要 `numpy`：請先依 `requirements.txt` 安裝（見 repo 根目錄 `README.md` 的 venv 範例）。

```bash
cd 04-orthogonality-and-least-squares/06-least-squares-normal-qr-svd/python
python3 least_squares_compare_numpy.py
```

## 核心做法（重點）

- 造三種設計矩陣 `A`：
  1. **Well-conditioned**：特徵彼此獨立
  2. **Ill-conditioned**：兩個特徵幾乎共線（multicollinearity）
  3. **Rank-deficient**：兩個特徵完全共線（rank 下降）
- 對每個 `A,b` 分別求 `x̂`（Normal/QR/SVD），並印出：
  - `‖Ax̂-b‖₂`
  - `‖Aᵀr‖₂`
  - `‖x̂‖₂`
  - `‖Δx‖/‖Δb‖` 與 `‖Δx‖/‖x̂‖`（穩定性）

## 詳細說明

### 1) 為什麼 Normal Equation 容易不穩？

Normal Equation 會先形成 `AᵀA`。在數值上，`AᵀA` 的條件數大約是：

`cond(AᵀA) ≈ cond(A)^2`

當 `A` 已經因共線性而病態時，平方化會讓有效精度更差，導致 `x̂` 更容易受浮點誤差影響。

### 2) 為什麼要看 `Aᵀr≈0`？

最小平方的最佳解 `x̂` 滿足殘差 `r=b-Ax̂` 與 column space 正交：

`Aᵀr = 0`

因此 `‖Aᵀr‖₂` 是一個很直接的驗算指標（越接近 0 越好；允許浮點誤差）。

### 3) 穩定性怎麼量化？

本單元固定生成一個很小的 `Δb`，再重新求解得到 `x̂(b+Δb)`，並報告：

- `gain = ‖Δx‖₂ / ‖Δb‖₂`（擾動放大倍率）
- `rel = ‖Δx‖₂ / ‖x̂‖₂`（相對變化）

注意：**病態問題本身就會敏感**，即使用 QR/SVD 也可能很不穩；但 Normal Equation 通常更糟。

### 4) SVD pseudo-inverse（可處理 rank-deficient）

以下節錄自 `least_squares_compare_numpy.py`：用 SVD 做 pseudo-inverse（會對很小的奇異值做截斷避免除以接近 0）：

```python
def solve_svd(A: np.ndarray, b: np.ndarray, rcond: float = RCOND) -> np.ndarray:  # EN: Solve least squares via SVD pseudo-inverse.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD A = U diag(s) V^T.
    cutoff = rcond * float(s.max())  # EN: Define absolute cutoff for singular values.
    keep = s > cutoff  # EN: Keep only "significant" singular values.
    U_r = U[:, keep]  # EN: Select kept left singular vectors.
    s_r = s[keep]  # EN: Select kept singular values.
    Vt_r = Vt[keep, :]  # EN: Select kept right singular vectors transposed.
    return Vt_r.T @ ((U_r.T @ b) / s_r)  # EN: x = V diag(1/s) U^T b (no explicit diag).
```

## 驗證方式（你應該看到的現象）

- Well-conditioned：三種方法的 `‖Ax̂-b‖₂` 與 `‖Aᵀr‖₂` 都很小，`x̂` 彼此接近。
- Ill-conditioned：`cond(A)` 很大、`cond(AᵀA)` 更大；Normal 的 `x̂` 更容易漂移（相對誤差/穩定性指標更差）。
- Rank-deficient：Normal/QR 可能直接失敗（矩陣奇異）；SVD 仍可回傳一個合理解（通常是 minimum-norm 解）。

