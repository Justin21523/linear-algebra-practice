# 實作說明：14-lsqr-damped-and-stopping-criteria（04-orthogonality-and-least-squares）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/14-lsqr-damped-and-stopping-criteria/`
- 概念說明：`04-orthogonality-and-least-squares/14-lsqr-damped-and-stopping-criteria/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/14-lsqr-damped-and-stopping-criteria/python/lsqr_damped_stopping_manual.py`

## 目標與背景

本單元是「LSQR 的實務版延伸」：

- 你要能用 LSQR 解 `min‖Ax-b‖`（不形成 `AᵀA`）
- 你也要能處理 **Ridge/Tikhonov 正則化**：`min (‖Ax-b‖² + damp²‖x‖²)`
- 同時需要比較完整的停止條件（stopping criteria），避免：
  - 跑太少沒收斂
  - 跑太久浪費計算
  - 只看單一指標導致誤判

## 如何執行

> 需要 `numpy`：請先依 `requirements.txt` 安裝（見 repo 根目錄 `README.md` 的 venv 範例）。

```bash
cd 04-orthogonality-and-least-squares/14-lsqr-damped-and-stopping-criteria/python
python3 lsqr_damped_stopping_manual.py
```

## 核心做法（重點）

### 1) Damped LSQR ⇔ 擴增系統

damped（Ridge）問題：

`min_x (‖Ax-b‖² + damp²‖x‖²)`

可改寫成擴增 least squares：

`min_x ‖[A; damp I]x - [b; 0]‖²`

因此你只要能寫出擴增矩陣的 matvec：

- `A_aug v = [A v; damp v]`
- `A_augᵀ u = Aᵀ u_top + damp u_bottom`

就能把「原本的 LSQR」直接套用到 damped 版本。

### 2) 更完整的 stopping criteria（本單元示範版）

本單元示範會同時看：

- 資料殘差：`‖Ax-b‖`
- 正規方程殘差（Ridge 梯度）：`‖Aᵀ(Ax-b) + damp² x‖`
- 混合門檻：`btol*‖b‖ + atol*‖A‖*‖x‖`

並用 power iteration 估計 `‖A_aug‖₂`（只用 matvec，不用 SVD）。

## 詳細說明

### 1) 為什麼「damp」等價於 Ridge？

Ridge（Tikhonov）常寫成：

`min_x (‖Ax-b‖² + λ‖x‖²)`

本單元採用 `damp` 參數，並令：

`λ = damp²`

所以 `damp` 只是把 `λ` 的平方根拆出來，方便寫成擴增矩陣 `[A; damp I]`。

### 2) 什麼時候應該加 damp？

當 `A` 病態或接近 rank-deficient（共線性/特徵高度相關）時：

- 不正則化解可能係數爆炸、對噪聲敏感
- 加上 `damp>0` 後解會更穩、且變成唯一解

你應該在輸出看到：

- `damp` 增大時 `‖x‖` 往往下降
- `‖Aᵀ(Ax-b)+damp²x‖` 更容易降到小值（因為它就是 Ridge 的一階條件）

### 3) 為什麼要同時看 `‖Ax-b‖` 與 `‖Aᵀr‖`？

對不正則化 least squares，最佳解滿足：

`Aᵀ(Ax-b)=0`

因此 `‖Aᵀr‖` 是最直接的最小化驗算。

但對 damped（Ridge）：

`Aᵀ(Ax-b) + damp² x = 0`

所以本單元用 `‖Aᵀ(Ax-b)+damp²x‖` 當作「Ridge 的最小化殘差指標」。

## 程式碼區段（節錄）

以下節錄自 `lsqr_damped_stopping_manual.py`：擴增矩陣的 matvec（不形成 `A_aug`）：

```python
def matvec_A_aug(v: np.ndarray) -> np.ndarray:
    top = A @ v
    bottom = damp * v
    return np.concatenate([top, bottom])

def matvec_AT_aug(u_aug: np.ndarray) -> np.ndarray:
    u_top = u_aug[:m]
    u_bottom = u_aug[m:]
    return (A.T @ u_top) + (damp * u_bottom)
```

## 驗證方式

- `damp=0`：與 `np.linalg.lstsq(A,b)` 的結果應接近
- `damp>0`：與用 SVD filter factors 計算的 ridge 參考解應接近
- 看輸出中的 checkpoint：`‖Ax-b‖` 與 `‖Aᵀ(Ax-b)+damp²x‖` 應逐步下降（至少趨勢上）

