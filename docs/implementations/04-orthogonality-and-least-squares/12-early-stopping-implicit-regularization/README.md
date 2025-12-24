# 實作說明：12-early-stopping-implicit-regularization（04-orthogonality-and-least-squares）

## 對應原始碼

- 單元路徑：`04-orthogonality-and-least-squares/12-early-stopping-implicit-regularization/`
- 概念說明：`04-orthogonality-and-least-squares/12-early-stopping-implicit-regularization/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/12-early-stopping-implicit-regularization/python/early_stopping_gd_numpy.py`

## 目標與背景

你已經看過顯式正則化（Ridge/TSVD）可以改善病態最小平方的穩定性與泛化。

本單元要再補上一個 ML 很常用、但容易被忽略的觀點：

> **早停（early stopping）本身就是一種隱式正則化**  
> 尤其當你用迭代法（如 GD）解最小平方時，「跑幾步」就等同於一個可調的超參數。

因此本單元把「迭代次數 `T`」當成超參數，像選 `λ/k` 一樣用驗證集來選 `T`。

## 如何執行

> 需要 `numpy`：請先依 `requirements.txt` 安裝（見 repo 根目錄 `README.md` 的 venv 範例）。

```bash
cd 04-orthogonality-and-least-squares/12-early-stopping-implicit-regularization/python
python3 early_stopping_gd_numpy.py
```

## 核心做法（重點）

1. 仍用 Vandermonde（多項式）設計矩陣，刻意製造病態/高相關特徵。
2. 用訓練集跑 GD（不加 λ）：
   - `f(x)=1/2‖Ax-b‖²`
   - `x ← x - α Aᵀ(Ax-b)`，其中 `α=1/L`、`L=‖A‖₂²`
3. 事先選一串 checkpoint `T`（例如 0/1/2/5/…/4000），跑到最大步數時順便把每個 checkpoint 的 train/val RMSE 記下來。
4. 用 `val RMSE` 最小的 checkpoint 當作最佳早停點（best `T`），並用 ASCII bar 直觀呈現曲線。

## 詳細說明

### 1) 為什麼早停會像正則化？

在病態問題裡，解會對小噪聲非常敏感；GD 一開始通常先往「主要方向」走（較大的特徵方向/奇異值方向），後面才慢慢把小奇異值方向也補齊。

但小奇異值方向常同時是：

- 最容易放大噪聲的方向
- 最容易造成係數爆炸、泛化變差的方向

所以如果你跑太久，訓練誤差繼續變好，但驗證誤差可能開始變差；提早停止就能避免把噪聲「學進去」。

### 2) 你要看的輸出是什麼？

程式會印一張表：

- `T`：迭代步數
- `train_rmse`
- `val_rmse`
- `‖x‖`：係數大小（通常會隨 `T` 增大）
- `curve`：ASCII bar（val RMSE 越低 bar 越長）

並標註 `<== best` 的最佳 `T`。

### 3) 步長為什麼用 `1/L`？

對 `f(x)=1/2‖Ax-b‖²`，梯度 Lipschitz 常數是：

`L = ‖A‖₂² = σ_max(A)²`

用 `α=1/L` 是一個保守且常見的選擇，能避免爆掉並讓目標值穩定下降（在凸二次問題上特別好理解）。

## 驗證方式（你應該看到的現象）

- train RMSE 隨 `T` 下降（或至少不會變差太多）
- val RMSE 通常不會一路下降，常見是中間某個 `T` 最好
- `‖x‖` 隨 `T` 增大，代表模型越來越「大/複雜」

