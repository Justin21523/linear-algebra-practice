# 實作說明：Randomized SVD vs Oja（07-svd/06-randomized-svd-vs-oja-benchmark）

## 對應原始碼

- 單元路徑：`07-svd/06-randomized-svd-vs-oja-benchmark/`
- 單元概念：`07-svd/06-randomized-svd-vs-oja-benchmark/README.md`
- 實作檔案：
  - `07-svd/06-randomized-svd-vs-oja-benchmark/python/benchmark_randomized_svd_vs_oja_numpy.py`
- 文件路徑：`docs/implementations/07-svd/06-randomized-svd-vs-oja-benchmark/README.md`

## 目標與背景

- 本實作示範：用**同一份中心化資料** `Xc`，對照 Randomized SVD 與 Oja’s online PCA 的「品質 vs 成本」。
- 這是 ML/數值線代常見的決策：你只想要 top-k 主成分，不想做完整 SVD，又希望可控地在「速度/品質」之間取捨。

## 如何執行

```bash
cd 07-svd/06-randomized-svd-vs-oja-benchmark/python
python3 benchmark_randomized_svd_vs_oja_numpy.py
```

## 核心做法（重點步驟）

1. 產生合成資料 `Xc`（已中心化），並用 covariance `C=(XcᵀXc)/(n-1)` 的 `eigh` 當參考 top-k 子空間。
2. Randomized SVD sweep：
   - 改 `k`、`p`（oversampling）、`q`（power iterations）
   - 量測：principal angles cosines、EVR（explained variance ratio）
   - 成本提示：用 `sketch_dim=k+p` 與 `approx_matvecs≈(2*(q+1))*sketch_dim` 當 proxy
3. Oja sweep：
   - 改 `k`、`eta0`、`decay_steps`（learning-rate schedule）
   - 量測：principal angles cosines、EVR
   - 成本提示：epochs（整份資料 pass 次數）與 steps（mini-batch 更新次數）

## 詳細說明（繁中）

### 品質指標：為什麼用 principal angles？

PCA 的答案本質上是「子空間」，不是單一向量。對 top-k 主成分，你要比的是：

- `span(V_ref_k)` vs `span(V_hat_k)`

principal angles 的 cosines 介於 `[0,1]`：

- 越接近 `1` → 子空間越接近（越好）
- 本程式會印 `min_cos`（最差的一個角）與 `mean_cos`（平均）

### 品質指標：EVR（explained variance ratio）

給定 covariance `C` 與一組正交基 `W`（`d×k`），子空間捕捉的變異比例是：

- `EVR = trace(Wᵀ C W) / trace(C)`

EVR 越大表示你用 k 維子空間保留了越多的總變異（通常越好）。

### 成本提示怎麼解讀？

本單元的成本只做「粗略 proxy」，用來幫你快速比較不同超參數設定：

- Randomized SVD：
  - `q` 每加 1，通常要多做一次 `Aᵀ` 與 `A` 的大型乘法（成本上升，但品質在 slow spectrum 可能提升很多）
- Oja：
  - `epochs` 越多、steps 越多 → 成本越高
  - learning-rate 太大可能不穩；太小又收斂太慢（因此要 sweep）

## 程式碼區段（節錄 + 解釋）

> 節錄 Randomized SVD 的核心流程（range finder + power iteration + 小矩陣 SVD）。

```text
Y = Xc @ Omega
Q = orthonormalize_columns(Y)
for _ in range(q):
    Z = Xc.T @ Q
    Y = Xc @ Z
    Q = orthonormalize_columns(Y)
B = Q.T @ Xc
_, _, Vt = np.linalg.svd(B, full_matrices=False)
V_hat = Vt[:k, :].T
```

- 這段在做什麼：先用 `Q` 近似 `Xc` 的 column space，再把大問題壓到小矩陣 `B` 上做 SVD，拿到近似 top-k 方向。

## 驗證方式與預期輸出

- 你會看到兩個表格（randSVD / Oja），每列含：
  - `min_cos`、`mean_cos`、`EVR`：品質
  - `seconds`：粗略時間（不同機器不可直接比，但同機器內部相對可看趨勢）
  - `cost`：成本 proxy（matvec/epochs/steps）
- 典型現象：
  - slow spectrum 下，randSVD 的 `q=1~2` 往往比 `q=0` 明顯更好
  - Oja 的品質高度依賴 learning-rate（需要 sweep 才能挑到好設定）

