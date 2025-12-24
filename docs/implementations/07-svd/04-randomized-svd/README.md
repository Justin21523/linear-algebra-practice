# 實作說明：Randomized SVD（07-svd/04-randomized-svd）

## 對應原始碼

- 單元路徑：`07-svd/04-randomized-svd/`
- 單元概念：`07-svd/04-randomized-svd/README.md`
- 實作檔案：
  - `07-svd/04-randomized-svd/python/randomized_svd_pca_numpy.py`
- 文件路徑：`docs/implementations/07-svd/04-randomized-svd/README.md`

## 目標與背景

- 本實作示範：用 **隨機投影** 近似矩陣的 column space，進而做 **近似 top-k SVD**，避免在大型矩陣上做完整分解。
- 適用情境：大型 PCA、低秩近似、去噪/壓縮、推薦系統 embedding 等（尤其是資料很大、只想要前幾個方向時）。
- 核心 trade-off：計算量（少做完整分解）vs 精度（用 `p`/`q` 補足）。

## 如何執行

```bash
cd 07-svd/04-randomized-svd/python
python3 randomized_svd_pca_numpy.py
```

## 核心做法（重點步驟）

1. 抽樣：產生高斯隨機矩陣 `Ω ∈ R^{n×(k+p)}`。
2. Range finder：`Y = AΩ`，再用 QR 得到正交基底 `Q`（`QᵀQ≈I`）。
3.（可選）Power iteration：重複 `Q ← orth(A(AᵀQ))` 共 `q` 次，強化主方向（特別是奇異值衰減很慢時）。
4. 壓縮：`B = QᵀA`（小矩陣）。
5. 小矩陣 SVD：`B = ŨΣVᵀ`，回推 `U ≈ QŨ`，最後取 top-k。

## 詳細說明（繁中）

### 為什麼 Randomized SVD 有效？

直覺是：如果 `Q` 的欄空間很好地涵蓋了 `A` 的 column space（至少涵蓋前 `k` 個主方向），那麼

- `QQᵀA` 就會是一個很好的近似（把 `A` 投影回 `span(Q)`）
- `B=QᵀA` 把問題變成「在小維度子空間內做 SVD」

Randomized SVD 的典型流程就是先用隨機投影找 `Q`（range finder），再在 `B` 上做完整 SVD，最後把結果「lift」回原空間。

### `p`（oversampling）與 `q`（power iterations）

- `p`：把 sketch 維度設為 `k+p`（例如 `k=10`、`p=10` → 20 維）。多出的維度能吸收「尾端能量」與隨機誤差，通常會讓近似更穩。
- `q`：當奇異值衰減很慢（spectrum “slow”）時，單次抽樣得到的 `Q` 可能不夠貼近真正的 top-k 子空間；透過
  `Q ← orth(A(AᵀQ))` 重複 `q` 次，相當於放大大奇異值、壓低小奇異值的影響，讓子空間更準（但每次會多 2 次 matvec：`Aᵀ` 與 `A`）。

## 程式碼區段（節錄 + 解釋）

> 節錄 `randomized_svd()` 的主流程（range finder + power iteration + 小矩陣 SVD）。

```text
Omega = rng.standard_normal((n, sketch_dim)).astype(float)
Y = A @ Omega
Q = orthonormalize_columns(Y)

for _ in range(q):
    Z = A.T @ Q
    Y = A @ Z
    Q = orthonormalize_columns(Y)

B = Q.T @ A
Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
U = Q @ Ub
```

- 這段在做什麼：用 `Ω` 抽樣得到 `Y=AΩ`，QR 得到 `Q`；`q>0` 時用 power iteration 反覆強化主方向；最後在小矩陣 `B` 上做 SVD，再用 `U=QŨ` 回推。
- 為什麼這樣寫：核心目標是把大矩陣 `A` 的低秩資訊「壓縮」到 `B`，讓昂貴的 SVD 發生在小矩陣上。
- 常見陷阱/邊界情況：
  - `k+p` 不要超過 `n`（程式裡用 `min(n, k+p)` 保護）。
  - `q` 太大會浪費成本；通常 `q=1~2` 就有感（尤其 slow spectrum）。
  - 若資料含噪、目標只是近似/降維，別過度追求非常小的誤差（本質上 noise 會限制可達精度）。

## 驗證方式與預期輸出

- 你可以檢查（程式會印出）：
  - `singular values rel_err`：top-k 奇異值相對誤差（越小越好）。
  - `reconstruction rel_err`：`‖A-U_kΣ_kV_kᵀ‖_F / ‖A‖_F`（越小越好）。
  - `gap to optimal rank-k`：相對於「最佳 rank-k 誤差」的差距（越接近 0 越好）。
  - `principal angle cosines`：子空間主角度的 cosines（越接近 1 表示越接近）。
- 預期特性：
  - spectrum “fast” 時，即使 `q=0` 也常有不錯結果；“slow” 時 `q=1~2` 會明顯改善。
  - `p` 從 5 提高到 10 通常也會更穩（但會增加計算量）。
- 浮點誤差容忍：本單元以相對誤差指標為主，不會要求到 `1e-12` 等極限精度；建議先看趨勢與相對改善量。
