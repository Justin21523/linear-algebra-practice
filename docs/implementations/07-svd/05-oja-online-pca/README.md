# 實作說明：Oja’s Online PCA（07-svd/05-oja-online-pca）

## 對應原始碼

- 單元路徑：`07-svd/05-oja-online-pca/`
- 單元概念：`07-svd/05-oja-online-pca/README.md`
- 實作檔案：
  - `07-svd/05-oja-online-pca/python/oja_online_pca_numpy.py`
- 文件路徑：`docs/implementations/07-svd/05-oja-online-pca/README.md`

## 目標與背景

- 本實作示範：在資料量很大（或資料以串流方式到來）時，用 **Oja’s rule** 近似學出 top-k 主成分子空間，而不是做完整 SVD。
- 適用情境：online learning、超大資料 PCA、特徵壓縮、embedding 前處理（只能看資料一次或只能用小批次）。
- 關鍵觀念：你在學的是「子空間」（subspace），因此驗證時要用 **principal angles** 或投影矩陣距離，而不是硬比向量逐元素。

## 如何執行

```bash
cd 07-svd/05-oja-online-pca/python
python3 oja_online_pca_numpy.py
```

## 核心做法（重點步驟）

1. 產生合成資料：先指定協方差特徵值（控制 eigengap），再抽樣得到資料矩陣 `X`。
2. 參考解：用樣本協方差 `C=(XᵀX)/(n-1)` 做 `eigh`，取 top-k eigenvectors 當作 `V_ref`.
3. block Oja 更新（mini-batch）：
   - `XW = X_batch W`
   - `Cw ≈ X_batchᵀ (X_batch W) / b`
   - `G = Cw - W(WᵀCw)`（維持在 Stiefel manifold 的切空間）
   - `W ← W + η G`，再用 QR 重新正交化
4. 定期驗證：principal angles cosines、explained variance ratio（EVR）。

## 詳細說明（繁中）

### Oja’s rule 在做什麼？

PCA 的 top-k 主成分子空間，是「讓投影後變異最大」的 k 維子空間。Oja’s rule 可以視為在做一個 **隨機梯度上升**：

- 目標：最大化 `trace(Wᵀ C W)`，並限制 `WᵀW=I`（`W` 是 `d×k`）
- 小批次下，用 `C_batch` 近似 `C`，做一次梯度更新

因為我們要求 `WᵀW=I`，所以更新不能直接用 `C_batch W`，需要把「破壞正交性」的方向投影掉（本實作用 `G = Cw - W(WᵀCw)`），最後再做 QR 讓 `W` 回到「正交欄」的集合上。

### 為什麼 eigengap 會影響收斂？

簡單說：當 `λ_k` 與 `λ_{k+1}` 很接近（eigengap 小），第 k 個與第 k+1 個方向的「能量」差不多，演算法更難穩定分辨哪個才是 top-k；你會看到 principal angles 的 cosines 收斂更慢、波動更大。

## 程式碼區段（節錄 + 解釋）

> 節錄 block Oja 的核心更新（mini-batch）。

```text
XW = Xb @ W
Cw = (Xb.T @ XW) / max(b, 1)
G = Cw - W @ (W.T @ Cw)
W = W + eta * G
W = orthonormalize_columns(W)
```

- 這段在做什麼：用 batch 估計 `C_batch W`，再用 `G` 做切空間修正，最後更新並重新正交化。
- 為什麼要 `orthonormalize_columns`：在線上更新中會累積數值誤差；QR 讓 `WᵀW≈I`，也讓 principal angles 的計算更可靠。

## 驗證方式與預期輸出

- `principal-angle cosines`：越接近 1，代表學到的子空間越接近參考 top-k。
- `EVR(top-k)`：越接近 `Reference EVR(top-k)` 越好（在噪聲存在下通常不會完全相同）。
- 預期現象：
  - “Easy eigengap” 會更快變好；“Hard eigengap” 會更慢且更抖。
  - learning rate 太大會不穩（本實作用衰減 `η_t = η0 / (1+t/T)` 讓後期更穩）。

