# 學習路徑：機器學習／數值線代導向

本路徑以「能寫出可驗算的程式」為目標：每章都用一組 **可控矩陣/資料** 做實驗，並用固定的 **驗證清單** 檢查性質（正交性、重建誤差、殘差、條件數影響等）。

## 使用方式（建議流程）

1. 先挑一章，選 1–2 個「矩陣設定」跑起來（小維度、可手算）。
2. 再把同一題改成「數值壞案例」（病態/近奇異/高相關/加噪），觀察誤差怎麼放大。
3. 每次都至少驗：`‖Ax-b‖`、`‖A-A_hat‖_F`、`QᵀQ≈I`、`(Aᵀr≈0)` 等（依章節選）。
4. 浮點比較請用容忍度（`atol`），不要直接比 `==`。

---

## 01 向量與矩陣（ML：特徵縮放、相似度）

### 要改哪些向量/矩陣
- 將向量尺度放大/縮小（`x` vs `1000x`）與加入小雜訊（`x+ε`）。
- 用「高相關特徵」資料：例如第二欄≈第一欄的線性組合。

### 要驗哪些性質
- 內積與範數：`‖x‖₂`、`cos(x,y)=xᵀy/(‖x‖‖y‖)` 的變化（特徵縮放是否影響相似度）。
- 矩陣乘法維度與線性性：`A(x+y)=Ax+Ay`。

---

## 02 解線性方程組（ML：回歸/解參數，數值穩定）

### 要改哪些矩陣
- 一般可逆矩陣、近奇異矩陣（行/列幾乎線性相依）、病態矩陣（例如 Hilbert）。
- 在 `b` 加小噪聲：`b+δ`（觀察解 `x` 的敏感度）。

### 要驗哪些性質
- 殘差：`r=Ax-b`、`‖r‖₂`；以及（若有真解）`‖x-x_true‖`。
- 條件數效應：比較不同矩陣時「同等噪聲」造成的解偏移大小。

---

## 03 向量空間與子空間（ML：共線性、可識別性）

### 要改哪些矩陣
- 設計矩陣 `X`：刻意做 rank-deficient（重複欄、欄線性組合、one-hot 少一欄）。
- 加入/移除一個特徵欄（觀察 rank 與解集合改變）。

### 要驗哪些性質
- `rank(X)`、`nullity(X)` 與 `rank + nullity = n_features`（Rank–Nullity）。
- 以數值方式驗證零空間：找 `v` 使 `Xv≈0`，並檢查殘差大小。

---

## 04 正交性與最小平方（ML：線性回歸/投影/QR）

### 要改哪些矩陣
- Overdetermined：`m>>n` 的 `A`（回歸常見），並製造共線性（兩欄高度相關）。
- 用「同一題」對比：正常矩陣 vs 病態矩陣（觀察 normal equation 放大誤差）。

### 要驗哪些性質
- 投影：`p = A x_hat` 落在 `Col(A)`；殘差 `r=b-p` 與 `Col(A)` 正交：`Aᵀr≈0`。
- QR：`QᵀQ≈I`、`A≈QR`；以及 `x_hat` 的穩定性比較（QR/SVD 通常優於 normal equation）。
- 正則化（Ridge / TSVD）：Ridge 驗 `Aᵀr≈λx` 並觀察 `‖x‖` 變小、擾動放大倍率下降；TSVD 驗 `U_kᵀr≈0`（只在保留子空間最優）。
- 參數選擇（ML 習慣）：用 hold-out 觀察 train/val 誤差曲線，挑 `λ` 或 `k` 的最佳點，並用重複加噪觀察 bias–variance 變化。
- 參數選擇（更穩）：用 k-fold CV（平均±標準差）選 `λ/k`，並用簡單曲線（ASCII/表格）把「超參數 vs val 誤差」視覺化。
- 大規模觀點（迭代法）：用 GD/CG 解 Ridge，驗 `‖∇f(x)‖`、`‖(AᵀA+λI)x-Aᵀb‖` 下降，並比較好/壞條件數下的迭代數。
- 預條件化（加速）：用 Jacobi 預條件器 `M=diag(AᵀA+λI)` 做 PCG，比較 `||r||` 下降與迭代數（PCG 通常更快）。
- 隱式正則化（Early stopping）：不加 λ 的 GD 也能用「迭代步數 T」控制泛化，用驗證集挑最佳 T，觀察 train/val RMSE 的 U 形。
- 大型最小平方（LSQR）：只用 `A v`/`Aᵀ u` 迭代解 `min‖Ax-b‖`，驗 `‖Ax-b‖` 與 `‖Aᵀ(Ax-b)‖`，並與 `lstsq` 對照。
- 預條件化（LSQR/LSMR 類）：用右預條件 `x=D⁻¹y`（column scaling）把問題改成解 `min‖(A D⁻¹)y-b‖`，比較迭代數與 `‖Aᵀr‖`（通常會更快）。
- 正則化最小平方（Damped LSQR）：把 `min(‖Ax-b‖² + damp²‖x‖²)` 改寫成擴增系統 `[A; damp I]`，並用 `atol/btol` 等 stopping criteria 驗收斂。
- 超參數選擇（CV 選 damp）：對候選 `damp` 做 k-fold CV（mean±std），用 validation RMSE 的 U-shape 挑最佳 `damp`（等價 Ridge 的 `λ=damp²`）。
- LSMR：可視為在 normal equations 上做 MINRES（但不形成 `AᵀA`），用 `‖Aᵀr‖` 作為核心收斂指標，並與 LSQR 比較迭代數/收斂行為。
- 進階預條件化（Randomized QR）：用 oversampled sketch `SA` 做 QR 取 `R` 當右預條件器（`x=R⁻¹y`），常能大幅降低 LSMR/LSQR 的迭代數。

---

## 05 行列式（ML：可逆性、體積、log-det）

### 要改哪些矩陣
- 可逆 vs 不可逆（det=0）；加上縮放因子（某列乘 `c`）。
- 正交矩陣（det 應為 ±1）與上三角矩陣（det 為對角線乘積）。

### 要驗哪些性質
- `det(AB)=det(A)det(B)`、列交換會讓 det 變號、列倍乘會讓 det 等比例縮放。
- （若實作到）以消去法/分解計算 det 時，避免直接展開（數值更穩）。

---

## 06 特徵值與特徵向量（ML：協方差/圖拉普拉斯/收斂）

### 要改哪些矩陣
- 對稱 PSD（如 `XᵀX`）、一般非對稱矩陣（觀察特徵向量不一定正交）。
- 造「特徵值很接近」的案例（power iteration 會變慢）。

### 要驗哪些性質
- 特徵對殘差：`‖Av-λv‖`；對稱矩陣下不同特徵向量應近似正交。
- 對角化（若可）：`A≈PDP^{-1}` 的重建誤差。

---

## 07 SVD（ML：PCA、低秩近似、去噪/壓縮）

### 要改哪些矩陣/資料
- 低秩矩陣 + 加噪：`A = L + noise`（觀察前幾個奇異值是否主導）。
- PCA：資料 `X` 先中心化，改變特徵尺度與噪聲強度。

### 要驗哪些性質
- `UᵀU≈I`、`VᵀV≈I`、重建：`‖A-UΣVᵀ‖_F`。
- 低秩近似：比較 `‖A-A_k‖_F` 隨 `k` 的變化；PCA 的 explained variance ratio 是否合理（總和≈1）。
- 大型 PCA（迭代特徵分解）：用 `Cv = Xcᵀ(Xc v)/(n-1)` 做 power iteration/Lanczos，驗 `‖Cv-λv‖` 與方向相似度（cosine similarity）。
- Top-k PCA（子空間法）：用 block power / deflation / Lanczos(Ritz) 求前 k 個主成分，驗 `‖Cv_i-λ_i v_i‖` 與子空間相似度（principal angles）。

### 對應單元
- `07-svd/01-svd-and-pca/`（同時示範「不用 `np.linalg.svd`」與 NumPy 版）
- `07-svd/02-power-iteration-and-lanczos-pca/`（只求第一主成分：Power iteration / Lanczos）
- `07-svd/03-top-k-pca-block-power-deflation-lanczos/`（只求前 k 個主成分：block power / deflation / Lanczos）
- `07-svd/04-randomized-svd/`（Randomized SVD：隨機投影做大型 top-k SVD/PCA 近似）
- `07-svd/05-oja-online-pca/`（Oja’s online PCA：串流/小批次學主成分，觀察 learning rate 與 eigengap）
- `07-svd/06-randomized-svd-vs-oja-benchmark/`（同資料對照：randSVD 的 p/q vs Oja 的 learning-rate，品質/成本表格）
