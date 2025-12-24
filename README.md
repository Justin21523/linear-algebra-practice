# tw-gradcs-linear-algebra

> 台灣資工所線性代數學習資源 | Linear Algebra for Taiwan CS Graduate School Exam

本專案以 **程式實作** 的方式學習線性代數，對應 Gilbert Strang《Introduction to Linear Algebra》的核心章節，涵蓋台灣各大學資工所入學考試的常考範圍，並延伸至機器學習與資料科學的應用。

## 為什麼線性代數這麼重要？

線性代數是資訊科學的基礎數學，在許多領域扮演核心角色：

| 領域 | 線代應用 |
|------|----------|
| **機器學習 (Machine Learning)** | 最小平方法、SVD、PCA 降維、神經網路權重矩陣 |
| **電腦圖學 (Computer Graphics)** | 座標變換、投影矩陣、旋轉矩陣 |
| **信號處理 (Signal Processing)** | 傅立葉轉換、濾波器設計 |
| **自然語言處理 (NLP)** | 詞向量 (Word Embeddings)、LSA |
| **推薦系統 (Recommender Systems)** | 矩陣分解、協同過濾 |
| **量子計算 (Quantum Computing)** | 量子態以向量表示、量子閘以矩陣表示 |

## 學習理念

- **不以算題為主**：著重程式實作，透過程式碼體會向量空間的幾何意義
- **多語言支援**：Python、Java、JavaScript、C、C++、C# 六種語言
- **雙語註解**：繁體中文為主、英文術語為輔
- **小維度範例**：以 2D、3D 例子搭配視覺化，讓抽象概念具體化

## 章節架構

```
01-vectors-and-matrices/          向量與矩陣 (Vectors and Matrices)
02-solving-linear-equations/      解線性方程組 (Solving Linear Equations)
03-vector-spaces-and-subspaces/   向量空間與子空間 (Vector Spaces and Subspaces)
04-orthogonality-and-least-squares/ 正交性與最小平方 (Orthogonality and Least Squares)
05-determinants/                  行列式 (Determinants)
06-eigenvalues-and-eigenvectors/  特徵值與特徵向量 (Eigenvalues and Eigenvectors)

# 規劃中（尚未加入 repo）
07-svd/                           奇異值分解 (Singular Value Decomposition)
08-linear-transformations/        線性變換 (Linear Transformations)
09-optimization-and-data/         最佳化與資料應用 (Optimization and Data) [進階]
```

## 文件導覽（必看）

- 單元級的「實作說明（繁體中文）」統一放在 `docs/implementations/<章節>/<單元>/README.md`
- 撰寫/維護規則請看 `docs/README.md`

## 每單元結構

```
/XX-topic-name/
├── README.md           # 觀念說明（繁中 + 英文術語）
├── python/             # Python 實作
│   ├── xxx_manual.py   # 手刻版本
│   └── xxx_numpy.py    # 使用 NumPy
├── cpp/                # C++ 實作
│   ├── xxx_manual.cpp  # 手刻版本
│   └── xxx_eigen.cpp   # 使用 Eigen library
├── java/
├── javascript/
├── c/
└── csharp/
```

## 台灣資工所考試重點對應

| 考試主題 | 對應章節 |
|----------|----------|
| 高斯消去法、LU 分解 | `02-solving-linear-equations` |
| 向量空間、基底、維度 | `03-vector-spaces-and-subspaces` |
| 四大基本子空間 | `03-vector-spaces-and-subspaces/05-four-fundamental-subspaces` |
| 正交投影、Gram-Schmidt | `04-orthogonality-and-least-squares` |
| 最小平方法 | `04-orthogonality-and-least-squares/03-least-squares-regression` |
| 行列式計算與性質 | `05-determinants` |
| 特徵值、特徵向量、對角化 | `06-eigenvalues-and-eigenvectors` |
| 奇異值分解 (SVD) | （規劃中）`07-svd` |
| 線性變換、相似矩陣 | （規劃中）`08-linear-transformations` |

## 使用方式

### Python 執行環境與安裝

- Python 版本：建議 `Python 3.12+`（至少 `Python 3.10+`，因為工具腳本使用 `|` 型別語法）
- 建議使用虛擬環境（venv）：
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
- 注意：`*_numpy.py` 需要 `numpy`；`*_manual.py` 通常不需要額外套件。

### Python（範例）
```bash
cd 06-eigenvalues-and-eigenvectors/02-diagonalization/python
python3 diagonalization_numpy.py
```

### C++（使用 Eigen）
```bash
cd 06-eigenvalues-and-eigenvectors/02-diagonalization/cpp
g++ -I /path/to/eigen diagonalization_eigen.cpp -o diagonalization
./diagonalization
```

## 參考資源

- Gilbert Strang, *Introduction to Linear Algebra*, 5th Edition
- MIT OpenCourseWare: [18.06 Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
- 3Blue1Brown: [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

## 授權

MIT License

---

*本專案旨在幫助台灣學生準備資工所考試，同時建立紮實的線性代數基礎以銜接後續的機器學習與資料科學學習。*
