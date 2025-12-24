# 矩陣乘法 (Matrix Multiplication)

> 線性代數最核心的運算——用四種觀點理解矩陣乘法

## 為什麼矩陣乘法這麼重要？

矩陣乘法是：
- **線性變換的組合** (Composition of linear transformations)
- **解線性方程組的語言** (Language for linear systems)
- **神經網路的基礎運算** (Foundation of neural networks)
- **圖形變換的核心** (Core of graphics transformations)

---

## 1. 矩陣乘法的定義 (Definition)

### 維度相容性

若 A 是 **m×n** 矩陣，B 是 **n×p** 矩陣，則：
- **AB 存在**，且是 **m×p** 矩陣
- A 的**行數**必須等於 B 的**列數**

```
A      ×      B      =      C
(m×n)       (n×p)         (m×p)
     └──┬──┘
     必須相等
```

### 元素計算公式

```
Cᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ = Aᵢ₁B₁ⱼ + Aᵢ₂B₂ⱼ + ... + AᵢₙBₙⱼ
```

C 的第 i 列第 j 行元素 = A 的第 i 列與 B 的第 j 行的**內積**

### 簡單範例

```
[1  2]   [5  6]   [1×5+2×7  1×6+2×8]   [19  22]
[3  4] × [7  8] = [3×5+4×7  3×6+4×8] = [43  50]
```

---

## 2. 四種理解矩陣乘法的觀點 (Four Ways to See Matrix Multiplication)

Gilbert Strang 強調要從**多種角度**理解矩陣乘法。

### 觀點一：內積 (Dot Product View)

**Cᵢⱼ = (A 的第 i 列) · (B 的第 j 行)**

```
C₁₁ = [1  2] · [5]  = 1×5 + 2×7 = 19
              [7]
```

這是最直接的計算方式。

### 觀點二：行的線性組合 (Column View)

**C 的每一行 = A 的各行的線性組合**

```
AB = A × [b₁ | b₂ | ... | bₚ]
   = [Ab₁ | Ab₂ | ... | Abₚ]
```

C 的第 j 行 = A × (B 的第 j 行)

```
                [5]
C 的第 1 行 = A × [7] = 5×[1] + 7×[2] = [19]
                       [3]    [4]   [43]
```

### 觀點三：列的線性組合 (Row View)

**C 的每一列 = B 的各列的線性組合**

```
      [─ a₁ ─]       [─ a₁B ─]
AB =  [─ a₂ ─] × B = [─ a₂B ─]
      [  ...  ]      [  ...   ]
```

C 的第 i 列 = (A 的第 i 列) × B

### 觀點四：外積的和 (Sum of Outer Products)

**AB = Σ (A 的第 k 行) × (B 的第 k 列)**

```
AB = a₁ × b₁ᵀ + a₂ × b₂ᵀ + ... + aₙ × bₙᵀ
```

其中 aₖ × bₖᵀ 是一個 m×p 的**外積**矩陣（秩為 1）。

```
[1]           [1×5  1×6]   [5   6]
[3] × [5  6] = [3×5  3×6] = [15  18]

[2]           [2×7  2×8]   [14  16]
[4] × [7  8] = [4×7  4×8] = [28  32]

AB = [5   6]  + [14  16] = [19  22]
     [15  18]   [28  32]   [43  50]
```

---

## 3. 矩陣與向量的乘法 (Matrix-Vector Multiplication)

### Ax 的兩種解讀

設 A 是 m×n 矩陣，x 是 n 維向量：

#### 解讀一：行的線性組合

```
Ax = x₁a₁ + x₂a₂ + ... + xₙaₙ
```

結果是 A 的**行向量**的線性組合。

```
[1  2] [3]     [1]     [2]   [7]
[3  4] [2] = 3×[3] + 2×[4] = [17]
```

#### 解讀二：列的內積

```
      [─ r₁ ─]       [r₁·x]
Ax =  [─ r₂ ─] × x = [r₂·x]
      [  ...  ]      [ ... ]
```

每個元素是 A 的一列與 x 的內積。

---

## 4. 矩陣乘法的性質 (Properties)

### ✓ 結合律 (Associative Law)

```
(AB)C = A(BC)
```

### ✓ 分配律 (Distributive Law)

```
A(B + C) = AB + AC
(A + B)C = AC + BC
```

### ✗ 交換律不成立 (NOT Commutative)

```
AB ≠ BA  （一般情況）
```

甚至 AB 存在時，BA 可能不存在！

```
A: 2×3，B: 3×4
AB: 2×4 ✓
BA: 不存在 ✗（4 ≠ 2）
```

### 轉置性質

```
(AB)ᵀ = BᵀAᵀ    ← 順序反轉！
```

---

## 5. 矩陣冪次 (Matrix Powers)

對於**方陣** A：

```
A² = A × A
A³ = A × A × A
Aⁿ = A × A × ... × A（n 次）
```

### 約定

```
A⁰ = I（單位矩陣）
A¹ = A
```

### 對角化的威力

若 A = PDP⁻¹（對角化），則：

```
Aⁿ = PDⁿP⁻¹
```

D 的 n 次方只需要對角線元素各自 n 次方，非常簡單！

---

## 6. 分塊矩陣乘法 (Block Matrix Multiplication)

矩陣可以分成小塊來計算：

```
[A₁₁  A₁₂] [B₁₁  B₁₂]   [A₁₁B₁₁+A₁₂B₂₁  A₁₁B₁₂+A₁₂B₂₂]
[A₂₁  A₂₂] [B₂₁  B₂₂] = [A₂₁B₁₁+A₂₂B₂₁  A₂₁B₁₂+A₂₂B₂₂]
```

這在處理大矩陣時非常實用。

---

## 7. 程式實作重點

### 時間複雜度

標準矩陣乘法：**O(n³)**

對於 n×n 矩陣相乘，需要約 n³ 次乘法。

### 實作效率

```python
# ❌ 慢：三層 Python 迴圈
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i][j] += A[i][k] * B[k][j]

# ✅ 快：使用 NumPy
C = A @ B  # 或 np.matmul(A, B)
```

NumPy 底層使用優化的 BLAS 函式庫。

### 本單元程式示範

| 檔案 | 內容 |
|------|------|
| `matrix_multiplication_manual.py` | 手刻實作 + 四種觀點示範 |
| `matrix_multiplication_numpy.py` | NumPy 實作 + 效能比較 |
| `matrix_multiplication.cpp` | C++ 實作 |

---

## 8. 考試重點

### 常見題型

1. **計算題**：給定矩陣求乘積
2. **維度題**：判斷 AB 是否存在、結果大小
3. **性質題**：證明 (AB)ᵀ = BᵀAᵀ
4. **觀點題**：用不同觀點解釋 Ax

### 關鍵檢查

```
A: m×n, B: n×p → AB: m×p
AB 存在不代表 BA 存在
AB = O 不代表 A = O 或 B = O
```

### 常見錯誤

- 誤以為 AB = BA
- 維度順序搞錯
- 忘記 (AB)ᵀ = BᵀAᵀ 的順序反轉

---

## 參考資料

- Strang, Chapter 1.4: Matrix Multiplication
- MIT 18.06 Lecture 3
- 3Blue1Brown: Matrix multiplication as composition
