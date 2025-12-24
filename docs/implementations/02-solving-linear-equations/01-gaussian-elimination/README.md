# 實作說明：01-gaussian-elimination（02-solving-linear-equations）
## 對應原始碼
- 單元路徑：`02-solving-linear-equations/01-gaussian-elimination/`
- 概念說明：`02-solving-linear-equations/01-gaussian-elimination/README.md`
- 程式實作：
  - `02-solving-linear-equations/01-gaussian-elimination/python/gaussian_elimination_manual.py`
  - `02-solving-linear-equations/01-gaussian-elimination/python/gaussian_elimination_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Python
```bash
cd 02-solving-linear-equations/01-gaussian-elimination/python
python3 gaussian_elimination_manual.py
python3 gaussian_elimination_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### 問題形式

- 線性方程組：`Ax = b`。
- 常見做法是把它寫成增廣矩陣：`[A | b]`，並用列運算把 `A` 變成上三角或階梯形。

### 高斯消去法（Gaussian Elimination）流程

1. **選主元（pivot）**：在第 `k` 欄選一個「不為 0 且夠大」的元素當主元，避免除以 0 / 數值不穩定。
2. **消去（elimination）**：對每個 `i > k`，用倍率 `m = A[i][k] / A[k][k]` 做列運算：`row_i -= m * row_k`，把主元下方消成 0。
3. **回代（back substitution）**：得到上三角 `U` 後，從最後一列開始解出 `x_n`，一路回推到 `x_1`。

### Manual vs NumPy

- Manual 版本會把每一步列運算展開，適合理解「列運算等價於方程組變形」。
- NumPy 版本通常用陣列操作/線代 API 來驗證結果（例如用 `np.linalg.solve` 對照）。

### 常見錯誤與檢查點

- **主元為 0**：必須換列（partial pivoting），否則會除以 0。
- **近似奇異（ill-conditioned）**：主元非常小會放大誤差；建議觀察 pivot 大小或條件數。
- **解的型態**：若在消去後出現 `0 = 非0`，代表無解；若出現整列 0，代表可能有無限多解（需要自由變數）。

## 程式碼區段（節錄）
以下節錄自 `02-solving-linear-equations/01-gaussian-elimination/python/gaussian_elimination_manual.py`（僅保留關鍵段落）：

```python
def print_matrix(name: str, A: Matrix, augmented: bool = False) -> None:  # EN: Define print_matrix and its behavior.
    """
    印出矩陣 (Print matrix)
    augmented=True 時，最後一行顯示為增廣部分
    """  # EN: Execute statement: """.
    rows = len(A)  # EN: Assign rows from expression: len(A).
    cols = len(A[0]) if rows > 0 else 0  # EN: Assign cols from expression: len(A[0]) if rows > 0 else 0.

    print(f"{name}:")  # EN: Print formatted output to the console.
    for i, row in enumerate(A):  # EN: Iterate with a for-loop: for i, row in enumerate(A):.
        print("  [", end="")  # EN: Print formatted output to the console.
        for j, val in enumerate(row):  # EN: Iterate with a for-loop: for j, val in enumerate(row):.
            if augmented and j == cols - 1:  # EN: Branch on a condition: if augmented and j == cols - 1:.
                print(" |", end="")  # EN: Print formatted output to the console.
            print(f"{val:8.4f}", end="")  # EN: Print formatted output to the console.
        print(" ]")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線 (Print separator)"""  # EN: Execute statement: """印出分隔線 (Print separator)""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def create_augmented_matrix(A: Matrix, b: Vector) -> Matrix:  # EN: Define create_augmented_matrix and its behavior.
    """
    建立增廣矩陣 [A | b]
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
