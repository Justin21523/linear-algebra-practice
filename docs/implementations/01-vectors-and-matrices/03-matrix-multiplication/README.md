# 實作說明：03-matrix-multiplication（01-vectors-and-matrices）
## 對應原始碼
- 單元路徑：`01-vectors-and-matrices/03-matrix-multiplication/`
- 概念說明：`01-vectors-and-matrices/03-matrix-multiplication/README.md`
- 程式實作：
  - `01-vectors-and-matrices/03-matrix-multiplication/python/matrix_multiplication_manual.py`
  - `01-vectors-and-matrices/03-matrix-multiplication/python/matrix_multiplication_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Python
```bash
cd 01-vectors-and-matrices/03-matrix-multiplication/python
python3 matrix_multiplication_manual.py
python3 matrix_multiplication_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### 乘法定義（最常用版本）

- 若 `A` 是 `m×n`、`B` 是 `n×p`，則 `C = AB` 的形狀是 `m×p`。
- 元素公式：`C[i][j] = Σ_k A[i][k] * B[k][j]`（把 A 的第 i 列與 B 的第 j 行做內積）。

### Manual 版本實作流程

1. **維度相容性檢查**：`cols(A) == rows(B)`，不成立直接拒絕。
2. **三層迴圈**：外層 `i`（列）、中層 `j`（行）、內層 `k`（加總索引）。
3. **中間量輸出**（教學用）：印出每次加總的項目，能把「抽象公式」對應到具體計算。

### NumPy 版本重點

- 矩陣乘法用 `A @ B` 或 `np.matmul(A, B)`，避免誤用逐元素乘法 `A * B`。
- 若 `A` 或 `B` 是 1D 向量，`@` 的行為會牽涉 broadcasting；本 repo 建議以 2D 矩陣（shape `(m,n)`）為主。

### 驗算建議（不只看數字）

- 檢查形狀：`(m×n)·(n×p) -> (m×p)`。
- 檢查特例：`A·I = A`、`I·A = A`（I 為單位矩陣）。
- 檢查非交換性：通常 `AB != BA`（用小矩陣即可示範）。

## 程式碼區段（節錄）
以下節錄自 `01-vectors-and-matrices/03-matrix-multiplication/python/matrix_multiplication_manual.py`（僅保留關鍵段落）：

```python
def get_shape(A: Matrix) -> tuple:  # EN: Define get_shape and its behavior.
    """取得矩陣大小 (Get matrix shape)"""  # EN: Execute statement: """取得矩陣大小 (Get matrix shape)""".
    rows = len(A)  # EN: Assign rows from expression: len(A).
    cols = len(A[0]) if rows > 0 else 0  # EN: Assign cols from expression: len(A[0]) if rows > 0 else 0.
    return (rows, cols)  # EN: Return a value: return (rows, cols).


def print_matrix(name: str, A: Matrix) -> None:  # EN: Define print_matrix and its behavior.
    """印出矩陣 (Print matrix)"""  # EN: Execute statement: """印出矩陣 (Print matrix)""".
    rows, cols = get_shape(A)  # EN: Execute statement: rows, cols = get_shape(A).
    print(f"{name} ({rows}×{cols}):")  # EN: Print formatted output to the console.
    for row in A:  # EN: Iterate with a for-loop: for row in A:.
        print("  [", end="")  # EN: Print formatted output to the console.
        print("  ".join(f"{x:8.4f}" for x in row), end="")  # EN: Print formatted output to the console.
        print(" ]")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.


def print_vector(name: str, v: Vector) -> None:  # EN: Define print_vector and its behavior.
    """印出向量 (Print vector)"""  # EN: Execute statement: """印出向量 (Print vector)""".
    formatted = ", ".join(f"{x:.4f}" for x in v)  # EN: Assign formatted from expression: ", ".join(f"{x:.4f}" for x in v).
    print(f"{name} = [{formatted}]")  # EN: Print formatted output to the console.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線 (Print separator)"""  # EN: Execute statement: """印出分隔線 (Print separator)""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
