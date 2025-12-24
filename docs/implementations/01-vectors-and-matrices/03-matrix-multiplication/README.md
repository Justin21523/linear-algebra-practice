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
python matrix_multiplication_manual.py
python matrix_multiplication_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

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
