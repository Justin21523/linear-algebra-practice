# 實作說明：02-matrix-operations（01-vectors-and-matrices）
## 對應原始碼
- 單元路徑：`01-vectors-and-matrices/02-matrix-operations/`
- 概念說明：`01-vectors-and-matrices/02-matrix-operations/README.md`
- 程式實作：
  - `01-vectors-and-matrices/02-matrix-operations/python/matrix_operations_manual.py`
  - `01-vectors-and-matrices/02-matrix-operations/python/matrix_operations_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Python
```bash
cd 01-vectors-and-matrices/02-matrix-operations/python
python3 matrix_operations_manual.py
python3 matrix_operations_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### 矩陣資料結構（Manual）

- 以 `List[List[float]]` 表示矩陣 `A`：外層是 row，內層是 column。
- `get_shape(A)` 回傳 `(rows, cols)`，是所有維度相容性檢查的基礎（特別是加減法與轉置）。

### 基本運算與檢查點

- 加法/減法：同型矩陣逐元素相加/相減；若 `rows/cols` 不一致應直接報錯。
- 純量乘法：逐元素乘上常數；可用 list comprehension 寫得很直觀。
- 轉置：把 `A[i][j]` 變成 `A_T[j][i]`；轉置後形狀從 `(m×n)` 變成 `(n×m)`。

### NumPy 版本重點

- 用 `np.array` 表示矩陣，形狀由 `A.shape` 決定，運算通常更直接：
  - 轉置：`A.T`
  - 逐元素加減/乘法：`A + B`、`A - B`、`c * A`
- NumPy 的維度錯誤通常會直接丟例外（或 broadcasting），因此建議仍保留「顯式檢查」的觀念。

### 常見錯誤

- **不規則矩陣（ragged array）**：每列長度不同會讓運算失去意義；manual 版本可在輸入時檢查。
- **把逐元素乘法當成矩陣乘法**：`A * B` 在 NumPy 是逐元素；矩陣乘法要用 `A @ B`（本單元先不處理矩陣乘法）。

## 程式碼區段（節錄）
以下節錄自 `01-vectors-and-matrices/02-matrix-operations/python/matrix_operations_manual.py`（僅保留關鍵段落）：

```python
def create_matrix(rows: int, cols: int, fill: float = 0.0) -> Matrix:  # EN: Define create_matrix and its behavior.
    """
    建立指定大小的矩陣 (Create matrix of specified size)
    """  # EN: Execute statement: """.
    return [[fill for _ in range(cols)] for _ in range(rows)]  # EN: Return a value: return [[fill for _ in range(cols)] for _ in range(rows)].


def get_shape(A: Matrix) -> tuple:  # EN: Define get_shape and its behavior.
    """
    取得矩陣大小 (Get matrix shape)
    回傳 (rows, cols)
    """  # EN: Execute statement: """.
    rows = len(A)  # EN: Assign rows from expression: len(A).
    cols = len(A[0]) if rows > 0 else 0  # EN: Assign cols from expression: len(A[0]) if rows > 0 else 0.
    return (rows, cols)  # EN: Return a value: return (rows, cols).


def print_matrix(name: str, A: Matrix) -> None:  # EN: Define print_matrix and its behavior.
    """
    印出矩陣 (Print matrix with nice formatting)
    """  # EN: Execute statement: """.
    rows, cols = get_shape(A)  # EN: Execute statement: rows, cols = get_shape(A).
    print(f"{name} ({rows}×{cols}):")  # EN: Print formatted output to the console.

    for row in A:  # EN: Iterate with a for-loop: for row in A:.
        print("  [", end="")  # EN: Print formatted output to the console.
        print("  ".join(f"{x:8.4f}" for x in row), end="")  # EN: Print formatted output to the console.
        print(" ]")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
