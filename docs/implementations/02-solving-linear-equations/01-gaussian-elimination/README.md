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
python gaussian_elimination_manual.py
python gaussian_elimination_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

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
