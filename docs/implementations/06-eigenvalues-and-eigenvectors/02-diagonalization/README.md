# 實作說明：02-diagonalization（06-eigenvalues-and-eigenvectors）
## 對應原始碼
- 單元路徑：`06-eigenvalues-and-eigenvectors/02-diagonalization/`
- 概念說明：`06-eigenvalues-and-eigenvectors/02-diagonalization/README.md`
- 程式實作：
  - `06-eigenvalues-and-eigenvectors/02-diagonalization/cpp/diagonalization_eigen.cpp`
  - `06-eigenvalues-and-eigenvectors/02-diagonalization/cpp/diagonalization_manual.cpp`
  - `06-eigenvalues-and-eigenvectors/02-diagonalization/python/diagonalization_manual.py`
  - `06-eigenvalues-and-eigenvectors/02-diagonalization/python/diagonalization_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Cpp
```bash
cd 06-eigenvalues-and-eigenvectors/02-diagonalization/cpp
g++ -std=c++17 -O2 diagonalization_eigen.cpp -o diagonalization_eigen && ./diagonalization_eigen
g++ -std=c++17 -O2 diagonalization_manual.cpp -o diagonalization_manual && ./diagonalization_manual
```
### Python
```bash
cd 06-eigenvalues-and-eigenvectors/02-diagonalization/python
python diagonalization_manual.py
python diagonalization_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 程式碼區段（節錄）
以下節錄自 `06-eigenvalues-and-eigenvectors/02-diagonalization/python/diagonalization_manual.py`（僅保留關鍵段落）：

```python
def print_matrix(name: str, matrix: list[list[float]]) -> None:  # EN: Define print_matrix and its behavior.
    """印出矩陣 (Print matrix)"""  # EN: Execute statement: """印出矩陣 (Print matrix)""".
    print(f"{name} =")  # EN: Print formatted output to the console.
    for row in matrix:  # EN: Iterate with a for-loop: for row in matrix:.
        print("  [", "  ".join(f"{x:8.4f}" for x in row), "]")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.


def print_vector(name: str, vec: list[float]) -> None:  # EN: Define print_vector and its behavior.
    """印出向量 (Print vector)"""  # EN: Execute statement: """印出向量 (Print vector)""".
    print(f"{name} = [{', '.join(f'{x:.4f}' for x in vec)}]")  # EN: Print formatted output to the console.


def matrix_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:  # EN: Define matrix_multiply and its behavior.
    """
    矩陣乘法 (Matrix multiplication)
    計算 A * B，其中 A 是 m×n，B 是 n×p，結果是 m×p
    """  # EN: Execute statement: """.
    m = len(A)  # EN: Assign m from expression: len(A).
    n = len(A[0])  # EN: Assign n from expression: len(A[0]).
    p = len(B[0])  # EN: Assign p from expression: len(B[0]).

    # 初始化結果矩陣 (Initialize result matrix)
    result = [[0.0] * p for _ in range(m)]  # EN: Assign result from expression: [[0.0] * p for _ in range(m)].

    for i in range(m):  # EN: Iterate with a for-loop: for i in range(m):.
        for j in range(p):  # EN: Iterate with a for-loop: for j in range(p):.
            for k in range(n):  # EN: Iterate with a for-loop: for k in range(n):.
                result[i][j] += A[i][k] * B[k][j]  # EN: Execute statement: result[i][j] += A[i][k] * B[k][j].
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
