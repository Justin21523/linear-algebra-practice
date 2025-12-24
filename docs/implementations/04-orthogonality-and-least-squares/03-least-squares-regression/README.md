# 實作說明：03-least-squares-regression（04-orthogonality-and-least-squares）
## 對應原始碼
- 單元路徑：`04-orthogonality-and-least-squares/03-least-squares-regression/`
- 概念說明：`04-orthogonality-and-least-squares/03-least-squares-regression/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/03-least-squares-regression/c/least_squares.c`
  - `04-orthogonality-and-least-squares/03-least-squares-regression/cpp/least_squares.cpp`
  - `04-orthogonality-and-least-squares/03-least-squares-regression/csharp/LeastSquares.cs`
  - `04-orthogonality-and-least-squares/03-least-squares-regression/java/LeastSquares.java`
  - `04-orthogonality-and-least-squares/03-least-squares-regression/javascript/least_squares.js`
  - `04-orthogonality-and-least-squares/03-least-squares-regression/python/least_squares_manual.py`
  - `04-orthogonality-and-least-squares/03-least-squares-regression/python/least_squares_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### C
```bash
cd 04-orthogonality-and-least-squares/03-least-squares-regression/c
gcc -std=c99 -O2 least_squares.c -o least_squares -lm && ./least_squares
```
### Cpp
```bash
cd 04-orthogonality-and-least-squares/03-least-squares-regression/cpp
g++ -std=c++17 -O2 least_squares.cpp -o least_squares && ./least_squares
```
### Csharp
```bash
cd 04-orthogonality-and-least-squares/03-least-squares-regression/csharp
csc LeastSquares.cs && ./LeastSquares.exe
```
### Java
```bash
cd 04-orthogonality-and-least-squares/03-least-squares-regression/java
javac LeastSquares.java && java LeastSquares
```
### Javascript
```bash
cd 04-orthogonality-and-least-squares/03-least-squares-regression/javascript
node least_squares.js
```
### Python
```bash
cd 04-orthogonality-and-least-squares/03-least-squares-regression/python
python least_squares_manual.py
python least_squares_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 程式碼區段（節錄）
以下節錄自 `04-orthogonality-and-least-squares/03-least-squares-regression/python/least_squares_manual.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


# ========================================
# 基本運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:  # EN: Define dot_product and its behavior.
    """內積"""  # EN: Execute statement: """內積""".
    return sum(xi * yi for xi, yi in zip(x, y))  # EN: Return a value: return sum(xi * yi for xi, yi in zip(x, y)).


def vector_norm(x: List[float]) -> float:  # EN: Define vector_norm and its behavior.
    """向量長度"""  # EN: Execute statement: """向量長度""".
    return math.sqrt(dot_product(x, x))  # EN: Return a value: return math.sqrt(dot_product(x, x)).


def matrix_transpose(A: List[List[float]]) -> List[List[float]]:  # EN: Define matrix_transpose and its behavior.
    """矩陣轉置"""  # EN: Execute statement: """矩陣轉置""".
    m, n = len(A), len(A[0])  # EN: Execute statement: m, n = len(A), len(A[0]).
    return [[A[i][j] for i in range(m)] for j in range(n)]  # EN: Return a value: return [[A[i][j] for i in range(m)] for j in range(n)].


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:  # EN: Define matrix_multiply and its behavior.
    """矩陣乘法"""  # EN: Execute statement: """矩陣乘法""".
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
