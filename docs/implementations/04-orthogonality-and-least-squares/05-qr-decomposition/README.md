# 實作說明：05-qr-decomposition（04-orthogonality-and-least-squares）
## 對應原始碼
- 單元路徑：`04-orthogonality-and-least-squares/05-qr-decomposition/`
- 概念說明：`04-orthogonality-and-least-squares/05-qr-decomposition/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/05-qr-decomposition/c/qr_decomposition.c`
  - `04-orthogonality-and-least-squares/05-qr-decomposition/cpp/qr_decomposition.cpp`
  - `04-orthogonality-and-least-squares/05-qr-decomposition/csharp/QRDecomposition.cs`
  - `04-orthogonality-and-least-squares/05-qr-decomposition/java/QRDecomposition.java`
  - `04-orthogonality-and-least-squares/05-qr-decomposition/javascript/qr_decomposition.js`
  - `04-orthogonality-and-least-squares/05-qr-decomposition/python/qr_decomposition_manual.py`
  - `04-orthogonality-and-least-squares/05-qr-decomposition/python/qr_decomposition_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### C
```bash
cd 04-orthogonality-and-least-squares/05-qr-decomposition/c
gcc -std=c99 -O2 qr_decomposition.c -o qr_decomposition -lm && ./qr_decomposition
```
### Cpp
```bash
cd 04-orthogonality-and-least-squares/05-qr-decomposition/cpp
g++ -std=c++17 -O2 qr_decomposition.cpp -o qr_decomposition && ./qr_decomposition
```
### Csharp
```bash
cd 04-orthogonality-and-least-squares/05-qr-decomposition/csharp
csc QRDecomposition.cs && ./QRDecomposition.exe
```
### Java
```bash
cd 04-orthogonality-and-least-squares/05-qr-decomposition/java
javac QRDecomposition.java && java QRDecomposition
```
### Javascript
```bash
cd 04-orthogonality-and-least-squares/05-qr-decomposition/javascript
node qr_decomposition.js
```
### Python
```bash
cd 04-orthogonality-and-least-squares/05-qr-decomposition/python
python qr_decomposition_manual.py
python qr_decomposition_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 程式碼區段（節錄）
以下節錄自 `04-orthogonality-and-least-squares/05-qr-decomposition/python/qr_decomposition_manual.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


# ========================================
# 基本向量運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:  # EN: Define dot_product and its behavior.
    return sum(xi * yi for xi, yi in zip(x, y))  # EN: Return a value: return sum(xi * yi for xi, yi in zip(x, y)).


def vector_norm(x: List[float]) -> float:  # EN: Define vector_norm and its behavior.
    return math.sqrt(dot_product(x, x))  # EN: Return a value: return math.sqrt(dot_product(x, x)).


def scalar_multiply(c: float, x: List[float]) -> List[float]:  # EN: Define scalar_multiply and its behavior.
    return [c * xi for xi in x]  # EN: Return a value: return [c * xi for xi in x].


def vector_subtract(x: List[float], y: List[float]) -> List[float]:  # EN: Define vector_subtract and its behavior.
    return [xi - yi for xi, yi in zip(x, y)]  # EN: Return a value: return [xi - yi for xi, yi in zip(x, y)].


def get_column(A: List[List[float]], j: int) -> List[float]:  # EN: Define get_column and its behavior.
    """取得矩陣的第 j 行"""  # EN: Execute statement: """取得矩陣的第 j 行""".
    return [A[i][j] for i in range(len(A))]  # EN: Return a value: return [A[i][j] for i in range(len(A))].
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
