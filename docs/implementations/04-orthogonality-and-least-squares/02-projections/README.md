# 實作說明：02-projections（04-orthogonality-and-least-squares）
## 對應原始碼
- 單元路徑：`04-orthogonality-and-least-squares/02-projections/`
- 概念說明：`04-orthogonality-and-least-squares/02-projections/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/02-projections/c/projection.c`
  - `04-orthogonality-and-least-squares/02-projections/cpp/projection.cpp`
  - `04-orthogonality-and-least-squares/02-projections/csharp/Projection.cs`
  - `04-orthogonality-and-least-squares/02-projections/java/Projection.java`
  - `04-orthogonality-and-least-squares/02-projections/javascript/projection.js`
  - `04-orthogonality-and-least-squares/02-projections/python/projection_manual.py`
  - `04-orthogonality-and-least-squares/02-projections/python/projection_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### C
```bash
cd 04-orthogonality-and-least-squares/02-projections/c
gcc -std=c99 -O2 projection.c -o projection -lm && ./projection
```
### Cpp
```bash
cd 04-orthogonality-and-least-squares/02-projections/cpp
g++ -std=c++17 -O2 projection.cpp -o projection && ./projection
```
### Csharp
```bash
cd 04-orthogonality-and-least-squares/02-projections/csharp
csc Projection.cs && ./Projection.exe
```
### Java
```bash
cd 04-orthogonality-and-least-squares/02-projections/java
javac Projection.java && java Projection
```
### Javascript
```bash
cd 04-orthogonality-and-least-squares/02-projections/javascript
node projection.js
```
### Python
```bash
cd 04-orthogonality-and-least-squares/02-projections/python
python projection_manual.py
python projection_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 程式碼區段（節錄）
以下節錄自 `04-orthogonality-and-least-squares/02-projections/python/projection_manual.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


# ========================================
# 基本向量和矩陣運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:  # EN: Define dot_product and its behavior.
    """內積"""  # EN: Execute statement: """內積""".
    return sum(xi * yi for xi, yi in zip(x, y))  # EN: Return a value: return sum(xi * yi for xi, yi in zip(x, y)).


def vector_norm(x: List[float]) -> float:  # EN: Define vector_norm and its behavior.
    """向量長度"""  # EN: Execute statement: """向量長度""".
    return math.sqrt(dot_product(x, x))  # EN: Return a value: return math.sqrt(dot_product(x, x)).


def scalar_multiply(c: float, x: List[float]) -> List[float]:  # EN: Define scalar_multiply and its behavior.
    """純量乘向量"""  # EN: Execute statement: """純量乘向量""".
    return [c * xi for xi in x]  # EN: Return a value: return [c * xi for xi in x].


def vector_add(x: List[float], y: List[float]) -> List[float]:  # EN: Define vector_add and its behavior.
    """向量加法"""  # EN: Execute statement: """向量加法""".
    return [xi + yi for xi, yi in zip(x, y)]  # EN: Return a value: return [xi + yi for xi, yi in zip(x, y)].
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
