# 實作說明：04-geometric-interpretation（05-determinants）
## 對應原始碼
- 單元路徑：`05-determinants/04-geometric-interpretation/`
- 概念說明：`05-determinants/04-geometric-interpretation/README.md`
- 程式實作：
  - `05-determinants/04-geometric-interpretation/c/geometric.c`
  - `05-determinants/04-geometric-interpretation/cpp/geometric.cpp`
  - `05-determinants/04-geometric-interpretation/csharp/Geometric.cs`
  - `05-determinants/04-geometric-interpretation/java/Geometric.java`
  - `05-determinants/04-geometric-interpretation/javascript/geometric.js`
  - `05-determinants/04-geometric-interpretation/python/geometric_manual.py`
  - `05-determinants/04-geometric-interpretation/python/geometric_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### C
```bash
cd 05-determinants/04-geometric-interpretation/c
gcc -std=c99 -O2 geometric.c -o geometric -lm && ./geometric
```
### Cpp
```bash
cd 05-determinants/04-geometric-interpretation/cpp
g++ -std=c++17 -O2 geometric.cpp -o geometric && ./geometric
```
### Csharp
```bash
cd 05-determinants/04-geometric-interpretation/csharp
csc Geometric.cs && ./Geometric.exe
```
### Java
```bash
cd 05-determinants/04-geometric-interpretation/java
javac Geometric.java && java Geometric
```
### Javascript
```bash
cd 05-determinants/04-geometric-interpretation/javascript
node geometric.js
```
### Python
```bash
cd 05-determinants/04-geometric-interpretation/python
python3 geometric_manual.py
python3 geometric_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### 幾何意義（2D/3D 最直觀）

- 2D：`det([a b])`（把 `a,b` 當成 columns）等於平行四邊形面積（帶方向）。
- 3D：`|det(A)|` 等於由三個向量張成的平行六面體體積。

### 符號（sign）代表什麼？

- `det(A) > 0`：保持方向（orientation preserved）
- `det(A) < 0`：翻轉方向（orientation reversed），例如鏡射（reflection）

### 程式上如何驗算

- 用小向量例子計算 det，並用圖形/幾何公式（如 2D 的叉積大小、3D 的 triple product）對照。
- 再用 NumPy 的 det 做第二次對照，確認實作無誤。

## 程式碼區段（節錄）
以下節錄自 `05-determinants/04-geometric-interpretation/python/geometric_manual.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def print_vector(name: str, v: List[float]) -> None:  # EN: Define print_vector and its behavior.
    formatted = [f"{x:.4f}" for x in v]  # EN: Assign formatted from expression: [f"{x:.4f}" for x in v].
    print(f"{name} = [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


def print_matrix(name: str, M: List[List[float]]) -> None:  # EN: Define print_matrix and its behavior.
    print(f"{name} =")  # EN: Print formatted output to the console.
    for row in M:  # EN: Iterate with a for-loop: for row in M:.
        formatted = [f"{x:8.4f}" for x in row]  # EN: Assign formatted from expression: [f"{x:8.4f}" for x in row].
        print(f"  [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


# ========================================
# 行列式計算
# ========================================

def det_2x2(A: List[List[float]]) -> float:  # EN: Define det_2x2 and its behavior.
    """2×2 行列式"""  # EN: Execute statement: """2×2 行列式""".
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]  # EN: Return a value: return A[0][0] * A[1][1] - A[0][1] * A[1][0].


def det_3x3(A: List[List[float]]) -> float:  # EN: Define det_3x3 and its behavior.
    """3×3 行列式"""  # EN: Execute statement: """3×3 行列式""".
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
