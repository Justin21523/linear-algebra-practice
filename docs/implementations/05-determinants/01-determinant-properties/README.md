# 實作說明：01-determinant-properties（05-determinants）
## 對應原始碼
- 單元路徑：`05-determinants/01-determinant-properties/`
- 概念說明：`05-determinants/01-determinant-properties/README.md`
- 程式實作：
  - `05-determinants/01-determinant-properties/c/determinant.c`
  - `05-determinants/01-determinant-properties/cpp/determinant.cpp`
  - `05-determinants/01-determinant-properties/csharp/Determinant.cs`
  - `05-determinants/01-determinant-properties/java/Determinant.java`
  - `05-determinants/01-determinant-properties/javascript/determinant.js`
  - `05-determinants/01-determinant-properties/python/determinant_manual.py`
  - `05-determinants/01-determinant-properties/python/determinant_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### C
```bash
cd 05-determinants/01-determinant-properties/c
gcc -std=c99 -O2 determinant.c -o determinant -lm && ./determinant
```
### Cpp
```bash
cd 05-determinants/01-determinant-properties/cpp
g++ -std=c++17 -O2 determinant.cpp -o determinant && ./determinant
```
### Csharp
```bash
cd 05-determinants/01-determinant-properties/csharp
csc Determinant.cs && ./Determinant.exe
```
### Java
```bash
cd 05-determinants/01-determinant-properties/java
javac Determinant.java && java Determinant
```
### Javascript
```bash
cd 05-determinants/01-determinant-properties/javascript
node determinant.js
```
### Python
```bash
cd 05-determinants/01-determinant-properties/python
python determinant_manual.py
python determinant_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 程式碼區段（節錄）
以下節錄自 `05-determinants/01-determinant-properties/python/determinant_manual.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def print_matrix(name: str, M: List[List[float]]) -> None:  # EN: Define print_matrix and its behavior.
    print(f"{name} =")  # EN: Print formatted output to the console.
    for row in M:  # EN: Iterate with a for-loop: for row in M:.
        formatted = [f"{x:8.4f}" for x in row]  # EN: Assign formatted from expression: [f"{x:8.4f}" for x in row].
        print(f"  [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


# ========================================
# 行列式計算
# ========================================

def det_2x2(A: List[List[float]]) -> float:  # EN: Define det_2x2 and its behavior.
    """計算 2×2 行列式"""  # EN: Execute statement: """計算 2×2 行列式""".
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]  # EN: Return a value: return A[0][0] * A[1][1] - A[0][1] * A[1][0].


def det_3x3(A: List[List[float]]) -> float:  # EN: Define det_3x3 and its behavior.
    """計算 3×3 行列式（Sarrus 法則或展開）"""  # EN: Execute statement: """計算 3×3 行列式（Sarrus 法則或展開）""".
    a, b, c = A[0]  # EN: Execute statement: a, b, c = A[0].
    d, e, f = A[1]  # EN: Execute statement: d, e, f = A[1].
    g, h, i = A[2]  # EN: Execute statement: g, h, i = A[2].

    return (a * e * i + b * f * g + c * d * h  # EN: Return a value: return (a * e * i + b * f * g + c * d * h.
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
