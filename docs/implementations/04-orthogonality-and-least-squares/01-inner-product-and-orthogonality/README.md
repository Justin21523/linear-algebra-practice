# 實作說明：01-inner-product-and-orthogonality（04-orthogonality-and-least-squares）
## 對應原始碼
- 單元路徑：`04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/`
- 概念說明：`04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/README.md`
- 程式實作：
  - `04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/c/inner_product.c`
  - `04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/cpp/inner_product.cpp`
  - `04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/csharp/InnerProduct.cs`
  - `04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/java/InnerProduct.java`
  - `04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/javascript/inner_product.js`
  - `04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/python/inner_product_manual.py`
  - `04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/python/inner_product_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### C
```bash
cd 04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/c
gcc -std=c99 -O2 inner_product.c -o inner_product -lm && ./inner_product
```
### Cpp
```bash
cd 04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/cpp
g++ -std=c++17 -O2 inner_product.cpp -o inner_product && ./inner_product
```
### Csharp
```bash
cd 04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/csharp
csc InnerProduct.cs && ./InnerProduct.exe
```
### Java
```bash
cd 04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/java
javac InnerProduct.java && java InnerProduct
```
### Javascript
```bash
cd 04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/javascript
node inner_product.js
```
### Python
```bash
cd 04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/python
python inner_product_manual.py
python inner_product_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 程式碼區段（節錄）
以下節錄自 `04-orthogonality-and-least-squares/01-inner-product-and-orthogonality/python/inner_product_manual.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


# ========================================
# 基本向量運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:  # EN: Define dot_product and its behavior.
    """
    計算兩向量的內積 (Dot Product)

    x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ

    時間複雜度：O(n)
    """  # EN: Execute statement: """.
    if len(x) != len(y):  # EN: Branch on a condition: if len(x) != len(y):.
        raise ValueError("向量維度必須相同")  # EN: Raise an exception: raise ValueError("向量維度必須相同").

    result = 0.0  # EN: Assign result from expression: 0.0.
    for i in range(len(x)):  # EN: Iterate with a for-loop: for i in range(len(x)):.
        result += x[i] * y[i]  # EN: Update result via += using: x[i] * y[i].
    return result  # EN: Return a value: return result.


def vector_norm(x: List[float]) -> float:  # EN: Define vector_norm and its behavior.
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
