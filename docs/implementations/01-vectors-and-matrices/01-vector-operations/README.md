# 實作說明：01-vector-operations（01-vectors-and-matrices）
## 對應原始碼
- 單元路徑：`01-vectors-and-matrices/01-vector-operations/`
- 概念說明：`01-vectors-and-matrices/01-vector-operations/README.md`
- 程式實作：
  - `01-vectors-and-matrices/01-vector-operations/cpp/vector_operations.cpp`
  - `01-vectors-and-matrices/01-vector-operations/python/vector_operations_manual.py`
  - `01-vectors-and-matrices/01-vector-operations/python/vector_operations_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Cpp
```bash
cd 01-vectors-and-matrices/01-vector-operations/cpp
g++ -std=c++17 -O2 vector_operations.cpp -o vector_operations && ./vector_operations
```
### Python
```bash
cd 01-vectors-and-matrices/01-vector-operations/python
python vector_operations_manual.py
python vector_operations_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 程式碼區段（節錄）
以下節錄自 `01-vectors-and-matrices/01-vector-operations/python/vector_operations_manual.py`（僅保留關鍵段落）：

```python
def vector_add(u: Vector, v: Vector) -> Vector:  # EN: Define vector_add and its behavior.
    """
    向量加法 (Vector addition)

    u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]
    """  # EN: Execute statement: """.
    if len(u) != len(v):  # EN: Branch on a condition: if len(u) != len(v):.
        raise ValueError("向量維度必須相同 (Vectors must have same dimension)")  # EN: Raise an exception: raise ValueError("向量維度必須相同 (Vectors must have same dimension)").

    return [u[i] + v[i] for i in range(len(u))]  # EN: Return a value: return [u[i] + v[i] for i in range(len(u))].


def vector_subtract(u: Vector, v: Vector) -> Vector:  # EN: Define vector_subtract and its behavior.
    """
    向量減法 (Vector subtraction)

    u - v = [u₁-v₁, u₂-v₂, ..., uₙ-vₙ]
    """  # EN: Execute statement: """.
    if len(u) != len(v):  # EN: Branch on a condition: if len(u) != len(v):.
        raise ValueError("向量維度必須相同 (Vectors must have same dimension)")  # EN: Raise an exception: raise ValueError("向量維度必須相同 (Vectors must have same dimension)").

    return [u[i] - v[i] for i in range(len(u))]  # EN: Return a value: return [u[i] - v[i] for i in range(len(u))].


def scalar_multiply(c: float, v: Vector) -> Vector:  # EN: Define scalar_multiply and its behavior.
    """
    純量乘法 (Scalar multiplication)

    c·v = [c·v₁, c·v₂, ..., c·vₙ]
    """  # EN: Execute statement: """.
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
