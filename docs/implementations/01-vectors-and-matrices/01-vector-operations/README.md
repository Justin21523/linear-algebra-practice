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
python3 vector_operations_manual.py
python3 vector_operations_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### 概念與公式對照

- 向量加/減：逐元素運算，前提是兩向量維度相同。
- 純量乘法：`c·v` 代表把每個分量都乘上 `c`。
- 向量長度（L2 norm）：`‖v‖ = sqrt(sum(v_i^2))`。
- 內積：`u·v = sum(u_i v_i)`，也等於 `‖u‖‖v‖cos(θ)`（用來算夾角）。
- 投影（projection）：把 `v` 投影到 `u` 的方向：`proj_u(v) = ((u·v)/(u·u)) u`。

### Manual 版本實作要點（`*_manual.py`）

1. **維度檢查**：加減法、內積都先檢查 `len(u) == len(v)`；不一致直接丟例外，避免「默默算錯」。
2. **零向量處理**：正規化 `v/‖v‖`、夾角公式 `(u·v)/(‖u‖‖v‖)` 都會除以範數；遇到零向量要明確拒絕。
3. **數值穩定**：夾角使用 `acos` 前，先把 `cosθ` 夾在 `[-1, 1]`，避免浮點誤差造成 `acos` domain error。
4. **投影分母**：`u·u` 為 0 代表 `u` 是零向量，投影沒有定義，需特別處理。

### NumPy 版本實作要點（`*_numpy.py`）

- 用 `np.array` 表示向量，搭配 `np.dot`、`np.linalg.norm` 完成同樣運算；優點是程式更短、也更接近線代記號。
- 留意形狀（shape）：本 repo 多用 1D 向量（shape `(n,)`），與「列向量/行向量」概念不同；若要做矩陣運算可改用 `(n,1)`。

### 如何快速驗算

- `dot(v, v)` 應等於 `‖v‖^2`。
- `normalize(v)` 的範數應接近 1（可允許 `1e-10` 級別誤差）。
- `angle_between(v, v)` 應接近 0；`angle_between(v, -v)` 應接近 `π`。

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
