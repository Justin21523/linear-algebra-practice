# 實作說明：04-basis-and-dimension（03-vector-spaces-and-subspaces）
## 對應原始碼
- 單元路徑：`03-vector-spaces-and-subspaces/04-basis-and-dimension/`
- 概念說明：`03-vector-spaces-and-subspaces/04-basis-and-dimension/README.md`
- 程式實作：
  - `03-vector-spaces-and-subspaces/04-basis-and-dimension/python/basis_dimension.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Python
```bash
cd 03-vector-spaces-and-subspaces/04-basis-and-dimension/python
python3 basis_dimension.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### 基底（Basis）與維度（Dimension）

- **基底**：一組向量，(1) 能生成整個空間（span），(2) 彼此線性獨立。
- **維度**：任何一組基底向量的個數（所有基底的大小都相同）。

### 典型程式化流程

1. 將向量（或矩陣的 columns）組成矩陣 `A`。
2. 用 RREF 找主元欄 → 主元欄對應的向量形成一組基底。
3. 基底數量 = `rank(A)`；若要找零空間維度，則用 `nullity(A) = n - rank(A)`。

### 驗算建議

- 用基底向量重建原向量：確認每個原向量都能寫成基底的線性組合（可用解方程/least squares 驗證）。
- 用 `rank + nullity = n` 做一致性檢查。

## 程式碼區段（節錄）
以下節錄自 `03-vector-spaces-and-subspaces/04-basis-and-dimension/python/basis_dimension.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def find_basis(vectors: List[np.ndarray]) -> Tuple[List[np.ndarray], List[int]]:  # EN: Define find_basis and its behavior.
    """
    從向量組中找出一組基底（最大獨立子集）

    Returns:
        (基底向量列表, 基底向量的原索引)
    """  # EN: Execute statement: """.
    if len(vectors) == 0:  # EN: Branch on a condition: if len(vectors) == 0:.
        return [], []  # EN: Return a value: return [], [].

    A = np.column_stack(vectors)  # EN: Assign A from expression: np.column_stack(vectors).
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.

    # RREF 找主元行
    A_work = A.astype(float).copy()  # EN: Assign A_work from expression: A.astype(float).copy().
    pivot_cols = []  # EN: Assign pivot_cols from expression: [].
    row = 0  # EN: Assign row from expression: 0.

    for col in range(n):  # EN: Iterate with a for-loop: for col in range(n):.
        if row >= m:  # EN: Branch on a condition: if row >= m:.
            break  # EN: Control flow statement: break.
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
