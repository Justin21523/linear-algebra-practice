# 實作說明：02-lu-decomposition（02-solving-linear-equations）
## 對應原始碼
- 單元路徑：`02-solving-linear-equations/02-lu-decomposition/`
- 概念說明：`02-solving-linear-equations/02-lu-decomposition/README.md`
- 程式實作：
  - `02-solving-linear-equations/02-lu-decomposition/python/lu_decomposition_manual.py`
  - `02-solving-linear-equations/02-lu-decomposition/python/lu_decomposition_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Python
```bash
cd 02-solving-linear-equations/02-lu-decomposition/python
python3 lu_decomposition_manual.py
python3 lu_decomposition_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### LU 分解在做什麼？

- 目標：把 `A` 拆成 `A = LU`
  - `L`：下三角矩陣（通常對角線為 1）
  - `U`：上三角矩陣
- 好處：當你要解很多個右手邊 `b`（`Ax=b1, Ax=b2, ...`），只要分解一次 `A`，後面每次都只需做兩次三角解：
  - `Ly = b`（前代 forward substitution）
  - `Ux = y`（回代 back substitution）

### 實作流程（概念對照）

1. **消去係數進 L**：高斯消去法的倍率 `m = A[i][k]/A[k][k]`，其實就是 `L[i][k]` 的元素。
2. **消去結果留在 U**：做完消去後，上三角部分就是 `U`。
3. **（需要時）Pivoting**：若主元為 0 或太小，需換列；此時會引入置換矩陣 `P`，變成 `PA = LU`。

### 常見錯誤與檢查點

- **A 不可逆/主元為 0**：分解會失敗或產生除以 0。
- **驗算**：用 `L @ U` 應能重建 `A`（允許浮點誤差）；再用 `Ly=b`、`Ux=y` 的結果去比對原方程 `Ax=b`。

## 程式碼區段（節錄）
以下節錄自 `02-solving-linear-equations/02-lu-decomposition/python/lu_decomposition_manual.py`（僅保留關鍵段落）：

```python
def print_matrix(name: str, A: Matrix) -> None:  # EN: Define print_matrix and its behavior.
    """印出矩陣"""  # EN: Execute statement: """印出矩陣""".
    rows = len(A)  # EN: Assign rows from expression: len(A).
    cols = len(A[0]) if rows > 0 else 0  # EN: Assign cols from expression: len(A[0]) if rows > 0 else 0.
    print(f"{name} ({rows}×{cols}):")  # EN: Print formatted output to the console.
    for row in A:  # EN: Iterate with a for-loop: for row in A:.
        print("  [", end="")  # EN: Print formatted output to the console.
        print("  ".join(f"{x:8.4f}" for x in row), end="")  # EN: Print formatted output to the console.
        print(" ]")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def create_identity(n: int) -> Matrix:  # EN: Define create_identity and its behavior.
    """建立 n×n 單位矩陣"""  # EN: Execute statement: """建立 n×n 單位矩陣""".
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]  # EN: Return a value: return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)].


def create_zero_matrix(n: int) -> Matrix:  # EN: Define create_zero_matrix and its behavior.
    """建立 n×n 零矩陣"""  # EN: Execute statement: """建立 n×n 零矩陣""".
    return [[0.0 for _ in range(n)] for _ in range(n)]  # EN: Return a value: return [[0.0 for _ in range(n)] for _ in range(n)].
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
