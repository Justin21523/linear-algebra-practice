# 實作說明：03-linear-independence（03-vector-spaces-and-subspaces）
## 對應原始碼
- 單元路徑：`03-vector-spaces-and-subspaces/03-linear-independence/`
- 概念說明：`03-vector-spaces-and-subspaces/03-linear-independence/README.md`
- 程式實作：
  - `03-vector-spaces-and-subspaces/03-linear-independence/python/linear_independence.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Python
```bash
cd 03-vector-spaces-and-subspaces/03-linear-independence/python
python linear_independence.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 程式碼區段（節錄）
以下節錄自 `03-vector-spaces-and-subspaces/03-linear-independence/python/linear_independence.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def is_linearly_independent(vectors: List[np.ndarray]) -> bool:  # EN: Define is_linearly_independent and its behavior.
    """
    判斷向量組是否線性獨立

    方法：將向量排成矩陣的行，檢查 rank 是否等於行數
    """  # EN: Execute statement: """.
    if len(vectors) == 0:  # EN: Branch on a condition: if len(vectors) == 0:.
        return True  # EN: Return a value: return True.

    A = np.column_stack(vectors)  # EN: Assign A from expression: np.column_stack(vectors).
    rank = np.linalg.matrix_rank(A)  # EN: Assign rank from expression: np.linalg.matrix_rank(A).

    return rank == len(vectors)  # EN: Return a value: return rank == len(vectors).


def find_dependency_relation(vectors: List[np.ndarray]) -> Optional[np.ndarray]:  # EN: Define find_dependency_relation and its behavior.
    """
    找出線性相依關係

    若相依，返回係數 c 使得 c₁v₁ + c₂v₂ + ... = 0
    若獨立，返回 None
    """  # EN: Execute statement: """.
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
