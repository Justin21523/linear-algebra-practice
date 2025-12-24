# 實作說明：02-column-space-and-null-space（03-vector-spaces-and-subspaces）
## 對應原始碼
- 單元路徑：`03-vector-spaces-and-subspaces/02-column-space-and-null-space/`
- 概念說明：`03-vector-spaces-and-subspaces/02-column-space-and-null-space/README.md`
- 程式實作：
  - `03-vector-spaces-and-subspaces/02-column-space-and-null-space/python/column_null_space.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Python
```bash
cd 03-vector-spaces-and-subspaces/02-column-space-and-null-space/python
python column_null_space.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 程式碼區段（節錄）
以下節錄自 `03-vector-spaces-and-subspaces/02-column-space-and-null-space/python/column_null_space.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def rref(A: np.ndarray) -> Tuple[np.ndarray, List[int]]:  # EN: Define rref and its behavior.
    """
    計算簡化列階梯形式 (Reduced Row Echelon Form)

    Returns:
        (RREF 矩陣, 主元行的索引列表)
    """  # EN: Execute statement: """.
    A = A.astype(float).copy()  # EN: Assign A from expression: A.astype(float).copy().
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.
    pivot_cols = []  # EN: Assign pivot_cols from expression: [].
    row = 0  # EN: Assign row from expression: 0.

    for col in range(n):  # EN: Iterate with a for-loop: for col in range(n):.
        if row >= m:  # EN: Branch on a condition: if row >= m:.
            break  # EN: Control flow statement: break.

        # 找主元
        max_row = row + np.argmax(np.abs(A[row:, col]))  # EN: Assign max_row from expression: row + np.argmax(np.abs(A[row:, col])).

        if np.abs(A[max_row, col]) < 1e-10:  # EN: Branch on a condition: if np.abs(A[max_row, col]) < 1e-10:.
            continue  # EN: Control flow statement: continue.
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
