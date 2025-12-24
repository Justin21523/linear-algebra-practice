# 實作說明：05-four-fundamental-subspaces（03-vector-spaces-and-subspaces）
## 對應原始碼
- 單元路徑：`03-vector-spaces-and-subspaces/05-four-fundamental-subspaces/`
- 概念說明：`03-vector-spaces-and-subspaces/05-four-fundamental-subspaces/README.md`
- 程式實作：
  - `03-vector-spaces-and-subspaces/05-four-fundamental-subspaces/python/four_subspaces.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Python
```bash
cd 03-vector-spaces-and-subspaces/05-four-fundamental-subspaces/python
python3 four_subspaces.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### Strang 的「四大基本子空間」

對一個 `m×n` 矩陣 `A`，有四個最重要的子空間：

- `C(A)`：Column space（在 `ℝ^m`）
- `N(A)`：Null space（在 `ℝ^n`）
- `C(A^T)`：Row space（在 `ℝ^n`）
- `N(A^T)`：Left null space（在 `ℝ^m`）

它們之間最關鍵的關係是「正交補」：

- `N(A)` 與 Row space 互為正交補
- `N(A^T)` 與 Column space 互為正交補

### 維度關係（一定要會）

若 `rank(A) = r`：

- `dim C(A) = r`
- `dim N(A) = n - r`
- `dim C(A^T) = r`
- `dim N(A^T) = m - r`

### 程式實作如何落地

- 先用 RREF 找主元欄與 rank。
- 由主元欄取得 `C(A)` 的基底；由 RREF 的非零列取得 Row space 的基底。
- 用「自由變數」方法建立 `N(A)` 的基底；`N(A^T)` 則可對 `A^T` 重複同樣步驟。

### 驗算建議

- 對 `x ∈ N(A)`，檢查 `A @ x ≈ 0`。
- 對 `y ∈ N(A^T)`，檢查 `A.T @ y ≈ 0`。
- 用內積檢查正交性（例如 Row space 的任一向量應與 `N(A)` 的基底向量近似正交）。

## 程式碼區段（節錄）
以下節錄自 `03-vector-spaces-and-subspaces/05-four-fundamental-subspaces/python/four_subspaces.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def rref_with_pivots(A: np.ndarray):  # EN: Define rref_with_pivots and its behavior.
    """計算 RREF 和主元行索引"""  # EN: Execute statement: """計算 RREF 和主元行索引""".
    A = A.astype(float).copy()  # EN: Assign A from expression: A.astype(float).copy().
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.
    pivot_cols = []  # EN: Assign pivot_cols from expression: [].
    row = 0  # EN: Assign row from expression: 0.

    for col in range(n):  # EN: Iterate with a for-loop: for col in range(n):.
        if row >= m:  # EN: Branch on a condition: if row >= m:.
            break  # EN: Control flow statement: break.

        max_row = row + np.argmax(np.abs(A[row:, col]))  # EN: Assign max_row from expression: row + np.argmax(np.abs(A[row:, col])).
        if np.abs(A[max_row, col]) < 1e-10:  # EN: Branch on a condition: if np.abs(A[max_row, col]) < 1e-10:.
            continue  # EN: Control flow statement: continue.

        A[[row, max_row]] = A[[max_row, row]]  # EN: Execute statement: A[[row, max_row]] = A[[max_row, row]].
        A[row] = A[row] / A[row, col]  # EN: Execute statement: A[row] = A[row] / A[row, col].

        for i in range(m):  # EN: Iterate with a for-loop: for i in range(m):.
            if i != row:  # EN: Branch on a condition: if i != row:.
                A[i] = A[i] - A[i, col] * A[row]  # EN: Execute statement: A[i] = A[i] - A[i, col] * A[row].
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
