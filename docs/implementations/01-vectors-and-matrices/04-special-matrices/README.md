# 實作說明：04-special-matrices（01-vectors-and-matrices）
## 對應原始碼
- 單元路徑：`01-vectors-and-matrices/04-special-matrices/`
- 概念說明：`01-vectors-and-matrices/04-special-matrices/README.md`
- 程式實作：
  - `01-vectors-and-matrices/04-special-matrices/python/special_matrices.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Python
```bash
cd 01-vectors-and-matrices/04-special-matrices/python
python special_matrices.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 程式碼區段（節錄）
以下節錄自 `01-vectors-and-matrices/04-special-matrices/python/special_matrices.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("特殊矩陣示範\nSpecial Matrices Demo")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 單位矩陣 (Identity Matrix)
    # ========================================
    print_separator("1. 單位矩陣 (Identity Matrix)")  # EN: Call print_separator(...) to perform an operation.

    I3 = np.eye(3)  # EN: Assign I3 from expression: np.eye(3).
    print(f"I₃ = np.eye(3):\n{I3}\n")  # EN: Print formatted output to the console.

    # 性質：AI = IA = A
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)  # EN: Assign A from expression: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float).
    print(f"A:\n{A}\n")  # EN: Print formatted output to the console.
    print(f"A @ I = A ? {np.allclose(A @ I3, A)}")  # EN: Print formatted output to the console.
    print(f"I @ A = A ? {np.allclose(I3 @ A, A)}")  # EN: Print formatted output to the console.

    # 性質：Ix = x
    x = np.array([1, 2, 3])  # EN: Assign x from expression: np.array([1, 2, 3]).
    print(f"\nx = {x}")  # EN: Print formatted output to the console.
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
