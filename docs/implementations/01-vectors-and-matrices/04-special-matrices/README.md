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
python3 special_matrices.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### 為什麼「特殊矩陣」重要？

很多線性代數題目其實在考你是否能辨識結構：

- **單位矩陣** `I`：`AI = A`、`IA = A`，是「乘法的單位元」。
- **對角矩陣**：非對角線元素皆為 0，計算常可大幅簡化（例如反矩陣、行列式）。
- **對稱矩陣**：`A^T = A`，常見於二次型、最小平方、特徵分解。
- **三角矩陣**：上/下三角結構讓解方程（回代）更快。
- **稀疏矩陣**：大多數元素為 0，實務上會用壓縮格式（此 repo 先以概念為主）。

### 實作重點（以判斷/生成為核心）

- 生成：用迴圈或 comprehension 直接控制「哪些位置為 0/1/指定值」。
- 判斷：以 `for i,j` 掃描元素，遇到不符合條件就立即回傳 `False`（早停）。
- 浮點比較：若有浮點值，判斷「是否為 0」建議用容忍度（tolerance），避免 `1e-16` 之類的誤差。

### 驗算建議

- 對稱：檢查 `A[i][j] == A[j][i]`。
- 對角：檢查 `i != j` 時 `A[i][j] == 0`。
- 單位：檢查對角線為 1、其餘為 0。

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
