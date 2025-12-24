# 實作說明：01-vector-space-definition（03-vector-spaces-and-subspaces）
## 對應原始碼
- 單元路徑：`03-vector-spaces-and-subspaces/01-vector-space-definition/`
- 概念說明：`03-vector-spaces-and-subspaces/01-vector-space-definition/README.md`
- 程式實作：
  - `03-vector-spaces-and-subspaces/01-vector-space-definition/python/vector_space_demo.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Python
```bash
cd 03-vector-spaces-and-subspaces/01-vector-space-definition/python
python3 vector_space_demo.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### 這個單元在做什麼？

向量空間（vector space）不是「一組向量」而已，而是一個集合 `V` 搭配兩個運算（向量加法、純量乘法），並且滿足一系列公理（axioms）。本單元用程式把這些公理逐條驗證，讓你把抽象定義對應到可計算的行為。

### 實作重點

- 以 `ℝ^2`（用 NumPy 向量）作為正例：`u+v` 仍在 `ℝ^2`、`c·u` 仍在 `ℝ^2`，並檢查交換律、結合律、分配律、零向量、加法逆元等。
- 以「不是向量空間」的集合作反例（例如限制某些分量必為正、或不含零向量的集合），用簡單例子展示「哪一條公理被破壞」。
- 浮點比較建議用 `np.allclose`（或設定容忍度），避免因為浮點誤差造成「應該相等卻判成不等」。

### 常見觀念提醒

- **子空間（subspace）判斷**通常用三點：含零向量、對加法封閉、對純量乘法封閉。
- 只要其中一條不成立，就不是子空間（例如「不含 0」會直接失敗）。

## 程式碼區段（節錄）
以下節錄自 `03-vector-spaces-and-subspaces/01-vector-space-definition/python/vector_space_demo.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def verify_vector_space_axioms() -> None:  # EN: Define verify_vector_space_axioms and its behavior.
    """
    驗證 ℝ² 滿足向量空間的公理
    Verify that ℝ² satisfies vector space axioms
    """  # EN: Execute statement: """.
    print_separator("1. 驗證向量空間公理（以 ℝ² 為例）")  # EN: Call print_separator(...) to perform an operation.

    u = np.array([1.0, 2.0])  # EN: Assign u from expression: np.array([1.0, 2.0]).
    v = np.array([3.0, 4.0])  # EN: Assign v from expression: np.array([3.0, 4.0]).
    w = np.array([5.0, 6.0])  # EN: Assign w from expression: np.array([5.0, 6.0]).
    c, d = 2.0, 3.0  # EN: Execute statement: c, d = 2.0, 3.0.

    print(f"u = {u}, v = {v}, w = {w}")  # EN: Print formatted output to the console.
    print(f"c = {c}, d = {d}")  # EN: Print formatted output to the console.

    print("\n【加法公理】")  # EN: Print formatted output to the console.

    # A1: 封閉性
    print(f"A1 封閉性: u + v = {u + v} ∈ ℝ² ✓")  # EN: Print formatted output to the console.

    # A2: 交換律
    print(f"A2 交換律: u + v = {u + v}, v + u = {v + u}")  # EN: Print formatted output to the console.
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
