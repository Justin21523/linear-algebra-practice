# 實作說明：03-cramers-rule（05-determinants）
## 對應原始碼
- 單元路徑：`05-determinants/03-cramers-rule/`
- 概念說明：`05-determinants/03-cramers-rule/README.md`
- 程式實作：
  - `05-determinants/03-cramers-rule/c/cramers_rule.c`
  - `05-determinants/03-cramers-rule/cpp/cramers_rule.cpp`
  - `05-determinants/03-cramers-rule/csharp/CramersRule.cs`
  - `05-determinants/03-cramers-rule/java/CramersRule.java`
  - `05-determinants/03-cramers-rule/javascript/cramers_rule.js`
  - `05-determinants/03-cramers-rule/python/cramers_rule_manual.py`
  - `05-determinants/03-cramers-rule/python/cramers_rule_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### C
```bash
cd 05-determinants/03-cramers-rule/c
gcc -std=c99 -O2 cramers_rule.c -o cramers_rule -lm && ./cramers_rule
```
### Cpp
```bash
cd 05-determinants/03-cramers-rule/cpp
g++ -std=c++17 -O2 cramers_rule.cpp -o cramers_rule && ./cramers_rule
```
### Csharp
```bash
cd 05-determinants/03-cramers-rule/csharp
csc CramersRule.cs && ./CramersRule.exe
```
### Java
```bash
cd 05-determinants/03-cramers-rule/java
javac CramersRule.java && java CramersRule
```
### Javascript
```bash
cd 05-determinants/03-cramers-rule/javascript
node cramers_rule.js
```
### Python
```bash
cd 05-determinants/03-cramers-rule/python
python3 cramers_rule_manual.py
python3 cramers_rule_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### Cramer’s Rule（克萊姆法則）

解 `Ax=b`（A 可逆）時：

- 先算 `det(A)`，必須 `det(A) != 0`
- 對每個未知數 `x_i`：
  - 把 `A` 的第 `i` 欄換成 `b` 得到 `A_i`
  - `x_i = det(A_i) / det(A)`

### 實作與限制

- 需要計算 `n+1` 次行列式（`A` 一次、每個 `A_i` 一次）。
- 若用 cofactor 展開求 det，複雜度非常高，只適合小矩陣（教學用途）。
- 實務上解方程更推薦 `solve`/消去法/LU/QR。

### 驗算建議

- 用求出的 `x` 檢查 `Ax ≈ b`。
- 與 `np.linalg.solve(A, b)` 對照（在 `det(A)` 不太小時應接近）。

## 程式碼區段（節錄）
以下節錄自 `05-determinants/03-cramers-rule/python/cramers_rule_manual.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def print_matrix(name: str, M: List[List[float]]) -> None:  # EN: Define print_matrix and its behavior.
    print(f"{name} =")  # EN: Print formatted output to the console.
    for row in M:  # EN: Iterate with a for-loop: for row in M:.
        formatted = [f"{x:8.4f}" for x in row]  # EN: Assign formatted from expression: [f"{x:8.4f}" for x in row].
        print(f"  [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


def print_vector(name: str, v: List[float]) -> None:  # EN: Define print_vector and its behavior.
    formatted = [f"{x:.4f}" for x in v]  # EN: Assign formatted from expression: [f"{x:.4f}" for x in v].
    print(f"{name} = [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


# ========================================
# 行列式計算
# ========================================

def det_2x2(A: List[List[float]]) -> float:  # EN: Define det_2x2 and its behavior.
    """2×2 行列式"""  # EN: Execute statement: """2×2 行列式""".
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]  # EN: Return a value: return A[0][0] * A[1][1] - A[0][1] * A[1][0].


def det_3x3(A: List[List[float]]) -> float:  # EN: Define det_3x3 and its behavior.
    """3×3 行列式"""  # EN: Execute statement: """3×3 行列式""".
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
