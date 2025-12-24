# 實作說明：02-cofactor-expansion（05-determinants）
## 對應原始碼
- 單元路徑：`05-determinants/02-cofactor-expansion/`
- 概念說明：`05-determinants/02-cofactor-expansion/README.md`
- 程式實作：
  - `05-determinants/02-cofactor-expansion/c/cofactor.c`
  - `05-determinants/02-cofactor-expansion/cpp/cofactor.cpp`
  - `05-determinants/02-cofactor-expansion/csharp/Cofactor.cs`
  - `05-determinants/02-cofactor-expansion/java/Cofactor.java`
  - `05-determinants/02-cofactor-expansion/javascript/cofactor.js`
  - `05-determinants/02-cofactor-expansion/python/cofactor_manual.py`
  - `05-determinants/02-cofactor-expansion/python/cofactor_numpy.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### C
```bash
cd 05-determinants/02-cofactor-expansion/c
gcc -std=c99 -O2 cofactor.c -o cofactor -lm && ./cofactor
```
### Cpp
```bash
cd 05-determinants/02-cofactor-expansion/cpp
g++ -std=c++17 -O2 cofactor.cpp -o cofactor && ./cofactor
```
### Csharp
```bash
cd 05-determinants/02-cofactor-expansion/csharp
csc Cofactor.cs && ./Cofactor.exe
```
### Java
```bash
cd 05-determinants/02-cofactor-expansion/java
javac Cofactor.java && java Cofactor
```
### Javascript
```bash
cd 05-determinants/02-cofactor-expansion/javascript
node cofactor.js
```
### Python
```bash
cd 05-determinants/02-cofactor-expansion/python
python3 cofactor_manual.py
python3 cofactor_numpy.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### Cofactor（代數餘子式）展開

對 `n×n` 矩陣 `A`，沿著第 `i` 列展開：

`det(A) = Σ_j a_{ij} C_{ij}`  
其中 `C_{ij} = (-1)^{i+j} det(M_{ij})`，`M_{ij}` 是刪掉第 i 列第 j 行後的 minor。

### 實作重點

- **遞迴（recursion）**：`det(n×n)` 會呼叫 `det((n-1)×(n-1))`，直到 2×2 或 1×1 為 base case。
- **時間複雜度很高**：cofactor 展開接近 `O(n!)`，適合教學與小矩陣，不適合大矩陣。
- **選擇 0 多的列/行展開**：如果某列很多 0，可以大幅減少計算量（本 repo 以概念清楚為主）。

### 驗算建議

- 與 `np.linalg.det` 對照（允許浮點誤差）。
- 對同一矩陣沿不同列/行展開，結果應一致。

## 程式碼區段（節錄）
以下節錄自 `05-determinants/02-cofactor-expansion/python/cofactor_manual.py`（僅保留關鍵段落）：

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


# ========================================
# 子行列式與餘因子
# ========================================

def get_minor_matrix(A: List[List[float]], row: int, col: int) -> List[List[float]]:  # EN: Define get_minor_matrix and its behavior.
    """取得去掉第 row 列、第 col 行後的子矩陣"""  # EN: Execute statement: """取得去掉第 row 列、第 col 行後的子矩陣""".
    n = len(A)  # EN: Assign n from expression: len(A).
    return [[A[i][j] for j in range(n) if j != col]  # EN: Return a value: return [[A[i][j] for j in range(n) if j != col].
            for i in range(n) if i != row]  # EN: Iterate with a for-loop: for i in range(n) if i != row].


def det_2x2(A: List[List[float]]) -> float:  # EN: Define det_2x2 and its behavior.
    """2×2 行列式"""  # EN: Execute statement: """2×2 行列式""".
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]  # EN: Return a value: return A[0][0] * A[1][1] - A[0][1] * A[1][0].
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
