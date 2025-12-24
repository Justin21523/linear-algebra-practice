# 實作說明：04-computational-complexity（02-solving-linear-equations）
## 對應原始碼
- 單元路徑：`02-solving-linear-equations/04-computational-complexity/`
- 概念說明：`02-solving-linear-equations/04-computational-complexity/README.md`
- 程式實作：
  - `02-solving-linear-equations/04-computational-complexity/python/complexity_demo.py`

## 目標與背景
- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。
- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。

## 如何執行
### Python
```bash
cd 02-solving-linear-equations/04-computational-complexity/python
python3 complexity_demo.py
```

## 核心做法（重點）
- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。
- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。

## 詳細說明

### 這個單元想建立的直覺

- 很多線代演算法的時間複雜度不是「看起來」那麼直觀，尤其是矩陣相關運算常見 `O(n^3)`。
- 當 `n` 變成 2 倍時，`O(n^3)` 可能變成 8 倍時間，這就是為什麼大矩陣運算要特別在意演算法與常數因子。

### 常見複雜度（重點記）

- 矩陣乘法（最基本三層迴圈）：`O(n^3)`
- 高斯消去法：`O(n^3)`
- 前代/回代（三角矩陣解）：`O(n^2)`

### 實作上通常怎麼做

- 以不同 `n` 產生測資，量測耗時，觀察成長趨勢。
- 需要注意：
  - 暖機（warm-up）與多次取平均
  - 不同語言/不同硬體會影響「常數因子」，但不影響大 O 的趨勢

## 程式碼區段（節錄）
以下節錄自 `02-solving-linear-equations/04-computational-complexity/python/complexity_demo.py`（僅保留關鍵段落）：

```python
def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def measure_time(func: Callable, *args, repeats: int = 3) -> float:  # EN: Define measure_time and its behavior.
    """測量函數執行時間（取多次平均）"""  # EN: Execute statement: """測量函數執行時間（取多次平均）""".
    times = []  # EN: Assign times from expression: [].
    for _ in range(repeats):  # EN: Iterate with a for-loop: for _ in range(repeats):.
        start = time.time()  # EN: Assign start from expression: time.time().
        func(*args)  # EN: Call func(...) to perform an operation.
        times.append(time.time() - start)  # EN: Execute statement: times.append(time.time() - start).
    return np.mean(times)  # EN: Return a value: return np.mean(times).


def estimate_complexity(sizes: List[int], times: List[float]) -> Tuple[float, str]:  # EN: Define estimate_complexity and its behavior.
    """
    估計複雜度指數

    假設 T(n) ∝ n^k，用最小平方法估計 k
    log(T) = k × log(n) + c
    """  # EN: Execute statement: """.
    log_n = np.log(sizes)  # EN: Assign log_n from expression: np.log(sizes).
    log_t = np.log(times)  # EN: Assign log_t from expression: np.log(times).

    # 線性迴歸
    k = np.polyfit(log_n, log_t, 1)[0]  # EN: Assign k from expression: np.polyfit(log_n, log_t, 1)[0].
```

## 驗證方式
- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。
- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。
