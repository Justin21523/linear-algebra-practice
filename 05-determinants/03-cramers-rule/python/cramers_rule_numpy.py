"""
克萊姆法則 - NumPy 版本 (Cramer's Rule - NumPy Implementation)

本程式示範：
1. NumPy 實作克萊姆法則
2. 與 np.linalg.solve 比較
3. 效率和數值穩定性測試
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.
import time  # EN: Import module(s): import time.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def cramers_rule(A: np.ndarray, b: np.ndarray) -> np.ndarray:  # EN: Define cramers_rule and its behavior.
    """用克萊姆法則解 Ax = b"""  # EN: Execute statement: """用克萊姆法則解 Ax = b""".
    n = A.shape[0]  # EN: Assign n from expression: A.shape[0].
    det_A = np.linalg.det(A)  # EN: Assign det_A from expression: np.linalg.det(A).

    if abs(det_A) < 1e-10:  # EN: Branch on a condition: if abs(det_A) < 1e-10:.
        raise ValueError("矩陣奇異")  # EN: Raise an exception: raise ValueError("矩陣奇異").

    x = np.zeros(n)  # EN: Assign x from expression: np.zeros(n).
    for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
        Aj = A.copy()  # EN: Assign Aj from expression: A.copy().
        Aj[:, j] = b  # EN: Execute statement: Aj[:, j] = b.
        x[j] = np.linalg.det(Aj) / det_A  # EN: Execute statement: x[j] = np.linalg.det(Aj) / det_A.

    return x  # EN: Return a value: return x.


def cramers_rule_single(A: np.ndarray, b: np.ndarray, j: int) -> float:  # EN: Define cramers_rule_single and its behavior.
    """只求第 j 個未知數"""  # EN: Execute statement: """只求第 j 個未知數""".
    det_A = np.linalg.det(A)  # EN: Assign det_A from expression: np.linalg.det(A).
    Aj = A.copy()  # EN: Assign Aj from expression: A.copy().
    Aj[:, j] = b  # EN: Execute statement: Aj[:, j] = b.
    return np.linalg.det(Aj) / det_A  # EN: Return a value: return np.linalg.det(Aj) / det_A.


def main():  # EN: Define main and its behavior.
    print_separator("克萊姆法則示範（NumPy 版）\nCramer's Rule Demo (NumPy)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 基本使用
    # ========================================
    print_separator("1. 基本使用")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [2, 1, -1],  # EN: Execute statement: [2, 1, -1],.
        [-3, -1, 2],  # EN: Execute statement: [-3, -1, 2],.
        [-2, 1, 2]  # EN: Execute statement: [-2, 1, 2].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    b = np.array([8, -11, -3], dtype=float)  # EN: Assign b from expression: np.array([8, -11, -3], dtype=float).

    print("方程組 Ax = b：")  # EN: Print formatted output to the console.
    print(f"A:\n{A}")  # EN: Print formatted output to the console.
    print(f"b: {b}")  # EN: Print formatted output to the console.

    x_cramer = cramers_rule(A, b)  # EN: Assign x_cramer from expression: cramers_rule(A, b).
    x_solve = np.linalg.solve(A, b)  # EN: Assign x_solve from expression: np.linalg.solve(A, b).

    print(f"\n克萊姆法則解：{x_cramer}")  # EN: Print formatted output to the console.
    print(f"np.linalg.solve：{x_solve}")  # EN: Print formatted output to the console.

    # 驗證
    print(f"\n驗證 Ax：{A @ x_cramer}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 詳細過程展示
    # ========================================
    print_separator("2. 詳細計算過程")  # EN: Call print_separator(...) to perform an operation.

    print(f"det(A) = {np.linalg.det(A):.4f}")  # EN: Print formatted output to the console.

    for j in range(3):  # EN: Iterate with a for-loop: for j in range(3):.
        Aj = A.copy()  # EN: Assign Aj from expression: A.copy().
        Aj[:, j] = b  # EN: Execute statement: Aj[:, j] = b.
        det_Aj = np.linalg.det(Aj)  # EN: Assign det_Aj from expression: np.linalg.det(Aj).
        xj = det_Aj / np.linalg.det(A)  # EN: Assign xj from expression: det_Aj / np.linalg.det(A).

        print(f"\nA{j+1}（第 {j+1} 行換成 b）:")  # EN: Print formatted output to the console.
        print(Aj)  # EN: Print formatted output to the console.
        print(f"det(A{j+1}) = {det_Aj:.4f}")  # EN: Print formatted output to the console.
        print(f"x{j+1} = {xj:.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 只求一個未知數
    # ========================================
    print_separator("3. 只求特定未知數")  # EN: Call print_separator(...) to perform an operation.

    print("假設只需要 x₂：")  # EN: Print formatted output to the console.
    x2 = cramers_rule_single(A, b, 1)  # EN: Assign x2 from expression: cramers_rule_single(A, b, 1).
    print(f"x₂ = {x2:.4f}")  # EN: Print formatted output to the console.
    print(f"（完整解中的 x₂ = {x_cramer[1]:.4f}）")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 與其他方法比較
    # ========================================
    print_separator("4. 效率比較")  # EN: Call print_separator(...) to perform an operation.

    sizes = [3, 5, 7, 9]  # EN: Assign sizes from expression: [3, 5, 7, 9].

    print("矩陣大小 | 克萊姆法則 | np.linalg.solve | 加速比")  # EN: Print formatted output to the console.
    print("-" * 55)  # EN: Print formatted output to the console.

    for n in sizes:  # EN: Iterate with a for-loop: for n in sizes:.
        np.random.seed(42)  # EN: Execute statement: np.random.seed(42).
        A_test = np.random.randn(n, n)  # EN: Assign A_test from expression: np.random.randn(n, n).
        # 確保非奇異
        A_test = A_test @ A_test.T + np.eye(n)  # EN: Assign A_test from expression: A_test @ A_test.T + np.eye(n).
        b_test = np.random.randn(n)  # EN: Assign b_test from expression: np.random.randn(n).

        # 克萊姆法則
        start = time.perf_counter()  # EN: Assign start from expression: time.perf_counter().
        for _ in range(100):  # EN: Iterate with a for-loop: for _ in range(100):.
            x_cramer = cramers_rule(A_test, b_test)  # EN: Assign x_cramer from expression: cramers_rule(A_test, b_test).
        t_cramer = (time.perf_counter() - start) / 100  # EN: Assign t_cramer from expression: (time.perf_counter() - start) / 100.

        # np.linalg.solve
        start = time.perf_counter()  # EN: Assign start from expression: time.perf_counter().
        for _ in range(100):  # EN: Iterate with a for-loop: for _ in range(100):.
            x_solve = np.linalg.solve(A_test, b_test)  # EN: Assign x_solve from expression: np.linalg.solve(A_test, b_test).
        t_solve = (time.perf_counter() - start) / 100  # EN: Assign t_solve from expression: (time.perf_counter() - start) / 100.

        speedup = t_cramer / t_solve  # EN: Assign speedup from expression: t_cramer / t_solve.
        print(f"{n:8d} | {t_cramer*1000:10.4f}ms | {t_solve*1000:14.4f}ms | {speedup:7.1f}x")  # EN: Print formatted output to the console.

    print("\n結論：克萊姆法則在大矩陣時效率極低")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 數值穩定性
    # ========================================
    print_separator("5. 數值穩定性測試")  # EN: Call print_separator(...) to perform an operation.

    # 構造病態矩陣
    n = 5  # EN: Assign n from expression: 5.
    epsilon = 1e-10  # EN: Assign epsilon from expression: 1e-10.

    A_bad = np.eye(n)  # EN: Assign A_bad from expression: np.eye(n).
    A_bad[0, n-1] = epsilon  # EN: Execute statement: A_bad[0, n-1] = epsilon.
    A_bad[n-1, 0] = epsilon  # EN: Execute statement: A_bad[n-1, 0] = epsilon.

    b_bad = np.ones(n)  # EN: Assign b_bad from expression: np.ones(n).

    print(f"病態矩陣（條件數）：{np.linalg.cond(A_bad):.2e}")  # EN: Print formatted output to the console.

    try:  # EN: Start a try block for exception handling.
        x_cramer = cramers_rule(A_bad, b_bad)  # EN: Assign x_cramer from expression: cramers_rule(A_bad, b_bad).
        x_solve = np.linalg.solve(A_bad, b_bad)  # EN: Assign x_solve from expression: np.linalg.solve(A_bad, b_bad).

        error_cramer = np.linalg.norm(A_bad @ x_cramer - b_bad)  # EN: Assign error_cramer from expression: np.linalg.norm(A_bad @ x_cramer - b_bad).
        error_solve = np.linalg.norm(A_bad @ x_solve - b_bad)  # EN: Assign error_solve from expression: np.linalg.norm(A_bad @ x_solve - b_bad).

        print(f"\n殘差 ‖Ax - b‖：")  # EN: Print formatted output to the console.
        print(f"  克萊姆法則：{error_cramer:.2e}")  # EN: Print formatted output to the console.
        print(f"  np.linalg.solve：{error_solve:.2e}")  # EN: Print formatted output to the console.
    except Exception as e:  # EN: Handle an exception case: except Exception as e:.
        print(f"錯誤：{e}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 應用：線性插值係數
    # ========================================
    print_separator("6. 應用：求多項式係數")  # EN: Call print_separator(...) to perform an operation.

    print("問題：找通過 (0,1), (1,3), (2,7) 的二次多項式 y = a + bx + cx²")  # EN: Print formatted output to the console.

    # Vandermonde 矩陣
    x_points = np.array([0, 1, 2], dtype=float)  # EN: Assign x_points from expression: np.array([0, 1, 2], dtype=float).
    y_points = np.array([1, 3, 7], dtype=float)  # EN: Assign y_points from expression: np.array([1, 3, 7], dtype=float).

    V = np.column_stack([np.ones(3), x_points, x_points**2])  # EN: Assign V from expression: np.column_stack([np.ones(3), x_points, x_points**2]).

    print(f"\nVandermonde 矩陣:\n{V}")  # EN: Print formatted output to the console.
    print(f"y 值：{y_points}")  # EN: Print formatted output to the console.

    coeffs = cramers_rule(V, y_points)  # EN: Assign coeffs from expression: cramers_rule(V, y_points).
    print(f"\n係數 [a, b, c] = {coeffs}")  # EN: Print formatted output to the console.
    print(f"多項式：y = {coeffs[0]:.1f} + {coeffs[1]:.1f}x + {coeffs[2]:.1f}x²")  # EN: Print formatted output to the console.

    # 驗證
    print("\n驗證：")  # EN: Print formatted output to the console.
    for xi, yi in zip(x_points, y_points):  # EN: Iterate with a for-loop: for xi, yi in zip(x_points, y_points):.
        y_calc = coeffs[0] + coeffs[1] * xi + coeffs[2] * xi**2  # EN: Assign y_calc from expression: coeffs[0] + coeffs[1] * xi + coeffs[2] * xi**2.
        print(f"  x={xi:.0f}: y = {y_calc:.1f} (預期 {yi:.1f})")  # EN: Print formatted output to the console.

    # 總結
    print_separator("NumPy 克萊姆法則總結")  # EN: Call print_separator(...) to perform an operation.
    print("""
實作方式：
  det_A = np.linalg.det(A)
  for j in range(n):
      Aj = A.copy()
      Aj[:, j] = b
      x[j] = np.linalg.det(Aj) / det_A

適用情況：
  - 小型系統（n ≤ 4）
  - 只需求特定未知數
  - 符號計算/教學

不適用情況：
  - 大型系統（效率低）
  - 病態矩陣（數值不穩）
  - 需要多次求解

實際使用：
  優先使用 np.linalg.solve（基於 LU 分解）
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
