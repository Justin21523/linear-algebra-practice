"""
克萊姆法則 - NumPy 版本 (Cramer's Rule - NumPy Implementation)

本程式示範：
1. NumPy 實作克萊姆法則
2. 與 np.linalg.solve 比較
3. 效率和數值穩定性測試
"""

import numpy as np
import time

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def cramers_rule(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """用克萊姆法則解 Ax = b"""
    n = A.shape[0]
    det_A = np.linalg.det(A)

    if abs(det_A) < 1e-10:
        raise ValueError("矩陣奇異")

    x = np.zeros(n)
    for j in range(n):
        Aj = A.copy()
        Aj[:, j] = b
        x[j] = np.linalg.det(Aj) / det_A

    return x


def cramers_rule_single(A: np.ndarray, b: np.ndarray, j: int) -> float:
    """只求第 j 個未知數"""
    det_A = np.linalg.det(A)
    Aj = A.copy()
    Aj[:, j] = b
    return np.linalg.det(Aj) / det_A


def main():
    print_separator("克萊姆法則示範（NumPy 版）\nCramer's Rule Demo (NumPy)")

    # ========================================
    # 1. 基本使用
    # ========================================
    print_separator("1. 基本使用")

    A = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)

    b = np.array([8, -11, -3], dtype=float)

    print("方程組 Ax = b：")
    print(f"A:\n{A}")
    print(f"b: {b}")

    x_cramer = cramers_rule(A, b)
    x_solve = np.linalg.solve(A, b)

    print(f"\n克萊姆法則解：{x_cramer}")
    print(f"np.linalg.solve：{x_solve}")

    # 驗證
    print(f"\n驗證 Ax：{A @ x_cramer}")

    # ========================================
    # 2. 詳細過程展示
    # ========================================
    print_separator("2. 詳細計算過程")

    print(f"det(A) = {np.linalg.det(A):.4f}")

    for j in range(3):
        Aj = A.copy()
        Aj[:, j] = b
        det_Aj = np.linalg.det(Aj)
        xj = det_Aj / np.linalg.det(A)

        print(f"\nA{j+1}（第 {j+1} 行換成 b）:")
        print(Aj)
        print(f"det(A{j+1}) = {det_Aj:.4f}")
        print(f"x{j+1} = {xj:.4f}")

    # ========================================
    # 3. 只求一個未知數
    # ========================================
    print_separator("3. 只求特定未知數")

    print("假設只需要 x₂：")
    x2 = cramers_rule_single(A, b, 1)
    print(f"x₂ = {x2:.4f}")
    print(f"（完整解中的 x₂ = {x_cramer[1]:.4f}）")

    # ========================================
    # 4. 與其他方法比較
    # ========================================
    print_separator("4. 效率比較")

    sizes = [3, 5, 7, 9]

    print("矩陣大小 | 克萊姆法則 | np.linalg.solve | 加速比")
    print("-" * 55)

    for n in sizes:
        np.random.seed(42)
        A_test = np.random.randn(n, n)
        # 確保非奇異
        A_test = A_test @ A_test.T + np.eye(n)
        b_test = np.random.randn(n)

        # 克萊姆法則
        start = time.perf_counter()
        for _ in range(100):
            x_cramer = cramers_rule(A_test, b_test)
        t_cramer = (time.perf_counter() - start) / 100

        # np.linalg.solve
        start = time.perf_counter()
        for _ in range(100):
            x_solve = np.linalg.solve(A_test, b_test)
        t_solve = (time.perf_counter() - start) / 100

        speedup = t_cramer / t_solve
        print(f"{n:8d} | {t_cramer*1000:10.4f}ms | {t_solve*1000:14.4f}ms | {speedup:7.1f}x")

    print("\n結論：克萊姆法則在大矩陣時效率極低")

    # ========================================
    # 5. 數值穩定性
    # ========================================
    print_separator("5. 數值穩定性測試")

    # 構造病態矩陣
    n = 5
    epsilon = 1e-10

    A_bad = np.eye(n)
    A_bad[0, n-1] = epsilon
    A_bad[n-1, 0] = epsilon

    b_bad = np.ones(n)

    print(f"病態矩陣（條件數）：{np.linalg.cond(A_bad):.2e}")

    try:
        x_cramer = cramers_rule(A_bad, b_bad)
        x_solve = np.linalg.solve(A_bad, b_bad)

        error_cramer = np.linalg.norm(A_bad @ x_cramer - b_bad)
        error_solve = np.linalg.norm(A_bad @ x_solve - b_bad)

        print(f"\n殘差 ‖Ax - b‖：")
        print(f"  克萊姆法則：{error_cramer:.2e}")
        print(f"  np.linalg.solve：{error_solve:.2e}")
    except Exception as e:
        print(f"錯誤：{e}")

    # ========================================
    # 6. 應用：線性插值係數
    # ========================================
    print_separator("6. 應用：求多項式係數")

    print("問題：找通過 (0,1), (1,3), (2,7) 的二次多項式 y = a + bx + cx²")

    # Vandermonde 矩陣
    x_points = np.array([0, 1, 2], dtype=float)
    y_points = np.array([1, 3, 7], dtype=float)

    V = np.column_stack([np.ones(3), x_points, x_points**2])

    print(f"\nVandermonde 矩陣:\n{V}")
    print(f"y 值：{y_points}")

    coeffs = cramers_rule(V, y_points)
    print(f"\n係數 [a, b, c] = {coeffs}")
    print(f"多項式：y = {coeffs[0]:.1f} + {coeffs[1]:.1f}x + {coeffs[2]:.1f}x²")

    # 驗證
    print("\n驗證：")
    for xi, yi in zip(x_points, y_points):
        y_calc = coeffs[0] + coeffs[1] * xi + coeffs[2] * xi**2
        print(f"  x={xi:.0f}: y = {y_calc:.1f} (預期 {yi:.1f})")

    # 總結
    print_separator("NumPy 克萊姆法則總結")
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
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
