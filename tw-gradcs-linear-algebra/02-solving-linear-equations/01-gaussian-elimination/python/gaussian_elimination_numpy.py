"""
高斯消去法：NumPy 版本 (Gaussian Elimination: NumPy Implementation)

本程式示範：
1. 使用 NumPy 實作高斯消去法
2. np.linalg.solve 求解線性系統
3. 比較不同求解方法
4. 數值穩定性示範

This program demonstrates Gaussian elimination using NumPy
and compares with built-in solvers.
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def gaussian_elimination_numpy(A: np.ndarray, b: np.ndarray,
                                verbose: bool = False) -> np.ndarray:
    """
    高斯消去法（NumPy 實作，含部分選主元）

    Gaussian elimination with partial pivoting using NumPy

    Parameters:
        A: 係數矩陣 (n×n)
        b: 右手邊向量 (n,)
        verbose: 是否印出過程

    Returns:
        解向量 x
    """
    n = A.shape[0]

    # 建立增廣矩陣 [A | b]
    # Create augmented matrix
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    if verbose:
        print("初始增廣矩陣 [A|b]:")
        print(Ab)
        print()

    # 前進消去 (Forward elimination)
    for k in range(n - 1):
        # 部分選主元 (Partial pivoting)
        max_row = k + np.argmax(np.abs(Ab[k:, k]))

        if np.abs(Ab[max_row, k]) < 1e-12:
            raise ValueError(f"矩陣奇異：第 {k} 行主元為零")

        # 換列
        if max_row != k:
            Ab[[k, max_row]] = Ab[[max_row, k]]
            if verbose:
                print(f"步驟 {k+1}: 交換第 {k+1} 列和第 {max_row+1} 列")

        # 消去
        for i in range(k + 1, n):
            if np.abs(Ab[i, k]) > 1e-12:
                multiplier = Ab[i, k] / Ab[k, k]
                Ab[i, k:] -= multiplier * Ab[k, k:]

        if verbose:
            print(f"步驟 {k+1} 後:")
            print(Ab)
            print()

    # 檢查最後一個主元
    if np.abs(Ab[n-1, n-1]) < 1e-12:
        raise ValueError("矩陣奇異：最後一個主元為零")

    # 回代 (Back substitution)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, n] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    return x


def main():
    """主程式"""

    print_separator("高斯消去法示範 - NumPy 版本\nGaussian Elimination Demo - NumPy")

    # ========================================
    # 範例 1：基本求解
    # ========================================
    print_separator("1. 基本求解範例")

    A = np.array([
        [2, 1, 1],
        [4, -6, 0],
        [-2, 7, 2]
    ], dtype=float)

    b = np.array([5, -2, 9], dtype=float)

    print(f"A =\n{A}\n")
    print(f"b = {b}\n")

    # 方法一：我們的實作
    x_manual = gaussian_elimination_numpy(A, b, verbose=True)
    print(f"我們的解: x = {x_manual}")

    # 方法二：np.linalg.solve
    x_numpy = np.linalg.solve(A, b)
    print(f"np.linalg.solve: x = {x_numpy}")

    # 驗證
    print(f"\n驗證 ‖Ax - b‖ = {np.linalg.norm(A @ x_manual - b):.2e}")

    # ========================================
    # 範例 2：比較不同求解方法
    # ========================================
    print_separator("2. 不同求解方法比較")

    print("""
NumPy 提供多種求解方法：

1. np.linalg.solve(A, b)
   - 使用 LU 分解
   - 最常用、效率高

2. np.linalg.lstsq(A, b)
   - 最小平方解
   - 即使 A 不是方陣也能用

3. np.linalg.inv(A) @ b
   - 先求反矩陣再相乘
   - ❌ 不推薦：效率低、數值不穩定
""")

    # 實際比較
    print("實際比較：")
    print(f"np.linalg.solve(A, b):      {np.linalg.solve(A, b)}")
    print(f"np.linalg.lstsq(A, b)[0]:   {np.linalg.lstsq(A, b, rcond=None)[0]}")
    print(f"np.linalg.inv(A) @ b:       {np.linalg.inv(A) @ b}")

    # ========================================
    # 範例 3：需要換列的情況
    # ========================================
    print_separator("3. 需要換列的情況（第一個元素為零）")

    A2 = np.array([
        [0, 1, 2],
        [1, 2, 1],
        [2, 3, 1]
    ], dtype=float)

    b2 = np.array([3, 4, 5], dtype=float)

    print(f"A =\n{A2}\n")
    print(f"b = {b2}\n")

    x2 = gaussian_elimination_numpy(A2, b2, verbose=True)
    print(f"解: x = {x2}")
    print(f"驗證: A @ x = {A2 @ x2}")

    # ========================================
    # 範例 4：數值穩定性示範
    # ========================================
    print_separator("4. 數值穩定性示範")

    print("考慮近乎奇異的矩陣：")

    # Hilbert 矩陣是著名的病態矩陣
    n = 5
    H = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])

    print(f"Hilbert 矩陣 H({n}×{n}):\n{H}\n")

    # 條件數
    cond = np.linalg.cond(H)
    print(f"條件數 (condition number): {cond:.2e}")
    print("條件數越大，矩陣越接近奇異，數值解越不穩定\n")

    b_h = np.ones(n)
    x_h = np.linalg.solve(H, b_h)
    print(f"解 x = {x_h}")
    print(f"殘差 ‖Hx - b‖ = {np.linalg.norm(H @ x_h - b_h):.2e}")

    # ========================================
    # 範例 5：奇異矩陣
    # ========================================
    print_separator("5. 奇異矩陣（無法求解）")

    A_singular = np.array([
        [1, 2, 3],
        [2, 4, 6],  # 第一列的 2 倍
        [1, 3, 4]
    ], dtype=float)

    print(f"A (奇異矩陣) =\n{A_singular}\n")
    print(f"行列式 det(A) = {np.linalg.det(A_singular):.6f}")
    print(f"秩 rank(A) = {np.linalg.matrix_rank(A_singular)}")

    try:
        x_singular = gaussian_elimination_numpy(A_singular, np.array([1, 2, 3]))
    except ValueError as e:
        print(f"\n錯誤：{e}")

    # ========================================
    # 範例 6：大型系統效能
    # ========================================
    print_separator("6. 大型系統效能提示")

    print("""
對於大型系統，NumPy 的效能非常重要：

時間複雜度：
- 高斯消去法：O(n³)
- 回代：O(n²)

效能提示：
1. 使用 np.linalg.solve，它底層使用優化的 LAPACK
2. 若需解多個 b，先做 LU 分解
3. 對於稀疏矩陣，使用 scipy.sparse.linalg
""")

    import time

    sizes = [100, 200, 500]
    for n in sizes:
        A_big = np.random.rand(n, n)
        b_big = np.random.rand(n)

        start = time.time()
        x_big = np.linalg.solve(A_big, b_big)
        elapsed = time.time() - start

        print(f"n = {n}: 求解時間 = {elapsed*1000:.2f} ms, "
              f"殘差 = {np.linalg.norm(A_big @ x_big - b_big):.2e}")

    # ========================================
    # 範例 7：多個右手邊
    # ========================================
    print_separator("7. 多個右手邊 (Multiple RHS)")

    print("若需解 AX = B（B 是矩陣，每行是一個 b）：")

    A = np.array([[2, 1], [1, 3]], dtype=float)
    B = np.array([[3, 1], [5, 2]], dtype=float)  # 兩個右手邊

    print(f"A =\n{A}\n")
    print(f"B =\n{B}\n")

    X = np.linalg.solve(A, B)
    print(f"X = solve(A, B):\n{X}")
    print(f"\n驗證 A @ X =\n{A @ X}")

    print()
    print("=" * 60)
    print("NumPy 高斯消去法示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
