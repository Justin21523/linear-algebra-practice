"""
高斯消去法：手刻版本 (Gaussian Elimination: Manual Implementation)

本程式示範：
1. 前進消去 (Forward elimination)
2. 部分選主元 (Partial pivoting)
3. 回代 (Back substitution)
4. 完整求解流程
5. 奇異情況檢測

This program demonstrates Gaussian elimination with partial pivoting
and back substitution for solving linear systems Ax = b.
"""

from typing import List, Tuple, Optional
import copy

# 型別別名 (Type alias)
Matrix = List[List[float]]
Vector = List[float]


def print_matrix(name: str, A: Matrix, augmented: bool = False) -> None:
    """
    印出矩陣 (Print matrix)
    augmented=True 時，最後一行顯示為增廣部分
    """
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0

    print(f"{name}:")
    for i, row in enumerate(A):
        print("  [", end="")
        for j, val in enumerate(row):
            if augmented and j == cols - 1:
                print(" |", end="")
            print(f"{val:8.4f}", end="")
        print(" ]")
    print()


def print_separator(title: str) -> None:
    """印出分隔線 (Print separator)"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def create_augmented_matrix(A: Matrix, b: Vector) -> Matrix:
    """
    建立增廣矩陣 [A | b]
    Create augmented matrix [A | b]
    """
    n = len(A)
    augmented = []
    for i in range(n):
        row = A[i].copy()
        row.append(b[i])
        augmented.append(row)
    return augmented


def swap_rows(A: Matrix, i: int, j: int) -> None:
    """
    交換第 i 列和第 j 列 (Swap rows i and j)
    """
    A[i], A[j] = A[j], A[i]


def find_pivot_row(A: Matrix, col: int, start_row: int) -> int:
    """
    部分選主元：找出從 start_row 開始，第 col 行中絕對值最大的元素所在列
    Partial pivoting: Find row with largest absolute value in column
    """
    n = len(A)
    max_val = abs(A[start_row][col])
    max_row = start_row

    for i in range(start_row + 1, n):
        if abs(A[i][col]) > max_val:
            max_val = abs(A[i][col])
            max_row = i

    return max_row


def forward_elimination(A: Matrix, verbose: bool = True) -> Tuple[Matrix, bool]:
    """
    前進消去 (Forward Elimination)

    將增廣矩陣化為上三角形式

    Parameters:
        A: 增廣矩陣 [A | b]
        verbose: 是否印出過程

    Returns:
        (上三角矩陣, 是否成功)
    """
    n = len(A)
    U = copy.deepcopy(A)  # 不修改原矩陣

    if verbose:
        print_separator("前進消去 (Forward Elimination)")
        print("初始增廣矩陣：")
        print_matrix("", U, augmented=True)

    for k in range(n - 1):  # 對每一行（除了最後一行）
        if verbose:
            print(f"--- 步驟 {k + 1}：消去第 {k + 1} 行以下的元素 ---")

        # 部分選主元 (Partial pivoting)
        pivot_row = find_pivot_row(U, k, k)

        if abs(U[pivot_row][k]) < 1e-12:
            print(f"警告：主元接近零，矩陣可能奇異")
            return U, False

        if pivot_row != k:
            if verbose:
                print(f"交換第 {k + 1} 列和第 {pivot_row + 1} 列")
            swap_rows(U, k, pivot_row)
            if verbose:
                print_matrix("", U, augmented=True)

        # 消去：使第 k 行以下的元素變為 0
        for i in range(k + 1, n):
            if abs(U[i][k]) > 1e-12:  # 只有非零才需要消去
                multiplier = U[i][k] / U[k][k]

                if verbose:
                    print(f"R{i + 1} ← R{i + 1} - ({multiplier:.4f}) × R{k + 1}")

                # 對整列（包含增廣部分）進行消去
                for j in range(k, len(U[0])):
                    U[i][j] = U[i][j] - multiplier * U[k][j]

        if verbose:
            print_matrix("", U, augmented=True)

    # 檢查最後一個主元
    if abs(U[n - 1][n - 1]) < 1e-12:
        if verbose:
            print("警告：最後一個主元為零，矩陣奇異")
        return U, False

    return U, True


def back_substitution(U: Matrix, verbose: bool = True) -> Optional[Vector]:
    """
    回代 (Back Substitution)

    從上三角系統求解 x

    Parameters:
        U: 上三角增廣矩陣
        verbose: 是否印出過程

    Returns:
        解向量 x，若無法求解則返回 None
    """
    n = len(U)
    x = [0.0] * n

    if verbose:
        print_separator("回代 (Back Substitution)")

    # 從最後一列開始回代
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < 1e-12:
            print(f"錯誤：主元 U[{i}][{i}] 為零，無法回代")
            return None

        # xᵢ = (cᵢ - Σⱼ uᵢⱼ·xⱼ) / uᵢᵢ
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += U[i][j] * x[j]

        x[i] = (U[i][n] - sum_val) / U[i][i]

        if verbose:
            if i == n - 1:
                print(f"x{i + 1} = {U[i][n]:.4f} / {U[i][i]:.4f} = {x[i]:.4f}")
            else:
                terms = " - ".join(f"{U[i][j]:.4f}×x{j + 1}" for j in range(i + 1, n))
                print(f"x{i + 1} = ({U[i][n]:.4f} - ({terms})) / {U[i][i]:.4f} = {x[i]:.4f}")

    return x


def gaussian_elimination_solve(A: Matrix, b: Vector, verbose: bool = True) -> Optional[Vector]:
    """
    高斯消去法完整求解 (Complete Gaussian Elimination Solver)

    求解 Ax = b

    Parameters:
        A: 係數矩陣
        b: 右手邊向量
        verbose: 是否印出詳細過程

    Returns:
        解向量 x，若無解則返回 None
    """
    if verbose:
        print_separator("高斯消去法求解 Ax = b")
        print("原始係數矩陣 A：")
        print_matrix("A", A)
        print(f"右手邊向量 b = {b}")

    # 建立增廣矩陣
    augmented = create_augmented_matrix(A, b)

    # 前進消去
    U, success = forward_elimination(augmented, verbose)

    if not success:
        print("消去失敗：矩陣可能奇異")
        return None

    # 回代
    x = back_substitution(U, verbose)

    if x is not None and verbose:
        print_separator("解 (Solution)")
        print(f"x = {[f'{xi:.4f}' for xi in x]}")

    return x


def verify_solution(A: Matrix, x: Vector, b: Vector) -> float:
    """
    驗證解的正確性 (Verify solution)
    計算殘差 ‖Ax - b‖
    """
    n = len(A)
    Ax = [0.0] * n

    for i in range(n):
        for j in range(n):
            Ax[i] += A[i][j] * x[j]

    # 計算殘差範數
    residual = sum((Ax[i] - b[i]) ** 2 for i in range(n)) ** 0.5

    return residual


def main():
    """主程式 (Main program)"""

    print_separator("高斯消去法示範\nGaussian Elimination Demo")

    # ========================================
    # 範例 1：標準 3×3 系統
    # ========================================
    print_separator("範例 1：標準 3×3 系統")

    A1 = [
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0]
    ]
    b1 = [5.0, -2.0, 9.0]

    x1 = gaussian_elimination_solve(A1, b1, verbose=True)

    if x1:
        residual = verify_solution(A1, x1, b1)
        print(f"\n驗證：‖Ax - b‖ = {residual:.2e}")

    # ========================================
    # 範例 2：需要換列的系統
    # ========================================
    print_separator("範例 2：需要換列的系統（主元為零）")

    A2 = [
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 1.0],
        [2.0, 3.0, 1.0]
    ]
    b2 = [3.0, 4.0, 5.0]

    x2 = gaussian_elimination_solve(A2, b2, verbose=True)

    if x2:
        residual = verify_solution(A2, x2, b2)
        print(f"\n驗證：‖Ax - b‖ = {residual:.2e}")

    # ========================================
    # 範例 3：2×2 簡單系統
    # ========================================
    print_separator("範例 3：2×2 簡單系統")

    A3 = [
        [3.0, 2.0],
        [1.0, 4.0]
    ]
    b3 = [7.0, 9.0]

    x3 = gaussian_elimination_solve(A3, b3, verbose=True)

    if x3:
        residual = verify_solution(A3, x3, b3)
        print(f"\n驗證：‖Ax - b‖ = {residual:.2e}")

    # ========================================
    # 範例 4：奇異矩陣（線性相依列）
    # ========================================
    print_separator("範例 4：奇異矩陣（線性相依列）")

    A4 = [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],  # 這列是第一列的 2 倍
        [1.0, 3.0, 4.0]
    ]
    b4 = [6.0, 12.0, 8.0]

    x4 = gaussian_elimination_solve(A4, b4, verbose=True)

    # ========================================
    # 消去矩陣示範 (Elimination Matrix Demo)
    # ========================================
    print_separator("消去矩陣示範 (Elimination Matrix)")

    print("""
消去操作可以用矩陣表示：

「R₂ ← R₂ - 2R₁」等價於左乘：

       [1   0  0]
E₂₁ =  [-2  1  0]
       [0   0  1]

整個消去過程：
E₃₂ · E₃₁ · E₂₁ · A = U

這就是 LU 分解的由來：
A = E₂₁⁻¹ · E₃₁⁻¹ · E₃₂⁻¹ · U = L · U
""")

    print("=" * 60)
    print("高斯消去法示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
