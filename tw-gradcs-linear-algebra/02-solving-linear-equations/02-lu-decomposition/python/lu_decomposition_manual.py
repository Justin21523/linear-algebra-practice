"""
LU 分解：手刻版本 (LU Decomposition: Manual Implementation)

本程式示範：
1. Doolittle LU 分解（L 對角線為 1）
2. 前進代入 (Forward substitution)
3. 回代 (Back substitution)
4. 完整求解流程 Ax = b
5. 利用 LU 計算行列式

This program demonstrates LU decomposition and solving linear systems.
"""

from typing import List, Tuple
import copy

# 型別別名
Matrix = List[List[float]]
Vector = List[float]


def print_matrix(name: str, A: Matrix) -> None:
    """印出矩陣"""
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    print(f"{name} ({rows}×{cols}):")
    for row in A:
        print("  [", end="")
        print("  ".join(f"{x:8.4f}" for x in row), end="")
        print(" ]")
    print()


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def create_identity(n: int) -> Matrix:
    """建立 n×n 單位矩陣"""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def create_zero_matrix(n: int) -> Matrix:
    """建立 n×n 零矩陣"""
    return [[0.0 for _ in range(n)] for _ in range(n)]


def lu_decomposition(A: Matrix, verbose: bool = True) -> Tuple[Matrix, Matrix]:
    """
    Doolittle LU 分解 (Doolittle LU Decomposition)

    將 A 分解為 A = LU
    - L：下三角矩陣，對角線為 1
    - U：上三角矩陣

    Parameters:
        A: n×n 方陣
        verbose: 是否印出過程

    Returns:
        (L, U) 元組
    """
    n = len(A)
    L = create_identity(n)  # L 對角線為 1
    U = create_zero_matrix(n)

    if verbose:
        print_separator("LU 分解過程 (Doolittle Algorithm)")
        print_matrix("原始矩陣 A", A)

    for k in range(n):
        if verbose:
            print(f"--- 步驟 {k + 1} ---")

        # 計算 U 的第 k 列
        for j in range(k, n):
            sum_val = sum(L[k][s] * U[s][j] for s in range(k))
            U[k][j] = A[k][j] - sum_val

            if verbose and j == k:
                print(f"u{k+1}{j+1} = a{k+1}{j+1} - Σ(l{k+1}s × us{j+1}) = {U[k][j]:.4f}")

        # 檢查主元
        if abs(U[k][k]) < 1e-12:
            raise ValueError(f"主元 u{k+1}{k+1} 為零，LU 分解失敗（可能需要選主元）")

        # 計算 L 的第 k 行
        for i in range(k + 1, n):
            sum_val = sum(L[i][s] * U[s][k] for s in range(k))
            L[i][k] = (A[i][k] - sum_val) / U[k][k]

            if verbose:
                print(f"l{i+1}{k+1} = (a{i+1}{k+1} - Σ) / u{k+1}{k+1} = {L[i][k]:.4f}")

        if verbose:
            print()

    if verbose:
        print_matrix("L（下三角，對角線為 1）", L)
        print_matrix("U（上三角）", U)

    return L, U


def forward_substitution(L: Matrix, b: Vector, verbose: bool = True) -> Vector:
    """
    前進代入 (Forward Substitution)

    解 Ly = b，其中 L 是下三角矩陣

    Parameters:
        L: 下三角矩陣
        b: 右手邊向量

    Returns:
        解向量 y
    """
    n = len(L)
    y = [0.0] * n

    if verbose:
        print_separator("前進代入 Ly = b")
        print(f"b = {[f'{bi:.4f}' for bi in b]}")
        print()

    for i in range(n):
        sum_val = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_val) / L[i][i]

        if verbose:
            print(f"y{i+1} = (b{i+1} - Σ) / L{i+1}{i+1} = {y[i]:.4f}")

    if verbose:
        print(f"\ny = {[f'{yi:.4f}' for yi in y]}")

    return y


def back_substitution(U: Matrix, y: Vector, verbose: bool = True) -> Vector:
    """
    回代 (Back Substitution)

    解 Ux = y，其中 U 是上三角矩陣

    Parameters:
        U: 上三角矩陣
        y: 右手邊向量

    Returns:
        解向量 x
    """
    n = len(U)
    x = [0.0] * n

    if verbose:
        print_separator("回代 Ux = y")
        print(f"y = {[f'{yi:.4f}' for yi in y]}")
        print()

    for i in range(n - 1, -1, -1):
        sum_val = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_val) / U[i][i]

        if verbose:
            print(f"x{i+1} = (y{i+1} - Σ) / U{i+1}{i+1} = {x[i]:.4f}")

    if verbose:
        print(f"\nx = {[f'{xi:.4f}' for xi in x]}")

    return x


def solve_with_lu(L: Matrix, U: Matrix, b: Vector, verbose: bool = True) -> Vector:
    """
    用 LU 分解求解 Ax = b

    步驟：
    1. 解 Ly = b（前進代入）
    2. 解 Ux = y（回代）
    """
    if verbose:
        print_separator("用 LU 分解求解 Ax = b")

    y = forward_substitution(L, b, verbose)
    x = back_substitution(U, y, verbose)

    return x


def lu_determinant(U: Matrix) -> float:
    """
    用 LU 分解計算行列式

    det(A) = det(L) × det(U) = 1 × (u₁₁ × u₂₂ × ... × uₙₙ)
    """
    n = len(U)
    det = 1.0
    for i in range(n):
        det *= U[i][i]
    return det


def verify_lu(A: Matrix, L: Matrix, U: Matrix) -> float:
    """驗證 LU = A"""
    n = len(A)
    error = 0.0

    for i in range(n):
        for j in range(n):
            lu_ij = sum(L[i][k] * U[k][j] for k in range(n))
            error += abs(A[i][j] - lu_ij)

    return error


def matrix_vector_multiply(A: Matrix, x: Vector) -> Vector:
    """計算 Ax"""
    n = len(A)
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]


def main():
    """主程式"""

    print_separator("LU 分解示範\nLU Decomposition Demo")

    # ========================================
    # 範例 1：標準 3×3 矩陣
    # ========================================
    print_separator("範例 1：標準 3×3 矩陣")

    A = [
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0]
    ]

    # LU 分解
    L, U = lu_decomposition(A, verbose=True)

    # 驗證 LU = A
    error = verify_lu(A, L, U)
    print(f"驗證：‖LU - A‖ = {error:.2e}")

    # 求解 Ax = b
    b = [5.0, -2.0, 9.0]
    print(f"\n求解 Ax = b，其中 b = {b}")

    x = solve_with_lu(L, U, b, verbose=True)

    # 驗證解
    Ax = matrix_vector_multiply(A, x)
    print_separator("驗證解")
    print(f"x = {[f'{xi:.4f}' for xi in x]}")
    print(f"Ax = {[f'{axi:.4f}' for axi in Ax]}")
    print(f"b  = {[f'{bi:.4f}' for bi in b]}")

    # ========================================
    # 範例 2：利用 LU 解多個右手邊
    # ========================================
    print_separator("範例 2：解多個右手邊（LU 分解的效率優勢）")

    print("已有 L 和 U，現在解不同的 b：")

    b_list = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]

    print("\n這三個 b 對應求 A 的反矩陣的三個行！")

    for i, b_i in enumerate(b_list):
        x_i = solve_with_lu(L, U, b_i, verbose=False)
        print(f"b{i+1} = {b_i} → x{i+1} = {[f'{xij:.4f}' for xij in x_i]}")

    # ========================================
    # 範例 3：計算行列式
    # ========================================
    print_separator("範例 3：用 LU 分解計算行列式")

    det = lu_determinant(U)
    print(f"det(A) = u₁₁ × u₂₂ × u₃₃")
    print(f"       = {U[0][0]:.4f} × {U[1][1]:.4f} × {U[2][2]:.4f}")
    print(f"       = {det:.4f}")

    # ========================================
    # 範例 4：2×2 簡單範例
    # ========================================
    print_separator("範例 4：2×2 簡單範例")

    A2 = [
        [4.0, 3.0],
        [6.0, 3.0]
    ]

    print_matrix("A", A2)

    L2, U2 = lu_decomposition(A2, verbose=True)

    print("手算驗證：")
    print("u₁₁ = 4, u₁₂ = 3")
    print("l₂₁ = 6/4 = 1.5")
    print("u₂₂ = 3 - 1.5×3 = -1.5")

    # ========================================
    # 範例 5：LU 分解與高斯消去的關係
    # ========================================
    print_separator("範例 5：LU 分解與高斯消去的關係")

    print("""
LU 分解的本質：

高斯消去過程中：
- E₂₁ · A = （消去第一個主元以下的元素後的矩陣）
- E₃₁ · E₂₁ · A = ...
- E₃₂ · E₃₁ · E₂₁ · A = U （最終的上三角矩陣）

因此：
A = E₂₁⁻¹ · E₃₁⁻¹ · E₃₂⁻¹ · U = L · U

其中 L = E₂₁⁻¹ · E₃₁⁻¹ · E₃₂⁻¹

L 的元素正好是消去過程中的乘數！

例如本例：
- l₂₁ = 4/2 = 2（消去 a₂₁ 的乘數）
- l₃₁ = -2/2 = -1（消去 a₃₁ 的乘數）
- l₃₂ = 8/(-8) = -1（消去 a₃₂ 的乘數）
""")

    print("=" * 60)
    print("LU 分解示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
