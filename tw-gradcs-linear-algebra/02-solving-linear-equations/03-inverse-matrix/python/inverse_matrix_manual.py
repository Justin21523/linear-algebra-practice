"""
反矩陣：手刻版本 (Inverse Matrix: Manual Implementation)

本程式示範：
1. 2×2 矩陣反矩陣公式
2. 高斯-乔丹消去法求反矩陣
3. 反矩陣的性質驗證

This program demonstrates computing inverse matrices using
the 2x2 formula and Gauss-Jordan elimination.
"""

from typing import List, Optional
import copy

Matrix = List[List[float]]


def print_matrix(name: str, A: Matrix, augmented: bool = False) -> None:
    """印出矩陣"""
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0

    print(f"{name}:")
    for row in A:
        print("  [", end="")
        for j, val in enumerate(row):
            if augmented and j == cols // 2:
                print(" |", end="")
            print(f"{val:8.4f}", end="")
        print(" ]")
    print()


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def matrix_multiply(A: Matrix, B: Matrix) -> Matrix:
    """矩陣乘法"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    result = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result


def inverse_2x2(A: Matrix) -> Optional[Matrix]:
    """
    2×2 矩陣的反矩陣公式

    A = [a  b]        A⁻¹ = 1/(ad-bc) × [d  -b]
        [c  d]                          [-c   a]
    """
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]

    det = a * d - b * c

    if abs(det) < 1e-12:
        print(f"行列式 = {det:.6f}，矩陣不可逆")
        return None

    return [
        [d / det, -b / det],
        [-c / det, a / det]
    ]


def gauss_jordan_inverse(A: Matrix, verbose: bool = True) -> Optional[Matrix]:
    """
    高斯-乔丹消去法求反矩陣 (Gauss-Jordan Elimination)

    將 [A | I] 化簡為 [I | A⁻¹]

    Parameters:
        A: n×n 方陣
        verbose: 是否印出過程

    Returns:
        A⁻¹ 或 None（若不可逆）
    """
    n = len(A)

    # 建立增廣矩陣 [A | I]
    augmented = []
    for i in range(n):
        row = A[i].copy()
        for j in range(n):
            row.append(1.0 if i == j else 0.0)
        augmented.append(row)

    if verbose:
        print_separator("高斯-乔丹消去法求反矩陣")
        print("初始增廣矩陣 [A | I]：")
        print_matrix("", augmented, augmented=True)

    # 前進消去 + 回消（同時進行）
    for k in range(n):
        # 部分選主元
        max_row = k
        max_val = abs(augmented[k][k])
        for i in range(k + 1, n):
            if abs(augmented[i][k]) > max_val:
                max_val = abs(augmented[i][k])
                max_row = i

        if max_val < 1e-12:
            print(f"第 {k+1} 個主元為零，矩陣不可逆")
            return None

        # 換列
        if max_row != k:
            augmented[k], augmented[max_row] = augmented[max_row], augmented[k]
            if verbose:
                print(f"交換第 {k+1} 列和第 {max_row+1} 列")

        # 將主元化為 1
        pivot = augmented[k][k]
        for j in range(2 * n):
            augmented[k][j] /= pivot

        if verbose:
            print(f"R{k+1} ← R{k+1} / {pivot:.4f}")

        # 消去該行的其他元素（上下都要消）
        for i in range(n):
            if i != k:
                factor = augmented[i][k]
                if abs(factor) > 1e-12:
                    for j in range(2 * n):
                        augmented[i][j] -= factor * augmented[k][j]

                    if verbose:
                        print(f"R{i+1} ← R{i+1} - ({factor:.4f}) × R{k+1}")

        if verbose:
            print_matrix("", augmented, augmented=True)

    # 提取右半部分作為 A⁻¹
    A_inv = [[augmented[i][j + n] for j in range(n)] for i in range(n)]

    return A_inv


def verify_inverse(A: Matrix, A_inv: Matrix) -> bool:
    """驗證 A × A⁻¹ = I"""
    n = len(A)
    product = matrix_multiply(A, A_inv)

    # 檢查是否為單位矩陣
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(product[i][j] - expected) > 1e-6:
                return False
    return True


def main():
    """主程式"""

    print_separator("反矩陣示範\nInverse Matrix Demo")

    # ========================================
    # 範例 1：2×2 反矩陣公式
    # ========================================
    print_separator("範例 1：2×2 反矩陣公式")

    A2 = [
        [4.0, 7.0],
        [2.0, 6.0]
    ]

    print_matrix("A", A2)

    det_A2 = A2[0][0] * A2[1][1] - A2[0][1] * A2[1][0]
    print(f"det(A) = 4×6 - 7×2 = {det_A2}")

    A2_inv = inverse_2x2(A2)
    if A2_inv:
        print_matrix("A⁻¹", A2_inv)

        print("驗證 A × A⁻¹ =")
        product = matrix_multiply(A2, A2_inv)
        print_matrix("", product)
        print(f"是單位矩陣？ {verify_inverse(A2, A2_inv)}")

    # ========================================
    # 範例 2：高斯-乔丹消去法 3×3
    # ========================================
    print_separator("範例 2：高斯-乔丹消去法 3×3")

    A3 = [
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0]
    ]

    print_matrix("A", A3)

    A3_inv = gauss_jordan_inverse(A3, verbose=True)

    if A3_inv:
        print_separator("結果")
        print_matrix("A⁻¹", A3_inv)

        print("驗證 A × A⁻¹ =")
        product = matrix_multiply(A3, A3_inv)
        print_matrix("", product)
        print(f"是單位矩陣？ {verify_inverse(A3, A3_inv)}")

    # ========================================
    # 範例 3：奇異矩陣（不可逆）
    # ========================================
    print_separator("範例 3：奇異矩陣（不可逆）")

    A_singular = [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],  # 第一列的 2 倍
        [1.0, 3.0, 4.0]
    ]

    print_matrix("A（奇異矩陣）", A_singular)
    print("第二列是第一列的 2 倍，所以矩陣不滿秩")

    result = gauss_jordan_inverse(A_singular, verbose=True)
    if result is None:
        print("確認：矩陣不可逆")

    # ========================================
    # 範例 4：反矩陣的性質
    # ========================================
    print_separator("範例 4：反矩陣的性質")

    B = [
        [1.0, 2.0],
        [3.0, 4.0]
    ]

    C = [
        [2.0, 0.0],
        [1.0, 2.0]
    ]

    print_matrix("B", B)
    print_matrix("C", C)

    B_inv = inverse_2x2(B)
    C_inv = inverse_2x2(C)

    if B_inv and C_inv:
        # 性質：(BC)⁻¹ = C⁻¹B⁻¹
        BC = matrix_multiply(B, C)
        BC_inv = inverse_2x2(BC)
        C_inv_B_inv = matrix_multiply(C_inv, B_inv)

        print("驗證 (BC)⁻¹ = C⁻¹B⁻¹：")
        print_matrix("(BC)⁻¹", BC_inv)
        print_matrix("C⁻¹B⁻¹", C_inv_B_inv)

        # 性質：(B⁻¹)⁻¹ = B
        B_inv_inv = inverse_2x2(B_inv)
        print("驗證 (B⁻¹)⁻¹ = B：")
        print_matrix("(B⁻¹)⁻¹", B_inv_inv)
        print_matrix("B", B)

    # ========================================
    # 範例 5：用反矩陣解方程組（不推薦）
    # ========================================
    print_separator("範例 5：用反矩陣解 Ax = b（僅作示範）")

    A = [
        [2.0, 1.0],
        [5.0, 3.0]
    ]
    b = [4.0, 11.0]

    print_matrix("A", A)
    print(f"b = {b}")

    A_inv = inverse_2x2(A)
    if A_inv:
        # x = A⁻¹ × b
        x = [
            A_inv[0][0] * b[0] + A_inv[0][1] * b[1],
            A_inv[1][0] * b[0] + A_inv[1][1] * b[1]
        ]

        print(f"\nx = A⁻¹ × b = {[f'{xi:.4f}' for xi in x]}")

        # 驗證
        Ax = [
            A[0][0] * x[0] + A[0][1] * x[1],
            A[1][0] * x[0] + A[1][1] * x[1]
        ]
        print(f"驗證 Ax = {[f'{axi:.4f}' for axi in Ax]}")

    print("""
⚠️ 注意：實際應用中不推薦這樣做！
   應該使用高斯消去法或 LU 分解求解。
""")

    print("=" * 60)
    print("反矩陣示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
