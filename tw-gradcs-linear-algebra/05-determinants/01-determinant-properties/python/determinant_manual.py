"""
行列式的性質 - 手刻版本 (Determinant Properties - Manual Implementation)

本程式示範：
1. 2×2, 3×3 行列式計算
2. 行列式的性質驗證
3. 列運算對行列式的影響
"""

from typing import List


def print_separator(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_matrix(name: str, M: List[List[float]]) -> None:
    print(f"{name} =")
    for row in M:
        formatted = [f"{x:8.4f}" for x in row]
        print(f"  [{', '.join(formatted)}]")


# ========================================
# 行列式計算
# ========================================

def det_2x2(A: List[List[float]]) -> float:
    """計算 2×2 行列式"""
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def det_3x3(A: List[List[float]]) -> float:
    """計算 3×3 行列式（Sarrus 法則或展開）"""
    a, b, c = A[0]
    d, e, f = A[1]
    g, h, i = A[2]

    return (a * e * i + b * f * g + c * d * h
            - c * e * g - b * d * i - a * f * h)


def det_nxn(A: List[List[float]]) -> float:
    """
    計算 n×n 行列式（列運算化為上三角）

    返回行列式值
    """
    n = len(A)

    # 複製矩陣
    M = [row[:] for row in A]

    sign = 1  # 追蹤列交換次數

    for col in range(n):
        # 找主元（pivot）
        pivot_row = None
        for row in range(col, n):
            if abs(M[row][col]) > 1e-10:
                pivot_row = row
                break

        if pivot_row is None:
            return 0.0  # 奇異矩陣

        # 列交換
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
            sign *= -1

        # 消去下方元素
        for row in range(col + 1, n):
            if abs(M[col][col]) > 1e-10:
                factor = M[row][col] / M[col][col]
                for j in range(col, n):
                    M[row][j] -= factor * M[col][j]

    # 對角線乘積
    det = sign
    for i in range(n):
        det *= M[i][i]

    return det


# ========================================
# 矩陣運算
# ========================================

def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩陣乘法"""
    m, k, n = len(A), len(B), len(B[0])
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for p in range(k):
                result[i][j] += A[i][p] * B[p][j]
    return result


def transpose(A: List[List[float]]) -> List[List[float]]:
    """矩陣轉置"""
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def scalar_multiply_matrix(c: float, A: List[List[float]]) -> List[List[float]]:
    """純量乘矩陣"""
    return [[c * x for x in row] for row in A]


def swap_rows(A: List[List[float]], i: int, j: int) -> List[List[float]]:
    """交換列"""
    result = [row[:] for row in A]
    result[i], result[j] = result[j], result[i]
    return result


def add_row_multiple(A: List[List[float]], target: int, source: int, c: float) -> List[List[float]]:
    """列運算：target 列 += c * source 列"""
    result = [row[:] for row in A]
    for j in range(len(A[0])):
        result[target][j] += c * result[source][j]
    return result


def main():
    print_separator("行列式性質示範（手刻版）\nDeterminant Properties Demo (Manual)")

    # ========================================
    # 1. 基本行列式計算
    # ========================================
    print_separator("1. 基本行列式計算")

    # 2×2
    A2 = [[3, 8], [4, 6]]
    print_matrix("A (2×2)", A2)
    print(f"det(A) = {A2[0][0]}×{A2[1][1]} - {A2[0][1]}×{A2[1][0]} = {det_2x2(A2)}")

    # 3×3
    A3 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ]
    print_matrix("\nA (3×3)", A3)
    print(f"det(A) = {det_3x3(A3)}")

    # n×n（使用列運算）
    print(f"使用列運算驗證：{det_nxn(A3):.4f}")

    # ========================================
    # 2. 性質 1：det(I) = 1
    # ========================================
    print_separator("2. 性質 1：det(I) = 1")

    I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    print_matrix("I₃", I3)
    print(f"det(I₃) = {det_3x3(I3)}")

    # ========================================
    # 3. 性質 2：列交換變號
    # ========================================
    print_separator("3. 性質 2：列交換變號")

    A = [[1, 2], [3, 4]]
    A_swap = swap_rows(A, 0, 1)

    print_matrix("A", A)
    print(f"det(A) = {det_2x2(A)}")

    print_matrix("\nA（交換列 1,2）", A_swap)
    print(f"det(交換後) = {det_2x2(A_swap)}")
    print("驗證：det 變號 ✓")

    # ========================================
    # 4. 性質 3：列加法不變
    # ========================================
    print_separator("4. 性質 3：r_i ← r_i + c·r_j 不變")

    A = [[1, 2], [3, 4]]
    A_add = add_row_multiple(A, 1, 0, -3)  # r2 <- r2 - 3*r1

    print_matrix("A", A)
    print(f"det(A) = {det_2x2(A)}")

    print_matrix("\nA（r₂ ← r₂ - 3r₁）", A_add)
    print(f"det(列運算後) = {det_2x2(A_add)}")
    print("驗證：det 不變 ✓")

    # ========================================
    # 5. 乘積公式：det(AB) = det(A)det(B)
    # ========================================
    print_separator("5. 乘積公式：det(AB) = det(A)det(B)")

    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    AB = matrix_multiply(A, B)

    print_matrix("A", A)
    print_matrix("B", B)
    print_matrix("AB", AB)

    det_A = det_2x2(A)
    det_B = det_2x2(B)
    det_AB = det_2x2(AB)

    print(f"\ndet(A) = {det_A}")
    print(f"det(B) = {det_B}")
    print(f"det(A)·det(B) = {det_A * det_B}")
    print(f"det(AB) = {det_AB}")
    print(f"驗證：{det_A * det_B} = {det_AB} ✓")

    # ========================================
    # 6. 轉置公式：det(Aᵀ) = det(A)
    # ========================================
    print_separator("6. 轉置公式：det(Aᵀ) = det(A)")

    A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    AT = transpose(A)

    print_matrix("A", A)
    print_matrix("Aᵀ", AT)

    print(f"\ndet(A) = {det_3x3(A)}")
    print(f"det(Aᵀ) = {det_3x3(AT)}")
    print("驗證：相等 ✓")

    # ========================================
    # 7. 純量乘法：det(cA) = cⁿdet(A)
    # ========================================
    print_separator("7. 純量乘法：det(cA) = cⁿ·det(A)")

    A = [[1, 2], [3, 4]]
    c = 2
    cA = scalar_multiply_matrix(c, A)

    print_matrix("A (2×2)", A)
    print(f"c = {c}")
    print_matrix("cA", cA)

    det_A = det_2x2(A)
    det_cA = det_2x2(cA)
    n = 2

    print(f"\ndet(A) = {det_A}")
    print(f"cⁿ·det(A) = {c}² × {det_A} = {c**n * det_A}")
    print(f"det(cA) = {det_cA}")
    print(f"驗證：{c**n * det_A} = {det_cA} ✓")

    # ========================================
    # 8. 奇異矩陣
    # ========================================
    print_separator("8. 奇異矩陣：det(A) = 0")

    A_singular = [[1, 2], [2, 4]]  # 列成比例
    print_matrix("A（列成比例）", A_singular)
    print(f"det(A) = {det_2x2(A_singular)}")
    print("此矩陣不可逆（奇異）")

    # ========================================
    # 9. 上三角矩陣
    # ========================================
    print_separator("9. 上三角矩陣：det = 對角線乘積")

    U = [[2, 3, 1], [0, 4, 5], [0, 0, 6]]
    print_matrix("U（上三角）", U)
    print(f"對角線乘積：2 × 4 × 6 = {2*4*6}")
    print(f"det(U) = {det_3x3(U)}")

    # 總結
    print_separator("總結")
    print("""
行列式三大性質：
1. det(I) = 1
2. 列交換 → det 變號
3. 對單列線性

重要公式：
- det(AB) = det(A)·det(B)
- det(Aᵀ) = det(A)
- det(A⁻¹) = 1/det(A)
- det(cA) = cⁿ·det(A)

可逆判定：
- A 可逆 ⟺ det(A) ≠ 0
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
