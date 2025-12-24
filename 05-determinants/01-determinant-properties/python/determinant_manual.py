"""
行列式的性質 - 手刻版本 (Determinant Properties - Manual Implementation)

本程式示範：
1. 2×2, 3×3 行列式計算
2. 行列式的性質驗證
3. 列運算對行列式的影響
"""  # EN: Execute statement: """.

from typing import List  # EN: Import symbol(s) from a module: from typing import List.


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
# 行列式計算
# ========================================

def det_2x2(A: List[List[float]]) -> float:  # EN: Define det_2x2 and its behavior.
    """計算 2×2 行列式"""  # EN: Execute statement: """計算 2×2 行列式""".
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]  # EN: Return a value: return A[0][0] * A[1][1] - A[0][1] * A[1][0].


def det_3x3(A: List[List[float]]) -> float:  # EN: Define det_3x3 and its behavior.
    """計算 3×3 行列式（Sarrus 法則或展開）"""  # EN: Execute statement: """計算 3×3 行列式（Sarrus 法則或展開）""".
    a, b, c = A[0]  # EN: Execute statement: a, b, c = A[0].
    d, e, f = A[1]  # EN: Execute statement: d, e, f = A[1].
    g, h, i = A[2]  # EN: Execute statement: g, h, i = A[2].

    return (a * e * i + b * f * g + c * d * h  # EN: Return a value: return (a * e * i + b * f * g + c * d * h.
            - c * e * g - b * d * i - a * f * h)  # EN: Execute statement: - c * e * g - b * d * i - a * f * h).


def det_nxn(A: List[List[float]]) -> float:  # EN: Define det_nxn and its behavior.
    """
    計算 n×n 行列式（列運算化為上三角）

    返回行列式值
    """  # EN: Execute statement: """.
    n = len(A)  # EN: Assign n from expression: len(A).

    # 複製矩陣
    M = [row[:] for row in A]  # EN: Assign M from expression: [row[:] for row in A].

    sign = 1  # 追蹤列交換次數  # EN: Assign sign from expression: 1 # 追蹤列交換次數.

    for col in range(n):  # EN: Iterate with a for-loop: for col in range(n):.
        # 找主元（pivot）
        pivot_row = None  # EN: Assign pivot_row from expression: None.
        for row in range(col, n):  # EN: Iterate with a for-loop: for row in range(col, n):.
            if abs(M[row][col]) > 1e-10:  # EN: Branch on a condition: if abs(M[row][col]) > 1e-10:.
                pivot_row = row  # EN: Assign pivot_row from expression: row.
                break  # EN: Control flow statement: break.

        if pivot_row is None:  # EN: Branch on a condition: if pivot_row is None:.
            return 0.0  # 奇異矩陣  # EN: Return a value: return 0.0 # 奇異矩陣.

        # 列交換
        if pivot_row != col:  # EN: Branch on a condition: if pivot_row != col:.
            M[col], M[pivot_row] = M[pivot_row], M[col]  # EN: Execute statement: M[col], M[pivot_row] = M[pivot_row], M[col].
            sign *= -1  # EN: Update sign via *= using: -1.

        # 消去下方元素
        for row in range(col + 1, n):  # EN: Iterate with a for-loop: for row in range(col + 1, n):.
            if abs(M[col][col]) > 1e-10:  # EN: Branch on a condition: if abs(M[col][col]) > 1e-10:.
                factor = M[row][col] / M[col][col]  # EN: Assign factor from expression: M[row][col] / M[col][col].
                for j in range(col, n):  # EN: Iterate with a for-loop: for j in range(col, n):.
                    M[row][j] -= factor * M[col][j]  # EN: Execute statement: M[row][j] -= factor * M[col][j].

    # 對角線乘積
    det = sign  # EN: Assign det from expression: sign.
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        det *= M[i][i]  # EN: Update det via *= using: M[i][i].

    return det  # EN: Return a value: return det.


# ========================================
# 矩陣運算
# ========================================

def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:  # EN: Define matrix_multiply and its behavior.
    """矩陣乘法"""  # EN: Execute statement: """矩陣乘法""".
    m, k, n = len(A), len(B), len(B[0])  # EN: Execute statement: m, k, n = len(A), len(B), len(B[0]).
    result = [[0.0] * n for _ in range(m)]  # EN: Assign result from expression: [[0.0] * n for _ in range(m)].
    for i in range(m):  # EN: Iterate with a for-loop: for i in range(m):.
        for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
            for p in range(k):  # EN: Iterate with a for-loop: for p in range(k):.
                result[i][j] += A[i][p] * B[p][j]  # EN: Execute statement: result[i][j] += A[i][p] * B[p][j].
    return result  # EN: Return a value: return result.


def transpose(A: List[List[float]]) -> List[List[float]]:  # EN: Define transpose and its behavior.
    """矩陣轉置"""  # EN: Execute statement: """矩陣轉置""".
    m, n = len(A), len(A[0])  # EN: Execute statement: m, n = len(A), len(A[0]).
    return [[A[i][j] for i in range(m)] for j in range(n)]  # EN: Return a value: return [[A[i][j] for i in range(m)] for j in range(n)].


def scalar_multiply_matrix(c: float, A: List[List[float]]) -> List[List[float]]:  # EN: Define scalar_multiply_matrix and its behavior.
    """純量乘矩陣"""  # EN: Execute statement: """純量乘矩陣""".
    return [[c * x for x in row] for row in A]  # EN: Return a value: return [[c * x for x in row] for row in A].


def swap_rows(A: List[List[float]], i: int, j: int) -> List[List[float]]:  # EN: Define swap_rows and its behavior.
    """交換列"""  # EN: Execute statement: """交換列""".
    result = [row[:] for row in A]  # EN: Assign result from expression: [row[:] for row in A].
    result[i], result[j] = result[j], result[i]  # EN: Execute statement: result[i], result[j] = result[j], result[i].
    return result  # EN: Return a value: return result.


def add_row_multiple(A: List[List[float]], target: int, source: int, c: float) -> List[List[float]]:  # EN: Define add_row_multiple and its behavior.
    """列運算：target 列 += c * source 列"""  # EN: Execute statement: """列運算：target 列 += c * source 列""".
    result = [row[:] for row in A]  # EN: Assign result from expression: [row[:] for row in A].
    for j in range(len(A[0])):  # EN: Iterate with a for-loop: for j in range(len(A[0])):.
        result[target][j] += c * result[source][j]  # EN: Execute statement: result[target][j] += c * result[source][j].
    return result  # EN: Return a value: return result.


def main():  # EN: Define main and its behavior.
    print_separator("行列式性質示範（手刻版）\nDeterminant Properties Demo (Manual)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 基本行列式計算
    # ========================================
    print_separator("1. 基本行列式計算")  # EN: Call print_separator(...) to perform an operation.

    # 2×2
    A2 = [[3, 8], [4, 6]]  # EN: Assign A2 from expression: [[3, 8], [4, 6]].
    print_matrix("A (2×2)", A2)  # EN: Call print_matrix(...) to perform an operation.
    print(f"det(A) = {A2[0][0]}×{A2[1][1]} - {A2[0][1]}×{A2[1][0]} = {det_2x2(A2)}")  # EN: Print formatted output to the console.

    # 3×3
    A3 = [  # EN: Assign A3 from expression: [.
        [1, 2, 3],  # EN: Execute statement: [1, 2, 3],.
        [4, 5, 6],  # EN: Execute statement: [4, 5, 6],.
        [7, 8, 10]  # EN: Execute statement: [7, 8, 10].
    ]  # EN: Execute statement: ].
    print_matrix("\nA (3×3)", A3)  # EN: Call print_matrix(...) to perform an operation.
    print(f"det(A) = {det_3x3(A3)}")  # EN: Print formatted output to the console.

    # n×n（使用列運算）
    print(f"使用列運算驗證：{det_nxn(A3):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 性質 1：det(I) = 1
    # ========================================
    print_separator("2. 性質 1：det(I) = 1")  # EN: Call print_separator(...) to perform an operation.

    I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # EN: Assign I3 from expression: [[1, 0, 0], [0, 1, 0], [0, 0, 1]].
    print_matrix("I₃", I3)  # EN: Call print_matrix(...) to perform an operation.
    print(f"det(I₃) = {det_3x3(I3)}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 性質 2：列交換變號
    # ========================================
    print_separator("3. 性質 2：列交換變號")  # EN: Call print_separator(...) to perform an operation.

    A = [[1, 2], [3, 4]]  # EN: Assign A from expression: [[1, 2], [3, 4]].
    A_swap = swap_rows(A, 0, 1)  # EN: Assign A_swap from expression: swap_rows(A, 0, 1).

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print(f"det(A) = {det_2x2(A)}")  # EN: Print formatted output to the console.

    print_matrix("\nA（交換列 1,2）", A_swap)  # EN: Call print_matrix(...) to perform an operation.
    print(f"det(交換後) = {det_2x2(A_swap)}")  # EN: Print formatted output to the console.
    print("驗證：det 變號 ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 性質 3：列加法不變
    # ========================================
    print_separator("4. 性質 3：r_i ← r_i + c·r_j 不變")  # EN: Call print_separator(...) to perform an operation.

    A = [[1, 2], [3, 4]]  # EN: Assign A from expression: [[1, 2], [3, 4]].
    A_add = add_row_multiple(A, 1, 0, -3)  # r2 <- r2 - 3*r1  # EN: Assign A_add from expression: add_row_multiple(A, 1, 0, -3) # r2 <- r2 - 3*r1.

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print(f"det(A) = {det_2x2(A)}")  # EN: Print formatted output to the console.

    print_matrix("\nA（r₂ ← r₂ - 3r₁）", A_add)  # EN: Call print_matrix(...) to perform an operation.
    print(f"det(列運算後) = {det_2x2(A_add)}")  # EN: Print formatted output to the console.
    print("驗證：det 不變 ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 乘積公式：det(AB) = det(A)det(B)
    # ========================================
    print_separator("5. 乘積公式：det(AB) = det(A)det(B)")  # EN: Call print_separator(...) to perform an operation.

    A = [[1, 2], [3, 4]]  # EN: Assign A from expression: [[1, 2], [3, 4]].
    B = [[5, 6], [7, 8]]  # EN: Assign B from expression: [[5, 6], [7, 8]].
    AB = matrix_multiply(A, B)  # EN: Assign AB from expression: matrix_multiply(A, B).

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("B", B)  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("AB", AB)  # EN: Call print_matrix(...) to perform an operation.

    det_A = det_2x2(A)  # EN: Assign det_A from expression: det_2x2(A).
    det_B = det_2x2(B)  # EN: Assign det_B from expression: det_2x2(B).
    det_AB = det_2x2(AB)  # EN: Assign det_AB from expression: det_2x2(AB).

    print(f"\ndet(A) = {det_A}")  # EN: Print formatted output to the console.
    print(f"det(B) = {det_B}")  # EN: Print formatted output to the console.
    print(f"det(A)·det(B) = {det_A * det_B}")  # EN: Print formatted output to the console.
    print(f"det(AB) = {det_AB}")  # EN: Print formatted output to the console.
    print(f"驗證：{det_A * det_B} = {det_AB} ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 轉置公式：det(Aᵀ) = det(A)
    # ========================================
    print_separator("6. 轉置公式：det(Aᵀ) = det(A)")  # EN: Call print_separator(...) to perform an operation.

    A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]  # EN: Assign A from expression: [[1, 2, 3], [4, 5, 6], [7, 8, 10]].
    AT = transpose(A)  # EN: Assign AT from expression: transpose(A).

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("Aᵀ", AT)  # EN: Call print_matrix(...) to perform an operation.

    print(f"\ndet(A) = {det_3x3(A)}")  # EN: Print formatted output to the console.
    print(f"det(Aᵀ) = {det_3x3(AT)}")  # EN: Print formatted output to the console.
    print("驗證：相等 ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 純量乘法：det(cA) = cⁿdet(A)
    # ========================================
    print_separator("7. 純量乘法：det(cA) = cⁿ·det(A)")  # EN: Call print_separator(...) to perform an operation.

    A = [[1, 2], [3, 4]]  # EN: Assign A from expression: [[1, 2], [3, 4]].
    c = 2  # EN: Assign c from expression: 2.
    cA = scalar_multiply_matrix(c, A)  # EN: Assign cA from expression: scalar_multiply_matrix(c, A).

    print_matrix("A (2×2)", A)  # EN: Call print_matrix(...) to perform an operation.
    print(f"c = {c}")  # EN: Print formatted output to the console.
    print_matrix("cA", cA)  # EN: Call print_matrix(...) to perform an operation.

    det_A = det_2x2(A)  # EN: Assign det_A from expression: det_2x2(A).
    det_cA = det_2x2(cA)  # EN: Assign det_cA from expression: det_2x2(cA).
    n = 2  # EN: Assign n from expression: 2.

    print(f"\ndet(A) = {det_A}")  # EN: Print formatted output to the console.
    print(f"cⁿ·det(A) = {c}² × {det_A} = {c**n * det_A}")  # EN: Print formatted output to the console.
    print(f"det(cA) = {det_cA}")  # EN: Print formatted output to the console.
    print(f"驗證：{c**n * det_A} = {det_cA} ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 奇異矩陣
    # ========================================
    print_separator("8. 奇異矩陣：det(A) = 0")  # EN: Call print_separator(...) to perform an operation.

    A_singular = [[1, 2], [2, 4]]  # 列成比例  # EN: Assign A_singular from expression: [[1, 2], [2, 4]] # 列成比例.
    print_matrix("A（列成比例）", A_singular)  # EN: Call print_matrix(...) to perform an operation.
    print(f"det(A) = {det_2x2(A_singular)}")  # EN: Print formatted output to the console.
    print("此矩陣不可逆（奇異）")  # EN: Print formatted output to the console.

    # ========================================
    # 9. 上三角矩陣
    # ========================================
    print_separator("9. 上三角矩陣：det = 對角線乘積")  # EN: Call print_separator(...) to perform an operation.

    U = [[2, 3, 1], [0, 4, 5], [0, 0, 6]]  # EN: Assign U from expression: [[2, 3, 1], [0, 4, 5], [0, 0, 6]].
    print_matrix("U（上三角）", U)  # EN: Call print_matrix(...) to perform an operation.
    print(f"對角線乘積：2 × 4 × 6 = {2*4*6}")  # EN: Print formatted output to the console.
    print(f"det(U) = {det_3x3(U)}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
