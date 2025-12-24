"""
反矩陣：手刻版本 (Inverse Matrix: Manual Implementation)

本程式示範：
1. 2×2 矩陣反矩陣公式
2. 高斯-乔丹消去法求反矩陣
3. 反矩陣的性質驗證

This program demonstrates computing inverse matrices using
the 2x2 formula and Gauss-Jordan elimination.
"""  # EN: Execute statement: """.

from typing import List, Optional  # EN: Import symbol(s) from a module: from typing import List, Optional.
import copy  # EN: Import module(s): import copy.

Matrix = List[List[float]]  # EN: Assign Matrix from expression: List[List[float]].


def print_matrix(name: str, A: Matrix, augmented: bool = False) -> None:  # EN: Define print_matrix and its behavior.
    """印出矩陣"""  # EN: Execute statement: """印出矩陣""".
    rows = len(A)  # EN: Assign rows from expression: len(A).
    cols = len(A[0]) if rows > 0 else 0  # EN: Assign cols from expression: len(A[0]) if rows > 0 else 0.

    print(f"{name}:")  # EN: Print formatted output to the console.
    for row in A:  # EN: Iterate with a for-loop: for row in A:.
        print("  [", end="")  # EN: Print formatted output to the console.
        for j, val in enumerate(row):  # EN: Iterate with a for-loop: for j, val in enumerate(row):.
            if augmented and j == cols // 2:  # EN: Branch on a condition: if augmented and j == cols // 2:.
                print(" |", end="")  # EN: Print formatted output to the console.
            print(f"{val:8.4f}", end="")  # EN: Print formatted output to the console.
        print(" ]")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def matrix_multiply(A: Matrix, B: Matrix) -> Matrix:  # EN: Define matrix_multiply and its behavior.
    """矩陣乘法"""  # EN: Execute statement: """矩陣乘法""".
    rows_A, cols_A = len(A), len(A[0])  # EN: Execute statement: rows_A, cols_A = len(A), len(A[0]).
    rows_B, cols_B = len(B), len(B[0])  # EN: Execute statement: rows_B, cols_B = len(B), len(B[0]).

    result = [[0.0] * cols_B for _ in range(rows_A)]  # EN: Assign result from expression: [[0.0] * cols_B for _ in range(rows_A)].
    for i in range(rows_A):  # EN: Iterate with a for-loop: for i in range(rows_A):.
        for j in range(cols_B):  # EN: Iterate with a for-loop: for j in range(cols_B):.
            for k in range(cols_A):  # EN: Iterate with a for-loop: for k in range(cols_A):.
                result[i][j] += A[i][k] * B[k][j]  # EN: Execute statement: result[i][j] += A[i][k] * B[k][j].
    return result  # EN: Return a value: return result.


def inverse_2x2(A: Matrix) -> Optional[Matrix]:  # EN: Define inverse_2x2 and its behavior.
    """
    2×2 矩陣的反矩陣公式

    A = [a  b]        A⁻¹ = 1/(ad-bc) × [d  -b]
        [c  d]                          [-c   a]
    """  # EN: Execute statement: """.
    a, b = A[0][0], A[0][1]  # EN: Execute statement: a, b = A[0][0], A[0][1].
    c, d = A[1][0], A[1][1]  # EN: Execute statement: c, d = A[1][0], A[1][1].

    det = a * d - b * c  # EN: Assign det from expression: a * d - b * c.

    if abs(det) < 1e-12:  # EN: Branch on a condition: if abs(det) < 1e-12:.
        print(f"行列式 = {det:.6f}，矩陣不可逆")  # EN: Print formatted output to the console.
        return None  # EN: Return a value: return None.

    return [  # EN: Return a value: return [.
        [d / det, -b / det],  # EN: Execute statement: [d / det, -b / det],.
        [-c / det, a / det]  # EN: Execute statement: [-c / det, a / det].
    ]  # EN: Execute statement: ].


def gauss_jordan_inverse(A: Matrix, verbose: bool = True) -> Optional[Matrix]:  # EN: Define gauss_jordan_inverse and its behavior.
    """
    高斯-乔丹消去法求反矩陣 (Gauss-Jordan Elimination)

    將 [A | I] 化簡為 [I | A⁻¹]

    Parameters:
        A: n×n 方陣
        verbose: 是否印出過程

    Returns:
        A⁻¹ 或 None（若不可逆）
    """  # EN: Execute statement: """.
    n = len(A)  # EN: Assign n from expression: len(A).

    # 建立增廣矩陣 [A | I]
    augmented = []  # EN: Assign augmented from expression: [].
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        row = A[i].copy()  # EN: Assign row from expression: A[i].copy().
        for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
            row.append(1.0 if i == j else 0.0)  # EN: Execute statement: row.append(1.0 if i == j else 0.0).
        augmented.append(row)  # EN: Execute statement: augmented.append(row).

    if verbose:  # EN: Branch on a condition: if verbose:.
        print_separator("高斯-乔丹消去法求反矩陣")  # EN: Call print_separator(...) to perform an operation.
        print("初始增廣矩陣 [A | I]：")  # EN: Print formatted output to the console.
        print_matrix("", augmented, augmented=True)  # EN: Call print_matrix(...) to perform an operation.

    # 前進消去 + 回消（同時進行）
    for k in range(n):  # EN: Iterate with a for-loop: for k in range(n):.
        # 部分選主元
        max_row = k  # EN: Assign max_row from expression: k.
        max_val = abs(augmented[k][k])  # EN: Assign max_val from expression: abs(augmented[k][k]).
        for i in range(k + 1, n):  # EN: Iterate with a for-loop: for i in range(k + 1, n):.
            if abs(augmented[i][k]) > max_val:  # EN: Branch on a condition: if abs(augmented[i][k]) > max_val:.
                max_val = abs(augmented[i][k])  # EN: Assign max_val from expression: abs(augmented[i][k]).
                max_row = i  # EN: Assign max_row from expression: i.

        if max_val < 1e-12:  # EN: Branch on a condition: if max_val < 1e-12:.
            print(f"第 {k+1} 個主元為零，矩陣不可逆")  # EN: Print formatted output to the console.
            return None  # EN: Return a value: return None.

        # 換列
        if max_row != k:  # EN: Branch on a condition: if max_row != k:.
            augmented[k], augmented[max_row] = augmented[max_row], augmented[k]  # EN: Execute statement: augmented[k], augmented[max_row] = augmented[max_row], augmented[k].
            if verbose:  # EN: Branch on a condition: if verbose:.
                print(f"交換第 {k+1} 列和第 {max_row+1} 列")  # EN: Print formatted output to the console.

        # 將主元化為 1
        pivot = augmented[k][k]  # EN: Assign pivot from expression: augmented[k][k].
        for j in range(2 * n):  # EN: Iterate with a for-loop: for j in range(2 * n):.
            augmented[k][j] /= pivot  # EN: Execute statement: augmented[k][j] /= pivot.

        if verbose:  # EN: Branch on a condition: if verbose:.
            print(f"R{k+1} ← R{k+1} / {pivot:.4f}")  # EN: Print formatted output to the console.

        # 消去該行的其他元素（上下都要消）
        for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
            if i != k:  # EN: Branch on a condition: if i != k:.
                factor = augmented[i][k]  # EN: Assign factor from expression: augmented[i][k].
                if abs(factor) > 1e-12:  # EN: Branch on a condition: if abs(factor) > 1e-12:.
                    for j in range(2 * n):  # EN: Iterate with a for-loop: for j in range(2 * n):.
                        augmented[i][j] -= factor * augmented[k][j]  # EN: Execute statement: augmented[i][j] -= factor * augmented[k][j].

                    if verbose:  # EN: Branch on a condition: if verbose:.
                        print(f"R{i+1} ← R{i+1} - ({factor:.4f}) × R{k+1}")  # EN: Print formatted output to the console.

        if verbose:  # EN: Branch on a condition: if verbose:.
            print_matrix("", augmented, augmented=True)  # EN: Call print_matrix(...) to perform an operation.

    # 提取右半部分作為 A⁻¹
    A_inv = [[augmented[i][j + n] for j in range(n)] for i in range(n)]  # EN: Assign A_inv from expression: [[augmented[i][j + n] for j in range(n)] for i in range(n)].

    return A_inv  # EN: Return a value: return A_inv.


def verify_inverse(A: Matrix, A_inv: Matrix) -> bool:  # EN: Define verify_inverse and its behavior.
    """驗證 A × A⁻¹ = I"""  # EN: Execute statement: """驗證 A × A⁻¹ = I""".
    n = len(A)  # EN: Assign n from expression: len(A).
    product = matrix_multiply(A, A_inv)  # EN: Assign product from expression: matrix_multiply(A, A_inv).

    # 檢查是否為單位矩陣
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
            expected = 1.0 if i == j else 0.0  # EN: Assign expected from expression: 1.0 if i == j else 0.0.
            if abs(product[i][j] - expected) > 1e-6:  # EN: Branch on a condition: if abs(product[i][j] - expected) > 1e-6:.
                return False  # EN: Return a value: return False.
    return True  # EN: Return a value: return True.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("反矩陣示範\nInverse Matrix Demo")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 範例 1：2×2 反矩陣公式
    # ========================================
    print_separator("範例 1：2×2 反矩陣公式")  # EN: Call print_separator(...) to perform an operation.

    A2 = [  # EN: Assign A2 from expression: [.
        [4.0, 7.0],  # EN: Execute statement: [4.0, 7.0],.
        [2.0, 6.0]  # EN: Execute statement: [2.0, 6.0].
    ]  # EN: Execute statement: ].

    print_matrix("A", A2)  # EN: Call print_matrix(...) to perform an operation.

    det_A2 = A2[0][0] * A2[1][1] - A2[0][1] * A2[1][0]  # EN: Assign det_A2 from expression: A2[0][0] * A2[1][1] - A2[0][1] * A2[1][0].
    print(f"det(A) = 4×6 - 7×2 = {det_A2}")  # EN: Print formatted output to the console.

    A2_inv = inverse_2x2(A2)  # EN: Assign A2_inv from expression: inverse_2x2(A2).
    if A2_inv:  # EN: Branch on a condition: if A2_inv:.
        print_matrix("A⁻¹", A2_inv)  # EN: Call print_matrix(...) to perform an operation.

        print("驗證 A × A⁻¹ =")  # EN: Print formatted output to the console.
        product = matrix_multiply(A2, A2_inv)  # EN: Assign product from expression: matrix_multiply(A2, A2_inv).
        print_matrix("", product)  # EN: Call print_matrix(...) to perform an operation.
        print(f"是單位矩陣？ {verify_inverse(A2, A2_inv)}")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 2：高斯-乔丹消去法 3×3
    # ========================================
    print_separator("範例 2：高斯-乔丹消去法 3×3")  # EN: Call print_separator(...) to perform an operation.

    A3 = [  # EN: Assign A3 from expression: [.
        [2.0, 1.0, 1.0],  # EN: Execute statement: [2.0, 1.0, 1.0],.
        [4.0, -6.0, 0.0],  # EN: Execute statement: [4.0, -6.0, 0.0],.
        [-2.0, 7.0, 2.0]  # EN: Execute statement: [-2.0, 7.0, 2.0].
    ]  # EN: Execute statement: ].

    print_matrix("A", A3)  # EN: Call print_matrix(...) to perform an operation.

    A3_inv = gauss_jordan_inverse(A3, verbose=True)  # EN: Assign A3_inv from expression: gauss_jordan_inverse(A3, verbose=True).

    if A3_inv:  # EN: Branch on a condition: if A3_inv:.
        print_separator("結果")  # EN: Call print_separator(...) to perform an operation.
        print_matrix("A⁻¹", A3_inv)  # EN: Call print_matrix(...) to perform an operation.

        print("驗證 A × A⁻¹ =")  # EN: Print formatted output to the console.
        product = matrix_multiply(A3, A3_inv)  # EN: Assign product from expression: matrix_multiply(A3, A3_inv).
        print_matrix("", product)  # EN: Call print_matrix(...) to perform an operation.
        print(f"是單位矩陣？ {verify_inverse(A3, A3_inv)}")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 3：奇異矩陣（不可逆）
    # ========================================
    print_separator("範例 3：奇異矩陣（不可逆）")  # EN: Call print_separator(...) to perform an operation.

    A_singular = [  # EN: Assign A_singular from expression: [.
        [1.0, 2.0, 3.0],  # EN: Execute statement: [1.0, 2.0, 3.0],.
        [2.0, 4.0, 6.0],  # 第一列的 2 倍  # EN: Execute statement: [2.0, 4.0, 6.0], # 第一列的 2 倍.
        [1.0, 3.0, 4.0]  # EN: Execute statement: [1.0, 3.0, 4.0].
    ]  # EN: Execute statement: ].

    print_matrix("A（奇異矩陣）", A_singular)  # EN: Call print_matrix(...) to perform an operation.
    print("第二列是第一列的 2 倍，所以矩陣不滿秩")  # EN: Print formatted output to the console.

    result = gauss_jordan_inverse(A_singular, verbose=True)  # EN: Assign result from expression: gauss_jordan_inverse(A_singular, verbose=True).
    if result is None:  # EN: Branch on a condition: if result is None:.
        print("確認：矩陣不可逆")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 4：反矩陣的性質
    # ========================================
    print_separator("範例 4：反矩陣的性質")  # EN: Call print_separator(...) to perform an operation.

    B = [  # EN: Assign B from expression: [.
        [1.0, 2.0],  # EN: Execute statement: [1.0, 2.0],.
        [3.0, 4.0]  # EN: Execute statement: [3.0, 4.0].
    ]  # EN: Execute statement: ].

    C = [  # EN: Assign C from expression: [.
        [2.0, 0.0],  # EN: Execute statement: [2.0, 0.0],.
        [1.0, 2.0]  # EN: Execute statement: [1.0, 2.0].
    ]  # EN: Execute statement: ].

    print_matrix("B", B)  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("C", C)  # EN: Call print_matrix(...) to perform an operation.

    B_inv = inverse_2x2(B)  # EN: Assign B_inv from expression: inverse_2x2(B).
    C_inv = inverse_2x2(C)  # EN: Assign C_inv from expression: inverse_2x2(C).

    if B_inv and C_inv:  # EN: Branch on a condition: if B_inv and C_inv:.
        # 性質：(BC)⁻¹ = C⁻¹B⁻¹
        BC = matrix_multiply(B, C)  # EN: Assign BC from expression: matrix_multiply(B, C).
        BC_inv = inverse_2x2(BC)  # EN: Assign BC_inv from expression: inverse_2x2(BC).
        C_inv_B_inv = matrix_multiply(C_inv, B_inv)  # EN: Assign C_inv_B_inv from expression: matrix_multiply(C_inv, B_inv).

        print("驗證 (BC)⁻¹ = C⁻¹B⁻¹：")  # EN: Print formatted output to the console.
        print_matrix("(BC)⁻¹", BC_inv)  # EN: Call print_matrix(...) to perform an operation.
        print_matrix("C⁻¹B⁻¹", C_inv_B_inv)  # EN: Call print_matrix(...) to perform an operation.

        # 性質：(B⁻¹)⁻¹ = B
        B_inv_inv = inverse_2x2(B_inv)  # EN: Assign B_inv_inv from expression: inverse_2x2(B_inv).
        print("驗證 (B⁻¹)⁻¹ = B：")  # EN: Print formatted output to the console.
        print_matrix("(B⁻¹)⁻¹", B_inv_inv)  # EN: Call print_matrix(...) to perform an operation.
        print_matrix("B", B)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 範例 5：用反矩陣解方程組（不推薦）
    # ========================================
    print_separator("範例 5：用反矩陣解 Ax = b（僅作示範）")  # EN: Call print_separator(...) to perform an operation.

    A = [  # EN: Assign A from expression: [.
        [2.0, 1.0],  # EN: Execute statement: [2.0, 1.0],.
        [5.0, 3.0]  # EN: Execute statement: [5.0, 3.0].
    ]  # EN: Execute statement: ].
    b = [4.0, 11.0]  # EN: Assign b from expression: [4.0, 11.0].

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print(f"b = {b}")  # EN: Print formatted output to the console.

    A_inv = inverse_2x2(A)  # EN: Assign A_inv from expression: inverse_2x2(A).
    if A_inv:  # EN: Branch on a condition: if A_inv:.
        # x = A⁻¹ × b
        x = [  # EN: Assign x from expression: [.
            A_inv[0][0] * b[0] + A_inv[0][1] * b[1],  # EN: Execute statement: A_inv[0][0] * b[0] + A_inv[0][1] * b[1],.
            A_inv[1][0] * b[0] + A_inv[1][1] * b[1]  # EN: Execute statement: A_inv[1][0] * b[0] + A_inv[1][1] * b[1].
        ]  # EN: Execute statement: ].

        print(f"\nx = A⁻¹ × b = {[f'{xi:.4f}' for xi in x]}")  # EN: Print formatted output to the console.

        # 驗證
        Ax = [  # EN: Assign Ax from expression: [.
            A[0][0] * x[0] + A[0][1] * x[1],  # EN: Execute statement: A[0][0] * x[0] + A[0][1] * x[1],.
            A[1][0] * x[0] + A[1][1] * x[1]  # EN: Execute statement: A[1][0] * x[0] + A[1][1] * x[1].
        ]  # EN: Execute statement: ].
        print(f"驗證 Ax = {[f'{axi:.4f}' for axi in Ax]}")  # EN: Print formatted output to the console.

    print("""
⚠️ 注意：實際應用中不推薦這樣做！
   應該使用高斯消去法或 LU 分解求解。
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("反矩陣示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
