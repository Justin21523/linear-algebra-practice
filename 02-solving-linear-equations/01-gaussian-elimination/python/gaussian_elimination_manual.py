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
"""  # EN: Execute statement: """.

from typing import List, Tuple, Optional  # EN: Import symbol(s) from a module: from typing import List, Tuple, Optional.
import copy  # EN: Import module(s): import copy.

# 型別別名 (Type alias)
Matrix = List[List[float]]  # EN: Assign Matrix from expression: List[List[float]].
Vector = List[float]  # EN: Assign Vector from expression: List[float].


def print_matrix(name: str, A: Matrix, augmented: bool = False) -> None:  # EN: Define print_matrix and its behavior.
    """
    印出矩陣 (Print matrix)
    augmented=True 時，最後一行顯示為增廣部分
    """  # EN: Execute statement: """.
    rows = len(A)  # EN: Assign rows from expression: len(A).
    cols = len(A[0]) if rows > 0 else 0  # EN: Assign cols from expression: len(A[0]) if rows > 0 else 0.

    print(f"{name}:")  # EN: Print formatted output to the console.
    for i, row in enumerate(A):  # EN: Iterate with a for-loop: for i, row in enumerate(A):.
        print("  [", end="")  # EN: Print formatted output to the console.
        for j, val in enumerate(row):  # EN: Iterate with a for-loop: for j, val in enumerate(row):.
            if augmented and j == cols - 1:  # EN: Branch on a condition: if augmented and j == cols - 1:.
                print(" |", end="")  # EN: Print formatted output to the console.
            print(f"{val:8.4f}", end="")  # EN: Print formatted output to the console.
        print(" ]")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線 (Print separator)"""  # EN: Execute statement: """印出分隔線 (Print separator)""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def create_augmented_matrix(A: Matrix, b: Vector) -> Matrix:  # EN: Define create_augmented_matrix and its behavior.
    """
    建立增廣矩陣 [A | b]
    Create augmented matrix [A | b]
    """  # EN: Execute statement: """.
    n = len(A)  # EN: Assign n from expression: len(A).
    augmented = []  # EN: Assign augmented from expression: [].
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        row = A[i].copy()  # EN: Assign row from expression: A[i].copy().
        row.append(b[i])  # EN: Execute statement: row.append(b[i]).
        augmented.append(row)  # EN: Execute statement: augmented.append(row).
    return augmented  # EN: Return a value: return augmented.


def swap_rows(A: Matrix, i: int, j: int) -> None:  # EN: Define swap_rows and its behavior.
    """
    交換第 i 列和第 j 列 (Swap rows i and j)
    """  # EN: Execute statement: """.
    A[i], A[j] = A[j], A[i]  # EN: Execute statement: A[i], A[j] = A[j], A[i].


def find_pivot_row(A: Matrix, col: int, start_row: int) -> int:  # EN: Define find_pivot_row and its behavior.
    """
    部分選主元：找出從 start_row 開始，第 col 行中絕對值最大的元素所在列
    Partial pivoting: Find row with largest absolute value in column
    """  # EN: Execute statement: """.
    n = len(A)  # EN: Assign n from expression: len(A).
    max_val = abs(A[start_row][col])  # EN: Assign max_val from expression: abs(A[start_row][col]).
    max_row = start_row  # EN: Assign max_row from expression: start_row.

    for i in range(start_row + 1, n):  # EN: Iterate with a for-loop: for i in range(start_row + 1, n):.
        if abs(A[i][col]) > max_val:  # EN: Branch on a condition: if abs(A[i][col]) > max_val:.
            max_val = abs(A[i][col])  # EN: Assign max_val from expression: abs(A[i][col]).
            max_row = i  # EN: Assign max_row from expression: i.

    return max_row  # EN: Return a value: return max_row.


def forward_elimination(A: Matrix, verbose: bool = True) -> Tuple[Matrix, bool]:  # EN: Define forward_elimination and its behavior.
    """
    前進消去 (Forward Elimination)

    將增廣矩陣化為上三角形式

    Parameters:
        A: 增廣矩陣 [A | b]
        verbose: 是否印出過程

    Returns:
        (上三角矩陣, 是否成功)
    """  # EN: Execute statement: """.
    n = len(A)  # EN: Assign n from expression: len(A).
    U = copy.deepcopy(A)  # 不修改原矩陣  # EN: Assign U from expression: copy.deepcopy(A) # 不修改原矩陣.

    if verbose:  # EN: Branch on a condition: if verbose:.
        print_separator("前進消去 (Forward Elimination)")  # EN: Call print_separator(...) to perform an operation.
        print("初始增廣矩陣：")  # EN: Print formatted output to the console.
        print_matrix("", U, augmented=True)  # EN: Call print_matrix(...) to perform an operation.

    for k in range(n - 1):  # 對每一行（除了最後一行）  # EN: Iterate with a for-loop: for k in range(n - 1): # 對每一行（除了最後一行）.
        if verbose:  # EN: Branch on a condition: if verbose:.
            print(f"--- 步驟 {k + 1}：消去第 {k + 1} 行以下的元素 ---")  # EN: Print formatted output to the console.

        # 部分選主元 (Partial pivoting)
        pivot_row = find_pivot_row(U, k, k)  # EN: Assign pivot_row from expression: find_pivot_row(U, k, k).

        if abs(U[pivot_row][k]) < 1e-12:  # EN: Branch on a condition: if abs(U[pivot_row][k]) < 1e-12:.
            print(f"警告：主元接近零，矩陣可能奇異")  # EN: Print formatted output to the console.
            return U, False  # EN: Return a value: return U, False.

        if pivot_row != k:  # EN: Branch on a condition: if pivot_row != k:.
            if verbose:  # EN: Branch on a condition: if verbose:.
                print(f"交換第 {k + 1} 列和第 {pivot_row + 1} 列")  # EN: Print formatted output to the console.
            swap_rows(U, k, pivot_row)  # EN: Call swap_rows(...) to perform an operation.
            if verbose:  # EN: Branch on a condition: if verbose:.
                print_matrix("", U, augmented=True)  # EN: Call print_matrix(...) to perform an operation.

        # 消去：使第 k 行以下的元素變為 0
        for i in range(k + 1, n):  # EN: Iterate with a for-loop: for i in range(k + 1, n):.
            if abs(U[i][k]) > 1e-12:  # 只有非零才需要消去  # EN: Branch on a condition: if abs(U[i][k]) > 1e-12: # 只有非零才需要消去.
                multiplier = U[i][k] / U[k][k]  # EN: Assign multiplier from expression: U[i][k] / U[k][k].

                if verbose:  # EN: Branch on a condition: if verbose:.
                    print(f"R{i + 1} ← R{i + 1} - ({multiplier:.4f}) × R{k + 1}")  # EN: Print formatted output to the console.

                # 對整列（包含增廣部分）進行消去
                for j in range(k, len(U[0])):  # EN: Iterate with a for-loop: for j in range(k, len(U[0])):.
                    U[i][j] = U[i][j] - multiplier * U[k][j]  # EN: Execute statement: U[i][j] = U[i][j] - multiplier * U[k][j].

        if verbose:  # EN: Branch on a condition: if verbose:.
            print_matrix("", U, augmented=True)  # EN: Call print_matrix(...) to perform an operation.

    # 檢查最後一個主元
    if abs(U[n - 1][n - 1]) < 1e-12:  # EN: Branch on a condition: if abs(U[n - 1][n - 1]) < 1e-12:.
        if verbose:  # EN: Branch on a condition: if verbose:.
            print("警告：最後一個主元為零，矩陣奇異")  # EN: Print formatted output to the console.
        return U, False  # EN: Return a value: return U, False.

    return U, True  # EN: Return a value: return U, True.


def back_substitution(U: Matrix, verbose: bool = True) -> Optional[Vector]:  # EN: Define back_substitution and its behavior.
    """
    回代 (Back Substitution)

    從上三角系統求解 x

    Parameters:
        U: 上三角增廣矩陣
        verbose: 是否印出過程

    Returns:
        解向量 x，若無法求解則返回 None
    """  # EN: Execute statement: """.
    n = len(U)  # EN: Assign n from expression: len(U).
    x = [0.0] * n  # EN: Assign x from expression: [0.0] * n.

    if verbose:  # EN: Branch on a condition: if verbose:.
        print_separator("回代 (Back Substitution)")  # EN: Call print_separator(...) to perform an operation.

    # 從最後一列開始回代
    for i in range(n - 1, -1, -1):  # EN: Iterate with a for-loop: for i in range(n - 1, -1, -1):.
        if abs(U[i][i]) < 1e-12:  # EN: Branch on a condition: if abs(U[i][i]) < 1e-12:.
            print(f"錯誤：主元 U[{i}][{i}] 為零，無法回代")  # EN: Print formatted output to the console.
            return None  # EN: Return a value: return None.

        # xᵢ = (cᵢ - Σⱼ uᵢⱼ·xⱼ) / uᵢᵢ
        sum_val = 0.0  # EN: Assign sum_val from expression: 0.0.
        for j in range(i + 1, n):  # EN: Iterate with a for-loop: for j in range(i + 1, n):.
            sum_val += U[i][j] * x[j]  # EN: Update sum_val via += using: U[i][j] * x[j].

        x[i] = (U[i][n] - sum_val) / U[i][i]  # EN: Execute statement: x[i] = (U[i][n] - sum_val) / U[i][i].

        if verbose:  # EN: Branch on a condition: if verbose:.
            if i == n - 1:  # EN: Branch on a condition: if i == n - 1:.
                print(f"x{i + 1} = {U[i][n]:.4f} / {U[i][i]:.4f} = {x[i]:.4f}")  # EN: Print formatted output to the console.
            else:  # EN: Execute the fallback branch when prior conditions are false.
                terms = " - ".join(f"{U[i][j]:.4f}×x{j + 1}" for j in range(i + 1, n))  # EN: Assign terms from expression: " - ".join(f"{U[i][j]:.4f}×x{j + 1}" for j in range(i + 1, n)).
                print(f"x{i + 1} = ({U[i][n]:.4f} - ({terms})) / {U[i][i]:.4f} = {x[i]:.4f}")  # EN: Print formatted output to the console.

    return x  # EN: Return a value: return x.


def gaussian_elimination_solve(A: Matrix, b: Vector, verbose: bool = True) -> Optional[Vector]:  # EN: Define gaussian_elimination_solve and its behavior.
    """
    高斯消去法完整求解 (Complete Gaussian Elimination Solver)

    求解 Ax = b

    Parameters:
        A: 係數矩陣
        b: 右手邊向量
        verbose: 是否印出詳細過程

    Returns:
        解向量 x，若無解則返回 None
    """  # EN: Execute statement: """.
    if verbose:  # EN: Branch on a condition: if verbose:.
        print_separator("高斯消去法求解 Ax = b")  # EN: Call print_separator(...) to perform an operation.
        print("原始係數矩陣 A：")  # EN: Print formatted output to the console.
        print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
        print(f"右手邊向量 b = {b}")  # EN: Print formatted output to the console.

    # 建立增廣矩陣
    augmented = create_augmented_matrix(A, b)  # EN: Assign augmented from expression: create_augmented_matrix(A, b).

    # 前進消去
    U, success = forward_elimination(augmented, verbose)  # EN: Execute statement: U, success = forward_elimination(augmented, verbose).

    if not success:  # EN: Branch on a condition: if not success:.
        print("消去失敗：矩陣可能奇異")  # EN: Print formatted output to the console.
        return None  # EN: Return a value: return None.

    # 回代
    x = back_substitution(U, verbose)  # EN: Assign x from expression: back_substitution(U, verbose).

    if x is not None and verbose:  # EN: Branch on a condition: if x is not None and verbose:.
        print_separator("解 (Solution)")  # EN: Call print_separator(...) to perform an operation.
        print(f"x = {[f'{xi:.4f}' for xi in x]}")  # EN: Print formatted output to the console.

    return x  # EN: Return a value: return x.


def verify_solution(A: Matrix, x: Vector, b: Vector) -> float:  # EN: Define verify_solution and its behavior.
    """
    驗證解的正確性 (Verify solution)
    計算殘差 ‖Ax - b‖
    """  # EN: Execute statement: """.
    n = len(A)  # EN: Assign n from expression: len(A).
    Ax = [0.0] * n  # EN: Assign Ax from expression: [0.0] * n.

    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
            Ax[i] += A[i][j] * x[j]  # EN: Execute statement: Ax[i] += A[i][j] * x[j].

    # 計算殘差範數
    residual = sum((Ax[i] - b[i]) ** 2 for i in range(n)) ** 0.5  # EN: Assign residual from expression: sum((Ax[i] - b[i]) ** 2 for i in range(n)) ** 0.5.

    return residual  # EN: Return a value: return residual.


def main():  # EN: Define main and its behavior.
    """主程式 (Main program)"""  # EN: Execute statement: """主程式 (Main program)""".

    print_separator("高斯消去法示範\nGaussian Elimination Demo")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 範例 1：標準 3×3 系統
    # ========================================
    print_separator("範例 1：標準 3×3 系統")  # EN: Call print_separator(...) to perform an operation.

    A1 = [  # EN: Assign A1 from expression: [.
        [2.0, 1.0, 1.0],  # EN: Execute statement: [2.0, 1.0, 1.0],.
        [4.0, -6.0, 0.0],  # EN: Execute statement: [4.0, -6.0, 0.0],.
        [-2.0, 7.0, 2.0]  # EN: Execute statement: [-2.0, 7.0, 2.0].
    ]  # EN: Execute statement: ].
    b1 = [5.0, -2.0, 9.0]  # EN: Assign b1 from expression: [5.0, -2.0, 9.0].

    x1 = gaussian_elimination_solve(A1, b1, verbose=True)  # EN: Assign x1 from expression: gaussian_elimination_solve(A1, b1, verbose=True).

    if x1:  # EN: Branch on a condition: if x1:.
        residual = verify_solution(A1, x1, b1)  # EN: Assign residual from expression: verify_solution(A1, x1, b1).
        print(f"\n驗證：‖Ax - b‖ = {residual:.2e}")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 2：需要換列的系統
    # ========================================
    print_separator("範例 2：需要換列的系統（主元為零）")  # EN: Call print_separator(...) to perform an operation.

    A2 = [  # EN: Assign A2 from expression: [.
        [0.0, 1.0, 2.0],  # EN: Execute statement: [0.0, 1.0, 2.0],.
        [1.0, 2.0, 1.0],  # EN: Execute statement: [1.0, 2.0, 1.0],.
        [2.0, 3.0, 1.0]  # EN: Execute statement: [2.0, 3.0, 1.0].
    ]  # EN: Execute statement: ].
    b2 = [3.0, 4.0, 5.0]  # EN: Assign b2 from expression: [3.0, 4.0, 5.0].

    x2 = gaussian_elimination_solve(A2, b2, verbose=True)  # EN: Assign x2 from expression: gaussian_elimination_solve(A2, b2, verbose=True).

    if x2:  # EN: Branch on a condition: if x2:.
        residual = verify_solution(A2, x2, b2)  # EN: Assign residual from expression: verify_solution(A2, x2, b2).
        print(f"\n驗證：‖Ax - b‖ = {residual:.2e}")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 3：2×2 簡單系統
    # ========================================
    print_separator("範例 3：2×2 簡單系統")  # EN: Call print_separator(...) to perform an operation.

    A3 = [  # EN: Assign A3 from expression: [.
        [3.0, 2.0],  # EN: Execute statement: [3.0, 2.0],.
        [1.0, 4.0]  # EN: Execute statement: [1.0, 4.0].
    ]  # EN: Execute statement: ].
    b3 = [7.0, 9.0]  # EN: Assign b3 from expression: [7.0, 9.0].

    x3 = gaussian_elimination_solve(A3, b3, verbose=True)  # EN: Assign x3 from expression: gaussian_elimination_solve(A3, b3, verbose=True).

    if x3:  # EN: Branch on a condition: if x3:.
        residual = verify_solution(A3, x3, b3)  # EN: Assign residual from expression: verify_solution(A3, x3, b3).
        print(f"\n驗證：‖Ax - b‖ = {residual:.2e}")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 4：奇異矩陣（線性相依列）
    # ========================================
    print_separator("範例 4：奇異矩陣（線性相依列）")  # EN: Call print_separator(...) to perform an operation.

    A4 = [  # EN: Assign A4 from expression: [.
        [1.0, 2.0, 3.0],  # EN: Execute statement: [1.0, 2.0, 3.0],.
        [2.0, 4.0, 6.0],  # 這列是第一列的 2 倍  # EN: Execute statement: [2.0, 4.0, 6.0], # 這列是第一列的 2 倍.
        [1.0, 3.0, 4.0]  # EN: Execute statement: [1.0, 3.0, 4.0].
    ]  # EN: Execute statement: ].
    b4 = [6.0, 12.0, 8.0]  # EN: Assign b4 from expression: [6.0, 12.0, 8.0].

    x4 = gaussian_elimination_solve(A4, b4, verbose=True)  # EN: Assign x4 from expression: gaussian_elimination_solve(A4, b4, verbose=True).

    # ========================================
    # 消去矩陣示範 (Elimination Matrix Demo)
    # ========================================
    print_separator("消去矩陣示範 (Elimination Matrix)")  # EN: Call print_separator(...) to perform an operation.

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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("高斯消去法示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
