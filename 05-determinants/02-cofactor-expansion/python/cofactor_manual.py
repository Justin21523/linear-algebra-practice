"""
餘因子展開 - 手刻版本 (Cofactor Expansion - Manual Implementation)

本程式示範：
1. 子行列式與餘因子計算
2. 餘因子展開求行列式
3. 餘因子矩陣與伴隨矩陣
4. 用伴隨矩陣求逆矩陣
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
# 子行列式與餘因子
# ========================================

def get_minor_matrix(A: List[List[float]], row: int, col: int) -> List[List[float]]:  # EN: Define get_minor_matrix and its behavior.
    """取得去掉第 row 列、第 col 行後的子矩陣"""  # EN: Execute statement: """取得去掉第 row 列、第 col 行後的子矩陣""".
    n = len(A)  # EN: Assign n from expression: len(A).
    return [[A[i][j] for j in range(n) if j != col]  # EN: Return a value: return [[A[i][j] for j in range(n) if j != col].
            for i in range(n) if i != row]  # EN: Iterate with a for-loop: for i in range(n) if i != row].


def det_2x2(A: List[List[float]]) -> float:  # EN: Define det_2x2 and its behavior.
    """2×2 行列式"""  # EN: Execute statement: """2×2 行列式""".
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]  # EN: Return a value: return A[0][0] * A[1][1] - A[0][1] * A[1][0].


def minor(A: List[List[float]], i: int, j: int) -> float:  # EN: Define minor and its behavior.
    """計算子行列式 Mᵢⱼ"""  # EN: Execute statement: """計算子行列式 Mᵢⱼ""".
    sub_matrix = get_minor_matrix(A, i, j)  # EN: Assign sub_matrix from expression: get_minor_matrix(A, i, j).
    return determinant(sub_matrix)  # EN: Return a value: return determinant(sub_matrix).


def cofactor(A: List[List[float]], i: int, j: int) -> float:  # EN: Define cofactor and its behavior.
    """計算餘因子 Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ"""  # EN: Execute statement: """計算餘因子 Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ""".
    sign = (-1) ** (i + j)  # EN: Assign sign from expression: (-1) ** (i + j).
    return sign * minor(A, i, j)  # EN: Return a value: return sign * minor(A, i, j).


def determinant(A: List[List[float]]) -> float:  # EN: Define determinant and its behavior.
    """用餘因子展開計算行列式（遞迴）"""  # EN: Execute statement: """用餘因子展開計算行列式（遞迴）""".
    n = len(A)  # EN: Assign n from expression: len(A).

    # 基本情況
    if n == 1:  # EN: Branch on a condition: if n == 1:.
        return A[0][0]  # EN: Return a value: return A[0][0].
    if n == 2:  # EN: Branch on a condition: if n == 2:.
        return det_2x2(A)  # EN: Return a value: return det_2x2(A).

    # 沿第一列展開
    det = 0.0  # EN: Assign det from expression: 0.0.
    for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
        det += A[0][j] * cofactor(A, 0, j)  # EN: Update det via += using: A[0][j] * cofactor(A, 0, j).

    return det  # EN: Return a value: return det.


def determinant_by_row(A: List[List[float]], row: int) -> float:  # EN: Define determinant_by_row and its behavior.
    """沿指定列展開"""  # EN: Execute statement: """沿指定列展開""".
    n = len(A)  # EN: Assign n from expression: len(A).
    det = 0.0  # EN: Assign det from expression: 0.0.
    for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
        det += A[row][j] * cofactor(A, row, j)  # EN: Update det via += using: A[row][j] * cofactor(A, row, j).
    return det  # EN: Return a value: return det.


def determinant_by_col(A: List[List[float]], col: int) -> float:  # EN: Define determinant_by_col and its behavior.
    """沿指定行展開"""  # EN: Execute statement: """沿指定行展開""".
    n = len(A)  # EN: Assign n from expression: len(A).
    det = 0.0  # EN: Assign det from expression: 0.0.
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        det += A[i][col] * cofactor(A, i, col)  # EN: Update det via += using: A[i][col] * cofactor(A, i, col).
    return det  # EN: Return a value: return det.


# ========================================
# 餘因子矩陣與伴隨矩陣
# ========================================

def cofactor_matrix(A: List[List[float]]) -> List[List[float]]:  # EN: Define cofactor_matrix and its behavior.
    """計算餘因子矩陣"""  # EN: Execute statement: """計算餘因子矩陣""".
    n = len(A)  # EN: Assign n from expression: len(A).
    C = [[0.0] * n for _ in range(n)]  # EN: Assign C from expression: [[0.0] * n for _ in range(n)].
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
            C[i][j] = cofactor(A, i, j)  # EN: Execute statement: C[i][j] = cofactor(A, i, j).
    return C  # EN: Return a value: return C.


def transpose(A: List[List[float]]) -> List[List[float]]:  # EN: Define transpose and its behavior.
    """矩陣轉置"""  # EN: Execute statement: """矩陣轉置""".
    m, n = len(A), len(A[0])  # EN: Execute statement: m, n = len(A), len(A[0]).
    return [[A[i][j] for i in range(m)] for j in range(n)]  # EN: Return a value: return [[A[i][j] for i in range(m)] for j in range(n)].


def adjugate(A: List[List[float]]) -> List[List[float]]:  # EN: Define adjugate and its behavior.
    """計算伴隨矩陣 adj(A) = Cᵀ"""  # EN: Execute statement: """計算伴隨矩陣 adj(A) = Cᵀ""".
    return transpose(cofactor_matrix(A))  # EN: Return a value: return transpose(cofactor_matrix(A)).


def scalar_multiply_matrix(c: float, A: List[List[float]]) -> List[List[float]]:  # EN: Define scalar_multiply_matrix and its behavior.
    """純量乘矩陣"""  # EN: Execute statement: """純量乘矩陣""".
    return [[c * x for x in row] for row in A]  # EN: Return a value: return [[c * x for x in row] for row in A].


def inverse_by_adjugate(A: List[List[float]]) -> List[List[float]]:  # EN: Define inverse_by_adjugate and its behavior.
    """用伴隨矩陣計算逆矩陣：A⁻¹ = adj(A) / det(A)"""  # EN: Execute statement: """用伴隨矩陣計算逆矩陣：A⁻¹ = adj(A) / det(A)""".
    det = determinant(A)  # EN: Assign det from expression: determinant(A).
    if abs(det) < 1e-10:  # EN: Branch on a condition: if abs(det) < 1e-10:.
        raise ValueError("矩陣不可逆（det = 0）")  # EN: Raise an exception: raise ValueError("矩陣不可逆（det = 0）").
    adj = adjugate(A)  # EN: Assign adj from expression: adjugate(A).
    return scalar_multiply_matrix(1.0 / det, adj)  # EN: Return a value: return scalar_multiply_matrix(1.0 / det, adj).


# ========================================
# 驗證函數
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


def main():  # EN: Define main and its behavior.
    print_separator("餘因子展開示範（手刻版）\nCofactor Expansion Demo (Manual)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 基本概念：子行列式與餘因子
    # ========================================
    print_separator("1. 子行列式與餘因子")  # EN: Call print_separator(...) to perform an operation.

    A = [  # EN: Assign A from expression: [.
        [1, 2, 3],  # EN: Execute statement: [1, 2, 3],.
        [4, 5, 6],  # EN: Execute statement: [4, 5, 6],.
        [7, 8, 9]  # EN: Execute statement: [7, 8, 9].
    ]  # EN: Execute statement: ].

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.

    # 計算 M₁₂（去掉第1列第2行）
    sub = get_minor_matrix(A, 0, 1)  # EN: Assign sub from expression: get_minor_matrix(A, 0, 1).
    print("\n去掉第 1 列、第 2 行：")  # EN: Print formatted output to the console.
    print_matrix("子矩陣", sub)  # EN: Call print_matrix(...) to perform an operation.
    print(f"M₁₂ = det(子矩陣) = {det_2x2(sub)}")  # EN: Print formatted output to the console.

    # 計算餘因子
    print(f"\nC₁₂ = (-1)^(1+2) × M₁₂ = -1 × {det_2x2(sub)} = {cofactor(A, 0, 1)}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 餘因子展開計算行列式
    # ========================================
    print_separator("2. 餘因子展開計算行列式")  # EN: Call print_separator(...) to perform an operation.

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.

    print("\n沿第一列展開：")  # EN: Print formatted output to the console.
    print(f"det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃")  # EN: Print formatted output to the console.
    print(f"       = {A[0][0]} × {cofactor(A, 0, 0)} + {A[0][1]} × {cofactor(A, 0, 1)} + {A[0][2]} × {cofactor(A, 0, 2)}")  # EN: Print formatted output to the console.
    print(f"       = {A[0][0] * cofactor(A, 0, 0)} + {A[0][1] * cofactor(A, 0, 1)} + {A[0][2] * cofactor(A, 0, 2)}")  # EN: Print formatted output to the console.
    print(f"       = {determinant(A)}")  # EN: Print formatted output to the console.

    print(f"\n沿第一行展開：det(A) = {determinant_by_col(A, 0)}")  # EN: Print formatted output to the console.
    print(f"沿第二列展開：det(A) = {determinant_by_row(A, 1)}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 選擇最佳展開位置
    # ========================================
    print_separator("3. 選擇有零元素的列展開")  # EN: Call print_separator(...) to perform an operation.

    B = [  # EN: Assign B from expression: [.
        [1, 0, 0],  # EN: Execute statement: [1, 0, 0],.
        [2, 3, 0],  # EN: Execute statement: [2, 3, 0],.
        [4, 5, 6]  # EN: Execute statement: [4, 5, 6].
    ]  # EN: Execute statement: ].

    print_matrix("B（下三角）", B)  # EN: Call print_matrix(...) to perform an operation.
    print("\n沿第一列展開（兩個零）：")  # EN: Print formatted output to the console.
    print(f"det(B) = {B[0][0]} × C₁₁ + 0 + 0")  # EN: Print formatted output to the console.
    print(f"       = 1 × det([[3, 0], [5, 6]])")  # EN: Print formatted output to the console.
    print(f"       = 1 × (3×6 - 0×5)")  # EN: Print formatted output to the console.
    print(f"       = {determinant(B)}")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 餘因子矩陣
    # ========================================
    print_separator("4. 餘因子矩陣")  # EN: Call print_separator(...) to perform an operation.

    A = [  # EN: Assign A from expression: [.
        [2, 1, 3],  # EN: Execute statement: [2, 1, 3],.
        [1, 0, 2],  # EN: Execute statement: [1, 0, 2],.
        [4, 1, 5]  # EN: Execute statement: [4, 1, 5].
    ]  # EN: Execute statement: ].

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print(f"\ndet(A) = {determinant(A)}")  # EN: Print formatted output to the console.

    C = cofactor_matrix(A)  # EN: Assign C from expression: cofactor_matrix(A).
    print("\n餘因子矩陣 C：")  # EN: Print formatted output to the console.
    print_matrix("C", C)  # EN: Call print_matrix(...) to perform an operation.

    print("\n各餘因子計算過程：")  # EN: Print formatted output to the console.
    for i in range(3):  # EN: Iterate with a for-loop: for i in range(3):.
        for j in range(3):  # EN: Iterate with a for-loop: for j in range(3):.
            sign = "+" if (i + j) % 2 == 0 else "-"  # EN: Assign sign from expression: "+" if (i + j) % 2 == 0 else "-".
            print(f"C_{i+1}{j+1} = {sign}M_{i+1}{j+1} = {cofactor(A, i, j):.1f}")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 伴隨矩陣
    # ========================================
    print_separator("5. 伴隨矩陣 adj(A) = Cᵀ")  # EN: Call print_separator(...) to perform an operation.

    adj_A = adjugate(A)  # EN: Assign adj_A from expression: adjugate(A).
    print_matrix("adj(A)", adj_A)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 6. 用伴隨矩陣求逆矩陣
    # ========================================
    print_separator("6. 用伴隨矩陣求逆矩陣")  # EN: Call print_separator(...) to perform an operation.

    print("A⁻¹ = adj(A) / det(A)")  # EN: Print formatted output to the console.
    print(f"    = adj(A) / {determinant(A)}")  # EN: Print formatted output to the console.

    A_inv = inverse_by_adjugate(A)  # EN: Assign A_inv from expression: inverse_by_adjugate(A).
    print_matrix("\nA⁻¹", A_inv)  # EN: Call print_matrix(...) to perform an operation.

    # 驗證
    I = matrix_multiply(A, A_inv)  # EN: Assign I from expression: matrix_multiply(A, A_inv).
    print("\n驗證 A × A⁻¹：")  # EN: Print formatted output to the console.
    print_matrix("A × A⁻¹", I)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 7. 4×4 行列式範例
    # ========================================
    print_separator("7. 4×4 行列式範例")  # EN: Call print_separator(...) to perform an operation.

    D = [  # EN: Assign D from expression: [.
        [1, 2, 0, 0],  # EN: Execute statement: [1, 2, 0, 0],.
        [3, 4, 0, 0],  # EN: Execute statement: [3, 4, 0, 0],.
        [0, 0, 5, 6],  # EN: Execute statement: [0, 0, 5, 6],.
        [0, 0, 7, 8]  # EN: Execute statement: [0, 0, 7, 8].
    ]  # EN: Execute statement: ].

    print_matrix("D（塊對角）", D)  # EN: Call print_matrix(...) to perform an operation.
    print(f"\ndet(D) = {determinant(D)}")  # EN: Print formatted output to the console.
    print("（= det([[1,2],[3,4]]) × det([[5,6],[7,8]])）")  # EN: Print formatted output to the console.
    print(f"（= {det_2x2([[1,2],[3,4]])} × {det_2x2([[5,6],[7,8]])} = {det_2x2([[1,2],[3,4]]) * det_2x2([[5,6],[7,8]])}）")  # EN: Print formatted output to the console.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
    print("""
餘因子展開公式：
  Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ
  det(A) = Σⱼ aᵢⱼ Cᵢⱼ（沿第 i 列）

伴隨矩陣：
  adj(A) = Cᵀ

逆矩陣公式：
  A⁻¹ = adj(A) / det(A)

計算技巧：
  - 選擇零最多的列/行展開
  - 三角矩陣 → det = 對角線乘積

時間複雜度：O(n!)，只適合小矩陣
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
