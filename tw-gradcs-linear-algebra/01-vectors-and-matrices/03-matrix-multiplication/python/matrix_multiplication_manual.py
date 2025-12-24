"""
矩陣乘法：手刻版本 - 四種觀點 (Matrix Multiplication: Four Views)

本程式示範矩陣乘法的四種理解方式：
1. 內積觀點 (Dot product view)
2. 行的線性組合 (Column view)
3. 列的線性組合 (Row view)
4. 外積的和 (Sum of outer products)

This program demonstrates the four ways to understand matrix multiplication,
as emphasized by Gilbert Strang.
"""

from typing import List

# 型別別名 (Type alias)
Matrix = List[List[float]]
Vector = List[float]


def get_shape(A: Matrix) -> tuple:
    """取得矩陣大小 (Get matrix shape)"""
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    return (rows, cols)


def print_matrix(name: str, A: Matrix) -> None:
    """印出矩陣 (Print matrix)"""
    rows, cols = get_shape(A)
    print(f"{name} ({rows}×{cols}):")
    for row in A:
        print("  [", end="")
        print("  ".join(f"{x:8.4f}" for x in row), end="")
        print(" ]")
    print()


def print_vector(name: str, v: Vector) -> None:
    """印出向量 (Print vector)"""
    formatted = ", ".join(f"{x:.4f}" for x in v)
    print(f"{name} = [{formatted}]")


def print_separator(title: str) -> None:
    """印出分隔線 (Print separator)"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def create_zero_matrix(rows: int, cols: int) -> Matrix:
    """建立零矩陣 (Create zero matrix)"""
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def get_row(A: Matrix, i: int) -> Vector:
    """取出第 i 列 (Get row i)"""
    return A[i].copy()


def get_column(A: Matrix, j: int) -> Vector:
    """取出第 j 行 (Get column j)"""
    return [A[i][j] for i in range(len(A))]


def dot_product(u: Vector, v: Vector) -> float:
    """內積 (Dot product)"""
    return sum(u[i] * v[i] for i in range(len(u)))


def scalar_multiply_vector(c: float, v: Vector) -> Vector:
    """純量乘向量 (Scalar multiply vector)"""
    return [c * x for x in v]


def vector_add(u: Vector, v: Vector) -> Vector:
    """向量加法 (Vector addition)"""
    return [u[i] + v[i] for i in range(len(u))]


def outer_product(u: Vector, v: Vector) -> Matrix:
    """
    外積 (Outer product)
    u ⊗ v = uvᵀ
    結果是 len(u) × len(v) 矩陣
    """
    rows = len(u)
    cols = len(v)
    result = create_zero_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = u[i] * v[j]
    return result


def matrix_add(A: Matrix, B: Matrix) -> Matrix:
    """矩陣加法 (Matrix addition)"""
    rows, cols = get_shape(A)
    result = create_zero_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] + B[i][j]
    return result


# ============================================================
# 四種矩陣乘法的實作
# Four implementations of matrix multiplication
# ============================================================

def matrix_multiply_standard(A: Matrix, B: Matrix) -> Matrix:
    """
    標準矩陣乘法（三層迴圈）
    Standard matrix multiplication (triple nested loop)

    Cᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ
    """
    rows_A, cols_A = get_shape(A)
    rows_B, cols_B = get_shape(B)

    if cols_A != rows_B:
        raise ValueError(f"維度不相容：{rows_A}×{cols_A} 和 {rows_B}×{cols_B}")

    C = create_zero_matrix(rows_A, cols_B)

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]

    return C


def matrix_multiply_dot_view(A: Matrix, B: Matrix) -> Matrix:
    """
    觀點一：內積觀點 (Dot Product View)

    Cᵢⱼ = (A 的第 i 列) · (B 的第 j 行)

    每個元素是一個內積的結果
    """
    rows_A, cols_A = get_shape(A)
    rows_B, cols_B = get_shape(B)

    if cols_A != rows_B:
        raise ValueError("維度不相容")

    C = create_zero_matrix(rows_A, cols_B)

    for i in range(rows_A):
        row_i = get_row(A, i)
        for j in range(cols_B):
            col_j = get_column(B, j)
            C[i][j] = dot_product(row_i, col_j)

    return C


def matrix_multiply_column_view(A: Matrix, B: Matrix) -> Matrix:
    """
    觀點二：行的線性組合 (Column View)

    C 的第 j 行 = A 的各行的線性組合
                = B[0,j]*A[:,0] + B[1,j]*A[:,1] + ...

    係數來自 B 的第 j 行
    """
    rows_A, cols_A = get_shape(A)
    rows_B, cols_B = get_shape(B)

    if cols_A != rows_B:
        raise ValueError("維度不相容")

    C = create_zero_matrix(rows_A, cols_B)

    # 對每一行 j
    for j in range(cols_B):
        # C 的第 j 行是 A 的各行的線性組合
        col_j = [0.0] * rows_A

        for k in range(cols_A):
            # 係數是 B[k][j]
            coef = B[k][j]
            # A 的第 k 行
            a_col_k = get_column(A, k)
            # 加權累加
            col_j = vector_add(col_j, scalar_multiply_vector(coef, a_col_k))

        # 把結果放入 C 的第 j 行
        for i in range(rows_A):
            C[i][j] = col_j[i]

    return C


def matrix_multiply_row_view(A: Matrix, B: Matrix) -> Matrix:
    """
    觀點三：列的線性組合 (Row View)

    C 的第 i 列 = B 的各列的線性組合
               = A[i,0]*B[0,:] + A[i,1]*B[1,:] + ...

    係數來自 A 的第 i 列
    """
    rows_A, cols_A = get_shape(A)
    rows_B, cols_B = get_shape(B)

    if cols_A != rows_B:
        raise ValueError("維度不相容")

    C = create_zero_matrix(rows_A, cols_B)

    # 對每一列 i
    for i in range(rows_A):
        # C 的第 i 列是 B 的各列的線性組合
        row_i = [0.0] * cols_B

        for k in range(cols_A):
            # 係數是 A[i][k]
            coef = A[i][k]
            # B 的第 k 列
            b_row_k = get_row(B, k)
            # 加權累加
            row_i = vector_add(row_i, scalar_multiply_vector(coef, b_row_k))

        # 把結果放入 C 的第 i 列
        C[i] = row_i

    return C


def matrix_multiply_outer_view(A: Matrix, B: Matrix) -> Matrix:
    """
    觀點四：外積的和 (Sum of Outer Products)

    AB = Σₖ (A 的第 k 行) ⊗ (B 的第 k 列)

    每個外積是一個秩 1 的矩陣
    """
    rows_A, cols_A = get_shape(A)
    rows_B, cols_B = get_shape(B)

    if cols_A != rows_B:
        raise ValueError("維度不相容")

    C = create_zero_matrix(rows_A, cols_B)

    for k in range(cols_A):
        # A 的第 k 行
        a_col_k = get_column(A, k)
        # B 的第 k 列
        b_row_k = get_row(B, k)
        # 外積
        outer = outer_product(a_col_k, b_row_k)
        # 累加
        C = matrix_add(C, outer)

    return C


def matrix_vector_multiply(A: Matrix, x: Vector) -> Vector:
    """
    矩陣與向量相乘 (Matrix-vector multiplication)
    Ax = x₁a₁ + x₂a₂ + ... (A 的行的線性組合)
    """
    rows, cols = get_shape(A)

    if cols != len(x):
        raise ValueError("維度不相容")

    result = [0.0] * rows
    for j in range(cols):
        col_j = get_column(A, j)
        result = vector_add(result, scalar_multiply_vector(x[j], col_j))

    return result


def main():
    """主程式 (Main program)"""

    print_separator("矩陣乘法示範 - 四種觀點\nMatrix Multiplication - Four Views")

    # ========================================
    # 定義範例矩陣 (Define example matrices)
    # ========================================
    A = [
        [1.0, 2.0],
        [3.0, 4.0]
    ]

    B = [
        [5.0, 6.0],
        [7.0, 8.0]
    ]

    print_matrix("A", A)
    print_matrix("B", B)

    # ========================================
    # 觀點一：內積 (Dot Product View)
    # ========================================
    print_separator("觀點一：內積 (Dot Product View)")

    print("Cᵢⱼ = (A 的第 i 列) · (B 的第 j 行)")
    print()

    # 手動展示計算過程
    print("C₁₁ = [1, 2] · [5, 7] = 1×5 + 2×7 = 19")
    print("C₁₂ = [1, 2] · [6, 8] = 1×6 + 2×8 = 22")
    print("C₂₁ = [3, 4] · [5, 7] = 3×5 + 4×7 = 43")
    print("C₂₂ = [3, 4] · [6, 8] = 3×6 + 4×8 = 50")
    print()

    C_dot = matrix_multiply_dot_view(A, B)
    print_matrix("C = AB（內積觀點）", C_dot)

    # ========================================
    # 觀點二：行的線性組合 (Column View)
    # ========================================
    print_separator("觀點二：行的線性組合 (Column View)")

    print("C 的每一行 = A 的各行的線性組合")
    print()

    print("C 的第 1 行 = 5×[1,3] + 7×[2,4] = [5,15] + [14,28] = [19,43]")
    print("C 的第 2 行 = 6×[1,3] + 8×[2,4] = [6,18] + [16,32] = [22,50]")
    print()

    C_col = matrix_multiply_column_view(A, B)
    print_matrix("C = AB（行觀點）", C_col)

    # ========================================
    # 觀點三：列的線性組合 (Row View)
    # ========================================
    print_separator("觀點三：列的線性組合 (Row View)")

    print("C 的每一列 = B 的各列的線性組合")
    print()

    print("C 的第 1 列 = 1×[5,6] + 2×[7,8] = [5,6] + [14,16] = [19,22]")
    print("C 的第 2 列 = 3×[5,6] + 4×[7,8] = [15,18] + [28,32] = [43,50]")
    print()

    C_row = matrix_multiply_row_view(A, B)
    print_matrix("C = AB（列觀點）", C_row)

    # ========================================
    # 觀點四：外積的和 (Outer Product View)
    # ========================================
    print_separator("觀點四：外積的和 (Sum of Outer Products)")

    print("AB = (A 的第 1 行)⊗(B 的第 1 列) + (A 的第 2 行)⊗(B 的第 2 列)")
    print()

    # 第一個外積
    a_col_1 = get_column(A, 0)  # [1, 3]
    b_row_1 = get_row(B, 0)     # [5, 6]
    outer_1 = outer_product(a_col_1, b_row_1)

    print(f"[1, 3]ᵀ × [5, 6] =")
    print_matrix("外積 1", outer_1)

    # 第二個外積
    a_col_2 = get_column(A, 1)  # [2, 4]
    b_row_2 = get_row(B, 1)     # [7, 8]
    outer_2 = outer_product(a_col_2, b_row_2)

    print(f"[2, 4]ᵀ × [7, 8] =")
    print_matrix("外積 2", outer_2)

    print("AB = 外積1 + 外積2 =")
    C_outer = matrix_multiply_outer_view(A, B)
    print_matrix("C", C_outer)

    # ========================================
    # 驗證四種方法結果相同
    # ========================================
    print_separator("驗證：四種方法結果相同")

    def matrices_equal(M1: Matrix, M2: Matrix) -> bool:
        rows, cols = get_shape(M1)
        for i in range(rows):
            for j in range(cols):
                if abs(M1[i][j] - M2[i][j]) > 1e-10:
                    return False
        return True

    print(f"內積觀點 == 行觀點？ {matrices_equal(C_dot, C_col)}")
    print(f"行觀點 == 列觀點？ {matrices_equal(C_col, C_row)}")
    print(f"列觀點 == 外積觀點？ {matrices_equal(C_row, C_outer)}")

    # ========================================
    # 矩陣乘向量 (Matrix-Vector Multiplication)
    # ========================================
    print_separator("矩陣乘向量 Ax (Matrix-Vector Multiplication)")

    x = [3.0, 2.0]
    print_matrix("A", A)
    print_vector("x", x)

    print("\n觀點：Ax = x₁×(A的第1行) + x₂×(A的第2行)")
    print(f"Ax = {x[0]}×[1,3] + {x[1]}×[2,4]")
    print(f"   = [3,9] + [4,8]")
    print(f"   = [7, 17]")

    result = matrix_vector_multiply(A, x)
    print_vector("\nAx", result)

    # ========================================
    # 不滿足交換律 (Non-Commutative)
    # ========================================
    print_separator("矩陣乘法不滿足交換律")

    print_matrix("AB", matrix_multiply_standard(A, B))
    print_matrix("BA", matrix_multiply_standard(B, A))
    print("AB ≠ BA（一般情況下）")

    # ========================================
    # 維度不相容的例子
    # ========================================
    print_separator("維度相容性 (Dimension Compatibility)")

    M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # 2×3
    N = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # 3×2

    print_matrix("M (2×3)", M)
    print_matrix("N (3×2)", N)

    MN = matrix_multiply_standard(M, N)
    print("M × N (2×3 × 3×2 = 2×2):")
    print_matrix("MN", MN)

    NM = matrix_multiply_standard(N, M)
    print("N × M (3×2 × 2×3 = 3×3):")
    print_matrix("NM", NM)

    print("注意：MN 是 2×2，NM 是 3×3，大小不同！")

    print()
    print("=" * 60)
    print("矩陣乘法示範完成！")
    print("Matrix multiplication demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
