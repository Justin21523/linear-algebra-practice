"""
矩陣運算：手刻版本 (Matrix Operations: Manual Implementation)

本程式示範：
1. 矩陣建立與印出 (Matrix creation and display)
2. 矩陣加法、減法 (Addition, subtraction)
3. 純量乘法 (Scalar multiplication)
4. 矩陣轉置 (Transpose)
5. 對稱性檢查 (Symmetry check)
6. 列/行提取 (Row/column extraction)

This program demonstrates basic matrix operations without using NumPy.
"""

from typing import List

# 型別別名 (Type alias)
Matrix = List[List[float]]
Vector = List[float]


def create_matrix(rows: int, cols: int, fill: float = 0.0) -> Matrix:
    """
    建立指定大小的矩陣 (Create matrix of specified size)
    """
    return [[fill for _ in range(cols)] for _ in range(rows)]


def get_shape(A: Matrix) -> tuple:
    """
    取得矩陣大小 (Get matrix shape)
    回傳 (rows, cols)
    """
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    return (rows, cols)


def print_matrix(name: str, A: Matrix) -> None:
    """
    印出矩陣 (Print matrix with nice formatting)
    """
    rows, cols = get_shape(A)
    print(f"{name} ({rows}×{cols}):")

    for row in A:
        print("  [", end="")
        print("  ".join(f"{x:8.4f}" for x in row), end="")
        print(" ]")
    print()


def matrix_add(A: Matrix, B: Matrix) -> Matrix:
    """
    矩陣加法 (Matrix addition)

    A + B: 對應元素相加
    條件：A 和 B 必須大小相同
    """
    rows_A, cols_A = get_shape(A)
    rows_B, cols_B = get_shape(B)

    if rows_A != rows_B or cols_A != cols_B:
        raise ValueError(f"矩陣大小不符：{rows_A}×{cols_A} vs {rows_B}×{cols_B}")

    result = create_matrix(rows_A, cols_A)
    for i in range(rows_A):
        for j in range(cols_A):
            result[i][j] = A[i][j] + B[i][j]

    return result


def matrix_subtract(A: Matrix, B: Matrix) -> Matrix:
    """
    矩陣減法 (Matrix subtraction)

    A - B: 對應元素相減
    """
    rows_A, cols_A = get_shape(A)
    rows_B, cols_B = get_shape(B)

    if rows_A != rows_B or cols_A != cols_B:
        raise ValueError(f"矩陣大小不符：{rows_A}×{cols_A} vs {rows_B}×{cols_B}")

    result = create_matrix(rows_A, cols_A)
    for i in range(rows_A):
        for j in range(cols_A):
            result[i][j] = A[i][j] - B[i][j]

    return result


def scalar_multiply(c: float, A: Matrix) -> Matrix:
    """
    純量乘法 (Scalar multiplication)

    c·A: 每個元素乘以 c
    """
    rows, cols = get_shape(A)
    result = create_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            result[i][j] = c * A[i][j]

    return result


def transpose(A: Matrix) -> Matrix:
    """
    矩陣轉置 (Matrix transpose)

    Aᵀ: 第 i 列變成第 i 行
    (Aᵀ)ᵢⱼ = Aⱼᵢ

    m×n 矩陣的轉置是 n×m 矩陣
    """
    rows, cols = get_shape(A)

    # 轉置後：原本的 cols 變成 rows
    result = create_matrix(cols, rows)

    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]

    return result


def is_symmetric(A: Matrix, tolerance: float = 1e-10) -> bool:
    """
    檢查矩陣是否對稱 (Check if matrix is symmetric)

    對稱條件：A = Aᵀ（即 aᵢⱼ = aⱼᵢ 對所有 i, j）
    只有方陣才可能對稱
    """
    rows, cols = get_shape(A)

    # 必須是方陣
    if rows != cols:
        return False

    # 檢查 aᵢⱼ = aⱼᵢ
    for i in range(rows):
        for j in range(i + 1, cols):  # 只需檢查上三角
            if abs(A[i][j] - A[j][i]) > tolerance:
                return False

    return True


def get_row(A: Matrix, row_index: int) -> Vector:
    """
    取出矩陣的某一列 (Get a row from matrix)
    """
    rows, _ = get_shape(A)
    if row_index < 0 or row_index >= rows:
        raise IndexError(f"列索引超出範圍：{row_index}")

    return A[row_index].copy()


def get_column(A: Matrix, col_index: int) -> Vector:
    """
    取出矩陣的某一行 (Get a column from matrix)
    """
    rows, cols = get_shape(A)
    if col_index < 0 or col_index >= cols:
        raise IndexError(f"行索引超出範圍：{col_index}")

    return [A[i][col_index] for i in range(rows)]


def matrices_equal(A: Matrix, B: Matrix, tolerance: float = 1e-10) -> bool:
    """
    檢查兩矩陣是否相等 (Check if two matrices are equal)
    """
    rows_A, cols_A = get_shape(A)
    rows_B, cols_B = get_shape(B)

    if rows_A != rows_B or cols_A != cols_B:
        return False

    for i in range(rows_A):
        for j in range(cols_A):
            if abs(A[i][j] - B[i][j]) > tolerance:
                return False

    return True


def print_separator(title: str) -> None:
    """印出分隔線 (Print separator)"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    """主程式 (Main program)"""

    print_separator("矩陣運算示範 - 手刻版本\nMatrix Operations Demo - Manual Implementation")

    # ========================================
    # 1. 建立矩陣 (Creating Matrices)
    # ========================================
    print_separator("1. 建立矩陣 (Creating Matrices)")

    A = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    B = [
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ]

    print_matrix("A", A)
    print_matrix("B", B)

    # ========================================
    # 2. 矩陣加法 (Matrix Addition)
    # ========================================
    print_separator("2. 矩陣加法 (Matrix Addition)")

    C = matrix_add(A, B)
    print("A + B =")
    print_matrix("C", C)

    # ========================================
    # 3. 矩陣減法 (Matrix Subtraction)
    # ========================================
    print_separator("3. 矩陣減法 (Matrix Subtraction)")

    D = matrix_subtract(B, A)
    print("B - A =")
    print_matrix("D", D)

    # ========================================
    # 4. 純量乘法 (Scalar Multiplication)
    # ========================================
    print_separator("4. 純量乘法 (Scalar Multiplication)")

    c = 2.0
    E = scalar_multiply(c, A)
    print(f"{c} × A =")
    print_matrix("E", E)

    # 負數純量
    F = scalar_multiply(-1, A)
    print("-A =")
    print_matrix("F", F)

    # ========================================
    # 5. 矩陣轉置 (Transpose)
    # ========================================
    print_separator("5. 矩陣轉置 (Transpose)")

    print("原矩陣 A (2×3):")
    print_matrix("A", A)

    At = transpose(A)
    print("轉置後 Aᵀ (3×2):")
    print_matrix("Aᵀ", At)

    # 驗證 (Aᵀ)ᵀ = A
    Att = transpose(At)
    print("驗證 (Aᵀ)ᵀ = A:")
    print_matrix("(Aᵀ)ᵀ", Att)
    print(f"(Aᵀ)ᵀ == A ? {matrices_equal(Att, A)}")

    # ========================================
    # 6. 轉置的性質 (Transpose Properties)
    # ========================================
    print_separator("6. 轉置的性質 (Transpose Properties)")

    # (A + B)ᵀ = Aᵀ + Bᵀ
    sum_then_transpose = transpose(matrix_add(A, B))
    transpose_then_sum = matrix_add(transpose(A), transpose(B))

    print("驗證 (A + B)ᵀ = Aᵀ + Bᵀ:")
    print_matrix("(A + B)ᵀ", sum_then_transpose)
    print_matrix("Aᵀ + Bᵀ", transpose_then_sum)
    print(f"相等？ {matrices_equal(sum_then_transpose, transpose_then_sum)}")

    # (cA)ᵀ = cAᵀ
    print("\n驗證 (cA)ᵀ = cAᵀ (c = 2):")
    scaled_then_transpose = transpose(scalar_multiply(2, A))
    transpose_then_scaled = scalar_multiply(2, transpose(A))
    print(f"相等？ {matrices_equal(scaled_then_transpose, transpose_then_scaled)}")

    # ========================================
    # 7. 對稱矩陣 (Symmetric Matrix)
    # ========================================
    print_separator("7. 對稱矩陣 (Symmetric Matrix)")

    # 對稱矩陣範例
    S = [
        [1.0, 2.0, 3.0],
        [2.0, 5.0, 6.0],
        [3.0, 6.0, 9.0]
    ]

    print_matrix("S（對稱矩陣）", S)
    print(f"S 是對稱矩陣？ {is_symmetric(S)}")

    St = transpose(S)
    print_matrix("Sᵀ", St)
    print(f"S == Sᵀ ? {matrices_equal(S, St)}")

    # 非對稱矩陣
    N = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]

    print_matrix("N（非對稱矩陣）", N)
    print(f"N 是對稱矩陣？ {is_symmetric(N)}")

    # ========================================
    # 8. 列與行的提取 (Row and Column Extraction)
    # ========================================
    print_separator("8. 列與行的提取 (Row and Column Extraction)")

    print_matrix("A", A)

    row_0 = get_row(A, 0)
    row_1 = get_row(A, 1)
    print(f"第 0 列 (row 0): {row_0}")
    print(f"第 1 列 (row 1): {row_1}")

    col_0 = get_column(A, 0)
    col_1 = get_column(A, 1)
    col_2 = get_column(A, 2)
    print(f"\n第 0 行 (col 0): {col_0}")
    print(f"第 1 行 (col 1): {col_1}")
    print(f"第 2 行 (col 2): {col_2}")

    # ========================================
    # 9. 構造對稱矩陣 (Constructing Symmetric Matrix)
    # ========================================
    print_separator("9. 構造對稱矩陣：AᵀA 總是對稱")

    # 任意矩陣 M
    M = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    print_matrix("M (3×2)", M)

    Mt = transpose(M)
    print_matrix("Mᵀ (2×3)", Mt)

    # 注意：這裡我們還沒實作矩陣乘法，先用概念說明
    print("MᵀM 會是 2×2 的對稱矩陣")
    print("MMᵀ 會是 3×3 的對稱矩陣")
    print("（矩陣乘法會在下一個單元介紹）")

    # ========================================
    # 10. 特殊矩陣預覽 (Special Matrices Preview)
    # ========================================
    print_separator("10. 特殊矩陣預覽 (Special Matrices Preview)")

    # 零矩陣
    zero = create_matrix(2, 3, 0.0)
    print_matrix("零矩陣 O (2×3)", zero)

    # 方陣範例
    square = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    print_matrix("方陣 (3×3 square matrix)", square)

    print("其他特殊矩陣（單位矩陣、對角矩陣等）")
    print("將在 04-special-matrices 單元詳細介紹")

    print()
    print("=" * 60)
    print("所有矩陣運算示範完成！")
    print("All matrix operations demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
