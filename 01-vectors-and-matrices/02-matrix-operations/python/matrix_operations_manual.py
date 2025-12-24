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
"""  # EN: Execute statement: """.

from typing import List  # EN: Import symbol(s) from a module: from typing import List.

# 型別別名 (Type alias)
Matrix = List[List[float]]  # EN: Assign Matrix from expression: List[List[float]].
Vector = List[float]  # EN: Assign Vector from expression: List[float].


def create_matrix(rows: int, cols: int, fill: float = 0.0) -> Matrix:  # EN: Define create_matrix and its behavior.
    """
    建立指定大小的矩陣 (Create matrix of specified size)
    """  # EN: Execute statement: """.
    return [[fill for _ in range(cols)] for _ in range(rows)]  # EN: Return a value: return [[fill for _ in range(cols)] for _ in range(rows)].


def get_shape(A: Matrix) -> tuple:  # EN: Define get_shape and its behavior.
    """
    取得矩陣大小 (Get matrix shape)
    回傳 (rows, cols)
    """  # EN: Execute statement: """.
    rows = len(A)  # EN: Assign rows from expression: len(A).
    cols = len(A[0]) if rows > 0 else 0  # EN: Assign cols from expression: len(A[0]) if rows > 0 else 0.
    return (rows, cols)  # EN: Return a value: return (rows, cols).


def print_matrix(name: str, A: Matrix) -> None:  # EN: Define print_matrix and its behavior.
    """
    印出矩陣 (Print matrix with nice formatting)
    """  # EN: Execute statement: """.
    rows, cols = get_shape(A)  # EN: Execute statement: rows, cols = get_shape(A).
    print(f"{name} ({rows}×{cols}):")  # EN: Print formatted output to the console.

    for row in A:  # EN: Iterate with a for-loop: for row in A:.
        print("  [", end="")  # EN: Print formatted output to the console.
        print("  ".join(f"{x:8.4f}" for x in row), end="")  # EN: Print formatted output to the console.
        print(" ]")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.


def matrix_add(A: Matrix, B: Matrix) -> Matrix:  # EN: Define matrix_add and its behavior.
    """
    矩陣加法 (Matrix addition)

    A + B: 對應元素相加
    條件：A 和 B 必須大小相同
    """  # EN: Execute statement: """.
    rows_A, cols_A = get_shape(A)  # EN: Execute statement: rows_A, cols_A = get_shape(A).
    rows_B, cols_B = get_shape(B)  # EN: Execute statement: rows_B, cols_B = get_shape(B).

    if rows_A != rows_B or cols_A != cols_B:  # EN: Branch on a condition: if rows_A != rows_B or cols_A != cols_B:.
        raise ValueError(f"矩陣大小不符：{rows_A}×{cols_A} vs {rows_B}×{cols_B}")  # EN: Raise an exception: raise ValueError(f"矩陣大小不符：{rows_A}×{cols_A} vs {rows_B}×{cols_B}").

    result = create_matrix(rows_A, cols_A)  # EN: Assign result from expression: create_matrix(rows_A, cols_A).
    for i in range(rows_A):  # EN: Iterate with a for-loop: for i in range(rows_A):.
        for j in range(cols_A):  # EN: Iterate with a for-loop: for j in range(cols_A):.
            result[i][j] = A[i][j] + B[i][j]  # EN: Execute statement: result[i][j] = A[i][j] + B[i][j].

    return result  # EN: Return a value: return result.


def matrix_subtract(A: Matrix, B: Matrix) -> Matrix:  # EN: Define matrix_subtract and its behavior.
    """
    矩陣減法 (Matrix subtraction)

    A - B: 對應元素相減
    """  # EN: Execute statement: """.
    rows_A, cols_A = get_shape(A)  # EN: Execute statement: rows_A, cols_A = get_shape(A).
    rows_B, cols_B = get_shape(B)  # EN: Execute statement: rows_B, cols_B = get_shape(B).

    if rows_A != rows_B or cols_A != cols_B:  # EN: Branch on a condition: if rows_A != rows_B or cols_A != cols_B:.
        raise ValueError(f"矩陣大小不符：{rows_A}×{cols_A} vs {rows_B}×{cols_B}")  # EN: Raise an exception: raise ValueError(f"矩陣大小不符：{rows_A}×{cols_A} vs {rows_B}×{cols_B}").

    result = create_matrix(rows_A, cols_A)  # EN: Assign result from expression: create_matrix(rows_A, cols_A).
    for i in range(rows_A):  # EN: Iterate with a for-loop: for i in range(rows_A):.
        for j in range(cols_A):  # EN: Iterate with a for-loop: for j in range(cols_A):.
            result[i][j] = A[i][j] - B[i][j]  # EN: Execute statement: result[i][j] = A[i][j] - B[i][j].

    return result  # EN: Return a value: return result.


def scalar_multiply(c: float, A: Matrix) -> Matrix:  # EN: Define scalar_multiply and its behavior.
    """
    純量乘法 (Scalar multiplication)

    c·A: 每個元素乘以 c
    """  # EN: Execute statement: """.
    rows, cols = get_shape(A)  # EN: Execute statement: rows, cols = get_shape(A).
    result = create_matrix(rows, cols)  # EN: Assign result from expression: create_matrix(rows, cols).

    for i in range(rows):  # EN: Iterate with a for-loop: for i in range(rows):.
        for j in range(cols):  # EN: Iterate with a for-loop: for j in range(cols):.
            result[i][j] = c * A[i][j]  # EN: Execute statement: result[i][j] = c * A[i][j].

    return result  # EN: Return a value: return result.


def transpose(A: Matrix) -> Matrix:  # EN: Define transpose and its behavior.
    """
    矩陣轉置 (Matrix transpose)

    Aᵀ: 第 i 列變成第 i 行
    (Aᵀ)ᵢⱼ = Aⱼᵢ

    m×n 矩陣的轉置是 n×m 矩陣
    """  # EN: Execute statement: """.
    rows, cols = get_shape(A)  # EN: Execute statement: rows, cols = get_shape(A).

    # 轉置後：原本的 cols 變成 rows
    result = create_matrix(cols, rows)  # EN: Assign result from expression: create_matrix(cols, rows).

    for i in range(rows):  # EN: Iterate with a for-loop: for i in range(rows):.
        for j in range(cols):  # EN: Iterate with a for-loop: for j in range(cols):.
            result[j][i] = A[i][j]  # EN: Execute statement: result[j][i] = A[i][j].

    return result  # EN: Return a value: return result.


def is_symmetric(A: Matrix, tolerance: float = 1e-10) -> bool:  # EN: Define is_symmetric and its behavior.
    """
    檢查矩陣是否對稱 (Check if matrix is symmetric)

    對稱條件：A = Aᵀ（即 aᵢⱼ = aⱼᵢ 對所有 i, j）
    只有方陣才可能對稱
    """  # EN: Execute statement: """.
    rows, cols = get_shape(A)  # EN: Execute statement: rows, cols = get_shape(A).

    # 必須是方陣
    if rows != cols:  # EN: Branch on a condition: if rows != cols:.
        return False  # EN: Return a value: return False.

    # 檢查 aᵢⱼ = aⱼᵢ
    for i in range(rows):  # EN: Iterate with a for-loop: for i in range(rows):.
        for j in range(i + 1, cols):  # 只需檢查上三角  # EN: Iterate with a for-loop: for j in range(i + 1, cols): # 只需檢查上三角.
            if abs(A[i][j] - A[j][i]) > tolerance:  # EN: Branch on a condition: if abs(A[i][j] - A[j][i]) > tolerance:.
                return False  # EN: Return a value: return False.

    return True  # EN: Return a value: return True.


def get_row(A: Matrix, row_index: int) -> Vector:  # EN: Define get_row and its behavior.
    """
    取出矩陣的某一列 (Get a row from matrix)
    """  # EN: Execute statement: """.
    rows, _ = get_shape(A)  # EN: Execute statement: rows, _ = get_shape(A).
    if row_index < 0 or row_index >= rows:  # EN: Branch on a condition: if row_index < 0 or row_index >= rows:.
        raise IndexError(f"列索引超出範圍：{row_index}")  # EN: Raise an exception: raise IndexError(f"列索引超出範圍：{row_index}").

    return A[row_index].copy()  # EN: Return a value: return A[row_index].copy().


def get_column(A: Matrix, col_index: int) -> Vector:  # EN: Define get_column and its behavior.
    """
    取出矩陣的某一行 (Get a column from matrix)
    """  # EN: Execute statement: """.
    rows, cols = get_shape(A)  # EN: Execute statement: rows, cols = get_shape(A).
    if col_index < 0 or col_index >= cols:  # EN: Branch on a condition: if col_index < 0 or col_index >= cols:.
        raise IndexError(f"行索引超出範圍：{col_index}")  # EN: Raise an exception: raise IndexError(f"行索引超出範圍：{col_index}").

    return [A[i][col_index] for i in range(rows)]  # EN: Return a value: return [A[i][col_index] for i in range(rows)].


def matrices_equal(A: Matrix, B: Matrix, tolerance: float = 1e-10) -> bool:  # EN: Define matrices_equal and its behavior.
    """
    檢查兩矩陣是否相等 (Check if two matrices are equal)
    """  # EN: Execute statement: """.
    rows_A, cols_A = get_shape(A)  # EN: Execute statement: rows_A, cols_A = get_shape(A).
    rows_B, cols_B = get_shape(B)  # EN: Execute statement: rows_B, cols_B = get_shape(B).

    if rows_A != rows_B or cols_A != cols_B:  # EN: Branch on a condition: if rows_A != rows_B or cols_A != cols_B:.
        return False  # EN: Return a value: return False.

    for i in range(rows_A):  # EN: Iterate with a for-loop: for i in range(rows_A):.
        for j in range(cols_A):  # EN: Iterate with a for-loop: for j in range(cols_A):.
            if abs(A[i][j] - B[i][j]) > tolerance:  # EN: Branch on a condition: if abs(A[i][j] - B[i][j]) > tolerance:.
                return False  # EN: Return a value: return False.

    return True  # EN: Return a value: return True.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線 (Print separator)"""  # EN: Execute statement: """印出分隔線 (Print separator)""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式 (Main program)"""  # EN: Execute statement: """主程式 (Main program)""".

    print_separator("矩陣運算示範 - 手刻版本\nMatrix Operations Demo - Manual Implementation")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 建立矩陣 (Creating Matrices)
    # ========================================
    print_separator("1. 建立矩陣 (Creating Matrices)")  # EN: Call print_separator(...) to perform an operation.

    A = [  # EN: Assign A from expression: [.
        [1.0, 2.0, 3.0],  # EN: Execute statement: [1.0, 2.0, 3.0],.
        [4.0, 5.0, 6.0]  # EN: Execute statement: [4.0, 5.0, 6.0].
    ]  # EN: Execute statement: ].

    B = [  # EN: Assign B from expression: [.
        [7.0, 8.0, 9.0],  # EN: Execute statement: [7.0, 8.0, 9.0],.
        [10.0, 11.0, 12.0]  # EN: Execute statement: [10.0, 11.0, 12.0].
    ]  # EN: Execute statement: ].

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("B", B)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 2. 矩陣加法 (Matrix Addition)
    # ========================================
    print_separator("2. 矩陣加法 (Matrix Addition)")  # EN: Call print_separator(...) to perform an operation.

    C = matrix_add(A, B)  # EN: Assign C from expression: matrix_add(A, B).
    print("A + B =")  # EN: Print formatted output to the console.
    print_matrix("C", C)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 3. 矩陣減法 (Matrix Subtraction)
    # ========================================
    print_separator("3. 矩陣減法 (Matrix Subtraction)")  # EN: Call print_separator(...) to perform an operation.

    D = matrix_subtract(B, A)  # EN: Assign D from expression: matrix_subtract(B, A).
    print("B - A =")  # EN: Print formatted output to the console.
    print_matrix("D", D)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 4. 純量乘法 (Scalar Multiplication)
    # ========================================
    print_separator("4. 純量乘法 (Scalar Multiplication)")  # EN: Call print_separator(...) to perform an operation.

    c = 2.0  # EN: Assign c from expression: 2.0.
    E = scalar_multiply(c, A)  # EN: Assign E from expression: scalar_multiply(c, A).
    print(f"{c} × A =")  # EN: Print formatted output to the console.
    print_matrix("E", E)  # EN: Call print_matrix(...) to perform an operation.

    # 負數純量
    F = scalar_multiply(-1, A)  # EN: Assign F from expression: scalar_multiply(-1, A).
    print("-A =")  # EN: Print formatted output to the console.
    print_matrix("F", F)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 5. 矩陣轉置 (Transpose)
    # ========================================
    print_separator("5. 矩陣轉置 (Transpose)")  # EN: Call print_separator(...) to perform an operation.

    print("原矩陣 A (2×3):")  # EN: Print formatted output to the console.
    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.

    At = transpose(A)  # EN: Assign At from expression: transpose(A).
    print("轉置後 Aᵀ (3×2):")  # EN: Print formatted output to the console.
    print_matrix("Aᵀ", At)  # EN: Call print_matrix(...) to perform an operation.

    # 驗證 (Aᵀ)ᵀ = A
    Att = transpose(At)  # EN: Assign Att from expression: transpose(At).
    print("驗證 (Aᵀ)ᵀ = A:")  # EN: Print formatted output to the console.
    print_matrix("(Aᵀ)ᵀ", Att)  # EN: Call print_matrix(...) to perform an operation.
    print(f"(Aᵀ)ᵀ == A ? {matrices_equal(Att, A)}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 轉置的性質 (Transpose Properties)
    # ========================================
    print_separator("6. 轉置的性質 (Transpose Properties)")  # EN: Call print_separator(...) to perform an operation.

    # (A + B)ᵀ = Aᵀ + Bᵀ
    sum_then_transpose = transpose(matrix_add(A, B))  # EN: Assign sum_then_transpose from expression: transpose(matrix_add(A, B)).
    transpose_then_sum = matrix_add(transpose(A), transpose(B))  # EN: Assign transpose_then_sum from expression: matrix_add(transpose(A), transpose(B)).

    print("驗證 (A + B)ᵀ = Aᵀ + Bᵀ:")  # EN: Print formatted output to the console.
    print_matrix("(A + B)ᵀ", sum_then_transpose)  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("Aᵀ + Bᵀ", transpose_then_sum)  # EN: Call print_matrix(...) to perform an operation.
    print(f"相等？ {matrices_equal(sum_then_transpose, transpose_then_sum)}")  # EN: Print formatted output to the console.

    # (cA)ᵀ = cAᵀ
    print("\n驗證 (cA)ᵀ = cAᵀ (c = 2):")  # EN: Print formatted output to the console.
    scaled_then_transpose = transpose(scalar_multiply(2, A))  # EN: Assign scaled_then_transpose from expression: transpose(scalar_multiply(2, A)).
    transpose_then_scaled = scalar_multiply(2, transpose(A))  # EN: Assign transpose_then_scaled from expression: scalar_multiply(2, transpose(A)).
    print(f"相等？ {matrices_equal(scaled_then_transpose, transpose_then_scaled)}")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 對稱矩陣 (Symmetric Matrix)
    # ========================================
    print_separator("7. 對稱矩陣 (Symmetric Matrix)")  # EN: Call print_separator(...) to perform an operation.

    # 對稱矩陣範例
    S = [  # EN: Assign S from expression: [.
        [1.0, 2.0, 3.0],  # EN: Execute statement: [1.0, 2.0, 3.0],.
        [2.0, 5.0, 6.0],  # EN: Execute statement: [2.0, 5.0, 6.0],.
        [3.0, 6.0, 9.0]  # EN: Execute statement: [3.0, 6.0, 9.0].
    ]  # EN: Execute statement: ].

    print_matrix("S（對稱矩陣）", S)  # EN: Call print_matrix(...) to perform an operation.
    print(f"S 是對稱矩陣？ {is_symmetric(S)}")  # EN: Print formatted output to the console.

    St = transpose(S)  # EN: Assign St from expression: transpose(S).
    print_matrix("Sᵀ", St)  # EN: Call print_matrix(...) to perform an operation.
    print(f"S == Sᵀ ? {matrices_equal(S, St)}")  # EN: Print formatted output to the console.

    # 非對稱矩陣
    N = [  # EN: Assign N from expression: [.
        [1.0, 2.0, 3.0],  # EN: Execute statement: [1.0, 2.0, 3.0],.
        [4.0, 5.0, 6.0],  # EN: Execute statement: [4.0, 5.0, 6.0],.
        [7.0, 8.0, 9.0]  # EN: Execute statement: [7.0, 8.0, 9.0].
    ]  # EN: Execute statement: ].

    print_matrix("N（非對稱矩陣）", N)  # EN: Call print_matrix(...) to perform an operation.
    print(f"N 是對稱矩陣？ {is_symmetric(N)}")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 列與行的提取 (Row and Column Extraction)
    # ========================================
    print_separator("8. 列與行的提取 (Row and Column Extraction)")  # EN: Call print_separator(...) to perform an operation.

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.

    row_0 = get_row(A, 0)  # EN: Assign row_0 from expression: get_row(A, 0).
    row_1 = get_row(A, 1)  # EN: Assign row_1 from expression: get_row(A, 1).
    print(f"第 0 列 (row 0): {row_0}")  # EN: Print formatted output to the console.
    print(f"第 1 列 (row 1): {row_1}")  # EN: Print formatted output to the console.

    col_0 = get_column(A, 0)  # EN: Assign col_0 from expression: get_column(A, 0).
    col_1 = get_column(A, 1)  # EN: Assign col_1 from expression: get_column(A, 1).
    col_2 = get_column(A, 2)  # EN: Assign col_2 from expression: get_column(A, 2).
    print(f"\n第 0 行 (col 0): {col_0}")  # EN: Print formatted output to the console.
    print(f"第 1 行 (col 1): {col_1}")  # EN: Print formatted output to the console.
    print(f"第 2 行 (col 2): {col_2}")  # EN: Print formatted output to the console.

    # ========================================
    # 9. 構造對稱矩陣 (Constructing Symmetric Matrix)
    # ========================================
    print_separator("9. 構造對稱矩陣：AᵀA 總是對稱")  # EN: Call print_separator(...) to perform an operation.

    # 任意矩陣 M
    M = [  # EN: Assign M from expression: [.
        [1.0, 2.0],  # EN: Execute statement: [1.0, 2.0],.
        [3.0, 4.0],  # EN: Execute statement: [3.0, 4.0],.
        [5.0, 6.0]  # EN: Execute statement: [5.0, 6.0].
    ]  # EN: Execute statement: ].
    print_matrix("M (3×2)", M)  # EN: Call print_matrix(...) to perform an operation.

    Mt = transpose(M)  # EN: Assign Mt from expression: transpose(M).
    print_matrix("Mᵀ (2×3)", Mt)  # EN: Call print_matrix(...) to perform an operation.

    # 注意：這裡我們還沒實作矩陣乘法，先用概念說明
    print("MᵀM 會是 2×2 的對稱矩陣")  # EN: Print formatted output to the console.
    print("MMᵀ 會是 3×3 的對稱矩陣")  # EN: Print formatted output to the console.
    print("（矩陣乘法會在下一個單元介紹）")  # EN: Print formatted output to the console.

    # ========================================
    # 10. 特殊矩陣預覽 (Special Matrices Preview)
    # ========================================
    print_separator("10. 特殊矩陣預覽 (Special Matrices Preview)")  # EN: Call print_separator(...) to perform an operation.

    # 零矩陣
    zero = create_matrix(2, 3, 0.0)  # EN: Assign zero from expression: create_matrix(2, 3, 0.0).
    print_matrix("零矩陣 O (2×3)", zero)  # EN: Call print_matrix(...) to perform an operation.

    # 方陣範例
    square = [  # EN: Assign square from expression: [.
        [1.0, 2.0, 3.0],  # EN: Execute statement: [1.0, 2.0, 3.0],.
        [4.0, 5.0, 6.0],  # EN: Execute statement: [4.0, 5.0, 6.0],.
        [7.0, 8.0, 9.0]  # EN: Execute statement: [7.0, 8.0, 9.0].
    ]  # EN: Execute statement: ].
    print_matrix("方陣 (3×3 square matrix)", square)  # EN: Call print_matrix(...) to perform an operation.

    print("其他特殊矩陣（單位矩陣、對角矩陣等）")  # EN: Print formatted output to the console.
    print("將在 04-special-matrices 單元詳細介紹")  # EN: Print formatted output to the console.

    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print("所有矩陣運算示範完成！")  # EN: Print formatted output to the console.
    print("All matrix operations demonstrated!")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
