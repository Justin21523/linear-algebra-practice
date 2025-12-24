"""
矩陣乘法：手刻版本 - 四種觀點 (Matrix Multiplication: Four Views)

本程式示範矩陣乘法的四種理解方式：
1. 內積觀點 (Dot product view)
2. 行的線性組合 (Column view)
3. 列的線性組合 (Row view)
4. 外積的和 (Sum of outer products)

This program demonstrates the four ways to understand matrix multiplication,
as emphasized by Gilbert Strang.
"""  # EN: Execute statement: """.

from typing import List  # EN: Import symbol(s) from a module: from typing import List.

# 型別別名 (Type alias)
Matrix = List[List[float]]  # EN: Assign Matrix from expression: List[List[float]].
Vector = List[float]  # EN: Assign Vector from expression: List[float].


def get_shape(A: Matrix) -> tuple:  # EN: Define get_shape and its behavior.
    """取得矩陣大小 (Get matrix shape)"""  # EN: Execute statement: """取得矩陣大小 (Get matrix shape)""".
    rows = len(A)  # EN: Assign rows from expression: len(A).
    cols = len(A[0]) if rows > 0 else 0  # EN: Assign cols from expression: len(A[0]) if rows > 0 else 0.
    return (rows, cols)  # EN: Return a value: return (rows, cols).


def print_matrix(name: str, A: Matrix) -> None:  # EN: Define print_matrix and its behavior.
    """印出矩陣 (Print matrix)"""  # EN: Execute statement: """印出矩陣 (Print matrix)""".
    rows, cols = get_shape(A)  # EN: Execute statement: rows, cols = get_shape(A).
    print(f"{name} ({rows}×{cols}):")  # EN: Print formatted output to the console.
    for row in A:  # EN: Iterate with a for-loop: for row in A:.
        print("  [", end="")  # EN: Print formatted output to the console.
        print("  ".join(f"{x:8.4f}" for x in row), end="")  # EN: Print formatted output to the console.
        print(" ]")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.


def print_vector(name: str, v: Vector) -> None:  # EN: Define print_vector and its behavior.
    """印出向量 (Print vector)"""  # EN: Execute statement: """印出向量 (Print vector)""".
    formatted = ", ".join(f"{x:.4f}" for x in v)  # EN: Assign formatted from expression: ", ".join(f"{x:.4f}" for x in v).
    print(f"{name} = [{formatted}]")  # EN: Print formatted output to the console.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線 (Print separator)"""  # EN: Execute statement: """印出分隔線 (Print separator)""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def create_zero_matrix(rows: int, cols: int) -> Matrix:  # EN: Define create_zero_matrix and its behavior.
    """建立零矩陣 (Create zero matrix)"""  # EN: Execute statement: """建立零矩陣 (Create zero matrix)""".
    return [[0.0 for _ in range(cols)] for _ in range(rows)]  # EN: Return a value: return [[0.0 for _ in range(cols)] for _ in range(rows)].


def get_row(A: Matrix, i: int) -> Vector:  # EN: Define get_row and its behavior.
    """取出第 i 列 (Get row i)"""  # EN: Execute statement: """取出第 i 列 (Get row i)""".
    return A[i].copy()  # EN: Return a value: return A[i].copy().


def get_column(A: Matrix, j: int) -> Vector:  # EN: Define get_column and its behavior.
    """取出第 j 行 (Get column j)"""  # EN: Execute statement: """取出第 j 行 (Get column j)""".
    return [A[i][j] for i in range(len(A))]  # EN: Return a value: return [A[i][j] for i in range(len(A))].


def dot_product(u: Vector, v: Vector) -> float:  # EN: Define dot_product and its behavior.
    """內積 (Dot product)"""  # EN: Execute statement: """內積 (Dot product)""".
    return sum(u[i] * v[i] for i in range(len(u)))  # EN: Return a value: return sum(u[i] * v[i] for i in range(len(u))).


def scalar_multiply_vector(c: float, v: Vector) -> Vector:  # EN: Define scalar_multiply_vector and its behavior.
    """純量乘向量 (Scalar multiply vector)"""  # EN: Execute statement: """純量乘向量 (Scalar multiply vector)""".
    return [c * x for x in v]  # EN: Return a value: return [c * x for x in v].


def vector_add(u: Vector, v: Vector) -> Vector:  # EN: Define vector_add and its behavior.
    """向量加法 (Vector addition)"""  # EN: Execute statement: """向量加法 (Vector addition)""".
    return [u[i] + v[i] for i in range(len(u))]  # EN: Return a value: return [u[i] + v[i] for i in range(len(u))].


def outer_product(u: Vector, v: Vector) -> Matrix:  # EN: Define outer_product and its behavior.
    """
    外積 (Outer product)
    u ⊗ v = uvᵀ
    結果是 len(u) × len(v) 矩陣
    """  # EN: Execute statement: """.
    rows = len(u)  # EN: Assign rows from expression: len(u).
    cols = len(v)  # EN: Assign cols from expression: len(v).
    result = create_zero_matrix(rows, cols)  # EN: Assign result from expression: create_zero_matrix(rows, cols).
    for i in range(rows):  # EN: Iterate with a for-loop: for i in range(rows):.
        for j in range(cols):  # EN: Iterate with a for-loop: for j in range(cols):.
            result[i][j] = u[i] * v[j]  # EN: Execute statement: result[i][j] = u[i] * v[j].
    return result  # EN: Return a value: return result.


def matrix_add(A: Matrix, B: Matrix) -> Matrix:  # EN: Define matrix_add and its behavior.
    """矩陣加法 (Matrix addition)"""  # EN: Execute statement: """矩陣加法 (Matrix addition)""".
    rows, cols = get_shape(A)  # EN: Execute statement: rows, cols = get_shape(A).
    result = create_zero_matrix(rows, cols)  # EN: Assign result from expression: create_zero_matrix(rows, cols).
    for i in range(rows):  # EN: Iterate with a for-loop: for i in range(rows):.
        for j in range(cols):  # EN: Iterate with a for-loop: for j in range(cols):.
            result[i][j] = A[i][j] + B[i][j]  # EN: Execute statement: result[i][j] = A[i][j] + B[i][j].
    return result  # EN: Return a value: return result.


# ============================================================
# 四種矩陣乘法的實作
# Four implementations of matrix multiplication
# ============================================================

def matrix_multiply_standard(A: Matrix, B: Matrix) -> Matrix:  # EN: Define matrix_multiply_standard and its behavior.
    """
    標準矩陣乘法（三層迴圈）
    Standard matrix multiplication (triple nested loop)

    Cᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ
    """  # EN: Execute statement: """.
    rows_A, cols_A = get_shape(A)  # EN: Execute statement: rows_A, cols_A = get_shape(A).
    rows_B, cols_B = get_shape(B)  # EN: Execute statement: rows_B, cols_B = get_shape(B).

    if cols_A != rows_B:  # EN: Branch on a condition: if cols_A != rows_B:.
        raise ValueError(f"維度不相容：{rows_A}×{cols_A} 和 {rows_B}×{cols_B}")  # EN: Raise an exception: raise ValueError(f"維度不相容：{rows_A}×{cols_A} 和 {rows_B}×{cols_B}").

    C = create_zero_matrix(rows_A, cols_B)  # EN: Assign C from expression: create_zero_matrix(rows_A, cols_B).

    for i in range(rows_A):  # EN: Iterate with a for-loop: for i in range(rows_A):.
        for j in range(cols_B):  # EN: Iterate with a for-loop: for j in range(cols_B):.
            for k in range(cols_A):  # EN: Iterate with a for-loop: for k in range(cols_A):.
                C[i][j] += A[i][k] * B[k][j]  # EN: Execute statement: C[i][j] += A[i][k] * B[k][j].

    return C  # EN: Return a value: return C.


def matrix_multiply_dot_view(A: Matrix, B: Matrix) -> Matrix:  # EN: Define matrix_multiply_dot_view and its behavior.
    """
    觀點一：內積觀點 (Dot Product View)

    Cᵢⱼ = (A 的第 i 列) · (B 的第 j 行)

    每個元素是一個內積的結果
    """  # EN: Execute statement: """.
    rows_A, cols_A = get_shape(A)  # EN: Execute statement: rows_A, cols_A = get_shape(A).
    rows_B, cols_B = get_shape(B)  # EN: Execute statement: rows_B, cols_B = get_shape(B).

    if cols_A != rows_B:  # EN: Branch on a condition: if cols_A != rows_B:.
        raise ValueError("維度不相容")  # EN: Raise an exception: raise ValueError("維度不相容").

    C = create_zero_matrix(rows_A, cols_B)  # EN: Assign C from expression: create_zero_matrix(rows_A, cols_B).

    for i in range(rows_A):  # EN: Iterate with a for-loop: for i in range(rows_A):.
        row_i = get_row(A, i)  # EN: Assign row_i from expression: get_row(A, i).
        for j in range(cols_B):  # EN: Iterate with a for-loop: for j in range(cols_B):.
            col_j = get_column(B, j)  # EN: Assign col_j from expression: get_column(B, j).
            C[i][j] = dot_product(row_i, col_j)  # EN: Execute statement: C[i][j] = dot_product(row_i, col_j).

    return C  # EN: Return a value: return C.


def matrix_multiply_column_view(A: Matrix, B: Matrix) -> Matrix:  # EN: Define matrix_multiply_column_view and its behavior.
    """
    觀點二：行的線性組合 (Column View)

    C 的第 j 行 = A 的各行的線性組合
                = B[0,j]*A[:,0] + B[1,j]*A[:,1] + ...

    係數來自 B 的第 j 行
    """  # EN: Execute statement: """.
    rows_A, cols_A = get_shape(A)  # EN: Execute statement: rows_A, cols_A = get_shape(A).
    rows_B, cols_B = get_shape(B)  # EN: Execute statement: rows_B, cols_B = get_shape(B).

    if cols_A != rows_B:  # EN: Branch on a condition: if cols_A != rows_B:.
        raise ValueError("維度不相容")  # EN: Raise an exception: raise ValueError("維度不相容").

    C = create_zero_matrix(rows_A, cols_B)  # EN: Assign C from expression: create_zero_matrix(rows_A, cols_B).

    # 對每一行 j
    for j in range(cols_B):  # EN: Iterate with a for-loop: for j in range(cols_B):.
        # C 的第 j 行是 A 的各行的線性組合
        col_j = [0.0] * rows_A  # EN: Assign col_j from expression: [0.0] * rows_A.

        for k in range(cols_A):  # EN: Iterate with a for-loop: for k in range(cols_A):.
            # 係數是 B[k][j]
            coef = B[k][j]  # EN: Assign coef from expression: B[k][j].
            # A 的第 k 行
            a_col_k = get_column(A, k)  # EN: Assign a_col_k from expression: get_column(A, k).
            # 加權累加
            col_j = vector_add(col_j, scalar_multiply_vector(coef, a_col_k))  # EN: Assign col_j from expression: vector_add(col_j, scalar_multiply_vector(coef, a_col_k)).

        # 把結果放入 C 的第 j 行
        for i in range(rows_A):  # EN: Iterate with a for-loop: for i in range(rows_A):.
            C[i][j] = col_j[i]  # EN: Execute statement: C[i][j] = col_j[i].

    return C  # EN: Return a value: return C.


def matrix_multiply_row_view(A: Matrix, B: Matrix) -> Matrix:  # EN: Define matrix_multiply_row_view and its behavior.
    """
    觀點三：列的線性組合 (Row View)

    C 的第 i 列 = B 的各列的線性組合
               = A[i,0]*B[0,:] + A[i,1]*B[1,:] + ...

    係數來自 A 的第 i 列
    """  # EN: Execute statement: """.
    rows_A, cols_A = get_shape(A)  # EN: Execute statement: rows_A, cols_A = get_shape(A).
    rows_B, cols_B = get_shape(B)  # EN: Execute statement: rows_B, cols_B = get_shape(B).

    if cols_A != rows_B:  # EN: Branch on a condition: if cols_A != rows_B:.
        raise ValueError("維度不相容")  # EN: Raise an exception: raise ValueError("維度不相容").

    C = create_zero_matrix(rows_A, cols_B)  # EN: Assign C from expression: create_zero_matrix(rows_A, cols_B).

    # 對每一列 i
    for i in range(rows_A):  # EN: Iterate with a for-loop: for i in range(rows_A):.
        # C 的第 i 列是 B 的各列的線性組合
        row_i = [0.0] * cols_B  # EN: Assign row_i from expression: [0.0] * cols_B.

        for k in range(cols_A):  # EN: Iterate with a for-loop: for k in range(cols_A):.
            # 係數是 A[i][k]
            coef = A[i][k]  # EN: Assign coef from expression: A[i][k].
            # B 的第 k 列
            b_row_k = get_row(B, k)  # EN: Assign b_row_k from expression: get_row(B, k).
            # 加權累加
            row_i = vector_add(row_i, scalar_multiply_vector(coef, b_row_k))  # EN: Assign row_i from expression: vector_add(row_i, scalar_multiply_vector(coef, b_row_k)).

        # 把結果放入 C 的第 i 列
        C[i] = row_i  # EN: Execute statement: C[i] = row_i.

    return C  # EN: Return a value: return C.


def matrix_multiply_outer_view(A: Matrix, B: Matrix) -> Matrix:  # EN: Define matrix_multiply_outer_view and its behavior.
    """
    觀點四：外積的和 (Sum of Outer Products)

    AB = Σₖ (A 的第 k 行) ⊗ (B 的第 k 列)

    每個外積是一個秩 1 的矩陣
    """  # EN: Execute statement: """.
    rows_A, cols_A = get_shape(A)  # EN: Execute statement: rows_A, cols_A = get_shape(A).
    rows_B, cols_B = get_shape(B)  # EN: Execute statement: rows_B, cols_B = get_shape(B).

    if cols_A != rows_B:  # EN: Branch on a condition: if cols_A != rows_B:.
        raise ValueError("維度不相容")  # EN: Raise an exception: raise ValueError("維度不相容").

    C = create_zero_matrix(rows_A, cols_B)  # EN: Assign C from expression: create_zero_matrix(rows_A, cols_B).

    for k in range(cols_A):  # EN: Iterate with a for-loop: for k in range(cols_A):.
        # A 的第 k 行
        a_col_k = get_column(A, k)  # EN: Assign a_col_k from expression: get_column(A, k).
        # B 的第 k 列
        b_row_k = get_row(B, k)  # EN: Assign b_row_k from expression: get_row(B, k).
        # 外積
        outer = outer_product(a_col_k, b_row_k)  # EN: Assign outer from expression: outer_product(a_col_k, b_row_k).
        # 累加
        C = matrix_add(C, outer)  # EN: Assign C from expression: matrix_add(C, outer).

    return C  # EN: Return a value: return C.


def matrix_vector_multiply(A: Matrix, x: Vector) -> Vector:  # EN: Define matrix_vector_multiply and its behavior.
    """
    矩陣與向量相乘 (Matrix-vector multiplication)
    Ax = x₁a₁ + x₂a₂ + ... (A 的行的線性組合)
    """  # EN: Execute statement: """.
    rows, cols = get_shape(A)  # EN: Execute statement: rows, cols = get_shape(A).

    if cols != len(x):  # EN: Branch on a condition: if cols != len(x):.
        raise ValueError("維度不相容")  # EN: Raise an exception: raise ValueError("維度不相容").

    result = [0.0] * rows  # EN: Assign result from expression: [0.0] * rows.
    for j in range(cols):  # EN: Iterate with a for-loop: for j in range(cols):.
        col_j = get_column(A, j)  # EN: Assign col_j from expression: get_column(A, j).
        result = vector_add(result, scalar_multiply_vector(x[j], col_j))  # EN: Assign result from expression: vector_add(result, scalar_multiply_vector(x[j], col_j)).

    return result  # EN: Return a value: return result.


def main():  # EN: Define main and its behavior.
    """主程式 (Main program)"""  # EN: Execute statement: """主程式 (Main program)""".

    print_separator("矩陣乘法示範 - 四種觀點\nMatrix Multiplication - Four Views")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 定義範例矩陣 (Define example matrices)
    # ========================================
    A = [  # EN: Assign A from expression: [.
        [1.0, 2.0],  # EN: Execute statement: [1.0, 2.0],.
        [3.0, 4.0]  # EN: Execute statement: [3.0, 4.0].
    ]  # EN: Execute statement: ].

    B = [  # EN: Assign B from expression: [.
        [5.0, 6.0],  # EN: Execute statement: [5.0, 6.0],.
        [7.0, 8.0]  # EN: Execute statement: [7.0, 8.0].
    ]  # EN: Execute statement: ].

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("B", B)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 觀點一：內積 (Dot Product View)
    # ========================================
    print_separator("觀點一：內積 (Dot Product View)")  # EN: Call print_separator(...) to perform an operation.

    print("Cᵢⱼ = (A 的第 i 列) · (B 的第 j 行)")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    # 手動展示計算過程
    print("C₁₁ = [1, 2] · [5, 7] = 1×5 + 2×7 = 19")  # EN: Print formatted output to the console.
    print("C₁₂ = [1, 2] · [6, 8] = 1×6 + 2×8 = 22")  # EN: Print formatted output to the console.
    print("C₂₁ = [3, 4] · [5, 7] = 3×5 + 4×7 = 43")  # EN: Print formatted output to the console.
    print("C₂₂ = [3, 4] · [6, 8] = 3×6 + 4×8 = 50")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    C_dot = matrix_multiply_dot_view(A, B)  # EN: Assign C_dot from expression: matrix_multiply_dot_view(A, B).
    print_matrix("C = AB（內積觀點）", C_dot)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 觀點二：行的線性組合 (Column View)
    # ========================================
    print_separator("觀點二：行的線性組合 (Column View)")  # EN: Call print_separator(...) to perform an operation.

    print("C 的每一行 = A 的各行的線性組合")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    print("C 的第 1 行 = 5×[1,3] + 7×[2,4] = [5,15] + [14,28] = [19,43]")  # EN: Print formatted output to the console.
    print("C 的第 2 行 = 6×[1,3] + 8×[2,4] = [6,18] + [16,32] = [22,50]")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    C_col = matrix_multiply_column_view(A, B)  # EN: Assign C_col from expression: matrix_multiply_column_view(A, B).
    print_matrix("C = AB（行觀點）", C_col)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 觀點三：列的線性組合 (Row View)
    # ========================================
    print_separator("觀點三：列的線性組合 (Row View)")  # EN: Call print_separator(...) to perform an operation.

    print("C 的每一列 = B 的各列的線性組合")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    print("C 的第 1 列 = 1×[5,6] + 2×[7,8] = [5,6] + [14,16] = [19,22]")  # EN: Print formatted output to the console.
    print("C 的第 2 列 = 3×[5,6] + 4×[7,8] = [15,18] + [28,32] = [43,50]")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    C_row = matrix_multiply_row_view(A, B)  # EN: Assign C_row from expression: matrix_multiply_row_view(A, B).
    print_matrix("C = AB（列觀點）", C_row)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 觀點四：外積的和 (Outer Product View)
    # ========================================
    print_separator("觀點四：外積的和 (Sum of Outer Products)")  # EN: Call print_separator(...) to perform an operation.

    print("AB = (A 的第 1 行)⊗(B 的第 1 列) + (A 的第 2 行)⊗(B 的第 2 列)")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    # 第一個外積
    a_col_1 = get_column(A, 0)  # [1, 3]  # EN: Assign a_col_1 from expression: get_column(A, 0) # [1, 3].
    b_row_1 = get_row(B, 0)     # [5, 6]  # EN: Assign b_row_1 from expression: get_row(B, 0) # [5, 6].
    outer_1 = outer_product(a_col_1, b_row_1)  # EN: Assign outer_1 from expression: outer_product(a_col_1, b_row_1).

    print(f"[1, 3]ᵀ × [5, 6] =")  # EN: Print formatted output to the console.
    print_matrix("外積 1", outer_1)  # EN: Call print_matrix(...) to perform an operation.

    # 第二個外積
    a_col_2 = get_column(A, 1)  # [2, 4]  # EN: Assign a_col_2 from expression: get_column(A, 1) # [2, 4].
    b_row_2 = get_row(B, 1)     # [7, 8]  # EN: Assign b_row_2 from expression: get_row(B, 1) # [7, 8].
    outer_2 = outer_product(a_col_2, b_row_2)  # EN: Assign outer_2 from expression: outer_product(a_col_2, b_row_2).

    print(f"[2, 4]ᵀ × [7, 8] =")  # EN: Print formatted output to the console.
    print_matrix("外積 2", outer_2)  # EN: Call print_matrix(...) to perform an operation.

    print("AB = 外積1 + 外積2 =")  # EN: Print formatted output to the console.
    C_outer = matrix_multiply_outer_view(A, B)  # EN: Assign C_outer from expression: matrix_multiply_outer_view(A, B).
    print_matrix("C", C_outer)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 驗證四種方法結果相同
    # ========================================
    print_separator("驗證：四種方法結果相同")  # EN: Call print_separator(...) to perform an operation.

    def matrices_equal(M1: Matrix, M2: Matrix) -> bool:  # EN: Define matrices_equal and its behavior.
        rows, cols = get_shape(M1)  # EN: Execute statement: rows, cols = get_shape(M1).
        for i in range(rows):  # EN: Iterate with a for-loop: for i in range(rows):.
            for j in range(cols):  # EN: Iterate with a for-loop: for j in range(cols):.
                if abs(M1[i][j] - M2[i][j]) > 1e-10:  # EN: Branch on a condition: if abs(M1[i][j] - M2[i][j]) > 1e-10:.
                    return False  # EN: Return a value: return False.
        return True  # EN: Return a value: return True.

    print(f"內積觀點 == 行觀點？ {matrices_equal(C_dot, C_col)}")  # EN: Print formatted output to the console.
    print(f"行觀點 == 列觀點？ {matrices_equal(C_col, C_row)}")  # EN: Print formatted output to the console.
    print(f"列觀點 == 外積觀點？ {matrices_equal(C_row, C_outer)}")  # EN: Print formatted output to the console.

    # ========================================
    # 矩陣乘向量 (Matrix-Vector Multiplication)
    # ========================================
    print_separator("矩陣乘向量 Ax (Matrix-Vector Multiplication)")  # EN: Call print_separator(...) to perform an operation.

    x = [3.0, 2.0]  # EN: Assign x from expression: [3.0, 2.0].
    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print_vector("x", x)  # EN: Call print_vector(...) to perform an operation.

    print("\n觀點：Ax = x₁×(A的第1行) + x₂×(A的第2行)")  # EN: Print formatted output to the console.
    print(f"Ax = {x[0]}×[1,3] + {x[1]}×[2,4]")  # EN: Print formatted output to the console.
    print(f"   = [3,9] + [4,8]")  # EN: Print formatted output to the console.
    print(f"   = [7, 17]")  # EN: Print formatted output to the console.

    result = matrix_vector_multiply(A, x)  # EN: Assign result from expression: matrix_vector_multiply(A, x).
    print_vector("\nAx", result)  # EN: Call print_vector(...) to perform an operation.

    # ========================================
    # 不滿足交換律 (Non-Commutative)
    # ========================================
    print_separator("矩陣乘法不滿足交換律")  # EN: Call print_separator(...) to perform an operation.

    print_matrix("AB", matrix_multiply_standard(A, B))  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("BA", matrix_multiply_standard(B, A))  # EN: Call print_matrix(...) to perform an operation.
    print("AB ≠ BA（一般情況下）")  # EN: Print formatted output to the console.

    # ========================================
    # 維度不相容的例子
    # ========================================
    print_separator("維度相容性 (Dimension Compatibility)")  # EN: Call print_separator(...) to perform an operation.

    M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # 2×3  # EN: Assign M from expression: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] # 2×3.
    N = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # 3×2  # EN: Assign N from expression: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] # 3×2.

    print_matrix("M (2×3)", M)  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("N (3×2)", N)  # EN: Call print_matrix(...) to perform an operation.

    MN = matrix_multiply_standard(M, N)  # EN: Assign MN from expression: matrix_multiply_standard(M, N).
    print("M × N (2×3 × 3×2 = 2×2):")  # EN: Print formatted output to the console.
    print_matrix("MN", MN)  # EN: Call print_matrix(...) to perform an operation.

    NM = matrix_multiply_standard(N, M)  # EN: Assign NM from expression: matrix_multiply_standard(N, M).
    print("N × M (3×2 × 2×3 = 3×3):")  # EN: Print formatted output to the console.
    print_matrix("NM", NM)  # EN: Call print_matrix(...) to perform an operation.

    print("注意：MN 是 2×2，NM 是 3×3，大小不同！")  # EN: Print formatted output to the console.

    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print("矩陣乘法示範完成！")  # EN: Print formatted output to the console.
    print("Matrix multiplication demo completed!")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
