"""
矩陣乘法：NumPy 版本 (Matrix Multiplication: NumPy Implementation)

本程式示範：
1. NumPy 的矩陣乘法語法
2. @ 運算子 vs np.dot vs np.matmul
3. 廣播機制與批次乘法
4. 效能比較

This program demonstrates matrix multiplication using NumPy.
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.
import time  # EN: Import module(s): import time.

# 設定輸出格式
np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("矩陣乘法示範 - NumPy 版本\nMatrix Multiplication - NumPy")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 基本矩陣乘法 (Basic Matrix Multiplication)
    # ========================================
    print_separator("1. 基本矩陣乘法 (Basic Matrix Multiplication)")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 2],  # EN: Execute statement: [1, 2],.
        [3, 4]  # EN: Execute statement: [3, 4].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    B = np.array([  # EN: Assign B from expression: np.array([.
        [5, 6],  # EN: Execute statement: [5, 6],.
        [7, 8]  # EN: Execute statement: [7, 8].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A:\n{A}\n")  # EN: Print formatted output to the console.
    print(f"B:\n{B}\n")  # EN: Print formatted output to the console.

    # 方法一：@ 運算子（推薦，Python 3.5+）
    C1 = A @ B  # EN: Assign C1 from expression: A @ B.
    print(f"A @ B:\n{C1}\n")  # EN: Print formatted output to the console.

    # 方法二：np.matmul
    C2 = np.matmul(A, B)  # EN: Assign C2 from expression: np.matmul(A, B).
    print(f"np.matmul(A, B):\n{C2}\n")  # EN: Print formatted output to the console.

    # 方法三：np.dot（對 2D 陣列效果相同）
    C3 = np.dot(A, B)  # EN: Assign C3 from expression: np.dot(A, B).
    print(f"np.dot(A, B):\n{C3}\n")  # EN: Print formatted output to the console.

    print("三種方法結果相同（對於 2D 矩陣）")  # EN: Print formatted output to the console.

    # ========================================
    # ⚠️ 注意：* 是元素相乘，不是矩陣乘法
    # ========================================
    print_separator("2. ⚠️ 注意：* 是元素相乘")  # EN: Call print_separator(...) to perform an operation.

    print(f"A * B (元素相乘，不是矩陣乘法!):\n{A * B}\n")  # EN: Print formatted output to the console.
    print(f"A @ B (矩陣乘法):\n{A @ B}\n")  # EN: Print formatted output to the console.
    print("記住：@ 是矩陣乘法，* 是元素相乘！")  # EN: Print formatted output to the console.

    # ========================================
    # 矩陣與向量相乘 (Matrix-Vector Multiplication)
    # ========================================
    print_separator("3. 矩陣與向量相乘 (Matrix-Vector)")  # EN: Call print_separator(...) to perform an operation.

    x = np.array([3.0, 2.0])  # EN: Assign x from expression: np.array([3.0, 2.0]).
    print(f"A:\n{A}\n")  # EN: Print formatted output to the console.
    print(f"x: {x}\n")  # EN: Print formatted output to the console.
    print(f"A @ x = {A @ x}")  # EN: Print formatted output to the console.

    print("\n觀點：Ax 是 A 的行向量的線性組合")  # EN: Print formatted output to the console.
    print(f"= {x[0]}*A[:,0] + {x[1]}*A[:,1]")  # EN: Print formatted output to the console.
    print(f"= {x[0]}*{A[:,0]} + {x[1]}*{A[:,1]}")  # EN: Print formatted output to the console.
    print(f"= {x[0]*A[:,0]} + {x[1]*A[:,1]}")  # EN: Print formatted output to the console.
    print(f"= {x[0]*A[:,0] + x[1]*A[:,1]}")  # EN: Print formatted output to the console.

    # ========================================
    # 四種觀點的 NumPy 實作
    # ========================================
    print_separator("4. 四種觀點的 NumPy 實作")  # EN: Call print_separator(...) to perform an operation.

    # 觀點一：內積
    print("觀點一：內積")  # EN: Print formatted output to the console.
    C_dot = np.zeros((2, 2))  # EN: Assign C_dot from expression: np.zeros((2, 2)).
    for i in range(2):  # EN: Iterate with a for-loop: for i in range(2):.
        for j in range(2):  # EN: Iterate with a for-loop: for j in range(2):.
            C_dot[i, j] = np.dot(A[i, :], B[:, j])  # EN: Execute statement: C_dot[i, j] = np.dot(A[i, :], B[:, j]).
    print(f"結果:\n{C_dot}\n")  # EN: Print formatted output to the console.

    # 觀點二：行的線性組合
    print("觀點二：行的線性組合")  # EN: Print formatted output to the console.
    C_col = np.zeros((2, 2))  # EN: Assign C_col from expression: np.zeros((2, 2)).
    for j in range(2):  # EN: Iterate with a for-loop: for j in range(2):.
        # C 的第 j 行 = B[0,j]*A[:,0] + B[1,j]*A[:,1]
        C_col[:, j] = B[0, j] * A[:, 0] + B[1, j] * A[:, 1]  # EN: Execute statement: C_col[:, j] = B[0, j] * A[:, 0] + B[1, j] * A[:, 1].
    print(f"結果:\n{C_col}\n")  # EN: Print formatted output to the console.

    # 觀點三：列的線性組合
    print("觀點三：列的線性組合")  # EN: Print formatted output to the console.
    C_row = np.zeros((2, 2))  # EN: Assign C_row from expression: np.zeros((2, 2)).
    for i in range(2):  # EN: Iterate with a for-loop: for i in range(2):.
        # C 的第 i 列 = A[i,0]*B[0,:] + A[i,1]*B[1,:]
        C_row[i, :] = A[i, 0] * B[0, :] + A[i, 1] * B[1, :]  # EN: Execute statement: C_row[i, :] = A[i, 0] * B[0, :] + A[i, 1] * B[1, :].
    print(f"結果:\n{C_row}\n")  # EN: Print formatted output to the console.

    # 觀點四：外積的和
    print("觀點四：外積的和")  # EN: Print formatted output to the console.
    outer1 = np.outer(A[:, 0], B[0, :])  # EN: Assign outer1 from expression: np.outer(A[:, 0], B[0, :]).
    outer2 = np.outer(A[:, 1], B[1, :])  # EN: Assign outer2 from expression: np.outer(A[:, 1], B[1, :]).
    print(f"外積 1 (A[:,0] ⊗ B[0,:]):\n{outer1}\n")  # EN: Print formatted output to the console.
    print(f"外積 2 (A[:,1] ⊗ B[1,:]):\n{outer2}\n")  # EN: Print formatted output to the console.
    C_outer = outer1 + outer2  # EN: Assign C_outer from expression: outer1 + outer2.
    print(f"總和:\n{C_outer}\n")  # EN: Print formatted output to the console.

    # ========================================
    # 維度相容性 (Dimension Compatibility)
    # ========================================
    print_separator("5. 維度相容性 (Dimension Compatibility)")  # EN: Call print_separator(...) to perform an operation.

    M = np.array([[1, 2, 3], [4, 5, 6]])      # 2×3  # EN: Assign M from expression: np.array([[1, 2, 3], [4, 5, 6]]) # 2×3.
    N = np.array([[1, 2], [3, 4], [5, 6]])    # 3×2  # EN: Assign N from expression: np.array([[1, 2], [3, 4], [5, 6]]) # 3×2.

    print(f"M (shape {M.shape}):\n{M}\n")  # EN: Print formatted output to the console.
    print(f"N (shape {N.shape}):\n{N}\n")  # EN: Print formatted output to the console.

    MN = M @ N  # EN: Assign MN from expression: M @ N.
    print(f"M @ N (shape {MN.shape}):\n{MN}\n")  # EN: Print formatted output to the console.

    NM = N @ M  # EN: Assign NM from expression: N @ M.
    print(f"N @ M (shape {NM.shape}):\n{NM}\n")  # EN: Print formatted output to the console.

    print("M@N 是 2×2，N@M 是 3×3")  # EN: Print formatted output to the console.

    # ========================================
    # 轉置性質 (Transpose Property)
    # ========================================
    print_separator("6. 轉置性質：(AB)ᵀ = BᵀAᵀ")  # EN: Call print_separator(...) to perform an operation.

    print(f"(A @ B).T:\n{(A @ B).T}\n")  # EN: Print formatted output to the console.
    print(f"B.T @ A.T:\n{B.T @ A.T}\n")  # EN: Print formatted output to the console.
    print(f"相等？ {np.allclose((A @ B).T, B.T @ A.T)}")  # EN: Print formatted output to the console.

    # ========================================
    # 矩陣冪次 (Matrix Powers)
    # ========================================
    print_separator("7. 矩陣冪次 (Matrix Powers)")  # EN: Call print_separator(...) to perform an operation.

    print(f"A:\n{A}\n")  # EN: Print formatted output to the console.
    print(f"A² = A @ A:\n{A @ A}\n")  # EN: Print formatted output to the console.
    print(f"A³ = A @ A @ A:\n{A @ A @ A}\n")  # EN: Print formatted output to the console.

    # 使用 np.linalg.matrix_power
    print(f"np.linalg.matrix_power(A, 5):\n{np.linalg.matrix_power(A, 5)}\n")  # EN: Print formatted output to the console.

    # ========================================
    # 批次矩陣乘法 (Batch Matrix Multiplication)
    # ========================================
    print_separator("8. 批次矩陣乘法 (Batch Multiplication)")  # EN: Call print_separator(...) to perform an operation.

    # 3 個 2×2 矩陣
    batch_A = np.array([  # EN: Assign batch_A from expression: np.array([.
        [[1, 2], [3, 4]],  # EN: Execute statement: [[1, 2], [3, 4]],.
        [[5, 6], [7, 8]],  # EN: Execute statement: [[5, 6], [7, 8]],.
        [[9, 10], [11, 12]]  # EN: Execute statement: [[9, 10], [11, 12]].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    batch_B = np.array([  # EN: Assign batch_B from expression: np.array([.
        [[1, 0], [0, 1]],  # EN: Execute statement: [[1, 0], [0, 1]],.
        [[2, 0], [0, 2]],  # EN: Execute statement: [[2, 0], [0, 2]],.
        [[3, 0], [0, 3]]  # EN: Execute statement: [[3, 0], [0, 3]].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"batch_A shape: {batch_A.shape}")  # EN: Print formatted output to the console.
    print(f"batch_B shape: {batch_B.shape}")  # EN: Print formatted output to the console.

    # @ 運算子支援批次乘法
    batch_C = batch_A @ batch_B  # EN: Assign batch_C from expression: batch_A @ batch_B.
    print(f"\nbatch_A @ batch_B (逐一相乘):")  # EN: Print formatted output to the console.
    for i in range(3):  # EN: Iterate with a for-loop: for i in range(3):.
        print(f"  結果 {i}:\n{batch_C[i]}")  # EN: Print formatted output to the console.

    # ========================================
    # 效能比較 (Performance Comparison)
    # ========================================
    print_separator("9. 效能比較 (Performance)")  # EN: Call print_separator(...) to perform an operation.

    n = 200  # EN: Assign n from expression: 200.
    A_large = np.random.rand(n, n)  # EN: Assign A_large from expression: np.random.rand(n, n).
    B_large = np.random.rand(n, n)  # EN: Assign B_large from expression: np.random.rand(n, n).

    # NumPy 矩陣乘法
    start = time.time()  # EN: Assign start from expression: time.time().
    for _ in range(10):  # EN: Iterate with a for-loop: for _ in range(10):.
        C_numpy = A_large @ B_large  # EN: Assign C_numpy from expression: A_large @ B_large.
    numpy_time = (time.time() - start) / 10  # EN: Assign numpy_time from expression: (time.time() - start) / 10.

    # Python 迴圈（僅做小規模示範）
    n_small = 50  # EN: Assign n_small from expression: 50.
    A_small = np.random.rand(n_small, n_small)  # EN: Assign A_small from expression: np.random.rand(n_small, n_small).
    B_small = np.random.rand(n_small, n_small)  # EN: Assign B_small from expression: np.random.rand(n_small, n_small).

    start = time.time()  # EN: Assign start from expression: time.time().
    C_loop = np.zeros((n_small, n_small))  # EN: Assign C_loop from expression: np.zeros((n_small, n_small)).
    for i in range(n_small):  # EN: Iterate with a for-loop: for i in range(n_small):.
        for j in range(n_small):  # EN: Iterate with a for-loop: for j in range(n_small):.
            for k in range(n_small):  # EN: Iterate with a for-loop: for k in range(n_small):.
                C_loop[i, j] += A_small[i, k] * B_small[k, j]  # EN: Execute statement: C_loop[i, j] += A_small[i, k] * B_small[k, j].
    loop_time = time.time() - start  # EN: Assign loop_time from expression: time.time() - start.

    print(f"NumPy {n}×{n} 矩陣乘法: {numpy_time*1000:.2f} ms")  # EN: Print formatted output to the console.
    print(f"Python 迴圈 {n_small}×{n_small} 矩陣乘法: {loop_time*1000:.2f} ms")  # EN: Print formatted output to the console.
    print(f"\n估計 NumPy 快約 {(loop_time / numpy_time * (n/n_small)**3):.0f} 倍")  # EN: Print formatted output to the console.
    print("永遠優先使用 NumPy 的向量化運算！")  # EN: Print formatted output to the console.

    # ========================================
    # einsum：愛因斯坦求和 (Einstein Summation)
    # ========================================
    print_separator("10. einsum：愛因斯坦求和 (Advanced)")  # EN: Call print_separator(...) to perform an operation.

    print("np.einsum 是強大的張量運算工具")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    # 矩陣乘法
    C_ein = np.einsum('ik,kj->ij', A, B)  # EN: Assign C_ein from expression: np.einsum('ik,kj->ij', A, B).
    print(f"矩陣乘法 np.einsum('ik,kj->ij', A, B):\n{C_ein}\n")  # EN: Print formatted output to the console.

    # 內積
    u = np.array([1, 2, 3])  # EN: Assign u from expression: np.array([1, 2, 3]).
    v = np.array([4, 5, 6])  # EN: Assign v from expression: np.array([4, 5, 6]).
    dot_ein = np.einsum('i,i->', u, v)  # EN: Assign dot_ein from expression: np.einsum('i,i->', u, v).
    print(f"內積 np.einsum('i,i->', u, v) = {dot_ein}")  # EN: Print formatted output to the console.
    print(f"驗證 np.dot(u, v) = {np.dot(u, v)}")  # EN: Print formatted output to the console.

    # 外積
    outer_ein = np.einsum('i,j->ij', u, v)  # EN: Assign outer_ein from expression: np.einsum('i,j->ij', u, v).
    print(f"\n外積 np.einsum('i,j->ij', u, v):\n{outer_ein}")  # EN: Print formatted output to the console.

    # 跡（trace）
    trace_ein = np.einsum('ii->', A)  # EN: Assign trace_ein from expression: np.einsum('ii->', A).
    print(f"\n跡 np.einsum('ii->', A) = {trace_ein}")  # EN: Print formatted output to the console.
    print(f"驗證 np.trace(A) = {np.trace(A)}")  # EN: Print formatted output to the console.

    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print("NumPy 矩陣乘法示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
