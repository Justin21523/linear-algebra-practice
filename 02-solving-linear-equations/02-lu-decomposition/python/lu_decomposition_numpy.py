"""
LU 分解：NumPy/SciPy 版本 (LU Decomposition: NumPy/SciPy Implementation)

本程式示範：
1. scipy.linalg.lu 進行 PLU 分解
2. scipy.linalg.lu_factor 和 lu_solve
3. Cholesky 分解（對稱正定矩陣）
4. 效能比較

This program demonstrates LU decomposition using NumPy and SciPy.
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.
from scipy import linalg  # EN: Import symbol(s) from a module: from scipy import linalg.
import time  # EN: Import module(s): import time.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("LU 分解示範 - NumPy/SciPy 版本\nLU Decomposition Demo - NumPy/SciPy")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. scipy.linalg.lu：完整 PLU 分解
    # ========================================
    print_separator("1. scipy.linalg.lu：PLU 分解")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [2, 1, 1],  # EN: Execute statement: [2, 1, 1],.
        [4, -6, 0],  # EN: Execute statement: [4, -6, 0],.
        [-2, 7, 2]  # EN: Execute statement: [-2, 7, 2].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A =\n{A}\n")  # EN: Print formatted output to the console.

    # PLU 分解：PA = LU
    P, L, U = linalg.lu(A)  # EN: Execute statement: P, L, U = linalg.lu(A).

    print(f"P（置換矩陣）=\n{P}\n")  # EN: Print formatted output to the console.
    print(f"L（下三角）=\n{L}\n")  # EN: Print formatted output to the console.
    print(f"U（上三角）=\n{U}\n")  # EN: Print formatted output to the console.

    # 驗證 PLU = A（注意：P 是 P^T）
    print(f"驗證：P @ L @ U =\n{P @ L @ U}")  # EN: Print formatted output to the console.
    print(f"P @ L @ U == A ? {np.allclose(P @ L @ U, A)}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. scipy.linalg.lu_factor + lu_solve
    # ========================================
    print_separator("2. lu_factor + lu_solve（高效求解多個 b）")  # EN: Call print_separator(...) to perform an operation.

    print("lu_factor 返回緊湊格式，適合重複求解：")  # EN: Print formatted output to the console.

    # lu_factor 返回 (lu, piv)
    lu, piv = linalg.lu_factor(A)  # EN: Execute statement: lu, piv = linalg.lu_factor(A).
    print(f"lu（緊湊格式）=\n{lu}\n")  # EN: Print formatted output to the console.
    print(f"piv（置換索引）= {piv}")  # EN: Print formatted output to the console.

    # 求解多個右手邊
    b1 = np.array([5, -2, 9])  # EN: Assign b1 from expression: np.array([5, -2, 9]).
    b2 = np.array([1, 0, 0])  # EN: Assign b2 from expression: np.array([1, 0, 0]).
    b3 = np.array([0, 1, 0])  # EN: Assign b3 from expression: np.array([0, 1, 0]).

    print(f"\n求解 Ax = b1, b1 = {b1}")  # EN: Print formatted output to the console.
    x1 = linalg.lu_solve((lu, piv), b1)  # EN: Assign x1 from expression: linalg.lu_solve((lu, piv), b1).
    print(f"x1 = {x1}")  # EN: Print formatted output to the console.
    print(f"驗證：A @ x1 = {A @ x1}")  # EN: Print formatted output to the console.

    print(f"\n求解 Ax = b2, b2 = {b2}")  # EN: Print formatted output to the console.
    x2 = linalg.lu_solve((lu, piv), b2)  # EN: Assign x2 from expression: linalg.lu_solve((lu, piv), b2).
    print(f"x2 = {x2}")  # EN: Print formatted output to the console.

    print(f"\n求解 Ax = b3, b3 = {b3}")  # EN: Print formatted output to the console.
    x3 = linalg.lu_solve((lu, piv), b3)  # EN: Assign x3 from expression: linalg.lu_solve((lu, piv), b3).
    print(f"x3 = {x3}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 用 LU 分解計算行列式
    # ========================================
    print_separator("3. 用 LU 分解計算行列式")  # EN: Call print_separator(...) to perform an operation.

    print("det(A) = ±(U 的對角線乘積)")  # EN: Print formatted output to the console.
    print("符號取決於置換矩陣 P 的符號")  # EN: Print formatted output to the console.

    det_from_u = np.prod(np.diag(U))  # EN: Assign det_from_u from expression: np.prod(np.diag(U)).
    det_from_p = np.linalg.det(P)  # P 的行列式是 ±1  # EN: Assign det_from_p from expression: np.linalg.det(P) # P 的行列式是 ±1.

    print(f"U 的對角線 = {np.diag(U)}")  # EN: Print formatted output to the console.
    print(f"U 對角線乘積 = {det_from_u:.4f}")  # EN: Print formatted output to the console.
    print(f"det(P) = {det_from_p:.4f}")  # EN: Print formatted output to the console.
    print(f"det(A) = det(P) × det(L) × det(U) = {det_from_p * det_from_u:.4f}")  # EN: Print formatted output to the console.
    print(f"np.linalg.det(A) = {np.linalg.det(A):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 用 LU 分解計算反矩陣
    # ========================================
    print_separator("4. 用 LU 分解計算反矩陣")  # EN: Call print_separator(...) to perform an operation.

    print("解 AX = I 的每一行，得到 A⁻¹ 的每一行")  # EN: Print formatted output to the console.

    I = np.eye(3)  # EN: Assign I from expression: np.eye(3).
    A_inv = np.zeros((3, 3))  # EN: Assign A_inv from expression: np.zeros((3, 3)).

    for j in range(3):  # EN: Iterate with a for-loop: for j in range(3):.
        A_inv[:, j] = linalg.lu_solve((lu, piv), I[:, j])  # EN: Execute statement: A_inv[:, j] = linalg.lu_solve((lu, piv), I[:, j]).

    print(f"A⁻¹ =\n{A_inv}\n")  # EN: Print formatted output to the console.
    print(f"驗證：A @ A⁻¹ =\n{A @ A_inv}")  # EN: Print formatted output to the console.

    # ========================================
    # 5. Cholesky 分解（對稱正定矩陣）
    # ========================================
    print_separator("5. Cholesky 分解（對稱正定矩陣）")  # EN: Call print_separator(...) to perform an operation.

    print("對於對稱正定矩陣，A = LLᵀ（更高效）")  # EN: Print formatted output to the console.

    # 建立對稱正定矩陣
    B = np.array([  # EN: Assign B from expression: np.array([.
        [4, 2, 2],  # EN: Execute statement: [4, 2, 2],.
        [2, 5, 1],  # EN: Execute statement: [2, 5, 1],.
        [2, 1, 6]  # EN: Execute statement: [2, 1, 6].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"B（對稱正定）=\n{B}\n")  # EN: Print formatted output to the console.

    # 檢查正定性
    eigenvalues = np.linalg.eigvalsh(B)  # EN: Assign eigenvalues from expression: np.linalg.eigvalsh(B).
    print(f"特徵值（都應為正）: {eigenvalues}")  # EN: Print formatted output to the console.

    # Cholesky 分解
    L_chol = linalg.cholesky(B, lower=True)  # EN: Assign L_chol from expression: linalg.cholesky(B, lower=True).
    print(f"\nCholesky 分解 L =\n{L_chol}\n")  # EN: Print formatted output to the console.

    print(f"驗證：L @ Lᵀ =\n{L_chol @ L_chol.T}")  # EN: Print formatted output to the console.
    print(f"L @ Lᵀ == B ? {np.allclose(L_chol @ L_chol.T, B)}")  # EN: Print formatted output to the console.

    # 用 Cholesky 求解
    b_chol = np.array([1, 2, 3])  # EN: Assign b_chol from expression: np.array([1, 2, 3]).
    x_chol = linalg.cho_solve((L_chol, True), b_chol)  # EN: Assign x_chol from expression: linalg.cho_solve((L_chol, True), b_chol).
    print(f"\n求解 Bx = {b_chol}")  # EN: Print formatted output to the console.
    print(f"x = {x_chol}")  # EN: Print formatted output to the console.
    print(f"驗證：B @ x = {B @ x_chol}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 效能比較
    # ========================================
    print_separator("6. 效能比較：多個右手邊")  # EN: Call print_separator(...) to perform an operation.

    n = 500  # EN: Assign n from expression: 500.
    k = 100  # 解 k 個不同的 b  # EN: Assign k from expression: 100 # 解 k 個不同的 b.

    A_big = np.random.rand(n, n)  # EN: Assign A_big from expression: np.random.rand(n, n).
    B_big = np.random.rand(n, k)  # k 個右手邊  # EN: Assign B_big from expression: np.random.rand(n, k) # k 個右手邊.

    print(f"矩陣大小：{n}×{n}")  # EN: Print formatted output to the console.
    print(f"右手邊數量：{k}")  # EN: Print formatted output to the console.

    # 方法一：每次都用 np.linalg.solve
    start = time.time()  # EN: Assign start from expression: time.time().
    for j in range(k):  # EN: Iterate with a for-loop: for j in range(k):.
        x = np.linalg.solve(A_big, B_big[:, j])  # EN: Assign x from expression: np.linalg.solve(A_big, B_big[:, j]).
    time_solve = time.time() - start  # EN: Assign time_solve from expression: time.time() - start.

    # 方法二：先 lu_factor，再 lu_solve
    start = time.time()  # EN: Assign start from expression: time.time().
    lu_big, piv_big = linalg.lu_factor(A_big)  # EN: Execute statement: lu_big, piv_big = linalg.lu_factor(A_big).
    for j in range(k):  # EN: Iterate with a for-loop: for j in range(k):.
        x = linalg.lu_solve((lu_big, piv_big), B_big[:, j])  # EN: Assign x from expression: linalg.lu_solve((lu_big, piv_big), B_big[:, j]).
    time_lu = time.time() - start  # EN: Assign time_lu from expression: time.time() - start.

    # 方法三：一次解所有（矩陣右手邊）
    start = time.time()  # EN: Assign start from expression: time.time().
    X = np.linalg.solve(A_big, B_big)  # EN: Assign X from expression: np.linalg.solve(A_big, B_big).
    time_batch = time.time() - start  # EN: Assign time_batch from expression: time.time() - start.

    print(f"\n方法 1（每次 solve）：{time_solve*1000:.2f} ms")  # EN: Print formatted output to the console.
    print(f"方法 2（lu_factor + lu_solve）：{time_lu*1000:.2f} ms")  # EN: Print formatted output to the console.
    print(f"方法 3（批次求解）：{time_batch*1000:.2f} ms")  # EN: Print formatted output to the console.
    print(f"\nLU 分解加速比：{time_solve/time_lu:.2f}x")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 稀疏矩陣的 LU 分解
    # ========================================
    print_separator("7. 稀疏矩陣提示")  # EN: Call print_separator(...) to perform an operation.

    print("""
對於大型稀疏矩陣，使用 scipy.sparse.linalg：

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu, spsolve

A_sparse = csr_matrix(A)
lu_sparse = splu(A_sparse)
x = lu_sparse.solve(b)

優點：
- 只存儲非零元素
- 利用稀疏結構加速計算
- 可處理百萬維度的矩陣
""")  # EN: Execute statement: """).

    # ========================================
    # 8. 小結
    # ========================================
    print_separator("8. 小結：何時使用 LU 分解")  # EN: Call print_separator(...) to perform an operation.

    print("""
使用場景：
1. 需要對同一個 A 解多個 b → lu_factor + lu_solve
2. 需要計算行列式 → det = ±prod(diag(U))
3. 需要計算反矩陣 → 解 n 個系統
4. 對稱正定矩陣 → 使用 Cholesky（更快、更穩定）

不需要 LU 分解：
- 只解一次 Ax = b → 直接用 np.linalg.solve
- 矩陣很小（如 3×3）→ 直接計算
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("LU 分解示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
