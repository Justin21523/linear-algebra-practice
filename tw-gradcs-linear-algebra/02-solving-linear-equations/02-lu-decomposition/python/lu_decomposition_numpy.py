"""
LU 分解：NumPy/SciPy 版本 (LU Decomposition: NumPy/SciPy Implementation)

本程式示範：
1. scipy.linalg.lu 進行 PLU 分解
2. scipy.linalg.lu_factor 和 lu_solve
3. Cholesky 分解（對稱正定矩陣）
4. 效能比較

This program demonstrates LU decomposition using NumPy and SciPy.
"""

import numpy as np
from scipy import linalg
import time

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    """主程式"""

    print_separator("LU 分解示範 - NumPy/SciPy 版本\nLU Decomposition Demo - NumPy/SciPy")

    # ========================================
    # 1. scipy.linalg.lu：完整 PLU 分解
    # ========================================
    print_separator("1. scipy.linalg.lu：PLU 分解")

    A = np.array([
        [2, 1, 1],
        [4, -6, 0],
        [-2, 7, 2]
    ], dtype=float)

    print(f"A =\n{A}\n")

    # PLU 分解：PA = LU
    P, L, U = linalg.lu(A)

    print(f"P（置換矩陣）=\n{P}\n")
    print(f"L（下三角）=\n{L}\n")
    print(f"U（上三角）=\n{U}\n")

    # 驗證 PLU = A（注意：P 是 P^T）
    print(f"驗證：P @ L @ U =\n{P @ L @ U}")
    print(f"P @ L @ U == A ? {np.allclose(P @ L @ U, A)}")

    # ========================================
    # 2. scipy.linalg.lu_factor + lu_solve
    # ========================================
    print_separator("2. lu_factor + lu_solve（高效求解多個 b）")

    print("lu_factor 返回緊湊格式，適合重複求解：")

    # lu_factor 返回 (lu, piv)
    lu, piv = linalg.lu_factor(A)
    print(f"lu（緊湊格式）=\n{lu}\n")
    print(f"piv（置換索引）= {piv}")

    # 求解多個右手邊
    b1 = np.array([5, -2, 9])
    b2 = np.array([1, 0, 0])
    b3 = np.array([0, 1, 0])

    print(f"\n求解 Ax = b1, b1 = {b1}")
    x1 = linalg.lu_solve((lu, piv), b1)
    print(f"x1 = {x1}")
    print(f"驗證：A @ x1 = {A @ x1}")

    print(f"\n求解 Ax = b2, b2 = {b2}")
    x2 = linalg.lu_solve((lu, piv), b2)
    print(f"x2 = {x2}")

    print(f"\n求解 Ax = b3, b3 = {b3}")
    x3 = linalg.lu_solve((lu, piv), b3)
    print(f"x3 = {x3}")

    # ========================================
    # 3. 用 LU 分解計算行列式
    # ========================================
    print_separator("3. 用 LU 分解計算行列式")

    print("det(A) = ±(U 的對角線乘積)")
    print("符號取決於置換矩陣 P 的符號")

    det_from_u = np.prod(np.diag(U))
    det_from_p = np.linalg.det(P)  # P 的行列式是 ±1

    print(f"U 的對角線 = {np.diag(U)}")
    print(f"U 對角線乘積 = {det_from_u:.4f}")
    print(f"det(P) = {det_from_p:.4f}")
    print(f"det(A) = det(P) × det(L) × det(U) = {det_from_p * det_from_u:.4f}")
    print(f"np.linalg.det(A) = {np.linalg.det(A):.4f}")

    # ========================================
    # 4. 用 LU 分解計算反矩陣
    # ========================================
    print_separator("4. 用 LU 分解計算反矩陣")

    print("解 AX = I 的每一行，得到 A⁻¹ 的每一行")

    I = np.eye(3)
    A_inv = np.zeros((3, 3))

    for j in range(3):
        A_inv[:, j] = linalg.lu_solve((lu, piv), I[:, j])

    print(f"A⁻¹ =\n{A_inv}\n")
    print(f"驗證：A @ A⁻¹ =\n{A @ A_inv}")

    # ========================================
    # 5. Cholesky 分解（對稱正定矩陣）
    # ========================================
    print_separator("5. Cholesky 分解（對稱正定矩陣）")

    print("對於對稱正定矩陣，A = LLᵀ（更高效）")

    # 建立對稱正定矩陣
    B = np.array([
        [4, 2, 2],
        [2, 5, 1],
        [2, 1, 6]
    ], dtype=float)

    print(f"B（對稱正定）=\n{B}\n")

    # 檢查正定性
    eigenvalues = np.linalg.eigvalsh(B)
    print(f"特徵值（都應為正）: {eigenvalues}")

    # Cholesky 分解
    L_chol = linalg.cholesky(B, lower=True)
    print(f"\nCholesky 分解 L =\n{L_chol}\n")

    print(f"驗證：L @ Lᵀ =\n{L_chol @ L_chol.T}")
    print(f"L @ Lᵀ == B ? {np.allclose(L_chol @ L_chol.T, B)}")

    # 用 Cholesky 求解
    b_chol = np.array([1, 2, 3])
    x_chol = linalg.cho_solve((L_chol, True), b_chol)
    print(f"\n求解 Bx = {b_chol}")
    print(f"x = {x_chol}")
    print(f"驗證：B @ x = {B @ x_chol}")

    # ========================================
    # 6. 效能比較
    # ========================================
    print_separator("6. 效能比較：多個右手邊")

    n = 500
    k = 100  # 解 k 個不同的 b

    A_big = np.random.rand(n, n)
    B_big = np.random.rand(n, k)  # k 個右手邊

    print(f"矩陣大小：{n}×{n}")
    print(f"右手邊數量：{k}")

    # 方法一：每次都用 np.linalg.solve
    start = time.time()
    for j in range(k):
        x = np.linalg.solve(A_big, B_big[:, j])
    time_solve = time.time() - start

    # 方法二：先 lu_factor，再 lu_solve
    start = time.time()
    lu_big, piv_big = linalg.lu_factor(A_big)
    for j in range(k):
        x = linalg.lu_solve((lu_big, piv_big), B_big[:, j])
    time_lu = time.time() - start

    # 方法三：一次解所有（矩陣右手邊）
    start = time.time()
    X = np.linalg.solve(A_big, B_big)
    time_batch = time.time() - start

    print(f"\n方法 1（每次 solve）：{time_solve*1000:.2f} ms")
    print(f"方法 2（lu_factor + lu_solve）：{time_lu*1000:.2f} ms")
    print(f"方法 3（批次求解）：{time_batch*1000:.2f} ms")
    print(f"\nLU 分解加速比：{time_solve/time_lu:.2f}x")

    # ========================================
    # 7. 稀疏矩陣的 LU 分解
    # ========================================
    print_separator("7. 稀疏矩陣提示")

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
""")

    # ========================================
    # 8. 小結
    # ========================================
    print_separator("8. 小結：何時使用 LU 分解")

    print("""
使用場景：
1. 需要對同一個 A 解多個 b → lu_factor + lu_solve
2. 需要計算行列式 → det = ±prod(diag(U))
3. 需要計算反矩陣 → 解 n 個系統
4. 對稱正定矩陣 → 使用 Cholesky（更快、更穩定）

不需要 LU 分解：
- 只解一次 Ax = b → 直接用 np.linalg.solve
- 矩陣很小（如 3×3）→ 直接計算
""")

    print("=" * 60)
    print("LU 分解示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
