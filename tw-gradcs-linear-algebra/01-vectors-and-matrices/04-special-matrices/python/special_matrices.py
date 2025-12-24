"""
特殊矩陣示範 (Special Matrices Demo)

本程式示範各種特殊矩陣的建立與性質驗證：
1. 單位矩陣 (Identity matrix)
2. 零矩陣 (Zero matrix)
3. 對角矩陣 (Diagonal matrix)
4. 對稱矩陣 (Symmetric matrix)
5. 三角矩陣 (Triangular matrix)
6. 正交矩陣 (Orthogonal matrix)
7. 置換矩陣 (Permutation matrix)

This program demonstrates various special matrices and their properties.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    """主程式"""

    print_separator("特殊矩陣示範\nSpecial Matrices Demo")

    # ========================================
    # 1. 單位矩陣 (Identity Matrix)
    # ========================================
    print_separator("1. 單位矩陣 (Identity Matrix)")

    I3 = np.eye(3)
    print(f"I₃ = np.eye(3):\n{I3}\n")

    # 性質：AI = IA = A
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    print(f"A:\n{A}\n")
    print(f"A @ I = A ? {np.allclose(A @ I3, A)}")
    print(f"I @ A = A ? {np.allclose(I3 @ A, A)}")

    # 性質：Ix = x
    x = np.array([1, 2, 3])
    print(f"\nx = {x}")
    print(f"I @ x = {I3 @ x}")
    print(f"I @ x == x ? {np.allclose(I3 @ x, x)}")

    # ========================================
    # 2. 零矩陣 (Zero Matrix)
    # ========================================
    print_separator("2. 零矩陣 (Zero Matrix)")

    O = np.zeros((3, 3))
    print(f"O = np.zeros((3, 3)):\n{O}\n")

    # 性質：A + O = A
    print(f"A + O = A ? {np.allclose(A + O, A)}")

    # 性質：AO = O
    print(f"A @ O 是零矩陣？ {np.allclose(A @ O, O)}")

    # 注意：AB = O 不代表 A = O 或 B = O
    print("\n⚠️ 注意：AB = O 不代表 A = O 或 B = O")
    X = np.array([[1, 0], [0, 0]])
    Y = np.array([[0, 0], [1, 0]])
    print(f"X:\n{X}\n")
    print(f"Y:\n{Y}\n")
    print(f"X @ Y:\n{X @ Y}")
    print("X @ Y = O，但 X ≠ O 且 Y ≠ O")

    # ========================================
    # 3. 對角矩陣 (Diagonal Matrix)
    # ========================================
    print_separator("3. 對角矩陣 (Diagonal Matrix)")

    d = [1, 2, 3]
    D = np.diag(d)
    print(f"D = np.diag({d}):\n{D}\n")

    # 從矩陣提取對角線
    print(f"np.diag(A) (提取對角線): {np.diag(A)}")

    # 對角矩陣相乘
    D1 = np.diag([2, 3])
    D2 = np.diag([4, 5])
    print(f"\nD1 = diag(2, 3), D2 = diag(4, 5)")
    print(f"D1 @ D2 = diag(2×4, 3×5) = diag(8, 15):\n{D1 @ D2}")

    # 對角矩陣的冪
    print(f"\nD1² = diag(4, 9):\n{D1 @ D1}")
    print(f"D1³ = diag(8, 27):\n{np.linalg.matrix_power(D1, 3)}")

    # 對角矩陣的逆
    D_inv = np.diag([1/2, 1/3])
    print(f"\nD1⁻¹ = diag(1/2, 1/3):\n{D_inv}")
    print(f"D1 @ D1⁻¹ = I ? {np.allclose(D1 @ D_inv, np.eye(2))}")

    # ========================================
    # 4. 對稱矩陣 (Symmetric Matrix)
    # ========================================
    print_separator("4. 對稱矩陣 (Symmetric Matrix)")

    S = np.array([
        [4, 2, 1],
        [2, 5, 3],
        [1, 3, 6]
    ], dtype=float)

    print(f"S:\n{S}\n")
    print(f"S.T:\n{S.T}\n")
    print(f"S 是對稱的？ S == S.T ? {np.allclose(S, S.T)}")

    # 對稱矩陣的特徵值（都是實數）
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    print(f"\n對稱矩陣的特徵值（都是實數）: {eigenvalues}")

    # 特徵向量互相正交
    print(f"\n特徵向量矩陣 Q:\n{eigenvectors}")
    print(f"Q.T @ Q (應該是 I):\n{eigenvectors.T @ eigenvectors}")

    # AᵀA 總是對稱的
    print("\nAᵀA 總是對稱矩陣：")
    M = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    MtM = M.T @ M
    print(f"M (3×2):\n{M}\n")
    print(f"MᵀM (2×2):\n{MtM}")
    print(f"MᵀM 是對稱的？ {np.allclose(MtM, MtM.T)}")

    # ========================================
    # 5. 三角矩陣 (Triangular Matrix)
    # ========================================
    print_separator("5. 三角矩陣 (Triangular Matrix)")

    A_full = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    U = np.triu(A_full)  # 上三角
    L = np.tril(A_full)  # 下三角

    print(f"原矩陣 A:\n{A_full}\n")
    print(f"上三角 np.triu(A):\n{U}\n")
    print(f"下三角 np.tril(A):\n{L}\n")

    # 三角矩陣的行列式 = 對角線乘積
    U2 = np.array([[2, 3, 1], [0, 4, 5], [0, 0, 3]], dtype=float)
    print(f"U2 (上三角):\n{U2}\n")
    print(f"det(U2) = {np.linalg.det(U2):.4f}")
    print(f"對角線乘積 = {2 * 4 * 3} = 24")

    # 上三角 × 上三角 = 上三角
    U3 = np.triu(np.random.rand(3, 3))
    U4 = np.triu(np.random.rand(3, 3))
    print(f"\n上三角 × 上三角 的結果也是上三角？")
    result = U3 @ U4
    print(f"結果:\n{result}")
    print(f"是上三角？ {np.allclose(result, np.triu(result))}")

    # ========================================
    # 6. 正交矩陣 (Orthogonal Matrix)
    # ========================================
    print_separator("6. 正交矩陣 (Orthogonal Matrix)")

    # 旋轉矩陣是正交矩陣
    theta = np.pi / 4  # 45 度
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    print(f"旋轉矩陣 R(45°):\n{R}\n")

    # 性質：QᵀQ = I
    print(f"R.T @ R (應該是 I):\n{R.T @ R}\n")
    print(f"是正交矩陣？ QᵀQ = I ? {np.allclose(R.T @ R, np.eye(2))}")

    # 性質：Q⁻¹ = Qᵀ
    R_inv = np.linalg.inv(R)
    print(f"\nR⁻¹:\n{R_inv}")
    print(f"R.T:\n{R.T}")
    print(f"R⁻¹ == R.T ? {np.allclose(R_inv, R.T)}")

    # 性質：保持長度
    v = np.array([1, 0])
    Rv = R @ v
    print(f"\nv = {v}, ‖v‖ = {np.linalg.norm(v)}")
    print(f"Rv = {Rv}, ‖Rv‖ = {np.linalg.norm(Rv)}")
    print(f"保持長度？ {np.isclose(np.linalg.norm(v), np.linalg.norm(Rv))}")

    # 行列式 = ±1
    print(f"\ndet(R) = {np.linalg.det(R):.4f}")
    print("det = +1 表示旋轉，det = -1 表示鏡射")

    # 鏡射矩陣範例
    Reflect = np.array([[1, 0], [0, -1]])  # x 軸鏡射
    print(f"\n鏡射矩陣:\n{Reflect}")
    print(f"det = {np.linalg.det(Reflect):.4f}")

    # ========================================
    # 7. 置換矩陣 (Permutation Matrix)
    # ========================================
    print_separator("7. 置換矩陣 (Permutation Matrix)")

    # 交換第 1 和第 3 列
    P = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ], dtype=float)

    print(f"置換矩陣 P:\n{P}\n")

    x = np.array([1, 2, 3])
    print(f"x = {x}")
    print(f"Px = {P @ x} (交換第 1 和第 3 元素)")

    # 置換矩陣是正交矩陣
    print(f"\nP 是正交矩陣？ PᵀP = I ? {np.allclose(P.T @ P, np.eye(3))}")

    # P⁻¹ = Pᵀ
    print(f"P⁻¹ = Pᵀ ? {np.allclose(np.linalg.inv(P), P.T)}")

    # ========================================
    # 8. 冪等矩陣 (Idempotent Matrix)
    # ========================================
    print_separator("8. 冪等矩陣 (Idempotent Matrix): P² = P")

    # 投影矩陣是冪等的
    a = np.array([1, 1]) / np.sqrt(2)  # 單位向量
    P_proj = np.outer(a, a)  # 投影到 a 方向的投影矩陣

    print(f"投影向量 a = {a}")
    print(f"投影矩陣 P = aaᵀ:\n{P_proj}\n")

    print(f"P²:\n{P_proj @ P_proj}")
    print(f"P² == P ? {np.allclose(P_proj @ P_proj, P_proj)}")

    # 投影的幾何意義
    v = np.array([3, 1])
    proj_v = P_proj @ v
    print(f"\nv = {v}")
    print(f"Pv = {proj_v} (v 在 a 方向的投影)")
    print(f"P(Pv) = {P_proj @ proj_v} (投影兩次結果相同)")

    # ========================================
    # 9. 冪零矩陣 (Nilpotent Matrix)
    # ========================================
    print_separator("9. 冪零矩陣 (Nilpotent Matrix): Nᵏ = O")

    N = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=float)

    print(f"N:\n{N}\n")
    print(f"N²:\n{N @ N}\n")
    print(f"N³:\n{N @ N @ N}")
    print("\nN³ = O (零矩陣)")

    # ========================================
    # 10. 特殊矩陣的快速判斷
    # ========================================
    print_separator("10. NumPy 判斷矩陣類型")

    def analyze_matrix(M, name="M"):
        """分析矩陣的類型"""
        print(f"{name}:\n{M}\n")

        # 是否為方陣
        is_square = M.shape[0] == M.shape[1]
        print(f"  方陣？ {is_square}")

        if is_square:
            # 對稱
            is_symmetric = np.allclose(M, M.T)
            print(f"  對稱？ {is_symmetric}")

            # 正交
            is_orthogonal = np.allclose(M.T @ M, np.eye(M.shape[0]))
            print(f"  正交？ {is_orthogonal}")

            # 對角
            is_diagonal = np.allclose(M, np.diag(np.diag(M)))
            print(f"  對角？ {is_diagonal}")

            # 上三角
            is_upper = np.allclose(M, np.triu(M))
            print(f"  上三角？ {is_upper}")

            # 下三角
            is_lower = np.allclose(M, np.tril(M))
            print(f"  下三角？ {is_lower}")

        print()

    test_matrices = {
        "單位矩陣": np.eye(3),
        "對角矩陣": np.diag([1, 2, 3]),
        "對稱矩陣": np.array([[1, 2], [2, 1]]),
        "旋轉矩陣": R,
        "上三角矩陣": np.triu(np.ones((3, 3)))
    }

    for name, matrix in test_matrices.items():
        analyze_matrix(matrix, name)

    print("=" * 60)
    print("特殊矩陣示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
