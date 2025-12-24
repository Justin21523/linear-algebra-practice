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
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("特殊矩陣示範\nSpecial Matrices Demo")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 單位矩陣 (Identity Matrix)
    # ========================================
    print_separator("1. 單位矩陣 (Identity Matrix)")  # EN: Call print_separator(...) to perform an operation.

    I3 = np.eye(3)  # EN: Assign I3 from expression: np.eye(3).
    print(f"I₃ = np.eye(3):\n{I3}\n")  # EN: Print formatted output to the console.

    # 性質：AI = IA = A
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)  # EN: Assign A from expression: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float).
    print(f"A:\n{A}\n")  # EN: Print formatted output to the console.
    print(f"A @ I = A ? {np.allclose(A @ I3, A)}")  # EN: Print formatted output to the console.
    print(f"I @ A = A ? {np.allclose(I3 @ A, A)}")  # EN: Print formatted output to the console.

    # 性質：Ix = x
    x = np.array([1, 2, 3])  # EN: Assign x from expression: np.array([1, 2, 3]).
    print(f"\nx = {x}")  # EN: Print formatted output to the console.
    print(f"I @ x = {I3 @ x}")  # EN: Print formatted output to the console.
    print(f"I @ x == x ? {np.allclose(I3 @ x, x)}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 零矩陣 (Zero Matrix)
    # ========================================
    print_separator("2. 零矩陣 (Zero Matrix)")  # EN: Call print_separator(...) to perform an operation.

    O = np.zeros((3, 3))  # EN: Assign O from expression: np.zeros((3, 3)).
    print(f"O = np.zeros((3, 3)):\n{O}\n")  # EN: Print formatted output to the console.

    # 性質：A + O = A
    print(f"A + O = A ? {np.allclose(A + O, A)}")  # EN: Print formatted output to the console.

    # 性質：AO = O
    print(f"A @ O 是零矩陣？ {np.allclose(A @ O, O)}")  # EN: Print formatted output to the console.

    # 注意：AB = O 不代表 A = O 或 B = O
    print("\n⚠️ 注意：AB = O 不代表 A = O 或 B = O")  # EN: Print formatted output to the console.
    X = np.array([[1, 0], [0, 0]])  # EN: Assign X from expression: np.array([[1, 0], [0, 0]]).
    Y = np.array([[0, 0], [1, 0]])  # EN: Assign Y from expression: np.array([[0, 0], [1, 0]]).
    print(f"X:\n{X}\n")  # EN: Print formatted output to the console.
    print(f"Y:\n{Y}\n")  # EN: Print formatted output to the console.
    print(f"X @ Y:\n{X @ Y}")  # EN: Print formatted output to the console.
    print("X @ Y = O，但 X ≠ O 且 Y ≠ O")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 對角矩陣 (Diagonal Matrix)
    # ========================================
    print_separator("3. 對角矩陣 (Diagonal Matrix)")  # EN: Call print_separator(...) to perform an operation.

    d = [1, 2, 3]  # EN: Assign d from expression: [1, 2, 3].
    D = np.diag(d)  # EN: Assign D from expression: np.diag(d).
    print(f"D = np.diag({d}):\n{D}\n")  # EN: Print formatted output to the console.

    # 從矩陣提取對角線
    print(f"np.diag(A) (提取對角線): {np.diag(A)}")  # EN: Print formatted output to the console.

    # 對角矩陣相乘
    D1 = np.diag([2, 3])  # EN: Assign D1 from expression: np.diag([2, 3]).
    D2 = np.diag([4, 5])  # EN: Assign D2 from expression: np.diag([4, 5]).
    print(f"\nD1 = diag(2, 3), D2 = diag(4, 5)")  # EN: Print formatted output to the console.
    print(f"D1 @ D2 = diag(2×4, 3×5) = diag(8, 15):\n{D1 @ D2}")  # EN: Print formatted output to the console.

    # 對角矩陣的冪
    print(f"\nD1² = diag(4, 9):\n{D1 @ D1}")  # EN: Print formatted output to the console.
    print(f"D1³ = diag(8, 27):\n{np.linalg.matrix_power(D1, 3)}")  # EN: Print formatted output to the console.

    # 對角矩陣的逆
    D_inv = np.diag([1/2, 1/3])  # EN: Assign D_inv from expression: np.diag([1/2, 1/3]).
    print(f"\nD1⁻¹ = diag(1/2, 1/3):\n{D_inv}")  # EN: Print formatted output to the console.
    print(f"D1 @ D1⁻¹ = I ? {np.allclose(D1 @ D_inv, np.eye(2))}")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 對稱矩陣 (Symmetric Matrix)
    # ========================================
    print_separator("4. 對稱矩陣 (Symmetric Matrix)")  # EN: Call print_separator(...) to perform an operation.

    S = np.array([  # EN: Assign S from expression: np.array([.
        [4, 2, 1],  # EN: Execute statement: [4, 2, 1],.
        [2, 5, 3],  # EN: Execute statement: [2, 5, 3],.
        [1, 3, 6]  # EN: Execute statement: [1, 3, 6].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"S:\n{S}\n")  # EN: Print formatted output to the console.
    print(f"S.T:\n{S.T}\n")  # EN: Print formatted output to the console.
    print(f"S 是對稱的？ S == S.T ? {np.allclose(S, S.T)}")  # EN: Print formatted output to the console.

    # 對稱矩陣的特徵值（都是實數）
    eigenvalues, eigenvectors = np.linalg.eigh(S)  # EN: Execute statement: eigenvalues, eigenvectors = np.linalg.eigh(S).
    print(f"\n對稱矩陣的特徵值（都是實數）: {eigenvalues}")  # EN: Print formatted output to the console.

    # 特徵向量互相正交
    print(f"\n特徵向量矩陣 Q:\n{eigenvectors}")  # EN: Print formatted output to the console.
    print(f"Q.T @ Q (應該是 I):\n{eigenvectors.T @ eigenvectors}")  # EN: Print formatted output to the console.

    # AᵀA 總是對稱的
    print("\nAᵀA 總是對稱矩陣：")  # EN: Print formatted output to the console.
    M = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)  # EN: Assign M from expression: np.array([[1, 2], [3, 4], [5, 6]], dtype=float).
    MtM = M.T @ M  # EN: Assign MtM from expression: M.T @ M.
    print(f"M (3×2):\n{M}\n")  # EN: Print formatted output to the console.
    print(f"MᵀM (2×2):\n{MtM}")  # EN: Print formatted output to the console.
    print(f"MᵀM 是對稱的？ {np.allclose(MtM, MtM.T)}")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 三角矩陣 (Triangular Matrix)
    # ========================================
    print_separator("5. 三角矩陣 (Triangular Matrix)")  # EN: Call print_separator(...) to perform an operation.

    A_full = np.array([  # EN: Assign A_full from expression: np.array([.
        [1, 2, 3],  # EN: Execute statement: [1, 2, 3],.
        [4, 5, 6],  # EN: Execute statement: [4, 5, 6],.
        [7, 8, 9]  # EN: Execute statement: [7, 8, 9].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    U = np.triu(A_full)  # 上三角  # EN: Assign U from expression: np.triu(A_full) # 上三角.
    L = np.tril(A_full)  # 下三角  # EN: Assign L from expression: np.tril(A_full) # 下三角.

    print(f"原矩陣 A:\n{A_full}\n")  # EN: Print formatted output to the console.
    print(f"上三角 np.triu(A):\n{U}\n")  # EN: Print formatted output to the console.
    print(f"下三角 np.tril(A):\n{L}\n")  # EN: Print formatted output to the console.

    # 三角矩陣的行列式 = 對角線乘積
    U2 = np.array([[2, 3, 1], [0, 4, 5], [0, 0, 3]], dtype=float)  # EN: Assign U2 from expression: np.array([[2, 3, 1], [0, 4, 5], [0, 0, 3]], dtype=float).
    print(f"U2 (上三角):\n{U2}\n")  # EN: Print formatted output to the console.
    print(f"det(U2) = {np.linalg.det(U2):.4f}")  # EN: Print formatted output to the console.
    print(f"對角線乘積 = {2 * 4 * 3} = 24")  # EN: Print formatted output to the console.

    # 上三角 × 上三角 = 上三角
    U3 = np.triu(np.random.rand(3, 3))  # EN: Assign U3 from expression: np.triu(np.random.rand(3, 3)).
    U4 = np.triu(np.random.rand(3, 3))  # EN: Assign U4 from expression: np.triu(np.random.rand(3, 3)).
    print(f"\n上三角 × 上三角 的結果也是上三角？")  # EN: Print formatted output to the console.
    result = U3 @ U4  # EN: Assign result from expression: U3 @ U4.
    print(f"結果:\n{result}")  # EN: Print formatted output to the console.
    print(f"是上三角？ {np.allclose(result, np.triu(result))}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 正交矩陣 (Orthogonal Matrix)
    # ========================================
    print_separator("6. 正交矩陣 (Orthogonal Matrix)")  # EN: Call print_separator(...) to perform an operation.

    # 旋轉矩陣是正交矩陣
    theta = np.pi / 4  # 45 度  # EN: Assign theta from expression: np.pi / 4 # 45 度.
    R = np.array([  # EN: Assign R from expression: np.array([.
        [np.cos(theta), -np.sin(theta)],  # EN: Execute statement: [np.cos(theta), -np.sin(theta)],.
        [np.sin(theta), np.cos(theta)]  # EN: Execute statement: [np.sin(theta), np.cos(theta)].
    ])  # EN: Execute statement: ]).

    print(f"旋轉矩陣 R(45°):\n{R}\n")  # EN: Print formatted output to the console.

    # 性質：QᵀQ = I
    print(f"R.T @ R (應該是 I):\n{R.T @ R}\n")  # EN: Print formatted output to the console.
    print(f"是正交矩陣？ QᵀQ = I ? {np.allclose(R.T @ R, np.eye(2))}")  # EN: Print formatted output to the console.

    # 性質：Q⁻¹ = Qᵀ
    R_inv = np.linalg.inv(R)  # EN: Assign R_inv from expression: np.linalg.inv(R).
    print(f"\nR⁻¹:\n{R_inv}")  # EN: Print formatted output to the console.
    print(f"R.T:\n{R.T}")  # EN: Print formatted output to the console.
    print(f"R⁻¹ == R.T ? {np.allclose(R_inv, R.T)}")  # EN: Print formatted output to the console.

    # 性質：保持長度
    v = np.array([1, 0])  # EN: Assign v from expression: np.array([1, 0]).
    Rv = R @ v  # EN: Assign Rv from expression: R @ v.
    print(f"\nv = {v}, ‖v‖ = {np.linalg.norm(v)}")  # EN: Print formatted output to the console.
    print(f"Rv = {Rv}, ‖Rv‖ = {np.linalg.norm(Rv)}")  # EN: Print formatted output to the console.
    print(f"保持長度？ {np.isclose(np.linalg.norm(v), np.linalg.norm(Rv))}")  # EN: Print formatted output to the console.

    # 行列式 = ±1
    print(f"\ndet(R) = {np.linalg.det(R):.4f}")  # EN: Print formatted output to the console.
    print("det = +1 表示旋轉，det = -1 表示鏡射")  # EN: Print formatted output to the console.

    # 鏡射矩陣範例
    Reflect = np.array([[1, 0], [0, -1]])  # x 軸鏡射  # EN: Assign Reflect from expression: np.array([[1, 0], [0, -1]]) # x 軸鏡射.
    print(f"\n鏡射矩陣:\n{Reflect}")  # EN: Print formatted output to the console.
    print(f"det = {np.linalg.det(Reflect):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 置換矩陣 (Permutation Matrix)
    # ========================================
    print_separator("7. 置換矩陣 (Permutation Matrix)")  # EN: Call print_separator(...) to perform an operation.

    # 交換第 1 和第 3 列
    P = np.array([  # EN: Assign P from expression: np.array([.
        [0, 0, 1],  # EN: Execute statement: [0, 0, 1],.
        [0, 1, 0],  # EN: Execute statement: [0, 1, 0],.
        [1, 0, 0]  # EN: Execute statement: [1, 0, 0].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"置換矩陣 P:\n{P}\n")  # EN: Print formatted output to the console.

    x = np.array([1, 2, 3])  # EN: Assign x from expression: np.array([1, 2, 3]).
    print(f"x = {x}")  # EN: Print formatted output to the console.
    print(f"Px = {P @ x} (交換第 1 和第 3 元素)")  # EN: Print formatted output to the console.

    # 置換矩陣是正交矩陣
    print(f"\nP 是正交矩陣？ PᵀP = I ? {np.allclose(P.T @ P, np.eye(3))}")  # EN: Print formatted output to the console.

    # P⁻¹ = Pᵀ
    print(f"P⁻¹ = Pᵀ ? {np.allclose(np.linalg.inv(P), P.T)}")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 冪等矩陣 (Idempotent Matrix)
    # ========================================
    print_separator("8. 冪等矩陣 (Idempotent Matrix): P² = P")  # EN: Call print_separator(...) to perform an operation.

    # 投影矩陣是冪等的
    a = np.array([1, 1]) / np.sqrt(2)  # 單位向量  # EN: Assign a from expression: np.array([1, 1]) / np.sqrt(2) # 單位向量.
    P_proj = np.outer(a, a)  # 投影到 a 方向的投影矩陣  # EN: Assign P_proj from expression: np.outer(a, a) # 投影到 a 方向的投影矩陣.

    print(f"投影向量 a = {a}")  # EN: Print formatted output to the console.
    print(f"投影矩陣 P = aaᵀ:\n{P_proj}\n")  # EN: Print formatted output to the console.

    print(f"P²:\n{P_proj @ P_proj}")  # EN: Print formatted output to the console.
    print(f"P² == P ? {np.allclose(P_proj @ P_proj, P_proj)}")  # EN: Print formatted output to the console.

    # 投影的幾何意義
    v = np.array([3, 1])  # EN: Assign v from expression: np.array([3, 1]).
    proj_v = P_proj @ v  # EN: Assign proj_v from expression: P_proj @ v.
    print(f"\nv = {v}")  # EN: Print formatted output to the console.
    print(f"Pv = {proj_v} (v 在 a 方向的投影)")  # EN: Print formatted output to the console.
    print(f"P(Pv) = {P_proj @ proj_v} (投影兩次結果相同)")  # EN: Print formatted output to the console.

    # ========================================
    # 9. 冪零矩陣 (Nilpotent Matrix)
    # ========================================
    print_separator("9. 冪零矩陣 (Nilpotent Matrix): Nᵏ = O")  # EN: Call print_separator(...) to perform an operation.

    N = np.array([  # EN: Assign N from expression: np.array([.
        [0, 1, 0],  # EN: Execute statement: [0, 1, 0],.
        [0, 0, 1],  # EN: Execute statement: [0, 0, 1],.
        [0, 0, 0]  # EN: Execute statement: [0, 0, 0].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"N:\n{N}\n")  # EN: Print formatted output to the console.
    print(f"N²:\n{N @ N}\n")  # EN: Print formatted output to the console.
    print(f"N³:\n{N @ N @ N}")  # EN: Print formatted output to the console.
    print("\nN³ = O (零矩陣)")  # EN: Print formatted output to the console.

    # ========================================
    # 10. 特殊矩陣的快速判斷
    # ========================================
    print_separator("10. NumPy 判斷矩陣類型")  # EN: Call print_separator(...) to perform an operation.

    def analyze_matrix(M, name="M"):  # EN: Define analyze_matrix and its behavior.
        """分析矩陣的類型"""  # EN: Execute statement: """分析矩陣的類型""".
        print(f"{name}:\n{M}\n")  # EN: Print formatted output to the console.

        # 是否為方陣
        is_square = M.shape[0] == M.shape[1]  # EN: Assign is_square from expression: M.shape[0] == M.shape[1].
        print(f"  方陣？ {is_square}")  # EN: Print formatted output to the console.

        if is_square:  # EN: Branch on a condition: if is_square:.
            # 對稱
            is_symmetric = np.allclose(M, M.T)  # EN: Assign is_symmetric from expression: np.allclose(M, M.T).
            print(f"  對稱？ {is_symmetric}")  # EN: Print formatted output to the console.

            # 正交
            is_orthogonal = np.allclose(M.T @ M, np.eye(M.shape[0]))  # EN: Assign is_orthogonal from expression: np.allclose(M.T @ M, np.eye(M.shape[0])).
            print(f"  正交？ {is_orthogonal}")  # EN: Print formatted output to the console.

            # 對角
            is_diagonal = np.allclose(M, np.diag(np.diag(M)))  # EN: Assign is_diagonal from expression: np.allclose(M, np.diag(np.diag(M))).
            print(f"  對角？ {is_diagonal}")  # EN: Print formatted output to the console.

            # 上三角
            is_upper = np.allclose(M, np.triu(M))  # EN: Assign is_upper from expression: np.allclose(M, np.triu(M)).
            print(f"  上三角？ {is_upper}")  # EN: Print formatted output to the console.

            # 下三角
            is_lower = np.allclose(M, np.tril(M))  # EN: Assign is_lower from expression: np.allclose(M, np.tril(M)).
            print(f"  下三角？ {is_lower}")  # EN: Print formatted output to the console.

        print()  # EN: Print formatted output to the console.

    test_matrices = {  # EN: Assign test_matrices from expression: {.
        "單位矩陣": np.eye(3),  # EN: Execute statement: "單位矩陣": np.eye(3),.
        "對角矩陣": np.diag([1, 2, 3]),  # EN: Execute statement: "對角矩陣": np.diag([1, 2, 3]),.
        "對稱矩陣": np.array([[1, 2], [2, 1]]),  # EN: Execute statement: "對稱矩陣": np.array([[1, 2], [2, 1]]),.
        "旋轉矩陣": R,  # EN: Execute statement: "旋轉矩陣": R,.
        "上三角矩陣": np.triu(np.ones((3, 3)))  # EN: Execute statement: "上三角矩陣": np.triu(np.ones((3, 3))).
    }  # EN: Execute statement: }.

    for name, matrix in test_matrices.items():  # EN: Iterate with a for-loop: for name, matrix in test_matrices.items():.
        analyze_matrix(matrix, name)  # EN: Call analyze_matrix(...) to perform an operation.

    print("=" * 60)  # EN: Print formatted output to the console.
    print("特殊矩陣示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
