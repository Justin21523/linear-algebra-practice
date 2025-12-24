"""
行空間與零空間 (Column Space and Null Space)

本程式示範：
1. 計算矩陣的零空間 N(A)
2. 計算矩陣的行空間 C(A)
3. 驗證 Ax = b 的解結構
4. 秩-零度定理

This program demonstrates how to compute column space and null space,
and how they relate to solving Ax = b.
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.
from scipy.linalg import null_space  # EN: Import symbol(s) from a module: from scipy.linalg import null_space.
from typing import Tuple, List  # EN: Import symbol(s) from a module: from typing import Tuple, List.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def rref(A: np.ndarray) -> Tuple[np.ndarray, List[int]]:  # EN: Define rref and its behavior.
    """
    計算簡化列階梯形式 (Reduced Row Echelon Form)

    Returns:
        (RREF 矩陣, 主元行的索引列表)
    """  # EN: Execute statement: """.
    A = A.astype(float).copy()  # EN: Assign A from expression: A.astype(float).copy().
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.
    pivot_cols = []  # EN: Assign pivot_cols from expression: [].
    row = 0  # EN: Assign row from expression: 0.

    for col in range(n):  # EN: Iterate with a for-loop: for col in range(n):.
        if row >= m:  # EN: Branch on a condition: if row >= m:.
            break  # EN: Control flow statement: break.

        # 找主元
        max_row = row + np.argmax(np.abs(A[row:, col]))  # EN: Assign max_row from expression: row + np.argmax(np.abs(A[row:, col])).

        if np.abs(A[max_row, col]) < 1e-10:  # EN: Branch on a condition: if np.abs(A[max_row, col]) < 1e-10:.
            continue  # EN: Control flow statement: continue.

        # 換列
        A[[row, max_row]] = A[[max_row, row]]  # EN: Execute statement: A[[row, max_row]] = A[[max_row, row]].

        # 正規化
        A[row] = A[row] / A[row, col]  # EN: Execute statement: A[row] = A[row] / A[row, col].

        # 消去（上下都要）
        for i in range(m):  # EN: Iterate with a for-loop: for i in range(m):.
            if i != row:  # EN: Branch on a condition: if i != row:.
                A[i] = A[i] - A[i, col] * A[row]  # EN: Execute statement: A[i] = A[i] - A[i, col] * A[row].

        pivot_cols.append(col)  # EN: Execute statement: pivot_cols.append(col).
        row += 1  # EN: Update row via += using: 1.

    return A, pivot_cols  # EN: Return a value: return A, pivot_cols.


def compute_null_space_manual(A: np.ndarray) -> np.ndarray:  # EN: Define compute_null_space_manual and its behavior.
    """
    手動計算零空間的基底（使用 RREF）

    Compute null space basis using RREF
    """  # EN: Execute statement: """.
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.
    R, pivot_cols = rref(A)  # EN: Execute statement: R, pivot_cols = rref(A).

    # 自由變數的索引
    free_cols = [j for j in range(n) if j not in pivot_cols]  # EN: Assign free_cols from expression: [j for j in range(n) if j not in pivot_cols].

    if len(free_cols) == 0:  # EN: Branch on a condition: if len(free_cols) == 0:.
        return np.zeros((n, 0))  # 零空間只有 {0}  # EN: Return a value: return np.zeros((n, 0)) # 零空間只有 {0}.

    # 對每個自由變數，生成一個特解
    null_vectors = []  # EN: Assign null_vectors from expression: [].

    for free_col in free_cols:  # EN: Iterate with a for-loop: for free_col in free_cols:.
        x = np.zeros(n)  # EN: Assign x from expression: np.zeros(n).
        x[free_col] = 1  # 設自由變數為 1  # EN: Execute statement: x[free_col] = 1 # 設自由變數為 1.

        # 回代求主元變數
        for i, pivot_col in enumerate(pivot_cols):  # EN: Iterate with a for-loop: for i, pivot_col in enumerate(pivot_cols):.
            # x[pivot_col] + sum of (R[i,j] * x[j] for free j) = 0
            x[pivot_col] = -np.sum(R[i, free_col:] * x[free_col:])  # EN: Execute statement: x[pivot_col] = -np.sum(R[i, free_col:] * x[free_col:]).
            # 修正：只考慮自由列
            x[pivot_col] = -R[i, free_col]  # EN: Execute statement: x[pivot_col] = -R[i, free_col].

        null_vectors.append(x)  # EN: Execute statement: null_vectors.append(x).

    return np.column_stack(null_vectors) if null_vectors else np.zeros((n, 0))  # EN: Return a value: return np.column_stack(null_vectors) if null_vectors else np.zeros((n, ….


def compute_column_space_basis(A: np.ndarray) -> np.ndarray:  # EN: Define compute_column_space_basis and its behavior.
    """
    計算行空間的基底

    行空間的基底 = A 中對應 RREF 主元行的那些行
    """  # EN: Execute statement: """.
    _, pivot_cols = rref(A)  # EN: Execute statement: _, pivot_cols = rref(A).

    if len(pivot_cols) == 0:  # EN: Branch on a condition: if len(pivot_cols) == 0:.
        return np.zeros((A.shape[0], 0))  # EN: Return a value: return np.zeros((A.shape[0], 0)).

    return A[:, pivot_cols]  # EN: Return a value: return A[:, pivot_cols].


def verify_null_space(A: np.ndarray, N: np.ndarray) -> bool:  # EN: Define verify_null_space and its behavior.
    """驗證 N 的每一行都在零空間中"""  # EN: Execute statement: """驗證 N 的每一行都在零空間中""".
    if N.shape[1] == 0:  # EN: Branch on a condition: if N.shape[1] == 0:.
        return True  # EN: Return a value: return True.
    return np.allclose(A @ N, 0)  # EN: Return a value: return np.allclose(A @ N, 0).


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("行空間與零空間示範\nColumn Space and Null Space Demo")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 範例 1：基本零空間計算
    # ========================================
    print_separator("1. 計算零空間 N(A)")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 2, 2, 2],  # EN: Execute statement: [1, 2, 2, 2],.
        [2, 4, 6, 8],  # EN: Execute statement: [2, 4, 6, 8],.
        [3, 6, 8, 10]  # EN: Execute statement: [3, 6, 8, 10].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A =\n{A}\n")  # EN: Print formatted output to the console.

    # 使用 SciPy 計算零空間
    N = null_space(A)  # EN: Assign N from expression: null_space(A).
    print(f"零空間基底（scipy.linalg.null_space）:")  # EN: Print formatted output to the console.
    print(f"N =\n{N}\n")  # EN: Print formatted output to the console.

    print(f"零空間維度 dim N(A) = {N.shape[1]}")  # EN: Print formatted output to the console.

    # 驗證
    print(f"\n驗證 A @ N = 0?")  # EN: Print formatted output to the console.
    print(f"A @ N =\n{A @ N}")  # EN: Print formatted output to the console.
    print(f"全部接近零？ {np.allclose(A @ N, 0)}")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 2：行空間
    # ========================================
    print_separator("2. 計算行空間 C(A)")  # EN: Call print_separator(...) to perform an operation.

    # 計算 RREF 找主元行
    R, pivot_cols = rref(A)  # EN: Execute statement: R, pivot_cols = rref(A).

    print(f"RREF(A) =\n{R}\n")  # EN: Print formatted output to the console.
    print(f"主元行索引：{pivot_cols}")  # EN: Print formatted output to the console.

    # 行空間基底
    C_basis = A[:, pivot_cols]  # EN: Assign C_basis from expression: A[:, pivot_cols].
    print(f"\n行空間基底（A 的主元行）:")  # EN: Print formatted output to the console.
    print(f"C(A) 基底 =\n{C_basis}")  # EN: Print formatted output to the console.
    print(f"\n行空間維度 dim C(A) = rank(A) = {len(pivot_cols)}")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 3：秩-零度定理
    # ========================================
    print_separator("3. 秩-零度定理 (Rank-Nullity Theorem)")  # EN: Call print_separator(...) to perform an operation.

    m, n = A.shape  # EN: Execute statement: m, n = A.shape.
    rank_A = np.linalg.matrix_rank(A)  # EN: Assign rank_A from expression: np.linalg.matrix_rank(A).
    nullity_A = N.shape[1]  # EN: Assign nullity_A from expression: N.shape[1].

    print(f"A 是 {m}×{n} 矩陣")  # EN: Print formatted output to the console.
    print(f"rank(A) = dim C(A) = {rank_A}")  # EN: Print formatted output to the console.
    print(f"nullity(A) = dim N(A) = {nullity_A}")  # EN: Print formatted output to the console.
    print(f"\nrank(A) + nullity(A) = {rank_A} + {nullity_A} = {rank_A + nullity_A}")  # EN: Print formatted output to the console.
    print(f"n（行數）= {n}")  # EN: Print formatted output to the console.
    print(f"\n秩-零度定理成立？ {rank_A + nullity_A == n} ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 4：Ax = b 的解結構
    # ========================================
    print_separator("4. Ax = b 的解結構")  # EN: Call print_separator(...) to perform an operation.

    # 有解的情況
    b1 = np.array([1.0, 2.0, 3.0])  # EN: Assign b1 from expression: np.array([1.0, 2.0, 3.0]).
    print(f"b₁ = {b1}")  # EN: Print formatted output to the console.

    # 檢查 b 是否在行空間中
    # 方法：解 Ax = b，看殘差
    x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b1, rcond=None)  # EN: Execute statement: x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b1, rcond=None).

    if len(residuals) == 0 or residuals[0] < 1e-10:  # EN: Branch on a condition: if len(residuals) == 0 or residuals[0] < 1e-10:.
        print("b₁ ∈ C(A)，方程有解")  # EN: Print formatted output to the console.

        print(f"\n特解 x_p = {x_lstsq}")  # EN: Print formatted output to the console.
        print(f"驗證 A @ x_p = {A @ x_lstsq}")  # EN: Print formatted output to the console.

        # 通解
        print(f"\n通解 = x_p + N(A)")  # EN: Print formatted output to the console.
        print("即 x_p + c₁n₁ + c₂n₂，其中 c₁, c₂ 為任意實數")  # EN: Print formatted output to the console.

        # 驗證另一個解
        if N.shape[1] > 0:  # EN: Branch on a condition: if N.shape[1] > 0:.
            x_another = x_lstsq + 2 * N[:, 0]  # 加上零空間的一個向量  # EN: Assign x_another from expression: x_lstsq + 2 * N[:, 0] # 加上零空間的一個向量.
            print(f"\n另一個解：x = x_p + 2n₁ = {x_another}")  # EN: Print formatted output to the console.
            print(f"驗證 A @ x = {A @ x_another}")  # EN: Print formatted output to the console.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        print("b₁ ∉ C(A)，方程無解")  # EN: Print formatted output to the console.

    # 無解的情況
    print("\n" + "-" * 40)  # EN: Print formatted output to the console.
    b2 = np.array([1.0, 1.0, 1.0])  # EN: Assign b2 from expression: np.array([1.0, 1.0, 1.0]).
    print(f"b₂ = {b2}")  # EN: Print formatted output to the console.

    x_lstsq2, residuals2, _, _ = np.linalg.lstsq(A, b2, rcond=None)  # EN: Execute statement: x_lstsq2, residuals2, _, _ = np.linalg.lstsq(A, b2, rcond=None).
    residual_norm = np.linalg.norm(A @ x_lstsq2 - b2)  # EN: Assign residual_norm from expression: np.linalg.norm(A @ x_lstsq2 - b2).

    if residual_norm > 1e-10:  # EN: Branch on a condition: if residual_norm > 1e-10:.
        print(f"b₂ ∉ C(A)，方程無解（殘差 = {residual_norm:.4f}）")  # EN: Print formatted output to the console.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        print("b₂ ∈ C(A)")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 5：不同秩的矩陣
    # ========================================
    print_separator("5. 不同秩的矩陣比較")  # EN: Call print_separator(...) to perform an operation.

    matrices = {  # EN: Assign matrices from expression: {.
        "滿秩矩陣 (3×3)": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),  # EN: Execute statement: "滿秩矩陣 (3×3)": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),.
        "秩虧矩陣 (3×3)": np.array([[1, 2, 3], [2, 4, 6], [1, 2, 3]], dtype=float),  # EN: Execute statement: "秩虧矩陣 (3×3)": np.array([[1, 2, 3], [2, 4, 6], [1, 2, 3]], dtype=float),.
        "寬矩陣 (2×4)": np.array([[1, 2, 0, 1], [0, 0, 1, 1]], dtype=float),  # EN: Execute statement: "寬矩陣 (2×4)": np.array([[1, 2, 0, 1], [0, 0, 1, 1]], dtype=float),.
        "高矩陣 (4×2)": np.array([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=float),  # EN: Execute statement: "高矩陣 (4×2)": np.array([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=float),.
    }  # EN: Execute statement: }.

    for name, M in matrices.items():  # EN: Iterate with a for-loop: for name, M in matrices.items():.
        m, n = M.shape  # EN: Execute statement: m, n = M.shape.
        r = np.linalg.matrix_rank(M)  # EN: Assign r from expression: np.linalg.matrix_rank(M).
        N_M = null_space(M)  # EN: Assign N_M from expression: null_space(M).
        nullity = N_M.shape[1]  # EN: Assign nullity from expression: N_M.shape[1].

        print(f"\n{name}:")  # EN: Print formatted output to the console.
        print(f"  大小: {m}×{n}")  # EN: Print formatted output to the console.
        print(f"  rank = {r}")  # EN: Print formatted output to the console.
        print(f"  nullity = {nullity}")  # EN: Print formatted output to the console.
        print(f"  rank + nullity = {r + nullity} = n = {n} ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 6：零空間的幾何意義
    # ========================================
    print_separator("6. 零空間的幾何意義")  # EN: Call print_separator(...) to perform an operation.

    print("""
零空間 N(A) 的幾何意義：

1. N(A) 是所有被 A「壓縮到零」的向量
2. 若 x ∈ N(A)，則 A 作用在 x 上會得到零向量
3. 零空間描述了 A 的「退化方向」

dim N(A) = 0：A 是單射（injective），不同的 x 給出不同的 Ax
dim N(A) > 0：A 有退化方向，無限多個 x 可以給出相同的 Ax
""")  # EN: Execute statement: """).

    # 視覺化範例
    print("\n範例：投影矩陣的零空間")  # EN: Print formatted output to the console.
    # 投影到 x 軸
    P = np.array([[1, 0], [0, 0]], dtype=float)  # EN: Assign P from expression: np.array([[1, 0], [0, 0]], dtype=float).
    print(f"P =\n{P}")  # EN: Print formatted output to the console.
    print("P 將向量投影到 x 軸")  # EN: Print formatted output to the console.

    N_P = null_space(P)  # EN: Assign N_P from expression: null_space(P).
    print(f"\nN(P) 的基底：\n{N_P}")  # EN: Print formatted output to the console.
    print("零空間是 y 軸（所有 [0, t]ᵀ 向量）")  # EN: Print formatted output to the console.
    print("這些向量被投影後變成零向量")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 7：行空間的幾何意義
    # ========================================
    print_separator("7. 行空間的幾何意義")  # EN: Call print_separator(...) to perform an operation.

    print("""
行空間 C(A) 的幾何意義：

1. C(A) = {Ax : x ∈ ℝⁿ} = A 能「到達」的所有向量
2. C(A) 是 A 的像（image）或值域（range）
3. b ∈ C(A) ⟺ Ax = b 有解

dim C(A) = m：A 是滿射（surjective），可以到達 ℝᵐ 的每一點
dim C(A) < m：A 只能到達 ℝᵐ 的一個子空間
""")  # EN: Execute statement: """).

    print("\n範例：")  # EN: Print formatted output to the console.
    B = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)  # EN: Assign B from expression: np.array([[1, 1], [2, 2], [3, 3]], dtype=float).
    print(f"B =\n{B}")  # EN: Print formatted output to the console.

    C_B = compute_column_space_basis(B)  # EN: Assign C_B from expression: compute_column_space_basis(B).
    print(f"\nC(B) 的基底：\n{C_B}")  # EN: Print formatted output to the console.
    print("行空間是 ℝ³ 中通過原點的一條直線")  # EN: Print formatted output to the console.
    print("B 只能到達這條直線上的點")  # EN: Print formatted output to the console.

    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print("行空間與零空間示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
