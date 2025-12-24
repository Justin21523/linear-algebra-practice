"""
Gram-Schmidt 正交化 - NumPy 版本 (Gram-Schmidt Process - NumPy Implementation)

本程式示範：
1. 使用 NumPy 實作 Gram-Schmidt
2. np.linalg.qr 內建 QR 分解
3. 比較 CGS 和 MGS 的數值穩定性

NumPy 提供高效的向量化運算。
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


# ========================================
# Gram-Schmidt 實作
# ========================================

def classical_gram_schmidt(A: np.ndarray) -> np.ndarray:  # EN: Define classical_gram_schmidt and its behavior.
    """
    Classical Gram-Schmidt (CGS)

    輸入：A 的行向量組成向量組
    輸出：正交向量組 Q
    """  # EN: Execute statement: """.
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.
    Q = np.zeros((m, n))  # EN: Assign Q from expression: np.zeros((m, n)).

    for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
        q = A[:, j].copy()  # EN: Assign q from expression: A[:, j].copy().

        for i in range(j):  # EN: Iterate with a for-loop: for i in range(j):.
            q -= np.dot(Q[:, i], A[:, j]) / np.dot(Q[:, i], Q[:, i]) * Q[:, i]  # EN: Update q via -= using: np.dot(Q[:, i], A[:, j]) / np.dot(Q[:, i], Q[:, i]) * Q[:, i].

        Q[:, j] = q  # EN: Execute statement: Q[:, j] = q.

    return Q  # EN: Return a value: return Q.


def modified_gram_schmidt(A: np.ndarray) -> np.ndarray:  # EN: Define modified_gram_schmidt and its behavior.
    """
    Modified Gram-Schmidt (MGS)

    數值更穩定的版本
    """  # EN: Execute statement: """.
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.
    Q = A.astype(float).copy()  # EN: Assign Q from expression: A.astype(float).copy().

    for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
        for i in range(j):  # EN: Iterate with a for-loop: for i in range(j):.
            Q[:, j] -= np.dot(Q[:, i], Q[:, j]) / np.dot(Q[:, i], Q[:, i]) * Q[:, i]  # EN: Execute statement: Q[:, j] -= np.dot(Q[:, i], Q[:, j]) / np.dot(Q[:, i], Q[:, i]) * Q[:, i].

    return Q  # EN: Return a value: return Q.


def gram_schmidt_qr(A: np.ndarray) -> tuple:  # EN: Define gram_schmidt_qr and its behavior.
    """
    Gram-Schmidt QR 分解

    回傳標準正交 Q 和上三角 R
    """  # EN: Execute statement: """.
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.

    Q = modified_gram_schmidt(A)  # EN: Assign Q from expression: modified_gram_schmidt(A).
    R = np.zeros((n, n))  # EN: Assign R from expression: np.zeros((n, n)).

    # 標準化 Q，同時建立 R
    for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
        R[j, j] = np.linalg.norm(Q[:, j])  # EN: Execute statement: R[j, j] = np.linalg.norm(Q[:, j]).
        Q[:, j] /= R[j, j]  # EN: Execute statement: Q[:, j] /= R[j, j].

        for i in range(j + 1, n):  # EN: Iterate with a for-loop: for i in range(j + 1, n):.
            R[j, i] = np.dot(Q[:, j], A[:, i])  # EN: Execute statement: R[j, i] = np.dot(Q[:, j], A[:, i]).

    return Q, R  # EN: Return a value: return Q, R.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("Gram-Schmidt 正交化示範（NumPy 版）\nGram-Schmidt Process Demo (NumPy)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 基本 Gram-Schmidt
    # ========================================
    print_separator("1. Gram-Schmidt 正交化")  # EN: Call print_separator(...) to perform an operation.

    # 輸入矩陣（行向量）
    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 1, 0],  # EN: Execute statement: [1, 1, 0],.
        [1, 0, 1],  # EN: Execute statement: [1, 0, 1],.
        [0, 1, 1]  # EN: Execute statement: [0, 1, 1].
    ], dtype=float).T  # 轉置使每行成為行向量  # EN: Execute statement: ], dtype=float).T # 轉置使每行成為行向量.

    print(f"輸入矩陣 A（每行是一個向量）：\n{A}")  # EN: Print formatted output to the console.

    # CGS
    Q_cgs = classical_gram_schmidt(A)  # EN: Assign Q_cgs from expression: classical_gram_schmidt(A).
    print(f"\nCGS 結果：\n{Q_cgs}")  # EN: Print formatted output to the console.

    # 驗證正交性
    print(f"\nQᵀQ (CGS)：\n{Q_cgs.T @ Q_cgs}")  # EN: Print formatted output to the console.

    # MGS
    Q_mgs = modified_gram_schmidt(A)  # EN: Assign Q_mgs from expression: modified_gram_schmidt(A).
    print(f"\nMGS 結果：\n{Q_mgs}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. QR 分解
    # ========================================
    print_separator("2. QR 分解")  # EN: Call print_separator(...) to perform an operation.

    Q, R = gram_schmidt_qr(A)  # EN: Execute statement: Q, R = gram_schmidt_qr(A).

    print(f"Q（標準正交）：\n{Q}")  # EN: Print formatted output to the console.
    print(f"\nR（上三角）：\n{R}")  # EN: Print formatted output to the console.

    # 驗證
    print(f"\n驗證 QᵀQ = I：\n{Q.T @ Q}")  # EN: Print formatted output to the console.
    print(f"\n驗證 A = QR：\n{Q @ R}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 使用 np.linalg.qr
    # ========================================
    print_separator("3. 使用 np.linalg.qr")  # EN: Call print_separator(...) to perform an operation.

    Q_numpy, R_numpy = np.linalg.qr(A)  # EN: Execute statement: Q_numpy, R_numpy = np.linalg.qr(A).

    print(f"Q（NumPy）：\n{Q_numpy}")  # EN: Print formatted output to the console.
    print(f"\nR（NumPy）：\n{R_numpy}")  # EN: Print formatted output to the console.

    # 注意：NumPy 可能會給出不同的符號
    print("\n注意：Q 的符號可能與手算不同（但 QR 乘積相同）")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 數值穩定性比較
    # ========================================
    print_separator("4. 數值穩定性比較")  # EN: Call print_separator(...) to perform an operation.

    # 建立接近線性相依的向量
    epsilon = 1e-10  # EN: Assign epsilon from expression: 1e-10.
    A_bad = np.array([  # EN: Assign A_bad from expression: np.array([.
        [1, 1, 1],  # EN: Execute statement: [1, 1, 1],.
        [epsilon, 0, 0],  # EN: Execute statement: [epsilon, 0, 0],.
        [0, epsilon, 0]  # EN: Execute statement: [0, epsilon, 0].
    ], dtype=float).T  # EN: Execute statement: ], dtype=float).T.

    print(f"接近線性相依的矩陣 A：\n{A_bad}")  # EN: Print formatted output to the console.

    Q_cgs_bad = classical_gram_schmidt(A_bad)  # EN: Assign Q_cgs_bad from expression: classical_gram_schmidt(A_bad).
    Q_mgs_bad = modified_gram_schmidt(A_bad)  # EN: Assign Q_mgs_bad from expression: modified_gram_schmidt(A_bad).

    print(f"\nCGS 正交性檢查（QᵀQ 對角線外應為 0）：")  # EN: Print formatted output to the console.
    orthogonality_cgs = Q_cgs_bad.T @ Q_cgs_bad  # EN: Assign orthogonality_cgs from expression: Q_cgs_bad.T @ Q_cgs_bad.
    print(f"QᵀQ 非對角線元素的最大值：{np.max(np.abs(orthogonality_cgs - np.diag(np.diag(orthogonality_cgs)))):.2e}")  # EN: Print formatted output to the console.

    print(f"\nMGS 正交性檢查：")  # EN: Print formatted output to the console.
    orthogonality_mgs = Q_mgs_bad.T @ Q_mgs_bad  # EN: Assign orthogonality_mgs from expression: Q_mgs_bad.T @ Q_mgs_bad.
    print(f"QᵀQ 非對角線元素的最大值：{np.max(np.abs(orthogonality_mgs - np.diag(np.diag(orthogonality_mgs)))):.2e}")  # EN: Print formatted output to the console.

    print("\nMGS 通常比 CGS 更穩定")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 用 QR 解最小平方
    # ========================================
    print_separator("5. 用 QR 解最小平方")  # EN: Call print_separator(...) to perform an operation.

    # 數據
    t = np.array([0, 1, 2, 3, 4], dtype=float)  # EN: Assign t from expression: np.array([0, 1, 2, 3, 4], dtype=float).
    b = np.array([1, 2.5, 3.5, 5, 6.5], dtype=float)  # EN: Assign b from expression: np.array([1, 2.5, 3.5, 5, 6.5], dtype=float).

    # 設計矩陣
    A_ls = np.column_stack([np.ones_like(t), t])  # EN: Assign A_ls from expression: np.column_stack([np.ones_like(t), t]).

    print(f"設計矩陣 A：\n{A_ls}")  # EN: Print formatted output to the console.
    print(f"觀測值 b = {b}")  # EN: Print formatted output to the console.

    # QR 分解
    Q_ls, R_ls = np.linalg.qr(A_ls)  # EN: Execute statement: Q_ls, R_ls = np.linalg.qr(A_ls).

    # 解 Rx = Qᵀb
    x_qr = np.linalg.solve(R_ls, Q_ls.T @ b)  # EN: Assign x_qr from expression: np.linalg.solve(R_ls, Q_ls.T @ b).

    print(f"\nQ：\n{Q_ls}")  # EN: Print formatted output to the console.
    print(f"\nR：\n{R_ls}")  # EN: Print formatted output to the console.
    print(f"\nQᵀb = {Q_ls.T @ b}")  # EN: Print formatted output to the console.
    print(f"\n解 x = {x_qr}")  # EN: Print formatted output to the console.

    # 比較正規方程解
    x_normal = np.linalg.solve(A_ls.T @ A_ls, A_ls.T @ b)  # EN: Assign x_normal from expression: np.linalg.solve(A_ls.T @ A_ls, A_ls.T @ b).
    print(f"正規方程解 = {x_normal}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 正交多項式
    # ========================================
    print_separator("6. 正交多項式範例")  # EN: Call print_separator(...) to perform an operation.

    # 從 {1, x, x²} 建立正交多項式
    x = np.linspace(-1, 1, 100)  # EN: Assign x from expression: np.linspace(-1, 1, 100).
    A_poly = np.column_stack([np.ones_like(x), x, x**2])  # EN: Assign A_poly from expression: np.column_stack([np.ones_like(x), x, x**2]).

    Q_poly, R_poly = np.linalg.qr(A_poly)  # EN: Execute statement: Q_poly, R_poly = np.linalg.qr(A_poly).

    print("從 {1, x, x²} 正交化（類似 Legendre 多項式）")  # EN: Print formatted output to the console.
    print(f"\n驗證正交性：QᵀQ =\n{(Q_poly.T @ Q_poly)[:3, :3]}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("NumPy QR 函數總結")  # EN: Call print_separator(...) to perform an operation.
    print("""
Gram-Schmidt 實作：
  Q_cgs = classical_gram_schmidt(A)
  Q_mgs = modified_gram_schmidt(A)

NumPy QR 分解：
  Q, R = np.linalg.qr(A)

用 QR 解最小平方：
  x = np.linalg.solve(R, Q.T @ b)

驗證正交性：
  np.allclose(Q.T @ Q, np.eye(n))

MGS vs CGS：
  MGS 數值更穩定，特別是接近奇異的矩陣
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
