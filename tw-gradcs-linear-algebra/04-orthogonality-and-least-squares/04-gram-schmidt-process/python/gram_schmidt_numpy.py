"""
Gram-Schmidt 正交化 - NumPy 版本 (Gram-Schmidt Process - NumPy Implementation)

本程式示範：
1. 使用 NumPy 實作 Gram-Schmidt
2. np.linalg.qr 內建 QR 分解
3. 比較 CGS 和 MGS 的數值穩定性

NumPy 提供高效的向量化運算。
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ========================================
# Gram-Schmidt 實作
# ========================================

def classical_gram_schmidt(A: np.ndarray) -> np.ndarray:
    """
    Classical Gram-Schmidt (CGS)

    輸入：A 的行向量組成向量組
    輸出：正交向量組 Q
    """
    m, n = A.shape
    Q = np.zeros((m, n))

    for j in range(n):
        q = A[:, j].copy()

        for i in range(j):
            q -= np.dot(Q[:, i], A[:, j]) / np.dot(Q[:, i], Q[:, i]) * Q[:, i]

        Q[:, j] = q

    return Q


def modified_gram_schmidt(A: np.ndarray) -> np.ndarray:
    """
    Modified Gram-Schmidt (MGS)

    數值更穩定的版本
    """
    m, n = A.shape
    Q = A.astype(float).copy()

    for j in range(n):
        for i in range(j):
            Q[:, j] -= np.dot(Q[:, i], Q[:, j]) / np.dot(Q[:, i], Q[:, i]) * Q[:, i]

    return Q


def gram_schmidt_qr(A: np.ndarray) -> tuple:
    """
    Gram-Schmidt QR 分解

    回傳標準正交 Q 和上三角 R
    """
    m, n = A.shape

    Q = modified_gram_schmidt(A)
    R = np.zeros((n, n))

    # 標準化 Q，同時建立 R
    for j in range(n):
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] /= R[j, j]

        for i in range(j + 1, n):
            R[j, i] = np.dot(Q[:, j], A[:, i])

    return Q, R


def main():
    """主程式"""

    print_separator("Gram-Schmidt 正交化示範（NumPy 版）\nGram-Schmidt Process Demo (NumPy)")

    # ========================================
    # 1. 基本 Gram-Schmidt
    # ========================================
    print_separator("1. Gram-Schmidt 正交化")

    # 輸入矩陣（行向量）
    A = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=float).T  # 轉置使每行成為行向量

    print(f"輸入矩陣 A（每行是一個向量）：\n{A}")

    # CGS
    Q_cgs = classical_gram_schmidt(A)
    print(f"\nCGS 結果：\n{Q_cgs}")

    # 驗證正交性
    print(f"\nQᵀQ (CGS)：\n{Q_cgs.T @ Q_cgs}")

    # MGS
    Q_mgs = modified_gram_schmidt(A)
    print(f"\nMGS 結果：\n{Q_mgs}")

    # ========================================
    # 2. QR 分解
    # ========================================
    print_separator("2. QR 分解")

    Q, R = gram_schmidt_qr(A)

    print(f"Q（標準正交）：\n{Q}")
    print(f"\nR（上三角）：\n{R}")

    # 驗證
    print(f"\n驗證 QᵀQ = I：\n{Q.T @ Q}")
    print(f"\n驗證 A = QR：\n{Q @ R}")

    # ========================================
    # 3. 使用 np.linalg.qr
    # ========================================
    print_separator("3. 使用 np.linalg.qr")

    Q_numpy, R_numpy = np.linalg.qr(A)

    print(f"Q（NumPy）：\n{Q_numpy}")
    print(f"\nR（NumPy）：\n{R_numpy}")

    # 注意：NumPy 可能會給出不同的符號
    print("\n注意：Q 的符號可能與手算不同（但 QR 乘積相同）")

    # ========================================
    # 4. 數值穩定性比較
    # ========================================
    print_separator("4. 數值穩定性比較")

    # 建立接近線性相依的向量
    epsilon = 1e-10
    A_bad = np.array([
        [1, 1, 1],
        [epsilon, 0, 0],
        [0, epsilon, 0]
    ], dtype=float).T

    print(f"接近線性相依的矩陣 A：\n{A_bad}")

    Q_cgs_bad = classical_gram_schmidt(A_bad)
    Q_mgs_bad = modified_gram_schmidt(A_bad)

    print(f"\nCGS 正交性檢查（QᵀQ 對角線外應為 0）：")
    orthogonality_cgs = Q_cgs_bad.T @ Q_cgs_bad
    print(f"QᵀQ 非對角線元素的最大值：{np.max(np.abs(orthogonality_cgs - np.diag(np.diag(orthogonality_cgs)))):.2e}")

    print(f"\nMGS 正交性檢查：")
    orthogonality_mgs = Q_mgs_bad.T @ Q_mgs_bad
    print(f"QᵀQ 非對角線元素的最大值：{np.max(np.abs(orthogonality_mgs - np.diag(np.diag(orthogonality_mgs)))):.2e}")

    print("\nMGS 通常比 CGS 更穩定")

    # ========================================
    # 5. 用 QR 解最小平方
    # ========================================
    print_separator("5. 用 QR 解最小平方")

    # 數據
    t = np.array([0, 1, 2, 3, 4], dtype=float)
    b = np.array([1, 2.5, 3.5, 5, 6.5], dtype=float)

    # 設計矩陣
    A_ls = np.column_stack([np.ones_like(t), t])

    print(f"設計矩陣 A：\n{A_ls}")
    print(f"觀測值 b = {b}")

    # QR 分解
    Q_ls, R_ls = np.linalg.qr(A_ls)

    # 解 Rx = Qᵀb
    x_qr = np.linalg.solve(R_ls, Q_ls.T @ b)

    print(f"\nQ：\n{Q_ls}")
    print(f"\nR：\n{R_ls}")
    print(f"\nQᵀb = {Q_ls.T @ b}")
    print(f"\n解 x = {x_qr}")

    # 比較正規方程解
    x_normal = np.linalg.solve(A_ls.T @ A_ls, A_ls.T @ b)
    print(f"正規方程解 = {x_normal}")

    # ========================================
    # 6. 正交多項式
    # ========================================
    print_separator("6. 正交多項式範例")

    # 從 {1, x, x²} 建立正交多項式
    x = np.linspace(-1, 1, 100)
    A_poly = np.column_stack([np.ones_like(x), x, x**2])

    Q_poly, R_poly = np.linalg.qr(A_poly)

    print("從 {1, x, x²} 正交化（類似 Legendre 多項式）")
    print(f"\n驗證正交性：QᵀQ =\n{(Q_poly.T @ Q_poly)[:3, :3]}")

    # 總結
    print_separator("NumPy QR 函數總結")
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
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
