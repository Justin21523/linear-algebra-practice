"""
QR 分解 - NumPy 版本 (QR Decomposition - NumPy Implementation)

本程式示範：
1. np.linalg.qr 的使用
2. 瘦 QR vs 滿 QR
3. 用 QR 解最小平方
4. Householder 反射
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    print_separator("QR 分解示範（NumPy 版）\nQR Decomposition Demo (NumPy)")

    # ========================================
    # 1. 基本 QR 分解
    # ========================================
    print_separator("1. 基本 QR 分解")

    A = np.array([
        [1, 1],
        [1, 0],
        [0, 1]
    ], dtype=float)

    print(f"輸入矩陣 A：\n{A}")

    # 瘦 QR（預設）
    Q, R = np.linalg.qr(A)

    print(f"\n瘦 QR 分解：")
    print(f"Q (m×n)：\n{Q}")
    print(f"\nR (n×n)：\n{R}")

    # 驗證
    print(f"\n驗證 QᵀQ：\n{Q.T @ Q}")
    print(f"\n驗證 A = QR：\n{Q @ R}")

    # ========================================
    # 2. 滿 QR 分解
    # ========================================
    print_separator("2. 滿 QR 分解")

    Q_full, R_full = np.linalg.qr(A, mode='complete')

    print(f"滿 QR 分解：")
    print(f"Q (m×m)：\n{Q_full}")
    print(f"\nR (m×n)：\n{R_full}")

    print(f"\n驗證 QQᵀ = I：\n{Q_full @ Q_full.T}")

    # ========================================
    # 3. 用 QR 解最小平方
    # ========================================
    print_separator("3. 用 QR 解最小平方")

    # 數據
    t = np.array([0, 1, 2, 3, 4], dtype=float)
    b = np.array([1, 2.5, 3.5, 5, 6.5], dtype=float)

    print("數據點：")
    for ti, bi in zip(t, b):
        print(f"  ({ti}, {bi})")

    # 設計矩陣
    A_ls = np.column_stack([np.ones_like(t), t])
    print(f"\n設計矩陣 A：\n{A_ls}")
    print(f"觀測值 b = {b}")

    # QR 分解
    Q_ls, R_ls = np.linalg.qr(A_ls)

    print(f"\nQ：\n{Q_ls}")
    print(f"\nR：\n{R_ls}")

    # 方法 1：解 Rx = Qᵀb
    Qt_b = Q_ls.T @ b
    print(f"\nQᵀb = {Qt_b}")

    x_qr = np.linalg.solve(R_ls, Qt_b)
    print(f"解 x = {x_qr}")

    # 方法 2：使用 scipy.linalg.solve_triangular
    from scipy.linalg import solve_triangular
    x_tri = solve_triangular(R_ls, Qt_b)
    print(f"使用 solve_triangular: {x_tri}")

    # 方法 3：比較 lstsq
    x_lstsq, _, _, _ = np.linalg.lstsq(A_ls, b, rcond=None)
    print(f"lstsq 解: {x_lstsq}")

    print(f"\n最佳直線：y = {x_qr[0]:.4f} + {x_qr[1]:.4f}t")

    # ========================================
    # 4. Householder 反射
    # ========================================
    print_separator("4. Householder 反射原理")

    def householder(x):
        """計算 Householder 向量 v 和反射矩陣 H"""
        n = len(x)
        alpha = -np.sign(x[0]) * np.linalg.norm(x)

        v = x.copy()
        v[0] -= alpha

        v = v / np.linalg.norm(v)

        H = np.eye(n) - 2 * np.outer(v, v)

        return H, alpha

    # 示範
    x = np.array([1, 1, 0], dtype=float)
    print(f"原始向量 x = {x}")

    H, alpha = householder(x)
    print(f"\nHouseholder 矩陣 H：\n{H}")
    print(f"\nHx = {H @ x}")
    print(f"預期：[{alpha:.4f}, 0, 0]")

    # 驗證 H 的性質
    print(f"\nHᵀ = H？ {np.allclose(H, H.T)}")
    print(f"HᵀH = I？ {np.allclose(H @ H, np.eye(3))}")
    print(f"det(H) = {np.linalg.det(H):.4f}（-1 表示反射）")

    # ========================================
    # 5. 條件數和穩定性
    # ========================================
    print_separator("5. 條件數和穩定性")

    # 病態矩陣
    epsilon = 1e-10
    A_bad = np.array([
        [1, 1],
        [1, 1 + epsilon],
        [1, 1 + 2*epsilon]
    ])

    print(f"病態矩陣 A：\n{A_bad}")
    print(f"\ncond(A) = {np.linalg.cond(A_bad):.2e}")
    print(f"cond(AᵀA) = {np.linalg.cond(A_bad.T @ A_bad):.2e}")

    print("\n注意：cond(AᵀA) ≈ cond(A)²")
    print("這就是為什麼 QR 比正規方程更穩定")

    # ========================================
    # 6. 比較不同方法
    # ========================================
    print_separator("6. 比較不同方法的準確性")

    # 造一個有確切解的問題
    np.random.seed(42)
    m, n = 100, 10
    A_test = np.random.randn(m, n)
    x_true = np.ones(n)
    b_test = A_test @ x_true  # 無噪聲

    # 方法 1：正規方程
    x_normal = np.linalg.solve(A_test.T @ A_test, A_test.T @ b_test)

    # 方法 2：QR 分解
    Q_test, R_test = np.linalg.qr(A_test)
    x_qr_test = np.linalg.solve(R_test, Q_test.T @ b_test)

    # 方法 3：lstsq（使用 SVD）
    x_lstsq_test, _, _, _ = np.linalg.lstsq(A_test, b_test, rcond=None)

    print(f"真實解：x = [1, 1, ..., 1]")
    print(f"\n各方法的誤差（‖x - x_true‖）：")
    print(f"正規方程：{np.linalg.norm(x_normal - x_true):.2e}")
    print(f"QR 分解：{np.linalg.norm(x_qr_test - x_true):.2e}")
    print(f"SVD (lstsq)：{np.linalg.norm(x_lstsq_test - x_true):.2e}")

    # 總結
    print_separator("NumPy QR 函數總結")
    print("""
基本用法：
  Q, R = np.linalg.qr(A)           # 瘦 QR
  Q, R = np.linalg.qr(A, 'complete')  # 滿 QR

用 QR 解最小平方：
  Q, R = np.linalg.qr(A)
  x = np.linalg.solve(R, Q.T @ b)

Householder 反射：
  H = I - 2vvᵀ/vᵀv

穩定性比較：
  SVD > QR > 正規方程

工業標準：
  LAPACK 使用 Householder QR
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
