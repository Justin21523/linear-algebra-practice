"""
QR 分解 - NumPy 版本 (QR Decomposition - NumPy Implementation)

本程式示範：
1. np.linalg.qr 的使用
2. 瘦 QR vs 滿 QR
3. 用 QR 解最小平方
4. Householder 反射
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    print_separator("QR 分解示範（NumPy 版）\nQR Decomposition Demo (NumPy)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 基本 QR 分解
    # ========================================
    print_separator("1. 基本 QR 分解")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 1],  # EN: Execute statement: [1, 1],.
        [1, 0],  # EN: Execute statement: [1, 0],.
        [0, 1]  # EN: Execute statement: [0, 1].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"輸入矩陣 A：\n{A}")  # EN: Print formatted output to the console.

    # 瘦 QR（預設）
    Q, R = np.linalg.qr(A)  # EN: Execute statement: Q, R = np.linalg.qr(A).

    print(f"\n瘦 QR 分解：")  # EN: Print formatted output to the console.
    print(f"Q (m×n)：\n{Q}")  # EN: Print formatted output to the console.
    print(f"\nR (n×n)：\n{R}")  # EN: Print formatted output to the console.

    # 驗證
    print(f"\n驗證 QᵀQ：\n{Q.T @ Q}")  # EN: Print formatted output to the console.
    print(f"\n驗證 A = QR：\n{Q @ R}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 滿 QR 分解
    # ========================================
    print_separator("2. 滿 QR 分解")  # EN: Call print_separator(...) to perform an operation.

    Q_full, R_full = np.linalg.qr(A, mode='complete')  # EN: Execute statement: Q_full, R_full = np.linalg.qr(A, mode='complete').

    print(f"滿 QR 分解：")  # EN: Print formatted output to the console.
    print(f"Q (m×m)：\n{Q_full}")  # EN: Print formatted output to the console.
    print(f"\nR (m×n)：\n{R_full}")  # EN: Print formatted output to the console.

    print(f"\n驗證 QQᵀ = I：\n{Q_full @ Q_full.T}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 用 QR 解最小平方
    # ========================================
    print_separator("3. 用 QR 解最小平方")  # EN: Call print_separator(...) to perform an operation.

    # 數據
    t = np.array([0, 1, 2, 3, 4], dtype=float)  # EN: Assign t from expression: np.array([0, 1, 2, 3, 4], dtype=float).
    b = np.array([1, 2.5, 3.5, 5, 6.5], dtype=float)  # EN: Assign b from expression: np.array([1, 2.5, 3.5, 5, 6.5], dtype=float).

    print("數據點：")  # EN: Print formatted output to the console.
    for ti, bi in zip(t, b):  # EN: Iterate with a for-loop: for ti, bi in zip(t, b):.
        print(f"  ({ti}, {bi})")  # EN: Print formatted output to the console.

    # 設計矩陣
    A_ls = np.column_stack([np.ones_like(t), t])  # EN: Assign A_ls from expression: np.column_stack([np.ones_like(t), t]).
    print(f"\n設計矩陣 A：\n{A_ls}")  # EN: Print formatted output to the console.
    print(f"觀測值 b = {b}")  # EN: Print formatted output to the console.

    # QR 分解
    Q_ls, R_ls = np.linalg.qr(A_ls)  # EN: Execute statement: Q_ls, R_ls = np.linalg.qr(A_ls).

    print(f"\nQ：\n{Q_ls}")  # EN: Print formatted output to the console.
    print(f"\nR：\n{R_ls}")  # EN: Print formatted output to the console.

    # 方法 1：解 Rx = Qᵀb
    Qt_b = Q_ls.T @ b  # EN: Assign Qt_b from expression: Q_ls.T @ b.
    print(f"\nQᵀb = {Qt_b}")  # EN: Print formatted output to the console.

    x_qr = np.linalg.solve(R_ls, Qt_b)  # EN: Assign x_qr from expression: np.linalg.solve(R_ls, Qt_b).
    print(f"解 x = {x_qr}")  # EN: Print formatted output to the console.

    # 方法 2：使用 scipy.linalg.solve_triangular
    from scipy.linalg import solve_triangular  # EN: Import symbol(s) from a module: from scipy.linalg import solve_triangular.
    x_tri = solve_triangular(R_ls, Qt_b)  # EN: Assign x_tri from expression: solve_triangular(R_ls, Qt_b).
    print(f"使用 solve_triangular: {x_tri}")  # EN: Print formatted output to the console.

    # 方法 3：比較 lstsq
    x_lstsq, _, _, _ = np.linalg.lstsq(A_ls, b, rcond=None)  # EN: Execute statement: x_lstsq, _, _, _ = np.linalg.lstsq(A_ls, b, rcond=None).
    print(f"lstsq 解: {x_lstsq}")  # EN: Print formatted output to the console.

    print(f"\n最佳直線：y = {x_qr[0]:.4f} + {x_qr[1]:.4f}t")  # EN: Print formatted output to the console.

    # ========================================
    # 4. Householder 反射
    # ========================================
    print_separator("4. Householder 反射原理")  # EN: Call print_separator(...) to perform an operation.

    def householder(x):  # EN: Define householder and its behavior.
        """計算 Householder 向量 v 和反射矩陣 H"""  # EN: Execute statement: """計算 Householder 向量 v 和反射矩陣 H""".
        n = len(x)  # EN: Assign n from expression: len(x).
        alpha = -np.sign(x[0]) * np.linalg.norm(x)  # EN: Assign alpha from expression: -np.sign(x[0]) * np.linalg.norm(x).

        v = x.copy()  # EN: Assign v from expression: x.copy().
        v[0] -= alpha  # EN: Execute statement: v[0] -= alpha.

        v = v / np.linalg.norm(v)  # EN: Assign v from expression: v / np.linalg.norm(v).

        H = np.eye(n) - 2 * np.outer(v, v)  # EN: Assign H from expression: np.eye(n) - 2 * np.outer(v, v).

        return H, alpha  # EN: Return a value: return H, alpha.

    # 示範
    x = np.array([1, 1, 0], dtype=float)  # EN: Assign x from expression: np.array([1, 1, 0], dtype=float).
    print(f"原始向量 x = {x}")  # EN: Print formatted output to the console.

    H, alpha = householder(x)  # EN: Execute statement: H, alpha = householder(x).
    print(f"\nHouseholder 矩陣 H：\n{H}")  # EN: Print formatted output to the console.
    print(f"\nHx = {H @ x}")  # EN: Print formatted output to the console.
    print(f"預期：[{alpha:.4f}, 0, 0]")  # EN: Print formatted output to the console.

    # 驗證 H 的性質
    print(f"\nHᵀ = H？ {np.allclose(H, H.T)}")  # EN: Print formatted output to the console.
    print(f"HᵀH = I？ {np.allclose(H @ H, np.eye(3))}")  # EN: Print formatted output to the console.
    print(f"det(H) = {np.linalg.det(H):.4f}（-1 表示反射）")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 條件數和穩定性
    # ========================================
    print_separator("5. 條件數和穩定性")  # EN: Call print_separator(...) to perform an operation.

    # 病態矩陣
    epsilon = 1e-10  # EN: Assign epsilon from expression: 1e-10.
    A_bad = np.array([  # EN: Assign A_bad from expression: np.array([.
        [1, 1],  # EN: Execute statement: [1, 1],.
        [1, 1 + epsilon],  # EN: Execute statement: [1, 1 + epsilon],.
        [1, 1 + 2*epsilon]  # EN: Execute statement: [1, 1 + 2*epsilon].
    ])  # EN: Execute statement: ]).

    print(f"病態矩陣 A：\n{A_bad}")  # EN: Print formatted output to the console.
    print(f"\ncond(A) = {np.linalg.cond(A_bad):.2e}")  # EN: Print formatted output to the console.
    print(f"cond(AᵀA) = {np.linalg.cond(A_bad.T @ A_bad):.2e}")  # EN: Print formatted output to the console.

    print("\n注意：cond(AᵀA) ≈ cond(A)²")  # EN: Print formatted output to the console.
    print("這就是為什麼 QR 比正規方程更穩定")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 比較不同方法
    # ========================================
    print_separator("6. 比較不同方法的準確性")  # EN: Call print_separator(...) to perform an operation.

    # 造一個有確切解的問題
    np.random.seed(42)  # EN: Execute statement: np.random.seed(42).
    m, n = 100, 10  # EN: Execute statement: m, n = 100, 10.
    A_test = np.random.randn(m, n)  # EN: Assign A_test from expression: np.random.randn(m, n).
    x_true = np.ones(n)  # EN: Assign x_true from expression: np.ones(n).
    b_test = A_test @ x_true  # 無噪聲  # EN: Assign b_test from expression: A_test @ x_true # 無噪聲.

    # 方法 1：正規方程
    x_normal = np.linalg.solve(A_test.T @ A_test, A_test.T @ b_test)  # EN: Assign x_normal from expression: np.linalg.solve(A_test.T @ A_test, A_test.T @ b_test).

    # 方法 2：QR 分解
    Q_test, R_test = np.linalg.qr(A_test)  # EN: Execute statement: Q_test, R_test = np.linalg.qr(A_test).
    x_qr_test = np.linalg.solve(R_test, Q_test.T @ b_test)  # EN: Assign x_qr_test from expression: np.linalg.solve(R_test, Q_test.T @ b_test).

    # 方法 3：lstsq（使用 SVD）
    x_lstsq_test, _, _, _ = np.linalg.lstsq(A_test, b_test, rcond=None)  # EN: Execute statement: x_lstsq_test, _, _, _ = np.linalg.lstsq(A_test, b_test, rcond=None).

    print(f"真實解：x = [1, 1, ..., 1]")  # EN: Print formatted output to the console.
    print(f"\n各方法的誤差（‖x - x_true‖）：")  # EN: Print formatted output to the console.
    print(f"正規方程：{np.linalg.norm(x_normal - x_true):.2e}")  # EN: Print formatted output to the console.
    print(f"QR 分解：{np.linalg.norm(x_qr_test - x_true):.2e}")  # EN: Print formatted output to the console.
    print(f"SVD (lstsq)：{np.linalg.norm(x_lstsq_test - x_true):.2e}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("NumPy QR 函數總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
