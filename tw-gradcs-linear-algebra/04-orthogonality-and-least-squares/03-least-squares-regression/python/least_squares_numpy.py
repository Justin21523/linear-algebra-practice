"""
最小平方回歸 - NumPy 版本 (Least Squares Regression - NumPy Implementation)

本程式示範：
1. 使用 np.linalg.lstsq 解最小平方問題
2. 使用 QR 分解求解
3. 線性迴歸和多項式擬合
4. 殘差分析和可視化

NumPy 提供高效且數值穩定的求解方法。
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

    print_separator("最小平方回歸示範（NumPy 版）\nLeast Squares Regression Demo (NumPy)")

    # ========================================
    # 1. 簡單線性迴歸
    # ========================================
    print_separator("1. 簡單線性迴歸：y = C + Dt")

    # 數據點
    t = np.array([0, 1, 2], dtype=float)
    b = np.array([1, 3, 4], dtype=float)

    print("數據點：")
    for ti, bi in zip(t, b):
        print(f"  t = {ti}, b = {bi}")

    # 設計矩陣
    A = np.column_stack([np.ones_like(t), t])
    print(f"\n設計矩陣 A [1, t]：\n{A}")
    print(f"\n觀測值 b = {b}")

    # 方法 1：正規方程
    print("\n【方法 1：正規方程】")
    ATA = A.T @ A
    ATb = A.T @ b
    print(f"AᵀA =\n{ATA}")
    print(f"Aᵀb = {ATb}")

    x_hat_normal = np.linalg.solve(ATA, ATb)
    print(f"解 x̂ = {x_hat_normal}")

    # 方法 2：np.linalg.lstsq
    print("\n【方法 2：np.linalg.lstsq】")
    x_hat_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print(f"解 x̂ = {x_hat_lstsq}")
    print(f"rank(A) = {rank}")

    # 方法 3：QR 分解
    print("\n【方法 3：QR 分解】")
    Q, R = np.linalg.qr(A)
    print(f"Q =\n{Q}")
    print(f"R =\n{R}")

    # Rx̂ = Qᵀb
    x_hat_qr = np.linalg.solve(R, Q.T @ b)
    print(f"解 x̂ = {x_hat_qr}")

    # 結果
    C, D = x_hat_lstsq
    print(f"\n最佳直線：y = {C:.4f} + {D:.4f}t")

    # 擬合和殘差
    y_hat = A @ x_hat_lstsq
    e = b - y_hat

    print(f"\n擬合值 ŷ = {y_hat}")
    print(f"殘差 e = {e}")
    print(f"殘差範數 ‖e‖ = {np.linalg.norm(e):.4f}")

    # 驗證 e ⊥ C(A)
    print(f"\n驗證 Aᵀe = {A.T @ e}（應為零向量）")

    # ========================================
    # 2. 更多數據點
    # ========================================
    print_separator("2. 更多數據點的線性迴歸")

    t2 = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    b2 = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.0], dtype=float)

    print("數據點：")
    for ti, bi in zip(t2, b2):
        print(f"  ({ti:.1f}, {bi:.1f})")

    A2 = np.column_stack([np.ones_like(t2), t2])
    x_hat2, _, _, _ = np.linalg.lstsq(A2, b2, rcond=None)

    C2, D2 = x_hat2
    y_hat2 = A2 @ x_hat2
    e2 = b2 - y_hat2

    # R²
    b_mean = np.mean(b2)
    tss = np.sum((b2 - b_mean) ** 2)
    rss = np.sum(e2 ** 2)
    r_squared = 1 - rss / tss

    print(f"\n最佳直線：y = {C2:.4f} + {D2:.4f}t")
    print(f"R² = {r_squared:.4f}")

    # ========================================
    # 3. 多項式擬合
    # ========================================
    print_separator("3. 多項式擬合")

    t3 = np.array([0, 1, 2, 3, 4], dtype=float)
    b3 = np.array([1, 2, 5, 10, 17], dtype=float)

    print("數據點：")
    for ti, bi in zip(t3, b3):
        print(f"  ({ti:.1f}, {bi:.1f})")

    # 使用 np.vander 建立 Vandermonde 矩陣
    degrees = [1, 2, 3]

    for deg in degrees:
        # 注意：np.vander 是從高次到低次，需要翻轉
        A_poly = np.vander(t3, deg + 1, increasing=True)

        x_hat_poly, _, _, _ = np.linalg.lstsq(A_poly, b3, rcond=None)
        y_hat_poly = A_poly @ x_hat_poly
        e_poly = b3 - y_hat_poly

        tss = np.sum((b3 - np.mean(b3)) ** 2)
        rss = np.sum(e_poly ** 2)
        r_sq = 1 - rss / tss

        # 顯示多項式
        terms = []
        for i, c in enumerate(x_hat_poly):
            if i == 0:
                terms.append(f"{c:.4f}")
            elif i == 1:
                terms.append(f"{c:.4f}t")
            else:
                terms.append(f"{c:.4f}t^{i}")

        poly_str = " + ".join(terms)
        print(f"\n{deg}次多項式：y = {poly_str}")
        print(f"R² = {r_sq:.4f}, ‖e‖ = {np.linalg.norm(e_poly):.4f}")

    # ========================================
    # 4. 使用 np.polyfit
    # ========================================
    print_separator("4. 使用 np.polyfit")

    # np.polyfit 是高次優先
    coeffs = np.polyfit(t3, b3, 2)
    print(f"二次多項式係數（高次優先）：{coeffs}")

    # 建立多項式函數
    poly = np.poly1d(coeffs)
    print(f"多項式：{poly}")

    # 預測
    t_test = np.array([5, 6])
    print(f"\n預測 t={t_test}：y = {poly(t_test)}")

    # ========================================
    # 5. 無解系統
    # ========================================
    print_separator("5. 無解系統的最佳近似")

    A4 = np.array([
        [1, 1],
        [1, -1],
        [2, 1]
    ], dtype=float)

    b4 = np.array([1, 1, 3], dtype=float)

    print("方程組：")
    print("  x + y = 1")
    print("  x - y = 1")
    print("  2x + y = 3")

    print(f"\nA =\n{A4}")
    print(f"b = {b4}")

    x_hat4, residuals4, rank4, s4 = np.linalg.lstsq(A4, b4, rcond=None)

    print(f"\n最小平方解：{x_hat4}")
    print(f"Ax̂ = {A4 @ x_hat4}")
    print(f"殘差 e = {b4 - A4 @ x_hat4}")
    print(f"殘差範數 = {np.linalg.norm(b4 - A4 @ x_hat4):.4f}")

    # ========================================
    # 6. 條件數和數值穩定性
    # ========================================
    print_separator("6. 條件數和數值穩定性")

    # 病態矩陣
    t_bad = np.array([0, 0.001, 0.002, 0.003], dtype=float)
    A_bad = np.column_stack([np.ones_like(t_bad), t_bad, t_bad**2])

    print(f"病態設計矩陣 A：\n{A_bad}")
    print(f"\n條件數 cond(A) = {np.linalg.cond(A_bad):.2e}")
    print("（條件數大表示數值不穩定）")

    print("\n解決方案：")
    print("1. 使用 QR 分解而非直接求 (AᵀA)⁻¹")
    print("2. 使用 SVD 分解")
    print("3. 正則化（嶺迴歸）")

    # ========================================
    # 7. 嶺迴歸
    # ========================================
    print_separator("7. 嶺迴歸（正則化）")

    lambda_reg = 0.1
    n = A4.shape[1]

    # (AᵀA + λI)x̂ = Aᵀb
    x_hat_ridge = np.linalg.solve(A4.T @ A4 + lambda_reg * np.eye(n), A4.T @ b4)

    print(f"正則化參數 λ = {lambda_reg}")
    print(f"嶺迴歸解：{x_hat_ridge}")
    print(f"一般最小平方解：{x_hat4}")

    # 總結
    print_separator("NumPy 最小平方函數總結")
    print("""
基本求解：
  np.linalg.lstsq(A, b)      # 推薦使用
  np.linalg.solve(ATA, ATb)  # 正規方程

多項式擬合：
  np.polyfit(t, b, degree)   # 返回係數（高次優先）
  np.vander(t, N)            # Vandermonde 矩陣

分解方法：
  Q, R = np.linalg.qr(A)     # QR 分解
  U, s, Vh = np.linalg.svd(A)  # SVD 分解

診斷：
  np.linalg.cond(A)          # 條件數
  np.linalg.matrix_rank(A)   # 秩
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
