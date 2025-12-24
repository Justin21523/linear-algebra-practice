"""
最小平方回歸 - NumPy 版本 (Least Squares Regression - NumPy Implementation)

本程式示範：
1. 使用 np.linalg.lstsq 解最小平方問題
2. 使用 QR 分解求解
3. 線性迴歸和多項式擬合
4. 殘差分析和可視化

NumPy 提供高效且數值穩定的求解方法。
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

    print_separator("最小平方回歸示範（NumPy 版）\nLeast Squares Regression Demo (NumPy)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 簡單線性迴歸
    # ========================================
    print_separator("1. 簡單線性迴歸：y = C + Dt")  # EN: Call print_separator(...) to perform an operation.

    # 數據點
    t = np.array([0, 1, 2], dtype=float)  # EN: Assign t from expression: np.array([0, 1, 2], dtype=float).
    b = np.array([1, 3, 4], dtype=float)  # EN: Assign b from expression: np.array([1, 3, 4], dtype=float).

    print("數據點：")  # EN: Print formatted output to the console.
    for ti, bi in zip(t, b):  # EN: Iterate with a for-loop: for ti, bi in zip(t, b):.
        print(f"  t = {ti}, b = {bi}")  # EN: Print formatted output to the console.

    # 設計矩陣
    A = np.column_stack([np.ones_like(t), t])  # EN: Assign A from expression: np.column_stack([np.ones_like(t), t]).
    print(f"\n設計矩陣 A [1, t]：\n{A}")  # EN: Print formatted output to the console.
    print(f"\n觀測值 b = {b}")  # EN: Print formatted output to the console.

    # 方法 1：正規方程
    print("\n【方法 1：正規方程】")  # EN: Print formatted output to the console.
    ATA = A.T @ A  # EN: Assign ATA from expression: A.T @ A.
    ATb = A.T @ b  # EN: Assign ATb from expression: A.T @ b.
    print(f"AᵀA =\n{ATA}")  # EN: Print formatted output to the console.
    print(f"Aᵀb = {ATb}")  # EN: Print formatted output to the console.

    x_hat_normal = np.linalg.solve(ATA, ATb)  # EN: Assign x_hat_normal from expression: np.linalg.solve(ATA, ATb).
    print(f"解 x̂ = {x_hat_normal}")  # EN: Print formatted output to the console.

    # 方法 2：np.linalg.lstsq
    print("\n【方法 2：np.linalg.lstsq】")  # EN: Print formatted output to the console.
    x_hat_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)  # EN: Execute statement: x_hat_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None).
    print(f"解 x̂ = {x_hat_lstsq}")  # EN: Print formatted output to the console.
    print(f"rank(A) = {rank}")  # EN: Print formatted output to the console.

    # 方法 3：QR 分解
    print("\n【方法 3：QR 分解】")  # EN: Print formatted output to the console.
    Q, R = np.linalg.qr(A)  # EN: Execute statement: Q, R = np.linalg.qr(A).
    print(f"Q =\n{Q}")  # EN: Print formatted output to the console.
    print(f"R =\n{R}")  # EN: Print formatted output to the console.

    # Rx̂ = Qᵀb
    x_hat_qr = np.linalg.solve(R, Q.T @ b)  # EN: Assign x_hat_qr from expression: np.linalg.solve(R, Q.T @ b).
    print(f"解 x̂ = {x_hat_qr}")  # EN: Print formatted output to the console.

    # 結果
    C, D = x_hat_lstsq  # EN: Execute statement: C, D = x_hat_lstsq.
    print(f"\n最佳直線：y = {C:.4f} + {D:.4f}t")  # EN: Print formatted output to the console.

    # 擬合和殘差
    y_hat = A @ x_hat_lstsq  # EN: Assign y_hat from expression: A @ x_hat_lstsq.
    e = b - y_hat  # EN: Assign e from expression: b - y_hat.

    print(f"\n擬合值 ŷ = {y_hat}")  # EN: Print formatted output to the console.
    print(f"殘差 e = {e}")  # EN: Print formatted output to the console.
    print(f"殘差範數 ‖e‖ = {np.linalg.norm(e):.4f}")  # EN: Print formatted output to the console.

    # 驗證 e ⊥ C(A)
    print(f"\n驗證 Aᵀe = {A.T @ e}（應為零向量）")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 更多數據點
    # ========================================
    print_separator("2. 更多數據點的線性迴歸")  # EN: Call print_separator(...) to perform an operation.

    t2 = np.array([0, 1, 2, 3, 4, 5], dtype=float)  # EN: Assign t2 from expression: np.array([0, 1, 2, 3, 4, 5], dtype=float).
    b2 = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.0], dtype=float)  # EN: Assign b2 from expression: np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.0], dtype=float).

    print("數據點：")  # EN: Print formatted output to the console.
    for ti, bi in zip(t2, b2):  # EN: Iterate with a for-loop: for ti, bi in zip(t2, b2):.
        print(f"  ({ti:.1f}, {bi:.1f})")  # EN: Print formatted output to the console.

    A2 = np.column_stack([np.ones_like(t2), t2])  # EN: Assign A2 from expression: np.column_stack([np.ones_like(t2), t2]).
    x_hat2, _, _, _ = np.linalg.lstsq(A2, b2, rcond=None)  # EN: Execute statement: x_hat2, _, _, _ = np.linalg.lstsq(A2, b2, rcond=None).

    C2, D2 = x_hat2  # EN: Execute statement: C2, D2 = x_hat2.
    y_hat2 = A2 @ x_hat2  # EN: Assign y_hat2 from expression: A2 @ x_hat2.
    e2 = b2 - y_hat2  # EN: Assign e2 from expression: b2 - y_hat2.

    # R²
    b_mean = np.mean(b2)  # EN: Assign b_mean from expression: np.mean(b2).
    tss = np.sum((b2 - b_mean) ** 2)  # EN: Assign tss from expression: np.sum((b2 - b_mean) ** 2).
    rss = np.sum(e2 ** 2)  # EN: Assign rss from expression: np.sum(e2 ** 2).
    r_squared = 1 - rss / tss  # EN: Assign r_squared from expression: 1 - rss / tss.

    print(f"\n最佳直線：y = {C2:.4f} + {D2:.4f}t")  # EN: Print formatted output to the console.
    print(f"R² = {r_squared:.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 多項式擬合
    # ========================================
    print_separator("3. 多項式擬合")  # EN: Call print_separator(...) to perform an operation.

    t3 = np.array([0, 1, 2, 3, 4], dtype=float)  # EN: Assign t3 from expression: np.array([0, 1, 2, 3, 4], dtype=float).
    b3 = np.array([1, 2, 5, 10, 17], dtype=float)  # EN: Assign b3 from expression: np.array([1, 2, 5, 10, 17], dtype=float).

    print("數據點：")  # EN: Print formatted output to the console.
    for ti, bi in zip(t3, b3):  # EN: Iterate with a for-loop: for ti, bi in zip(t3, b3):.
        print(f"  ({ti:.1f}, {bi:.1f})")  # EN: Print formatted output to the console.

    # 使用 np.vander 建立 Vandermonde 矩陣
    degrees = [1, 2, 3]  # EN: Assign degrees from expression: [1, 2, 3].

    for deg in degrees:  # EN: Iterate with a for-loop: for deg in degrees:.
        # 注意：np.vander 是從高次到低次，需要翻轉
        A_poly = np.vander(t3, deg + 1, increasing=True)  # EN: Assign A_poly from expression: np.vander(t3, deg + 1, increasing=True).

        x_hat_poly, _, _, _ = np.linalg.lstsq(A_poly, b3, rcond=None)  # EN: Execute statement: x_hat_poly, _, _, _ = np.linalg.lstsq(A_poly, b3, rcond=None).
        y_hat_poly = A_poly @ x_hat_poly  # EN: Assign y_hat_poly from expression: A_poly @ x_hat_poly.
        e_poly = b3 - y_hat_poly  # EN: Assign e_poly from expression: b3 - y_hat_poly.

        tss = np.sum((b3 - np.mean(b3)) ** 2)  # EN: Assign tss from expression: np.sum((b3 - np.mean(b3)) ** 2).
        rss = np.sum(e_poly ** 2)  # EN: Assign rss from expression: np.sum(e_poly ** 2).
        r_sq = 1 - rss / tss  # EN: Assign r_sq from expression: 1 - rss / tss.

        # 顯示多項式
        terms = []  # EN: Assign terms from expression: [].
        for i, c in enumerate(x_hat_poly):  # EN: Iterate with a for-loop: for i, c in enumerate(x_hat_poly):.
            if i == 0:  # EN: Branch on a condition: if i == 0:.
                terms.append(f"{c:.4f}")  # EN: Execute statement: terms.append(f"{c:.4f}").
            elif i == 1:  # EN: Branch on a condition: elif i == 1:.
                terms.append(f"{c:.4f}t")  # EN: Execute statement: terms.append(f"{c:.4f}t").
            else:  # EN: Execute the fallback branch when prior conditions are false.
                terms.append(f"{c:.4f}t^{i}")  # EN: Execute statement: terms.append(f"{c:.4f}t^{i}").

        poly_str = " + ".join(terms)  # EN: Assign poly_str from expression: " + ".join(terms).
        print(f"\n{deg}次多項式：y = {poly_str}")  # EN: Print formatted output to the console.
        print(f"R² = {r_sq:.4f}, ‖e‖ = {np.linalg.norm(e_poly):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 使用 np.polyfit
    # ========================================
    print_separator("4. 使用 np.polyfit")  # EN: Call print_separator(...) to perform an operation.

    # np.polyfit 是高次優先
    coeffs = np.polyfit(t3, b3, 2)  # EN: Assign coeffs from expression: np.polyfit(t3, b3, 2).
    print(f"二次多項式係數（高次優先）：{coeffs}")  # EN: Print formatted output to the console.

    # 建立多項式函數
    poly = np.poly1d(coeffs)  # EN: Assign poly from expression: np.poly1d(coeffs).
    print(f"多項式：{poly}")  # EN: Print formatted output to the console.

    # 預測
    t_test = np.array([5, 6])  # EN: Assign t_test from expression: np.array([5, 6]).
    print(f"\n預測 t={t_test}：y = {poly(t_test)}")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 無解系統
    # ========================================
    print_separator("5. 無解系統的最佳近似")  # EN: Call print_separator(...) to perform an operation.

    A4 = np.array([  # EN: Assign A4 from expression: np.array([.
        [1, 1],  # EN: Execute statement: [1, 1],.
        [1, -1],  # EN: Execute statement: [1, -1],.
        [2, 1]  # EN: Execute statement: [2, 1].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    b4 = np.array([1, 1, 3], dtype=float)  # EN: Assign b4 from expression: np.array([1, 1, 3], dtype=float).

    print("方程組：")  # EN: Print formatted output to the console.
    print("  x + y = 1")  # EN: Print formatted output to the console.
    print("  x - y = 1")  # EN: Print formatted output to the console.
    print("  2x + y = 3")  # EN: Print formatted output to the console.

    print(f"\nA =\n{A4}")  # EN: Print formatted output to the console.
    print(f"b = {b4}")  # EN: Print formatted output to the console.

    x_hat4, residuals4, rank4, s4 = np.linalg.lstsq(A4, b4, rcond=None)  # EN: Execute statement: x_hat4, residuals4, rank4, s4 = np.linalg.lstsq(A4, b4, rcond=None).

    print(f"\n最小平方解：{x_hat4}")  # EN: Print formatted output to the console.
    print(f"Ax̂ = {A4 @ x_hat4}")  # EN: Print formatted output to the console.
    print(f"殘差 e = {b4 - A4 @ x_hat4}")  # EN: Print formatted output to the console.
    print(f"殘差範數 = {np.linalg.norm(b4 - A4 @ x_hat4):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 條件數和數值穩定性
    # ========================================
    print_separator("6. 條件數和數值穩定性")  # EN: Call print_separator(...) to perform an operation.

    # 病態矩陣
    t_bad = np.array([0, 0.001, 0.002, 0.003], dtype=float)  # EN: Assign t_bad from expression: np.array([0, 0.001, 0.002, 0.003], dtype=float).
    A_bad = np.column_stack([np.ones_like(t_bad), t_bad, t_bad**2])  # EN: Assign A_bad from expression: np.column_stack([np.ones_like(t_bad), t_bad, t_bad**2]).

    print(f"病態設計矩陣 A：\n{A_bad}")  # EN: Print formatted output to the console.
    print(f"\n條件數 cond(A) = {np.linalg.cond(A_bad):.2e}")  # EN: Print formatted output to the console.
    print("（條件數大表示數值不穩定）")  # EN: Print formatted output to the console.

    print("\n解決方案：")  # EN: Print formatted output to the console.
    print("1. 使用 QR 分解而非直接求 (AᵀA)⁻¹")  # EN: Print formatted output to the console.
    print("2. 使用 SVD 分解")  # EN: Print formatted output to the console.
    print("3. 正則化（嶺迴歸）")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 嶺迴歸
    # ========================================
    print_separator("7. 嶺迴歸（正則化）")  # EN: Call print_separator(...) to perform an operation.

    lambda_reg = 0.1  # EN: Assign lambda_reg from expression: 0.1.
    n = A4.shape[1]  # EN: Assign n from expression: A4.shape[1].

    # (AᵀA + λI)x̂ = Aᵀb
    x_hat_ridge = np.linalg.solve(A4.T @ A4 + lambda_reg * np.eye(n), A4.T @ b4)  # EN: Assign x_hat_ridge from expression: np.linalg.solve(A4.T @ A4 + lambda_reg * np.eye(n), A4.T @ b4).

    print(f"正則化參數 λ = {lambda_reg}")  # EN: Print formatted output to the console.
    print(f"嶺迴歸解：{x_hat_ridge}")  # EN: Print formatted output to the console.
    print(f"一般最小平方解：{x_hat4}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("NumPy 最小平方函數總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
