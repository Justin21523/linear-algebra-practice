"""
反矩陣：NumPy 版本 (Inverse Matrix: NumPy Implementation)

本程式示範：
1. np.linalg.inv 求反矩陣
2. 為什麼不該用反矩陣求解方程組
3. 數值穩定性問題
4. 條件數與可逆性判斷

This program demonstrates inverse matrices in NumPy and
warns about numerical issues.
"""

import numpy as np
import time

np.set_printoptions(precision=6, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    """主程式"""

    print_separator("反矩陣示範 - NumPy 版本\nInverse Matrix Demo - NumPy")

    # ========================================
    # 1. 基本用法
    # ========================================
    print_separator("1. 基本用法：np.linalg.inv")

    A = np.array([
        [4, 7],
        [2, 6]
    ], dtype=float)

    print(f"A =\n{A}\n")

    A_inv = np.linalg.inv(A)
    print(f"A⁻¹ = np.linalg.inv(A) =\n{A_inv}\n")

    print(f"驗證 A @ A⁻¹ =\n{A @ A_inv}")
    print(f"\n是單位矩陣？ {np.allclose(A @ A_inv, np.eye(2))}")

    # ========================================
    # 2. 行列式與可逆性
    # ========================================
    print_separator("2. 行列式與可逆性")

    print(f"det(A) = {np.linalg.det(A):.4f}")
    print(f"det(A⁻¹) = {np.linalg.det(A_inv):.4f}")
    print(f"det(A) × det(A⁻¹) = {np.linalg.det(A) * np.linalg.det(A_inv):.4f} (應該是 1)")

    # 奇異矩陣
    print("\n奇異矩陣：")
    A_singular = np.array([[1, 2], [2, 4]], dtype=float)
    print(f"A =\n{A_singular}")
    print(f"det(A) = {np.linalg.det(A_singular):.6f}")

    try:
        np.linalg.inv(A_singular)
    except np.linalg.LinAlgError as e:
        print(f"錯誤：{e}")

    # ========================================
    # 3. ⚠️ 為什麼不該用反矩陣解方程組
    # ========================================
    print_separator("3. ⚠️ 為什麼不該用 A⁻¹b 解方程組")

    n = 100
    A_big = np.random.rand(n, n) + np.eye(n)  # 確保可逆
    b_big = np.random.rand(n)

    # 方法 1：x = A⁻¹b（不好）
    start = time.time()
    x_inv = np.linalg.inv(A_big) @ b_big
    time_inv = time.time() - start

    # 方法 2：np.linalg.solve（好）
    start = time.time()
    x_solve = np.linalg.solve(A_big, b_big)
    time_solve = time.time() - start

    print(f"矩陣大小：{n}×{n}")
    print(f"\n方法 1（A⁻¹b）時間：{time_inv*1000:.3f} ms")
    print(f"方法 2（solve）時間：{time_solve*1000:.3f} ms")
    print(f"速度比：{time_inv/time_solve:.2f}x")

    print(f"\n殘差比較：")
    print(f"  A⁻¹b 殘差：‖Ax - b‖ = {np.linalg.norm(A_big @ x_inv - b_big):.2e}")
    print(f"  solve 殘差：‖Ax - b‖ = {np.linalg.norm(A_big @ x_solve - b_big):.2e}")

    print("""
結論：
❌ np.linalg.inv(A) @ b 較慢、較不穩定
✅ np.linalg.solve(A, b) 較快、較穩定
""")

    # ========================================
    # 4. 條件數 (Condition Number)
    # ========================================
    print_separator("4. 條件數 (Condition Number)")

    print("""
條件數 cond(A) = ‖A‖ × ‖A⁻¹‖

- cond(A) ≈ 1：良態矩陣（well-conditioned）
- cond(A) 很大：病態矩陣（ill-conditioned）
- cond(A) = ∞：奇異矩陣

經驗法則：若 cond(A) ≈ 10^k，則解可能丟失約 k 位有效數字
""")

    # 良態矩陣
    A_good = np.array([[2, 1], [1, 2]], dtype=float)
    print(f"良態矩陣 A =\n{A_good}")
    print(f"cond(A) = {np.linalg.cond(A_good):.4f}\n")

    # 病態矩陣 (Hilbert 矩陣)
    n_hilbert = 10
    H = np.array([[1/(i+j+1) for j in range(n_hilbert)]
                  for i in range(n_hilbert)])
    print(f"Hilbert 矩陣 H({n_hilbert}×{n_hilbert}) 的條件數：")
    print(f"cond(H) = {np.linalg.cond(H):.2e}")
    print("這意味著數值解可能非常不準確！")

    # ========================================
    # 5. 反矩陣的性質
    # ========================================
    print_separator("5. 反矩陣的性質驗證")

    B = np.array([[1, 2], [3, 4]], dtype=float)
    C = np.array([[2, 0], [1, 2]], dtype=float)

    B_inv = np.linalg.inv(B)
    C_inv = np.linalg.inv(C)

    print("驗證 (BC)⁻¹ = C⁻¹B⁻¹：")
    BC_inv = np.linalg.inv(B @ C)
    C_inv_B_inv = C_inv @ B_inv
    print(f"(BC)⁻¹ =\n{BC_inv}\n")
    print(f"C⁻¹B⁻¹ =\n{C_inv_B_inv}")
    print(f"相等？ {np.allclose(BC_inv, C_inv_B_inv)}\n")

    print("驗證 (Bᵀ)⁻¹ = (B⁻¹)ᵀ：")
    BT_inv = np.linalg.inv(B.T)
    B_inv_T = B_inv.T
    print(f"(Bᵀ)⁻¹ =\n{BT_inv}\n")
    print(f"(B⁻¹)ᵀ =\n{B_inv_T}")
    print(f"相等？ {np.allclose(BT_inv, B_inv_T)}")

    # ========================================
    # 6. 特殊矩陣的反矩陣
    # ========================================
    print_separator("6. 特殊矩陣的反矩陣")

    # 對角矩陣
    D = np.diag([2, 3, 4])
    D_inv = np.linalg.inv(D)
    print("對角矩陣：D⁻¹ = diag(1/dᵢ)")
    print(f"D = diag(2, 3, 4)")
    print(f"D⁻¹ = diag{tuple(np.diag(D_inv))}\n")

    # 正交矩陣
    theta = np.pi / 4
    Q = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    Q_inv = np.linalg.inv(Q)
    print("正交矩陣：Q⁻¹ = Qᵀ")
    print(f"Q（旋轉 45°）=\n{Q}\n")
    print(f"Q⁻¹ =\n{Q_inv}\n")
    print(f"Qᵀ =\n{Q.T}")
    print(f"Q⁻¹ == Qᵀ ? {np.allclose(Q_inv, Q.T)}")

    # ========================================
    # 7. 偽逆矩陣 (Moore-Penrose Pseudoinverse)
    # ========================================
    print_separator("7. 偽逆矩陣（非方陣或奇異矩陣）")

    print("當 A 不是方陣或不可逆時，可以使用偽逆矩陣 A⁺")

    # 非方陣
    A_rect = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    A_pinv = np.linalg.pinv(A_rect)

    print(f"A (3×2) =\n{A_rect}\n")
    print(f"A⁺ (2×3) = np.linalg.pinv(A) =\n{A_pinv}\n")

    print(f"A⁺ @ A =\n{A_pinv @ A_rect}")
    print("（左偽逆：A⁺A ≈ I，但 AA⁺ ≠ I）")

    # 奇異矩陣
    print("\n奇異矩陣也可以用偽逆：")
    A_sing = np.array([[1, 2], [2, 4]], dtype=float)
    A_sing_pinv = np.linalg.pinv(A_sing)
    print(f"A（奇異）=\n{A_sing}")
    print(f"A⁺ =\n{A_sing_pinv}")

    # ========================================
    # 8. 小結
    # ========================================
    print_separator("8. 小結：何時使用反矩陣")

    print("""
✅ 適合使用反矩陣：
- 理論推導和符號計算
- 需要 A⁻¹ 的多個元素
- 特殊結構矩陣（對角、正交）

❌ 不該使用反矩陣：
- 解線性方程組 Ax = b → 用 solve()
- 大型稀疏矩陣 → 用迭代法
- 數值不穩定的情況

⚠️ 注意事項：
- 檢查條件數 cond(A)
- 小矩陣可以，大矩陣謹慎
- 優先使用 LU 分解或 QR 分解
""")

    print("=" * 60)
    print("反矩陣示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
