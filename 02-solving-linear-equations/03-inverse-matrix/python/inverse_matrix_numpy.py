"""
反矩陣：NumPy 版本 (Inverse Matrix: NumPy Implementation)

本程式示範：
1. np.linalg.inv 求反矩陣
2. 為什麼不該用反矩陣求解方程組
3. 數值穩定性問題
4. 條件數與可逆性判斷

This program demonstrates inverse matrices in NumPy and
warns about numerical issues.
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.
import time  # EN: Import module(s): import time.

np.set_printoptions(precision=6, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=6, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("反矩陣示範 - NumPy 版本\nInverse Matrix Demo - NumPy")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 基本用法
    # ========================================
    print_separator("1. 基本用法：np.linalg.inv")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [4, 7],  # EN: Execute statement: [4, 7],.
        [2, 6]  # EN: Execute statement: [2, 6].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A =\n{A}\n")  # EN: Print formatted output to the console.

    A_inv = np.linalg.inv(A)  # EN: Assign A_inv from expression: np.linalg.inv(A).
    print(f"A⁻¹ = np.linalg.inv(A) =\n{A_inv}\n")  # EN: Print formatted output to the console.

    print(f"驗證 A @ A⁻¹ =\n{A @ A_inv}")  # EN: Print formatted output to the console.
    print(f"\n是單位矩陣？ {np.allclose(A @ A_inv, np.eye(2))}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 行列式與可逆性
    # ========================================
    print_separator("2. 行列式與可逆性")  # EN: Call print_separator(...) to perform an operation.

    print(f"det(A) = {np.linalg.det(A):.4f}")  # EN: Print formatted output to the console.
    print(f"det(A⁻¹) = {np.linalg.det(A_inv):.4f}")  # EN: Print formatted output to the console.
    print(f"det(A) × det(A⁻¹) = {np.linalg.det(A) * np.linalg.det(A_inv):.4f} (應該是 1)")  # EN: Print formatted output to the console.

    # 奇異矩陣
    print("\n奇異矩陣：")  # EN: Print formatted output to the console.
    A_singular = np.array([[1, 2], [2, 4]], dtype=float)  # EN: Assign A_singular from expression: np.array([[1, 2], [2, 4]], dtype=float).
    print(f"A =\n{A_singular}")  # EN: Print formatted output to the console.
    print(f"det(A) = {np.linalg.det(A_singular):.6f}")  # EN: Print formatted output to the console.

    try:  # EN: Start a try block for exception handling.
        np.linalg.inv(A_singular)  # EN: Execute statement: np.linalg.inv(A_singular).
    except np.linalg.LinAlgError as e:  # EN: Handle an exception case: except np.linalg.LinAlgError as e:.
        print(f"錯誤：{e}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. ⚠️ 為什麼不該用反矩陣解方程組
    # ========================================
    print_separator("3. ⚠️ 為什麼不該用 A⁻¹b 解方程組")  # EN: Call print_separator(...) to perform an operation.

    n = 100  # EN: Assign n from expression: 100.
    A_big = np.random.rand(n, n) + np.eye(n)  # 確保可逆  # EN: Assign A_big from expression: np.random.rand(n, n) + np.eye(n) # 確保可逆.
    b_big = np.random.rand(n)  # EN: Assign b_big from expression: np.random.rand(n).

    # 方法 1：x = A⁻¹b（不好）
    start = time.time()  # EN: Assign start from expression: time.time().
    x_inv = np.linalg.inv(A_big) @ b_big  # EN: Assign x_inv from expression: np.linalg.inv(A_big) @ b_big.
    time_inv = time.time() - start  # EN: Assign time_inv from expression: time.time() - start.

    # 方法 2：np.linalg.solve（好）
    start = time.time()  # EN: Assign start from expression: time.time().
    x_solve = np.linalg.solve(A_big, b_big)  # EN: Assign x_solve from expression: np.linalg.solve(A_big, b_big).
    time_solve = time.time() - start  # EN: Assign time_solve from expression: time.time() - start.

    print(f"矩陣大小：{n}×{n}")  # EN: Print formatted output to the console.
    print(f"\n方法 1（A⁻¹b）時間：{time_inv*1000:.3f} ms")  # EN: Print formatted output to the console.
    print(f"方法 2（solve）時間：{time_solve*1000:.3f} ms")  # EN: Print formatted output to the console.
    print(f"速度比：{time_inv/time_solve:.2f}x")  # EN: Print formatted output to the console.

    print(f"\n殘差比較：")  # EN: Print formatted output to the console.
    print(f"  A⁻¹b 殘差：‖Ax - b‖ = {np.linalg.norm(A_big @ x_inv - b_big):.2e}")  # EN: Print formatted output to the console.
    print(f"  solve 殘差：‖Ax - b‖ = {np.linalg.norm(A_big @ x_solve - b_big):.2e}")  # EN: Print formatted output to the console.

    print("""
結論：
❌ np.linalg.inv(A) @ b 較慢、較不穩定
✅ np.linalg.solve(A, b) 較快、較穩定
""")  # EN: Execute statement: """).

    # ========================================
    # 4. 條件數 (Condition Number)
    # ========================================
    print_separator("4. 條件數 (Condition Number)")  # EN: Call print_separator(...) to perform an operation.

    print("""
條件數 cond(A) = ‖A‖ × ‖A⁻¹‖

- cond(A) ≈ 1：良態矩陣（well-conditioned）
- cond(A) 很大：病態矩陣（ill-conditioned）
- cond(A) = ∞：奇異矩陣

經驗法則：若 cond(A) ≈ 10^k，則解可能丟失約 k 位有效數字
""")  # EN: Execute statement: """).

    # 良態矩陣
    A_good = np.array([[2, 1], [1, 2]], dtype=float)  # EN: Assign A_good from expression: np.array([[2, 1], [1, 2]], dtype=float).
    print(f"良態矩陣 A =\n{A_good}")  # EN: Print formatted output to the console.
    print(f"cond(A) = {np.linalg.cond(A_good):.4f}\n")  # EN: Print formatted output to the console.

    # 病態矩陣 (Hilbert 矩陣)
    n_hilbert = 10  # EN: Assign n_hilbert from expression: 10.
    H = np.array([[1/(i+j+1) for j in range(n_hilbert)]  # EN: Assign H from expression: np.array([[1/(i+j+1) for j in range(n_hilbert)].
                  for i in range(n_hilbert)])  # EN: Iterate with a for-loop: for i in range(n_hilbert)]).
    print(f"Hilbert 矩陣 H({n_hilbert}×{n_hilbert}) 的條件數：")  # EN: Print formatted output to the console.
    print(f"cond(H) = {np.linalg.cond(H):.2e}")  # EN: Print formatted output to the console.
    print("這意味著數值解可能非常不準確！")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 反矩陣的性質
    # ========================================
    print_separator("5. 反矩陣的性質驗證")  # EN: Call print_separator(...) to perform an operation.

    B = np.array([[1, 2], [3, 4]], dtype=float)  # EN: Assign B from expression: np.array([[1, 2], [3, 4]], dtype=float).
    C = np.array([[2, 0], [1, 2]], dtype=float)  # EN: Assign C from expression: np.array([[2, 0], [1, 2]], dtype=float).

    B_inv = np.linalg.inv(B)  # EN: Assign B_inv from expression: np.linalg.inv(B).
    C_inv = np.linalg.inv(C)  # EN: Assign C_inv from expression: np.linalg.inv(C).

    print("驗證 (BC)⁻¹ = C⁻¹B⁻¹：")  # EN: Print formatted output to the console.
    BC_inv = np.linalg.inv(B @ C)  # EN: Assign BC_inv from expression: np.linalg.inv(B @ C).
    C_inv_B_inv = C_inv @ B_inv  # EN: Assign C_inv_B_inv from expression: C_inv @ B_inv.
    print(f"(BC)⁻¹ =\n{BC_inv}\n")  # EN: Print formatted output to the console.
    print(f"C⁻¹B⁻¹ =\n{C_inv_B_inv}")  # EN: Print formatted output to the console.
    print(f"相等？ {np.allclose(BC_inv, C_inv_B_inv)}\n")  # EN: Print formatted output to the console.

    print("驗證 (Bᵀ)⁻¹ = (B⁻¹)ᵀ：")  # EN: Print formatted output to the console.
    BT_inv = np.linalg.inv(B.T)  # EN: Assign BT_inv from expression: np.linalg.inv(B.T).
    B_inv_T = B_inv.T  # EN: Assign B_inv_T from expression: B_inv.T.
    print(f"(Bᵀ)⁻¹ =\n{BT_inv}\n")  # EN: Print formatted output to the console.
    print(f"(B⁻¹)ᵀ =\n{B_inv_T}")  # EN: Print formatted output to the console.
    print(f"相等？ {np.allclose(BT_inv, B_inv_T)}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 特殊矩陣的反矩陣
    # ========================================
    print_separator("6. 特殊矩陣的反矩陣")  # EN: Call print_separator(...) to perform an operation.

    # 對角矩陣
    D = np.diag([2, 3, 4])  # EN: Assign D from expression: np.diag([2, 3, 4]).
    D_inv = np.linalg.inv(D)  # EN: Assign D_inv from expression: np.linalg.inv(D).
    print("對角矩陣：D⁻¹ = diag(1/dᵢ)")  # EN: Print formatted output to the console.
    print(f"D = diag(2, 3, 4)")  # EN: Print formatted output to the console.
    print(f"D⁻¹ = diag{tuple(np.diag(D_inv))}\n")  # EN: Print formatted output to the console.

    # 正交矩陣
    theta = np.pi / 4  # EN: Assign theta from expression: np.pi / 4.
    Q = np.array([  # EN: Assign Q from expression: np.array([.
        [np.cos(theta), -np.sin(theta)],  # EN: Execute statement: [np.cos(theta), -np.sin(theta)],.
        [np.sin(theta), np.cos(theta)]  # EN: Execute statement: [np.sin(theta), np.cos(theta)].
    ])  # EN: Execute statement: ]).
    Q_inv = np.linalg.inv(Q)  # EN: Assign Q_inv from expression: np.linalg.inv(Q).
    print("正交矩陣：Q⁻¹ = Qᵀ")  # EN: Print formatted output to the console.
    print(f"Q（旋轉 45°）=\n{Q}\n")  # EN: Print formatted output to the console.
    print(f"Q⁻¹ =\n{Q_inv}\n")  # EN: Print formatted output to the console.
    print(f"Qᵀ =\n{Q.T}")  # EN: Print formatted output to the console.
    print(f"Q⁻¹ == Qᵀ ? {np.allclose(Q_inv, Q.T)}")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 偽逆矩陣 (Moore-Penrose Pseudoinverse)
    # ========================================
    print_separator("7. 偽逆矩陣（非方陣或奇異矩陣）")  # EN: Call print_separator(...) to perform an operation.

    print("當 A 不是方陣或不可逆時，可以使用偽逆矩陣 A⁺")  # EN: Print formatted output to the console.

    # 非方陣
    A_rect = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)  # EN: Assign A_rect from expression: np.array([[1, 2], [3, 4], [5, 6]], dtype=float).
    A_pinv = np.linalg.pinv(A_rect)  # EN: Assign A_pinv from expression: np.linalg.pinv(A_rect).

    print(f"A (3×2) =\n{A_rect}\n")  # EN: Print formatted output to the console.
    print(f"A⁺ (2×3) = np.linalg.pinv(A) =\n{A_pinv}\n")  # EN: Print formatted output to the console.

    print(f"A⁺ @ A =\n{A_pinv @ A_rect}")  # EN: Print formatted output to the console.
    print("（左偽逆：A⁺A ≈ I，但 AA⁺ ≠ I）")  # EN: Print formatted output to the console.

    # 奇異矩陣
    print("\n奇異矩陣也可以用偽逆：")  # EN: Print formatted output to the console.
    A_sing = np.array([[1, 2], [2, 4]], dtype=float)  # EN: Assign A_sing from expression: np.array([[1, 2], [2, 4]], dtype=float).
    A_sing_pinv = np.linalg.pinv(A_sing)  # EN: Assign A_sing_pinv from expression: np.linalg.pinv(A_sing).
    print(f"A（奇異）=\n{A_sing}")  # EN: Print formatted output to the console.
    print(f"A⁺ =\n{A_sing_pinv}")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 小結
    # ========================================
    print_separator("8. 小結：何時使用反矩陣")  # EN: Call print_separator(...) to perform an operation.

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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("反矩陣示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
