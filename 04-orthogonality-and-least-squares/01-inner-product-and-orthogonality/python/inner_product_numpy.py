"""
內積與正交性 - NumPy 版本 (Inner Product and Orthogonality - NumPy Implementation)

本程式示範：
1. 使用 NumPy 計算內積、範數、夾角
2. 正交性與正交矩陣
3. Cauchy-Schwarz 不等式
4. 正交投影的基礎

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


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("內積與正交性示範（NumPy 版）\nInner Product & Orthogonality Demo (NumPy)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 內積計算
    # ========================================
    print_separator("1. 內積計算 (Dot Product)")  # EN: Call print_separator(...) to perform an operation.

    x = np.array([1, 2, 3], dtype=float)  # EN: Assign x from expression: np.array([1, 2, 3], dtype=float).
    y = np.array([4, 5, 6], dtype=float)  # EN: Assign y from expression: np.array([4, 5, 6], dtype=float).

    print(f"x = {x}")  # EN: Print formatted output to the console.
    print(f"y = {y}")  # EN: Print formatted output to the console.

    # 多種計算內積的方式
    dot1 = np.dot(x, y)  # EN: Assign dot1 from expression: np.dot(x, y).
    dot2 = x @ y  # EN: Assign dot2 from expression: x @ y.
    dot3 = np.inner(x, y)  # EN: Assign dot3 from expression: np.inner(x, y).
    dot4 = np.sum(x * y)  # EN: Assign dot4 from expression: np.sum(x * y).

    print(f"\n內積計算方式：")  # EN: Print formatted output to the console.
    print(f"np.dot(x, y)   = {dot1}")  # EN: Print formatted output to the console.
    print(f"x @ y          = {dot2}")  # EN: Print formatted output to the console.
    print(f"np.inner(x, y) = {dot3}")  # EN: Print formatted output to the console.
    print(f"np.sum(x * y)  = {dot4}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 向量長度（範數）
    # ========================================
    print_separator("2. 向量長度 (Vector Norm)")  # EN: Call print_separator(...) to perform an operation.

    v = np.array([3, 4], dtype=float)  # EN: Assign v from expression: np.array([3, 4], dtype=float).
    print(f"v = {v}")  # EN: Print formatted output to the console.

    # L2 範數（預設）
    norm_v = np.linalg.norm(v)  # EN: Assign norm_v from expression: np.linalg.norm(v).
    print(f"\n‖v‖₂ = {norm_v}")  # EN: Print formatted output to the console.

    # 其他範數
    print(f"‖v‖₁ (L1) = {np.linalg.norm(v, ord=1)}")  # EN: Print formatted output to the console.
    print(f"‖v‖∞ (L∞) = {np.linalg.norm(v, ord=np.inf)}")  # EN: Print formatted output to the console.

    # 正規化
    v_normalized = v / norm_v  # EN: Assign v_normalized from expression: v / norm_v.
    print(f"\n單位向量 v̂ = {v_normalized}")  # EN: Print formatted output to the console.
    print(f"‖v̂‖ = {np.linalg.norm(v_normalized)}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 向量夾角
    # ========================================
    print_separator("3. 向量夾角 (Vector Angle)")  # EN: Call print_separator(...) to perform an operation.

    a = np.array([1, 0])  # EN: Assign a from expression: np.array([1, 0]).
    b = np.array([1, 1])  # EN: Assign b from expression: np.array([1, 1]).

    print(f"a = {a}")  # EN: Print formatted output to the console.
    print(f"b = {b}")  # EN: Print formatted output to the console.

    # cos θ = (a · b) / (‖a‖ ‖b‖)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # EN: Assign cos_theta from expression: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)).
    theta = np.arccos(np.clip(cos_theta, -1, 1))  # EN: Assign theta from expression: np.arccos(np.clip(cos_theta, -1, 1)).

    print(f"\ncos θ = {cos_theta:.4f}")  # EN: Print formatted output to the console.
    print(f"θ = {theta:.4f} rad = {np.degrees(theta):.2f}°")  # EN: Print formatted output to the console.

    # 向量化計算多組夾角
    print("\n批次計算多組夾角：")  # EN: Print formatted output to the console.
    vectors1 = np.array([[1, 0], [1, 0], [1, 1]])  # EN: Assign vectors1 from expression: np.array([[1, 0], [1, 0], [1, 1]]).
    vectors2 = np.array([[0, 1], [1, 1], [-1, 1]])  # EN: Assign vectors2 from expression: np.array([[0, 1], [1, 1], [-1, 1]]).

    # 使用 einsum 計算內積
    dots = np.einsum('ij,ij->i', vectors1, vectors2)  # EN: Assign dots from expression: np.einsum('ij,ij->i', vectors1, vectors2).
    norms1 = np.linalg.norm(vectors1, axis=1)  # EN: Assign norms1 from expression: np.linalg.norm(vectors1, axis=1).
    norms2 = np.linalg.norm(vectors2, axis=1)  # EN: Assign norms2 from expression: np.linalg.norm(vectors2, axis=1).
    cos_thetas = dots / (norms1 * norms2)  # EN: Assign cos_thetas from expression: dots / (norms1 * norms2).
    angles = np.degrees(np.arccos(np.clip(cos_thetas, -1, 1)))  # EN: Assign angles from expression: np.degrees(np.arccos(np.clip(cos_thetas, -1, 1))).

    for i in range(len(vectors1)):  # EN: Iterate with a for-loop: for i in range(len(vectors1)):.
        print(f"  {vectors1[i]} 和 {vectors2[i]} 的夾角：{angles[i]:.1f}°")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 正交性判斷
    # ========================================
    print_separator("4. 正交性判斷 (Orthogonality Check)")  # EN: Call print_separator(...) to perform an operation.

    u1 = np.array([1, 2])  # EN: Assign u1 from expression: np.array([1, 2]).
    u2 = np.array([-2, 1])  # EN: Assign u2 from expression: np.array([-2, 1]).

    print(f"u₁ = {u1}")  # EN: Print formatted output to the console.
    print(f"u₂ = {u2}")  # EN: Print formatted output to the console.
    print(f"u₁ · u₂ = {np.dot(u1, u2)}")  # EN: Print formatted output to the console.
    print(f"u₁ ⊥ u₂？ {np.isclose(np.dot(u1, u2), 0)}")  # EN: Print formatted output to the console.

    # 在更高維度
    print("\n3D 空間中的正交向量組：")  # EN: Print formatted output to the console.
    v1 = np.array([1, 1, 0])  # EN: Assign v1 from expression: np.array([1, 1, 0]).
    v2 = np.array([-1, 1, 0])  # EN: Assign v2 from expression: np.array([-1, 1, 0]).
    v3 = np.array([0, 0, 1])  # EN: Assign v3 from expression: np.array([0, 0, 1]).

    print(f"v₁ = {v1}")  # EN: Print formatted output to the console.
    print(f"v₂ = {v2}")  # EN: Print formatted output to the console.
    print(f"v₃ = {v3}")  # EN: Print formatted output to the console.

    # 建立矩陣並計算內積矩陣
    V = np.column_stack([v1, v2, v3])  # EN: Assign V from expression: np.column_stack([v1, v2, v3]).
    inner_products = V.T @ V  # EN: Assign inner_products from expression: V.T @ V.

    print(f"\n內積矩陣 VᵀV：\n{inner_products}")  # EN: Print formatted output to the console.
    print("（對角線外元素應為 0 表示兩兩正交）")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 正交矩陣
    # ========================================
    print_separator("5. 正交矩陣 (Orthogonal Matrix)")  # EN: Call print_separator(...) to perform an operation.

    # 旋轉矩陣
    theta = np.pi / 4  # 45度  # EN: Assign theta from expression: np.pi / 4 # 45度.
    Q = np.array([  # EN: Assign Q from expression: np.array([.
        [np.cos(theta), -np.sin(theta)],  # EN: Execute statement: [np.cos(theta), -np.sin(theta)],.
        [np.sin(theta), np.cos(theta)]  # EN: Execute statement: [np.sin(theta), np.cos(theta)].
    ])  # EN: Execute statement: ]).

    print(f"旋轉矩陣 Q（θ = 45°）：\n{Q}")  # EN: Print formatted output to the console.
    print(f"\nQᵀQ =\n{Q.T @ Q}")  # EN: Print formatted output to the console.
    print(f"\nQQᵀ =\n{Q @ Q.T}")  # EN: Print formatted output to the console.
    print(f"\nQ 是正交矩陣？ {np.allclose(Q.T @ Q, np.eye(2))}")  # EN: Print formatted output to the console.

    # 驗證正交矩陣的性質
    x = np.array([3, 4])  # EN: Assign x from expression: np.array([3, 4]).
    Qx = Q @ x  # EN: Assign Qx from expression: Q @ x.

    print(f"\n【正交矩陣保長度】")  # EN: Print formatted output to the console.
    print(f"x = {x}")  # EN: Print formatted output to the console.
    print(f"Qx = {Qx}")  # EN: Print formatted output to the console.
    print(f"‖x‖ = {np.linalg.norm(x):.4f}")  # EN: Print formatted output to the console.
    print(f"‖Qx‖ = {np.linalg.norm(Qx):.4f}")  # EN: Print formatted output to the console.

    # 保內積
    y = np.array([1, 2])  # EN: Assign y from expression: np.array([1, 2]).
    Qy = Q @ y  # EN: Assign Qy from expression: Q @ y.

    print(f"\n【正交矩陣保內積】")  # EN: Print formatted output to the console.
    print(f"x · y = {np.dot(x, y)}")  # EN: Print formatted output to the console.
    print(f"(Qx) · (Qy) = {np.dot(Qx, Qy):.4f}")  # EN: Print formatted output to the console.

    # 行列式
    print(f"\n【行列式】")  # EN: Print formatted output to the console.
    print(f"det(Q) = {np.linalg.det(Q):.4f}")  # EN: Print formatted output to the console.
    print("(+1 表示旋轉，-1 表示反射)")  # EN: Print formatted output to the console.

    # 反射矩陣
    R = np.array([[1, 0], [0, -1]])  # 關於 x 軸反射  # EN: Assign R from expression: np.array([[1, 0], [0, -1]]) # 關於 x 軸反射.
    print(f"\n反射矩陣 R：\n{R}")  # EN: Print formatted output to the console.
    print(f"det(R) = {np.linalg.det(R):.4f}")  # EN: Print formatted output to the console.
    print(f"R 是正交矩陣？ {np.allclose(R.T @ R, np.eye(2))}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. Cauchy-Schwarz 不等式
    # ========================================
    print_separator("6. Cauchy-Schwarz 不等式")  # EN: Call print_separator(...) to perform an operation.

    x = np.array([1, 2, 3])  # EN: Assign x from expression: np.array([1, 2, 3]).
    y = np.array([4, 5, 6])  # EN: Assign y from expression: np.array([4, 5, 6]).

    print(f"x = {x}")  # EN: Print formatted output to the console.
    print(f"y = {y}")  # EN: Print formatted output to the console.

    left_side = np.abs(np.dot(x, y))  # EN: Assign left_side from expression: np.abs(np.dot(x, y)).
    right_side = np.linalg.norm(x) * np.linalg.norm(y)  # EN: Assign right_side from expression: np.linalg.norm(x) * np.linalg.norm(y).

    print(f"\n|x · y| = {left_side:.4f}")  # EN: Print formatted output to the console.
    print(f"‖x‖ ‖y‖ = {right_side:.4f}")  # EN: Print formatted output to the console.
    print(f"|x · y| ≤ ‖x‖ ‖y‖？ {left_side <= right_side + 1e-10}")  # EN: Print formatted output to the console.

    # 等號成立的情況（平行向量）
    print("\n【等號成立：平行向量】")  # EN: Print formatted output to the console.
    p = np.array([1, 2, 3])  # EN: Assign p from expression: np.array([1, 2, 3]).
    q = np.array([2, 4, 6])  # q = 2p  # EN: Assign q from expression: np.array([2, 4, 6]) # q = 2p.

    print(f"p = {p}")  # EN: Print formatted output to the console.
    print(f"q = 2p = {q}")  # EN: Print formatted output to the console.

    left = np.abs(np.dot(p, q))  # EN: Assign left from expression: np.abs(np.dot(p, q)).
    right = np.linalg.norm(p) * np.linalg.norm(q)  # EN: Assign right from expression: np.linalg.norm(p) * np.linalg.norm(q).

    print(f"|p · q| = {left:.4f}")  # EN: Print formatted output to the console.
    print(f"‖p‖ ‖q‖ = {right:.4f}")  # EN: Print formatted output to the console.
    print(f"等號成立？ {np.isclose(left, right)}")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 三角不等式
    # ========================================
    print_separator("7. 三角不等式")  # EN: Call print_separator(...) to perform an operation.

    x = np.array([3, 0])  # EN: Assign x from expression: np.array([3, 0]).
    y = np.array([0, 4])  # EN: Assign y from expression: np.array([0, 4]).

    print(f"x = {x}")  # EN: Print formatted output to the console.
    print(f"y = {y}")  # EN: Print formatted output to the console.
    print(f"x + y = {x + y}")  # EN: Print formatted output to the console.

    left_side = np.linalg.norm(x + y)  # EN: Assign left_side from expression: np.linalg.norm(x + y).
    right_side = np.linalg.norm(x) + np.linalg.norm(y)  # EN: Assign right_side from expression: np.linalg.norm(x) + np.linalg.norm(y).

    print(f"\n‖x + y‖ = {left_side:.4f}")  # EN: Print formatted output to the console.
    print(f"‖x‖ + ‖y‖ = {right_side:.4f}")  # EN: Print formatted output to the console.
    print(f"‖x + y‖ ≤ ‖x‖ + ‖y‖？ {left_side <= right_side + 1e-10}")  # EN: Print formatted output to the console.

    # 等號成立（同方向）
    print("\n【等號成立：同方向向量】")  # EN: Print formatted output to the console.
    a = np.array([1, 0])  # EN: Assign a from expression: np.array([1, 0]).
    b = np.array([2, 0])  # EN: Assign b from expression: np.array([2, 0]).

    print(f"a = {a}")  # EN: Print formatted output to the console.
    print(f"b = {b}")  # EN: Print formatted output to the console.
    print(f"‖a + b‖ = {np.linalg.norm(a + b):.4f}")  # EN: Print formatted output to the console.
    print(f"‖a‖ + ‖b‖ = {np.linalg.norm(a) + np.linalg.norm(b):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 標準正交基底
    # ========================================
    print_separator("8. 標準正交基底 (Orthonormal Basis)")  # EN: Call print_separator(...) to perform an operation.

    # 標準基底
    e1 = np.array([1, 0, 0])  # EN: Assign e1 from expression: np.array([1, 0, 0]).
    e2 = np.array([0, 1, 0])  # EN: Assign e2 from expression: np.array([0, 1, 0]).
    e3 = np.array([0, 0, 1])  # EN: Assign e3 from expression: np.array([0, 0, 1]).

    E = np.column_stack([e1, e2, e3])  # EN: Assign E from expression: np.column_stack([e1, e2, e3]).
    print(f"標準基底矩陣 E：\n{E}")  # EN: Print formatted output to the console.
    print(f"\nEᵀE =\n{E.T @ E}")  # EN: Print formatted output to the console.

    # 用標準正交基底表示向量
    v = np.array([3, 4, 5])  # EN: Assign v from expression: np.array([3, 4, 5]).
    print(f"\n向量 v = {v}")  # EN: Print formatted output to the console.
    print(f"v 在標準基底下的座標：")  # EN: Print formatted output to the console.
    for i, ei in enumerate([e1, e2, e3], 1):  # EN: Iterate with a for-loop: for i, ei in enumerate([e1, e2, e3], 1):.
        print(f"  v · e{i} = {np.dot(v, ei)}")  # EN: Print formatted output to the console.

    print("\n標準正交基底的優點：座標 = 內積！")  # EN: Print formatted output to the console.

    # ========================================
    # 9. 應用：向量分解
    # ========================================
    print_separator("9. 應用：向量在正交方向的分解")  # EN: Call print_separator(...) to perform an operation.

    # 把向量分解為平行和垂直分量
    v = np.array([3, 4])  # EN: Assign v from expression: np.array([3, 4]).
    direction = np.array([1, 0])  # x 軸方向  # EN: Assign direction from expression: np.array([1, 0]) # x 軸方向.

    print(f"向量 v = {v}")  # EN: Print formatted output to the console.
    print(f"方向 d = {direction}")  # EN: Print formatted output to the console.

    # 平行分量（投影）
    v_parallel = np.dot(v, direction) * direction  # EN: Assign v_parallel from expression: np.dot(v, direction) * direction.
    # 垂直分量
    v_perp = v - v_parallel  # EN: Assign v_perp from expression: v - v_parallel.

    print(f"\nv 在 d 方向的分量：{v_parallel}")  # EN: Print formatted output to the console.
    print(f"v 垂直於 d 的分量：{v_perp}")  # EN: Print formatted output to the console.
    print(f"\n驗證 v = v_parallel + v_perp：{np.allclose(v, v_parallel + v_perp)}")  # EN: Print formatted output to the console.
    print(f"驗證 v_parallel ⊥ v_perp：{np.isclose(np.dot(v_parallel, v_perp), 0)}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("NumPy 常用函數總結")  # EN: Call print_separator(...) to perform an operation.
    print("""
內積與範數：
  np.dot(x, y)          # 內積
  x @ y                 # 內積（Python 3.5+）
  np.linalg.norm(x)     # L2 範數
  np.linalg.norm(x, ord=1)  # L1 範數

正交矩陣：
  Q.T @ Q               # 應為單位矩陣
  np.allclose(Q.T @ Q, np.eye(n))  # 檢驗

角度計算：
  np.arccos(np.clip(cos_theta, -1, 1))  # 避免數值誤差

批次運算：
  np.einsum('ij,ij->i', A, B)  # 每列內積
  np.linalg.norm(A, axis=1)    # 每列範數
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
