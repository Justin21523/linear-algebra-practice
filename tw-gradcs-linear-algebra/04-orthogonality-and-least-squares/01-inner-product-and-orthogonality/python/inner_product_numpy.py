"""
內積與正交性 - NumPy 版本 (Inner Product and Orthogonality - NumPy Implementation)

本程式示範：
1. 使用 NumPy 計算內積、範數、夾角
2. 正交性與正交矩陣
3. Cauchy-Schwarz 不等式
4. 正交投影的基礎

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


def main():
    """主程式"""

    print_separator("內積與正交性示範（NumPy 版）\nInner Product & Orthogonality Demo (NumPy)")

    # ========================================
    # 1. 內積計算
    # ========================================
    print_separator("1. 內積計算 (Dot Product)")

    x = np.array([1, 2, 3], dtype=float)
    y = np.array([4, 5, 6], dtype=float)

    print(f"x = {x}")
    print(f"y = {y}")

    # 多種計算內積的方式
    dot1 = np.dot(x, y)
    dot2 = x @ y
    dot3 = np.inner(x, y)
    dot4 = np.sum(x * y)

    print(f"\n內積計算方式：")
    print(f"np.dot(x, y)   = {dot1}")
    print(f"x @ y          = {dot2}")
    print(f"np.inner(x, y) = {dot3}")
    print(f"np.sum(x * y)  = {dot4}")

    # ========================================
    # 2. 向量長度（範數）
    # ========================================
    print_separator("2. 向量長度 (Vector Norm)")

    v = np.array([3, 4], dtype=float)
    print(f"v = {v}")

    # L2 範數（預設）
    norm_v = np.linalg.norm(v)
    print(f"\n‖v‖₂ = {norm_v}")

    # 其他範數
    print(f"‖v‖₁ (L1) = {np.linalg.norm(v, ord=1)}")
    print(f"‖v‖∞ (L∞) = {np.linalg.norm(v, ord=np.inf)}")

    # 正規化
    v_normalized = v / norm_v
    print(f"\n單位向量 v̂ = {v_normalized}")
    print(f"‖v̂‖ = {np.linalg.norm(v_normalized)}")

    # ========================================
    # 3. 向量夾角
    # ========================================
    print_separator("3. 向量夾角 (Vector Angle)")

    a = np.array([1, 0])
    b = np.array([1, 1])

    print(f"a = {a}")
    print(f"b = {b}")

    # cos θ = (a · b) / (‖a‖ ‖b‖)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    theta = np.arccos(np.clip(cos_theta, -1, 1))

    print(f"\ncos θ = {cos_theta:.4f}")
    print(f"θ = {theta:.4f} rad = {np.degrees(theta):.2f}°")

    # 向量化計算多組夾角
    print("\n批次計算多組夾角：")
    vectors1 = np.array([[1, 0], [1, 0], [1, 1]])
    vectors2 = np.array([[0, 1], [1, 1], [-1, 1]])

    # 使用 einsum 計算內積
    dots = np.einsum('ij,ij->i', vectors1, vectors2)
    norms1 = np.linalg.norm(vectors1, axis=1)
    norms2 = np.linalg.norm(vectors2, axis=1)
    cos_thetas = dots / (norms1 * norms2)
    angles = np.degrees(np.arccos(np.clip(cos_thetas, -1, 1)))

    for i in range(len(vectors1)):
        print(f"  {vectors1[i]} 和 {vectors2[i]} 的夾角：{angles[i]:.1f}°")

    # ========================================
    # 4. 正交性判斷
    # ========================================
    print_separator("4. 正交性判斷 (Orthogonality Check)")

    u1 = np.array([1, 2])
    u2 = np.array([-2, 1])

    print(f"u₁ = {u1}")
    print(f"u₂ = {u2}")
    print(f"u₁ · u₂ = {np.dot(u1, u2)}")
    print(f"u₁ ⊥ u₂？ {np.isclose(np.dot(u1, u2), 0)}")

    # 在更高維度
    print("\n3D 空間中的正交向量組：")
    v1 = np.array([1, 1, 0])
    v2 = np.array([-1, 1, 0])
    v3 = np.array([0, 0, 1])

    print(f"v₁ = {v1}")
    print(f"v₂ = {v2}")
    print(f"v₃ = {v3}")

    # 建立矩陣並計算內積矩陣
    V = np.column_stack([v1, v2, v3])
    inner_products = V.T @ V

    print(f"\n內積矩陣 VᵀV：\n{inner_products}")
    print("（對角線外元素應為 0 表示兩兩正交）")

    # ========================================
    # 5. 正交矩陣
    # ========================================
    print_separator("5. 正交矩陣 (Orthogonal Matrix)")

    # 旋轉矩陣
    theta = np.pi / 4  # 45度
    Q = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    print(f"旋轉矩陣 Q（θ = 45°）：\n{Q}")
    print(f"\nQᵀQ =\n{Q.T @ Q}")
    print(f"\nQQᵀ =\n{Q @ Q.T}")
    print(f"\nQ 是正交矩陣？ {np.allclose(Q.T @ Q, np.eye(2))}")

    # 驗證正交矩陣的性質
    x = np.array([3, 4])
    Qx = Q @ x

    print(f"\n【正交矩陣保長度】")
    print(f"x = {x}")
    print(f"Qx = {Qx}")
    print(f"‖x‖ = {np.linalg.norm(x):.4f}")
    print(f"‖Qx‖ = {np.linalg.norm(Qx):.4f}")

    # 保內積
    y = np.array([1, 2])
    Qy = Q @ y

    print(f"\n【正交矩陣保內積】")
    print(f"x · y = {np.dot(x, y)}")
    print(f"(Qx) · (Qy) = {np.dot(Qx, Qy):.4f}")

    # 行列式
    print(f"\n【行列式】")
    print(f"det(Q) = {np.linalg.det(Q):.4f}")
    print("(+1 表示旋轉，-1 表示反射)")

    # 反射矩陣
    R = np.array([[1, 0], [0, -1]])  # 關於 x 軸反射
    print(f"\n反射矩陣 R：\n{R}")
    print(f"det(R) = {np.linalg.det(R):.4f}")
    print(f"R 是正交矩陣？ {np.allclose(R.T @ R, np.eye(2))}")

    # ========================================
    # 6. Cauchy-Schwarz 不等式
    # ========================================
    print_separator("6. Cauchy-Schwarz 不等式")

    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    print(f"x = {x}")
    print(f"y = {y}")

    left_side = np.abs(np.dot(x, y))
    right_side = np.linalg.norm(x) * np.linalg.norm(y)

    print(f"\n|x · y| = {left_side:.4f}")
    print(f"‖x‖ ‖y‖ = {right_side:.4f}")
    print(f"|x · y| ≤ ‖x‖ ‖y‖？ {left_side <= right_side + 1e-10}")

    # 等號成立的情況（平行向量）
    print("\n【等號成立：平行向量】")
    p = np.array([1, 2, 3])
    q = np.array([2, 4, 6])  # q = 2p

    print(f"p = {p}")
    print(f"q = 2p = {q}")

    left = np.abs(np.dot(p, q))
    right = np.linalg.norm(p) * np.linalg.norm(q)

    print(f"|p · q| = {left:.4f}")
    print(f"‖p‖ ‖q‖ = {right:.4f}")
    print(f"等號成立？ {np.isclose(left, right)}")

    # ========================================
    # 7. 三角不等式
    # ========================================
    print_separator("7. 三角不等式")

    x = np.array([3, 0])
    y = np.array([0, 4])

    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {x + y}")

    left_side = np.linalg.norm(x + y)
    right_side = np.linalg.norm(x) + np.linalg.norm(y)

    print(f"\n‖x + y‖ = {left_side:.4f}")
    print(f"‖x‖ + ‖y‖ = {right_side:.4f}")
    print(f"‖x + y‖ ≤ ‖x‖ + ‖y‖？ {left_side <= right_side + 1e-10}")

    # 等號成立（同方向）
    print("\n【等號成立：同方向向量】")
    a = np.array([1, 0])
    b = np.array([2, 0])

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"‖a + b‖ = {np.linalg.norm(a + b):.4f}")
    print(f"‖a‖ + ‖b‖ = {np.linalg.norm(a) + np.linalg.norm(b):.4f}")

    # ========================================
    # 8. 標準正交基底
    # ========================================
    print_separator("8. 標準正交基底 (Orthonormal Basis)")

    # 標準基底
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    E = np.column_stack([e1, e2, e3])
    print(f"標準基底矩陣 E：\n{E}")
    print(f"\nEᵀE =\n{E.T @ E}")

    # 用標準正交基底表示向量
    v = np.array([3, 4, 5])
    print(f"\n向量 v = {v}")
    print(f"v 在標準基底下的座標：")
    for i, ei in enumerate([e1, e2, e3], 1):
        print(f"  v · e{i} = {np.dot(v, ei)}")

    print("\n標準正交基底的優點：座標 = 內積！")

    # ========================================
    # 9. 應用：向量分解
    # ========================================
    print_separator("9. 應用：向量在正交方向的分解")

    # 把向量分解為平行和垂直分量
    v = np.array([3, 4])
    direction = np.array([1, 0])  # x 軸方向

    print(f"向量 v = {v}")
    print(f"方向 d = {direction}")

    # 平行分量（投影）
    v_parallel = np.dot(v, direction) * direction
    # 垂直分量
    v_perp = v - v_parallel

    print(f"\nv 在 d 方向的分量：{v_parallel}")
    print(f"v 垂直於 d 的分量：{v_perp}")
    print(f"\n驗證 v = v_parallel + v_perp：{np.allclose(v, v_parallel + v_perp)}")
    print(f"驗證 v_parallel ⊥ v_perp：{np.isclose(np.dot(v_parallel, v_perp), 0)}")

    # 總結
    print_separator("NumPy 常用函數總結")
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
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
