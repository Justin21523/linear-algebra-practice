"""
行列式的幾何解釋 - NumPy 版本 (Geometric Interpretation - NumPy Implementation)

本程式示範：
1. NumPy 計算幾何量
2. 線性變換視覺化
3. 雅可比行列式
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def parallelogram_area_2d(a: np.ndarray, b: np.ndarray) -> float:
    """2D 平行四邊形面積"""
    return abs(np.cross(a, b))


def parallelepiped_volume_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """3D 平行六面體體積"""
    # det([a | b | c]) = a · (b × c)
    return abs(np.dot(a, np.cross(b, c)))


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """三角形面積"""
    # 使用向量方法
    v1 = p2 - p1
    v2 = p3 - p1
    return abs(np.cross(v1, v2)) / 2


def main():
    print_separator("行列式幾何解釋示範（NumPy 版）\nGeometric Interpretation Demo (NumPy)")

    # ========================================
    # 1. 平行四邊形面積
    # ========================================
    print_separator("1. 平行四邊形面積")

    a = np.array([3, 0])
    b = np.array([1, 2])

    print(f"a = {a}")
    print(f"b = {b}")

    # 方法 1：使用叉積
    area_cross = abs(np.cross(a, b))
    print(f"\n方法 1（叉積）：|a × b| = {area_cross}")

    # 方法 2：使用行列式
    A = np.column_stack([a, b])
    area_det = abs(np.linalg.det(A))
    print(f"方法 2（行列式）：|det([a|b])| = {area_det}")

    # ========================================
    # 2. 平行六面體體積
    # ========================================
    print_separator("2. 平行六面體體積")

    a = np.array([1, 0, 0])
    b = np.array([0, 2, 0])
    c = np.array([0, 0, 3])

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")

    # 方法 1：純量三重積
    vol_triple = abs(np.dot(a, np.cross(b, c)))
    print(f"\n方法 1（三重積）：|a · (b × c)| = {vol_triple}")

    # 方法 2：行列式
    A = np.column_stack([a, b, c])
    vol_det = abs(np.linalg.det(A))
    print(f"方法 2（行列式）：|det([a|b|c])| = {vol_det}")

    # ========================================
    # 3. 線性變換的體積縮放
    # ========================================
    print_separator("3. 線性變換的體積縮放")

    # 單位正方形的頂點
    unit_square = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])

    print("單位正方形頂點：")
    print(unit_square)

    # 縮放矩陣
    A = np.array([[2, 0], [0, 3]])
    print(f"\n縮放矩陣 A:\n{A}")
    print(f"det(A) = {np.linalg.det(A):.4f}")

    transformed = (A @ unit_square.T).T
    print(f"\n變換後的頂點：\n{transformed}")
    print("面積從 1 變成 6")

    # ========================================
    # 4. 各種變換的行列式
    # ========================================
    print_separator("4. 各種變換的行列式")

    # 旋轉
    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    print(f"旋轉 45°：det(R) = {np.linalg.det(R):.4f}（面積不變）")

    # 剪切
    S = np.array([[1, 2], [0, 1]])
    print(f"剪切：det(S) = {np.linalg.det(S):.4f}（面積不變）")

    # 反射
    H = np.array([[1, 0], [0, -1]])
    print(f"反射：det(H) = {np.linalg.det(H):.4f}（面積不變，定向反轉）")

    # 投影
    P = np.array([[1, 0], [0, 0]])
    print(f"投影到 x 軸：det(P) = {np.linalg.det(P):.4f}（降維，面積 → 0）")

    # ========================================
    # 5. 三角形面積
    # ========================================
    print_separator("5. 三角形面積")

    p1 = np.array([0, 0])
    p2 = np.array([4, 0])
    p3 = np.array([0, 3])

    print(f"頂點：P1={p1}, P2={p2}, P3={p3}")

    # 方法 1：向量叉積
    v1, v2 = p2 - p1, p3 - p1
    area = abs(np.cross(v1, v2)) / 2
    print(f"\n面積 = |v1 × v2| / 2 = {area}")

    # 方法 2：行列式公式
    M = np.array([
        [p1[0], p1[1], 1],
        [p2[0], p2[1], 1],
        [p3[0], p3[1], 1]
    ])
    area_det = abs(np.linalg.det(M)) / 2
    print(f"面積 = |det(M)| / 2 = {area_det}")

    # ========================================
    # 6. 雅可比行列式
    # ========================================
    print_separator("6. 雅可比行列式（Jacobian）")

    print("極座標變換：x = r cos θ, y = r sin θ")

    def jacobian_polar(r: float, theta: float) -> float:
        """極座標的雅可比行列式"""
        J = np.array([
            [np.cos(theta), -r * np.sin(theta)],
            [np.sin(theta), r * np.cos(theta)]
        ])
        return np.linalg.det(J)

    print("\n不同 r 值的 Jacobian：")
    for r in [0.5, 1.0, 2.0, 3.0]:
        J = jacobian_polar(r, np.pi/4)
        print(f"  r = {r}: J = {J:.4f}")

    print("\n因此 dx dy = |J| dr dθ = r dr dθ")

    # ========================================
    # 7. 驗證變換前後的面積
    # ========================================
    print_separator("7. 驗證：變換前後面積比 = |det(A)|")

    # 原始三角形
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([0, 1])

    original_area = triangle_area(p1, p2, p3)
    print(f"原始三角形面積：{original_area}")

    # 變換矩陣
    A = np.array([[3, 1], [2, 2]])
    print(f"\n變換矩陣 A:\n{A}")
    print(f"det(A) = {np.linalg.det(A):.4f}")

    # 變換後的三角形
    p1_new = A @ p1
    p2_new = A @ p2
    p3_new = A @ p3

    new_area = triangle_area(p1_new, p2_new, p3_new)
    print(f"\n變換後三角形面積：{new_area}")
    print(f"面積比 = {new_area / original_area:.4f}")
    print(f"|det(A)| = {abs(np.linalg.det(A)):.4f}")
    print("驗證：面積比 = |det(A)| ✓")

    # ========================================
    # 8. 3D 體積變換
    # ========================================
    print_separator("8. 3D 體積變換")

    # 單位立方體
    print("單位立方體 → 變換後的平行六面體")

    A_3d = np.array([
        [2, 0, 0],
        [0, 3, 0],
        [0, 0, 4]
    ])

    print(f"\n變換矩陣 A:\n{A_3d}")
    print(f"det(A) = {np.linalg.det(A_3d):.4f}")
    print(f"\n單位立方體體積：1")
    print(f"變換後體積：{abs(np.linalg.det(A_3d)):.4f}（= 2×3×4 = 24）")

    # 總結
    print_separator("NumPy 幾何計算總結")
    print("""
面積/體積計算：
  2D 叉積：np.cross(a, b) → 純量
  3D 叉積：np.cross(b, c) → 向量
  三重積：np.dot(a, np.cross(b, c))
  行列式：np.linalg.det(A)

重要關係：
  平行四邊形面積 = |np.cross(a, b)|
  平行六面體體積 = |np.dot(a, np.cross(b, c))|
  三角形面積 = |np.cross(v1, v2)| / 2

變換性質：
  變換後面積/體積 = |det(A)| × 原面積/體積
  det > 0：保持定向
  det < 0：反轉定向
  det = 0：降維
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
