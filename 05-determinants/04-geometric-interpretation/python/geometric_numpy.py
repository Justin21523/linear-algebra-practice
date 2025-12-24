"""
行列式的幾何解釋 - NumPy 版本 (Geometric Interpretation - NumPy Implementation)

本程式示範：
1. NumPy 計算幾何量
2. 線性變換視覺化
3. 雅可比行列式
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def parallelogram_area_2d(a: np.ndarray, b: np.ndarray) -> float:  # EN: Define parallelogram_area_2d and its behavior.
    """2D 平行四邊形面積"""  # EN: Execute statement: """2D 平行四邊形面積""".
    return abs(np.cross(a, b))  # EN: Return a value: return abs(np.cross(a, b)).


def parallelepiped_volume_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:  # EN: Define parallelepiped_volume_3d and its behavior.
    """3D 平行六面體體積"""  # EN: Execute statement: """3D 平行六面體體積""".
    # det([a | b | c]) = a · (b × c)
    return abs(np.dot(a, np.cross(b, c)))  # EN: Return a value: return abs(np.dot(a, np.cross(b, c))).


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:  # EN: Define triangle_area and its behavior.
    """三角形面積"""  # EN: Execute statement: """三角形面積""".
    # 使用向量方法
    v1 = p2 - p1  # EN: Assign v1 from expression: p2 - p1.
    v2 = p3 - p1  # EN: Assign v2 from expression: p3 - p1.
    return abs(np.cross(v1, v2)) / 2  # EN: Return a value: return abs(np.cross(v1, v2)) / 2.


def main():  # EN: Define main and its behavior.
    print_separator("行列式幾何解釋示範（NumPy 版）\nGeometric Interpretation Demo (NumPy)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 平行四邊形面積
    # ========================================
    print_separator("1. 平行四邊形面積")  # EN: Call print_separator(...) to perform an operation.

    a = np.array([3, 0])  # EN: Assign a from expression: np.array([3, 0]).
    b = np.array([1, 2])  # EN: Assign b from expression: np.array([1, 2]).

    print(f"a = {a}")  # EN: Print formatted output to the console.
    print(f"b = {b}")  # EN: Print formatted output to the console.

    # 方法 1：使用叉積
    area_cross = abs(np.cross(a, b))  # EN: Assign area_cross from expression: abs(np.cross(a, b)).
    print(f"\n方法 1（叉積）：|a × b| = {area_cross}")  # EN: Print formatted output to the console.

    # 方法 2：使用行列式
    A = np.column_stack([a, b])  # EN: Assign A from expression: np.column_stack([a, b]).
    area_det = abs(np.linalg.det(A))  # EN: Assign area_det from expression: abs(np.linalg.det(A)).
    print(f"方法 2（行列式）：|det([a|b])| = {area_det}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 平行六面體體積
    # ========================================
    print_separator("2. 平行六面體體積")  # EN: Call print_separator(...) to perform an operation.

    a = np.array([1, 0, 0])  # EN: Assign a from expression: np.array([1, 0, 0]).
    b = np.array([0, 2, 0])  # EN: Assign b from expression: np.array([0, 2, 0]).
    c = np.array([0, 0, 3])  # EN: Assign c from expression: np.array([0, 0, 3]).

    print(f"a = {a}")  # EN: Print formatted output to the console.
    print(f"b = {b}")  # EN: Print formatted output to the console.
    print(f"c = {c}")  # EN: Print formatted output to the console.

    # 方法 1：純量三重積
    vol_triple = abs(np.dot(a, np.cross(b, c)))  # EN: Assign vol_triple from expression: abs(np.dot(a, np.cross(b, c))).
    print(f"\n方法 1（三重積）：|a · (b × c)| = {vol_triple}")  # EN: Print formatted output to the console.

    # 方法 2：行列式
    A = np.column_stack([a, b, c])  # EN: Assign A from expression: np.column_stack([a, b, c]).
    vol_det = abs(np.linalg.det(A))  # EN: Assign vol_det from expression: abs(np.linalg.det(A)).
    print(f"方法 2（行列式）：|det([a|b|c])| = {vol_det}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 線性變換的體積縮放
    # ========================================
    print_separator("3. 線性變換的體積縮放")  # EN: Call print_separator(...) to perform an operation.

    # 單位正方形的頂點
    unit_square = np.array([  # EN: Assign unit_square from expression: np.array([.
        [0, 0],  # EN: Execute statement: [0, 0],.
        [1, 0],  # EN: Execute statement: [1, 0],.
        [1, 1],  # EN: Execute statement: [1, 1],.
        [0, 1]  # EN: Execute statement: [0, 1].
    ])  # EN: Execute statement: ]).

    print("單位正方形頂點：")  # EN: Print formatted output to the console.
    print(unit_square)  # EN: Print formatted output to the console.

    # 縮放矩陣
    A = np.array([[2, 0], [0, 3]])  # EN: Assign A from expression: np.array([[2, 0], [0, 3]]).
    print(f"\n縮放矩陣 A:\n{A}")  # EN: Print formatted output to the console.
    print(f"det(A) = {np.linalg.det(A):.4f}")  # EN: Print formatted output to the console.

    transformed = (A @ unit_square.T).T  # EN: Assign transformed from expression: (A @ unit_square.T).T.
    print(f"\n變換後的頂點：\n{transformed}")  # EN: Print formatted output to the console.
    print("面積從 1 變成 6")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 各種變換的行列式
    # ========================================
    print_separator("4. 各種變換的行列式")  # EN: Call print_separator(...) to perform an operation.

    # 旋轉
    theta = np.pi / 4  # EN: Assign theta from expression: np.pi / 4.
    R = np.array([  # EN: Assign R from expression: np.array([.
        [np.cos(theta), -np.sin(theta)],  # EN: Execute statement: [np.cos(theta), -np.sin(theta)],.
        [np.sin(theta), np.cos(theta)]  # EN: Execute statement: [np.sin(theta), np.cos(theta)].
    ])  # EN: Execute statement: ]).
    print(f"旋轉 45°：det(R) = {np.linalg.det(R):.4f}（面積不變）")  # EN: Print formatted output to the console.

    # 剪切
    S = np.array([[1, 2], [0, 1]])  # EN: Assign S from expression: np.array([[1, 2], [0, 1]]).
    print(f"剪切：det(S) = {np.linalg.det(S):.4f}（面積不變）")  # EN: Print formatted output to the console.

    # 反射
    H = np.array([[1, 0], [0, -1]])  # EN: Assign H from expression: np.array([[1, 0], [0, -1]]).
    print(f"反射：det(H) = {np.linalg.det(H):.4f}（面積不變，定向反轉）")  # EN: Print formatted output to the console.

    # 投影
    P = np.array([[1, 0], [0, 0]])  # EN: Assign P from expression: np.array([[1, 0], [0, 0]]).
    print(f"投影到 x 軸：det(P) = {np.linalg.det(P):.4f}（降維，面積 → 0）")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 三角形面積
    # ========================================
    print_separator("5. 三角形面積")  # EN: Call print_separator(...) to perform an operation.

    p1 = np.array([0, 0])  # EN: Assign p1 from expression: np.array([0, 0]).
    p2 = np.array([4, 0])  # EN: Assign p2 from expression: np.array([4, 0]).
    p3 = np.array([0, 3])  # EN: Assign p3 from expression: np.array([0, 3]).

    print(f"頂點：P1={p1}, P2={p2}, P3={p3}")  # EN: Print formatted output to the console.

    # 方法 1：向量叉積
    v1, v2 = p2 - p1, p3 - p1  # EN: Execute statement: v1, v2 = p2 - p1, p3 - p1.
    area = abs(np.cross(v1, v2)) / 2  # EN: Assign area from expression: abs(np.cross(v1, v2)) / 2.
    print(f"\n面積 = |v1 × v2| / 2 = {area}")  # EN: Print formatted output to the console.

    # 方法 2：行列式公式
    M = np.array([  # EN: Assign M from expression: np.array([.
        [p1[0], p1[1], 1],  # EN: Execute statement: [p1[0], p1[1], 1],.
        [p2[0], p2[1], 1],  # EN: Execute statement: [p2[0], p2[1], 1],.
        [p3[0], p3[1], 1]  # EN: Execute statement: [p3[0], p3[1], 1].
    ])  # EN: Execute statement: ]).
    area_det = abs(np.linalg.det(M)) / 2  # EN: Assign area_det from expression: abs(np.linalg.det(M)) / 2.
    print(f"面積 = |det(M)| / 2 = {area_det}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 雅可比行列式
    # ========================================
    print_separator("6. 雅可比行列式（Jacobian）")  # EN: Call print_separator(...) to perform an operation.

    print("極座標變換：x = r cos θ, y = r sin θ")  # EN: Print formatted output to the console.

    def jacobian_polar(r: float, theta: float) -> float:  # EN: Define jacobian_polar and its behavior.
        """極座標的雅可比行列式"""  # EN: Execute statement: """極座標的雅可比行列式""".
        J = np.array([  # EN: Assign J from expression: np.array([.
            [np.cos(theta), -r * np.sin(theta)],  # EN: Execute statement: [np.cos(theta), -r * np.sin(theta)],.
            [np.sin(theta), r * np.cos(theta)]  # EN: Execute statement: [np.sin(theta), r * np.cos(theta)].
        ])  # EN: Execute statement: ]).
        return np.linalg.det(J)  # EN: Return a value: return np.linalg.det(J).

    print("\n不同 r 值的 Jacobian：")  # EN: Print formatted output to the console.
    for r in [0.5, 1.0, 2.0, 3.0]:  # EN: Iterate with a for-loop: for r in [0.5, 1.0, 2.0, 3.0]:.
        J = jacobian_polar(r, np.pi/4)  # EN: Assign J from expression: jacobian_polar(r, np.pi/4).
        print(f"  r = {r}: J = {J:.4f}")  # EN: Print formatted output to the console.

    print("\n因此 dx dy = |J| dr dθ = r dr dθ")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 驗證變換前後的面積
    # ========================================
    print_separator("7. 驗證：變換前後面積比 = |det(A)|")  # EN: Call print_separator(...) to perform an operation.

    # 原始三角形
    p1 = np.array([0, 0])  # EN: Assign p1 from expression: np.array([0, 0]).
    p2 = np.array([1, 0])  # EN: Assign p2 from expression: np.array([1, 0]).
    p3 = np.array([0, 1])  # EN: Assign p3 from expression: np.array([0, 1]).

    original_area = triangle_area(p1, p2, p3)  # EN: Assign original_area from expression: triangle_area(p1, p2, p3).
    print(f"原始三角形面積：{original_area}")  # EN: Print formatted output to the console.

    # 變換矩陣
    A = np.array([[3, 1], [2, 2]])  # EN: Assign A from expression: np.array([[3, 1], [2, 2]]).
    print(f"\n變換矩陣 A:\n{A}")  # EN: Print formatted output to the console.
    print(f"det(A) = {np.linalg.det(A):.4f}")  # EN: Print formatted output to the console.

    # 變換後的三角形
    p1_new = A @ p1  # EN: Assign p1_new from expression: A @ p1.
    p2_new = A @ p2  # EN: Assign p2_new from expression: A @ p2.
    p3_new = A @ p3  # EN: Assign p3_new from expression: A @ p3.

    new_area = triangle_area(p1_new, p2_new, p3_new)  # EN: Assign new_area from expression: triangle_area(p1_new, p2_new, p3_new).
    print(f"\n變換後三角形面積：{new_area}")  # EN: Print formatted output to the console.
    print(f"面積比 = {new_area / original_area:.4f}")  # EN: Print formatted output to the console.
    print(f"|det(A)| = {abs(np.linalg.det(A)):.4f}")  # EN: Print formatted output to the console.
    print("驗證：面積比 = |det(A)| ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 3D 體積變換
    # ========================================
    print_separator("8. 3D 體積變換")  # EN: Call print_separator(...) to perform an operation.

    # 單位立方體
    print("單位立方體 → 變換後的平行六面體")  # EN: Print formatted output to the console.

    A_3d = np.array([  # EN: Assign A_3d from expression: np.array([.
        [2, 0, 0],  # EN: Execute statement: [2, 0, 0],.
        [0, 3, 0],  # EN: Execute statement: [0, 3, 0],.
        [0, 0, 4]  # EN: Execute statement: [0, 0, 4].
    ])  # EN: Execute statement: ]).

    print(f"\n變換矩陣 A:\n{A_3d}")  # EN: Print formatted output to the console.
    print(f"det(A) = {np.linalg.det(A_3d):.4f}")  # EN: Print formatted output to the console.
    print(f"\n單位立方體體積：1")  # EN: Print formatted output to the console.
    print(f"變換後體積：{abs(np.linalg.det(A_3d)):.4f}（= 2×3×4 = 24）")  # EN: Print formatted output to the console.

    # 總結
    print_separator("NumPy 幾何計算總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
