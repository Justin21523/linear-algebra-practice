"""
行列式的幾何解釋 - 手刻版本 (Geometric Interpretation - Manual Implementation)

本程式示範：
1. 平行四邊形面積
2. 平行六面體體積
3. 三角形面積
4. 線性變換的體積縮放
"""

from typing import List, Tuple
import math


def print_separator(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_vector(name: str, v: List[float]) -> None:
    formatted = [f"{x:.4f}" for x in v]
    print(f"{name} = [{', '.join(formatted)}]")


def print_matrix(name: str, M: List[List[float]]) -> None:
    print(f"{name} =")
    for row in M:
        formatted = [f"{x:8.4f}" for x in row]
        print(f"  [{', '.join(formatted)}]")


# ========================================
# 行列式計算
# ========================================

def det_2x2(A: List[List[float]]) -> float:
    """2×2 行列式"""
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def det_3x3(A: List[List[float]]) -> float:
    """3×3 行列式"""
    a, b, c = A[0]
    d, e, f = A[1]
    g, h, i = A[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


# ========================================
# 幾何計算
# ========================================

def parallelogram_area(a: List[float], b: List[float]) -> float:
    """計算兩向量形成的平行四邊形面積"""
    # A = [a | b] 的行列式
    return abs(a[0] * b[1] - a[1] * b[0])


def parallelogram_signed_area(a: List[float], b: List[float]) -> float:
    """有號面積（用於判斷定向）"""
    return a[0] * b[1] - a[1] * b[0]


def parallelepiped_volume(a: List[float], b: List[float], c: List[float]) -> float:
    """計算三向量形成的平行六面體體積"""
    # det([a | b | c])
    matrix = [
        [a[0], b[0], c[0]],
        [a[1], b[1], c[1]],
        [a[2], b[2], c[2]]
    ]
    return abs(det_3x3(matrix))


def triangle_area(p1: Tuple[float, float],
                  p2: Tuple[float, float],
                  p3: Tuple[float, float]) -> float:
    """計算三角形面積（三頂點）"""
    # 使用 (1/2)|det([P2-P1 | P3-P1])|
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # 或使用擴展的 3×3 行列式公式
    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    return abs(det) / 2


def cross_product_2d(a: List[float], b: List[float]) -> float:
    """2D 叉積（z 分量）"""
    return a[0] * b[1] - a[1] * b[0]


def cross_product_3d(a: List[float], b: List[float]) -> List[float]:
    """3D 叉積"""
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]


def dot_product(a: List[float], b: List[float]) -> float:
    """內積"""
    return sum(ai * bi for ai, bi in zip(a, b))


def scalar_triple_product(a: List[float], b: List[float], c: List[float]) -> float:
    """純量三重積 a · (b × c)"""
    b_cross_c = cross_product_3d(b, c)
    return dot_product(a, b_cross_c)


def main():
    print_separator("行列式幾何解釋示範（手刻版）\nGeometric Interpretation Demo (Manual)")

    # ========================================
    # 1. 平行四邊形面積
    # ========================================
    print_separator("1. 平行四邊形面積")

    a = [3, 0]
    b = [1, 2]

    print_vector("a", a)
    print_vector("b", b)

    area = parallelogram_area(a, b)
    signed_area = parallelogram_signed_area(a, b)

    print(f"\n平行四邊形：")
    print(f"  det([a | b]) = {signed_area:.4f}")
    print(f"  面積 = |det| = {area:.4f}")

    # 另一組例子
    print("\n另一組向量：")
    a2 = [2, 1]
    b2 = [1, 3]
    print_vector("a", a2)
    print_vector("b", b2)
    print(f"  面積 = {parallelogram_area(a2, b2):.4f}")

    # ========================================
    # 2. 定向判斷
    # ========================================
    print_separator("2. 定向判斷")

    a = [1, 0]
    b = [0, 1]
    signed = parallelogram_signed_area(a, b)
    print_vector("a", a)
    print_vector("b", b)
    print(f"有號面積 = {signed:.4f}")
    print(f"定向：{'逆時針（正向）' if signed > 0 else '順時針（負向）'}")

    # 交換順序
    print("\n交換 a, b 順序：")
    signed = parallelogram_signed_area(b, a)
    print(f"有號面積 = {signed:.4f}")
    print(f"定向：{'逆時針（正向）' if signed > 0 else '順時針（負向）'}")

    # ========================================
    # 3. 平行六面體體積
    # ========================================
    print_separator("3. 平行六面體體積")

    a = [1, 0, 0]
    b = [0, 2, 0]
    c = [0, 0, 3]

    print_vector("a", a)
    print_vector("b", b)
    print_vector("c", c)

    vol = parallelepiped_volume(a, b, c)
    print(f"\n平行六面體（長方體）體積 = {vol:.4f}")
    print(f"（= 1 × 2 × 3 = 6）")

    # 一般情況
    print("\n一般平行六面體：")
    a = [1, 1, 0]
    b = [0, 1, 1]
    c = [1, 0, 1]
    print_vector("a", a)
    print_vector("b", b)
    print_vector("c", c)
    print(f"體積 = {parallelepiped_volume(a, b, c):.4f}")

    # ========================================
    # 4. 三角形面積
    # ========================================
    print_separator("4. 三角形面積")

    p1 = (0, 0)
    p2 = (4, 0)
    p3 = (0, 3)

    print(f"三角形頂點：")
    print(f"  P1 = {p1}")
    print(f"  P2 = {p2}")
    print(f"  P3 = {p3}")

    area = triangle_area(p1, p2, p3)
    print(f"\n面積 = {area:.4f}")
    print(f"（底 × 高 / 2 = 4 × 3 / 2 = 6）")

    # 一般三角形
    print("\n一般三角形：")
    p1 = (1, 1)
    p2 = (4, 2)
    p3 = (2, 5)
    print(f"  P1 = {p1}, P2 = {p2}, P3 = {p3}")
    print(f"  面積 = {triangle_area(p1, p2, p3):.4f}")

    # ========================================
    # 5. 純量三重積
    # ========================================
    print_separator("5. 純量三重積 a · (b × c)")

    a = [1, 0, 0]
    b = [0, 1, 0]
    c = [0, 0, 1]

    print_vector("a", a)
    print_vector("b", b)
    print_vector("c", c)

    b_cross_c = cross_product_3d(b, c)
    triple = scalar_triple_product(a, b, c)

    print(f"\nb × c = {b_cross_c}")
    print(f"a · (b × c) = {triple:.4f}")
    print(f"\n這等於 det([a | b | c]) = 平行六面體體積")

    # ========================================
    # 6. 線性變換的體積縮放
    # ========================================
    print_separator("6. 線性變換的體積縮放")

    # 縮放矩陣
    A = [[2, 0], [0, 3]]
    print_matrix("縮放矩陣 A", A)
    print(f"det(A) = {det_2x2(A):.4f}")
    print("\n單位正方形 → 2×3 長方形")
    print("面積從 1 變成 6（= |det(A)|）")

    # 旋轉矩陣
    print("\n旋轉矩陣（45°）：")
    theta = math.pi / 4
    R = [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]]
    print_matrix("R", R)
    print(f"det(R) = {det_2x2(R):.4f}")
    print("旋轉保持面積！")

    # 剪切矩陣
    print("\n剪切矩陣：")
    S = [[1, 2], [0, 1]]
    print_matrix("S", S)
    print(f"det(S) = {det_2x2(S):.4f}")
    print("剪切保持面積！")

    # 反射矩陣
    print("\n反射矩陣（沿 x 軸）：")
    H = [[1, 0], [0, -1]]
    print_matrix("H", H)
    print(f"det(H) = {det_2x2(H):.4f}")
    print("反射保持面積，但反轉定向（det < 0）")

    # ========================================
    # 7. 奇異矩陣：面積變零
    # ========================================
    print_separator("7. 奇異矩陣：面積變零")

    A_singular = [[1, 2], [2, 4]]
    print_matrix("A（奇異）", A_singular)
    print(f"det(A) = {det_2x2(A_singular):.4f}")
    print("\n線性變換後，所有點都落在一條直線上")
    print("2D 區域 → 1D 線段（面積 = 0）")

    # 總結
    print_separator("總結")
    print("""
行列式的幾何意義：

1. |det| = 體積/面積的縮放因子
   - 2D：平行四邊形面積
   - 3D：平行六面體體積

2. sign(det) = 定向
   - det > 0：保持定向
   - det < 0：反轉定向

3. det = 0 → 降維
   - 面積/體積變成零
   - 矩陣奇異

特殊矩陣：
   - 旋轉：det = 1
   - 反射：det = -1
   - 剪切：det = 1
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
