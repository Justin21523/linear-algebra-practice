"""
向量運算：手刻版本 (Vector Operations: Manual Implementation)

本程式示範：
1. 向量加法、減法 (Vector addition, subtraction)
2. 純量乘法 (Scalar multiplication)
3. 向量長度 (Vector norm/magnitude)
4. 向量正規化 (Vector normalization)
5. 內積 (Dot product)
6. 夾角計算 (Angle between vectors)
7. 投影 (Projection)

This program demonstrates basic vector operations without using NumPy.
"""

import math
from typing import List

# 型別別名 (Type alias)
Vector = List[float]


def vector_add(u: Vector, v: Vector) -> Vector:
    """
    向量加法 (Vector addition)

    u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]
    """
    if len(u) != len(v):
        raise ValueError("向量維度必須相同 (Vectors must have same dimension)")

    return [u[i] + v[i] for i in range(len(u))]


def vector_subtract(u: Vector, v: Vector) -> Vector:
    """
    向量減法 (Vector subtraction)

    u - v = [u₁-v₁, u₂-v₂, ..., uₙ-vₙ]
    """
    if len(u) != len(v):
        raise ValueError("向量維度必須相同 (Vectors must have same dimension)")

    return [u[i] - v[i] for i in range(len(u))]


def scalar_multiply(c: float, v: Vector) -> Vector:
    """
    純量乘法 (Scalar multiplication)

    c·v = [c·v₁, c·v₂, ..., c·vₙ]
    """
    return [c * v[i] for i in range(len(v))]


def vector_norm(v: Vector) -> float:
    """
    向量長度/範數 (Vector norm/magnitude)

    ‖v‖ = √(v₁² + v₂² + ... + vₙ²)
    """
    return math.sqrt(sum(x**2 for x in v))


def normalize(v: Vector) -> Vector:
    """
    向量正規化 (Vector normalization)

    û = v / ‖v‖

    將向量轉換為同方向的單位向量
    Converts vector to unit vector in the same direction
    """
    norm = vector_norm(v)

    if norm == 0:
        raise ValueError("零向量無法正規化 (Cannot normalize zero vector)")

    return [x / norm for x in v]


def dot_product(u: Vector, v: Vector) -> float:
    """
    內積 (Dot product / Inner product)

    u·v = u₁v₁ + u₂v₂ + ... + uₙvₙ

    幾何意義：u·v = ‖u‖‖v‖cos(θ)
    Geometric meaning: u·v = ‖u‖‖v‖cos(θ)
    """
    if len(u) != len(v):
        raise ValueError("向量維度必須相同 (Vectors must have same dimension)")

    return sum(u[i] * v[i] for i in range(len(u)))


def angle_between(u: Vector, v: Vector) -> float:
    """
    計算兩向量夾角（弧度）(Angle between vectors in radians)

    cos(θ) = (u·v) / (‖u‖‖v‖)
    θ = arccos((u·v) / (‖u‖‖v‖))
    """
    dot = dot_product(u, v)
    norm_u = vector_norm(u)
    norm_v = vector_norm(v)

    if norm_u == 0 or norm_v == 0:
        raise ValueError("零向量沒有定義夾角 (Angle undefined for zero vector)")

    # 處理數值誤差，確保 cos_theta 在 [-1, 1] 範圍內
    # Handle numerical errors, ensure cos_theta is in [-1, 1]
    cos_theta = dot / (norm_u * norm_v)
    cos_theta = max(-1.0, min(1.0, cos_theta))

    return math.acos(cos_theta)


def project(u: Vector, v: Vector) -> Vector:
    """
    向量投影 (Vector projection)

    proj_v(u) = ((u·v) / ‖v‖²) · v

    將向量 u 投影到向量 v 上
    Projects vector u onto vector v
    """
    dot_uv = dot_product(u, v)
    norm_v_squared = dot_product(v, v)  # ‖v‖² = v·v

    if norm_v_squared == 0:
        raise ValueError("無法投影到零向量 (Cannot project onto zero vector)")

    scalar = dot_uv / norm_v_squared
    return scalar_multiply(scalar, v)


def scalar_projection(u: Vector, v: Vector) -> float:
    """
    純量投影 / 投影長度 (Scalar projection / Component)

    comp_v(u) = (u·v) / ‖v‖

    向量 u 在向量 v 方向上的分量（帶正負號）
    The signed length of the projection of u onto v
    """
    dot_uv = dot_product(u, v)
    norm_v = vector_norm(v)

    if norm_v == 0:
        raise ValueError("無法計算到零向量的投影 (Cannot compute projection onto zero vector)")

    return dot_uv / norm_v


def is_orthogonal(u: Vector, v: Vector, tolerance: float = 1e-10) -> bool:
    """
    檢查兩向量是否正交 (Check if two vectors are orthogonal)

    若 u·v = 0，則 u ⊥ v
    """
    return abs(dot_product(u, v)) < tolerance


def print_vector(name: str, v: Vector) -> None:
    """印出向量 (Print vector)"""
    formatted = ", ".join(f"{x:.4f}" for x in v)
    print(f"{name} = [{formatted}]")


def print_separator(title: str) -> None:
    """印出分隔線 (Print separator)"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    """主程式：示範所有向量運算 (Main: Demonstrate all vector operations)"""

    print_separator("向量運算示範 - 手刻版本\nVector Operations Demo - Manual Implementation")

    # ========================================
    # 定義範例向量 (Define example vectors)
    # ========================================
    u = [3.0, 4.0]
    v = [1.0, 2.0]

    print("\n範例向量 (Example vectors):")
    print_vector("u", u)
    print_vector("v", v)

    # ========================================
    # 1. 向量加法 (Vector Addition)
    # ========================================
    print_separator("1. 向量加法 (Vector Addition)")

    result = vector_add(u, v)
    print(f"u + v = [{u[0]}, {u[1]}] + [{v[0]}, {v[1]}]")
    print_vector("u + v", result)

    # ========================================
    # 2. 向量減法 (Vector Subtraction)
    # ========================================
    print_separator("2. 向量減法 (Vector Subtraction)")

    result = vector_subtract(u, v)
    print(f"u - v = [{u[0]}, {u[1]}] - [{v[0]}, {v[1]}]")
    print_vector("u - v", result)

    # ========================================
    # 3. 純量乘法 (Scalar Multiplication)
    # ========================================
    print_separator("3. 純量乘法 (Scalar Multiplication)")

    c = 2.5
    result = scalar_multiply(c, u)
    print(f"{c} × u = {c} × [{u[0]}, {u[1]}]")
    print_vector(f"{c}u", result)

    # 負純量示範 (Negative scalar)
    result_neg = scalar_multiply(-1, u)
    print_vector("-u (反向量)", result_neg)

    # ========================================
    # 4. 向量長度 (Vector Norm)
    # ========================================
    print_separator("4. 向量長度 (Vector Norm)")

    norm_u = vector_norm(u)
    print(f"‖u‖ = √({u[0]}² + {u[1]}²) = √({u[0]**2} + {u[1]**2}) = √{u[0]**2 + u[1]**2}")
    print(f"‖u‖ = {norm_u:.4f}")

    # 經典的 3-4-5 直角三角形！
    print("\n這就是經典的 3-4-5 直角三角形！")
    print("This is the classic 3-4-5 right triangle!")

    # ========================================
    # 5. 向量正規化 (Normalization)
    # ========================================
    print_separator("5. 向量正規化 (Normalization)")

    u_hat = normalize(u)
    print(f"û = u / ‖u‖ = [{u[0]}, {u[1]}] / {norm_u}")
    print_vector("û (單位向量)", u_hat)
    print(f"‖û‖ = {vector_norm(u_hat):.4f} (應該是 1)")

    # ========================================
    # 6. 內積 (Dot Product)
    # ========================================
    print_separator("6. 內積 (Dot Product)")

    dot = dot_product(u, v)
    print(f"u·v = {u[0]}×{v[0]} + {u[1]}×{v[1]}")
    print(f"u·v = {u[0]*v[0]} + {u[1]*v[1]} = {dot}")

    # 驗證：‖v‖² = v·v
    print(f"\n驗證 ‖v‖² = v·v:")
    print(f"‖v‖² = {vector_norm(v)**2:.4f}")
    print(f"v·v  = {dot_product(v, v):.4f}")

    # ========================================
    # 7. 夾角計算 (Angle Calculation)
    # ========================================
    print_separator("7. 夾角計算 (Angle Between Vectors)")

    angle_rad = angle_between(u, v)
    angle_deg = math.degrees(angle_rad)

    print(f"θ = arccos((u·v) / (‖u‖×‖v‖))")
    print(f"θ = arccos({dot} / ({norm_u:.4f} × {vector_norm(v):.4f}))")
    print(f"θ = {angle_rad:.4f} 弧度 (radians)")
    print(f"θ = {angle_deg:.2f}° 度 (degrees)")

    # ========================================
    # 8. 正交檢驗 (Orthogonality Check)
    # ========================================
    print_separator("8. 正交檢驗 (Orthogonality Check)")

    # 兩個正交向量
    a = [1.0, 0.0]
    b = [0.0, 1.0]

    print_vector("a", a)
    print_vector("b", b)
    print(f"a·b = {dot_product(a, b)}")
    print(f"a 和 b 是否正交？ {is_orthogonal(a, b)}")

    print()
    print_vector("u", u)
    print_vector("v", v)
    print(f"u·v = {dot_product(u, v)}")
    print(f"u 和 v 是否正交？ {is_orthogonal(u, v)}")

    # ========================================
    # 9. 向量投影 (Vector Projection)
    # ========================================
    print_separator("9. 向量投影 (Vector Projection)")

    # 將 u 投影到 v 上
    proj = project(u, v)
    comp = scalar_projection(u, v)

    print(f"將 u 投影到 v 上 (Project u onto v):")
    print_vector("proj_v(u)", proj)
    print(f"comp_v(u) = {comp:.4f} (投影長度/純量投影)")

    # 驗證：投影向量與 (u - 投影) 正交
    perpendicular = vector_subtract(u, proj)
    print()
    print("驗證：u = proj + perp，其中 perp ⊥ v")
    print_vector("perp = u - proj", perpendicular)
    print(f"perp · v = {dot_product(perpendicular, v):.10f} (應該接近 0)")

    # ========================================
    # 10. 3D 向量示範 (3D Vector Demo)
    # ========================================
    print_separator("10. 3D 向量示範 (3D Vector Demo)")

    p = [1.0, 2.0, 3.0]
    q = [4.0, 5.0, 6.0]

    print_vector("p", p)
    print_vector("q", q)
    print_vector("p + q", vector_add(p, q))
    print(f"p · q = {dot_product(p, q)}")
    print(f"‖p‖ = {vector_norm(p):.4f}")
    print(f"夾角 = {math.degrees(angle_between(p, q)):.2f}°")

    # ========================================
    # 11. 線性組合示範 (Linear Combination Demo)
    # ========================================
    print_separator("11. 線性組合示範 (Linear Combination)")

    # e1 和 e2 是標準基底向量
    e1 = [1.0, 0.0]
    e2 = [0.0, 1.0]

    print("標準基底向量 (Standard basis vectors):")
    print_vector("e₁", e1)
    print_vector("e₂", e2)

    # 任何 2D 向量都可以表示為 e1 和 e2 的線性組合
    target = [3.0, 4.0]
    print(f"\n向量 [3, 4] = 3·e₁ + 4·e₂")

    combination = vector_add(
        scalar_multiply(3, e1),
        scalar_multiply(4, e2)
    )
    print_vector("3·e₁ + 4·e₂", combination)

    print()
    print("=" * 60)
    print("所有向量運算示範完成！")
    print("All vector operations demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
