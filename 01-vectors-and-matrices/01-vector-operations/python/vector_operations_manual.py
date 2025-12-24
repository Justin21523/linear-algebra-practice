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
"""  # EN: Execute statement: """.

import math  # EN: Import module(s): import math.
from typing import List  # EN: Import symbol(s) from a module: from typing import List.

# 型別別名 (Type alias)
Vector = List[float]  # EN: Assign Vector from expression: List[float].


def vector_add(u: Vector, v: Vector) -> Vector:  # EN: Define vector_add and its behavior.
    """
    向量加法 (Vector addition)

    u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]
    """  # EN: Execute statement: """.
    if len(u) != len(v):  # EN: Branch on a condition: if len(u) != len(v):.
        raise ValueError("向量維度必須相同 (Vectors must have same dimension)")  # EN: Raise an exception: raise ValueError("向量維度必須相同 (Vectors must have same dimension)").

    return [u[i] + v[i] for i in range(len(u))]  # EN: Return a value: return [u[i] + v[i] for i in range(len(u))].


def vector_subtract(u: Vector, v: Vector) -> Vector:  # EN: Define vector_subtract and its behavior.
    """
    向量減法 (Vector subtraction)

    u - v = [u₁-v₁, u₂-v₂, ..., uₙ-vₙ]
    """  # EN: Execute statement: """.
    if len(u) != len(v):  # EN: Branch on a condition: if len(u) != len(v):.
        raise ValueError("向量維度必須相同 (Vectors must have same dimension)")  # EN: Raise an exception: raise ValueError("向量維度必須相同 (Vectors must have same dimension)").

    return [u[i] - v[i] for i in range(len(u))]  # EN: Return a value: return [u[i] - v[i] for i in range(len(u))].


def scalar_multiply(c: float, v: Vector) -> Vector:  # EN: Define scalar_multiply and its behavior.
    """
    純量乘法 (Scalar multiplication)

    c·v = [c·v₁, c·v₂, ..., c·vₙ]
    """  # EN: Execute statement: """.
    return [c * v[i] for i in range(len(v))]  # EN: Return a value: return [c * v[i] for i in range(len(v))].


def vector_norm(v: Vector) -> float:  # EN: Define vector_norm and its behavior.
    """
    向量長度/範數 (Vector norm/magnitude)

    ‖v‖ = √(v₁² + v₂² + ... + vₙ²)
    """  # EN: Execute statement: """.
    return math.sqrt(sum(x**2 for x in v))  # EN: Return a value: return math.sqrt(sum(x**2 for x in v)).


def normalize(v: Vector) -> Vector:  # EN: Define normalize and its behavior.
    """
    向量正規化 (Vector normalization)

    û = v / ‖v‖

    將向量轉換為同方向的單位向量
    Converts vector to unit vector in the same direction
    """  # EN: Execute statement: """.
    norm = vector_norm(v)  # EN: Assign norm from expression: vector_norm(v).

    if norm == 0:  # EN: Branch on a condition: if norm == 0:.
        raise ValueError("零向量無法正規化 (Cannot normalize zero vector)")  # EN: Raise an exception: raise ValueError("零向量無法正規化 (Cannot normalize zero vector)").

    return [x / norm for x in v]  # EN: Return a value: return [x / norm for x in v].


def dot_product(u: Vector, v: Vector) -> float:  # EN: Define dot_product and its behavior.
    """
    內積 (Dot product / Inner product)

    u·v = u₁v₁ + u₂v₂ + ... + uₙvₙ

    幾何意義：u·v = ‖u‖‖v‖cos(θ)
    Geometric meaning: u·v = ‖u‖‖v‖cos(θ)
    """  # EN: Execute statement: """.
    if len(u) != len(v):  # EN: Branch on a condition: if len(u) != len(v):.
        raise ValueError("向量維度必須相同 (Vectors must have same dimension)")  # EN: Raise an exception: raise ValueError("向量維度必須相同 (Vectors must have same dimension)").

    return sum(u[i] * v[i] for i in range(len(u)))  # EN: Return a value: return sum(u[i] * v[i] for i in range(len(u))).


def angle_between(u: Vector, v: Vector) -> float:  # EN: Define angle_between and its behavior.
    """
    計算兩向量夾角（弧度）(Angle between vectors in radians)

    cos(θ) = (u·v) / (‖u‖‖v‖)
    θ = arccos((u·v) / (‖u‖‖v‖))
    """  # EN: Execute statement: """.
    dot = dot_product(u, v)  # EN: Assign dot from expression: dot_product(u, v).
    norm_u = vector_norm(u)  # EN: Assign norm_u from expression: vector_norm(u).
    norm_v = vector_norm(v)  # EN: Assign norm_v from expression: vector_norm(v).

    if norm_u == 0 or norm_v == 0:  # EN: Branch on a condition: if norm_u == 0 or norm_v == 0:.
        raise ValueError("零向量沒有定義夾角 (Angle undefined for zero vector)")  # EN: Raise an exception: raise ValueError("零向量沒有定義夾角 (Angle undefined for zero vector)").

    # 處理數值誤差，確保 cos_theta 在 [-1, 1] 範圍內
    # Handle numerical errors, ensure cos_theta is in [-1, 1]
    cos_theta = dot / (norm_u * norm_v)  # EN: Assign cos_theta from expression: dot / (norm_u * norm_v).
    cos_theta = max(-1.0, min(1.0, cos_theta))  # EN: Assign cos_theta from expression: max(-1.0, min(1.0, cos_theta)).

    return math.acos(cos_theta)  # EN: Return a value: return math.acos(cos_theta).


def project(u: Vector, v: Vector) -> Vector:  # EN: Define project and its behavior.
    """
    向量投影 (Vector projection)

    proj_v(u) = ((u·v) / ‖v‖²) · v

    將向量 u 投影到向量 v 上
    Projects vector u onto vector v
    """  # EN: Execute statement: """.
    dot_uv = dot_product(u, v)  # EN: Assign dot_uv from expression: dot_product(u, v).
    norm_v_squared = dot_product(v, v)  # ‖v‖² = v·v  # EN: Assign norm_v_squared from expression: dot_product(v, v) # ‖v‖² = v·v.

    if norm_v_squared == 0:  # EN: Branch on a condition: if norm_v_squared == 0:.
        raise ValueError("無法投影到零向量 (Cannot project onto zero vector)")  # EN: Raise an exception: raise ValueError("無法投影到零向量 (Cannot project onto zero vector)").

    scalar = dot_uv / norm_v_squared  # EN: Assign scalar from expression: dot_uv / norm_v_squared.
    return scalar_multiply(scalar, v)  # EN: Return a value: return scalar_multiply(scalar, v).


def scalar_projection(u: Vector, v: Vector) -> float:  # EN: Define scalar_projection and its behavior.
    """
    純量投影 / 投影長度 (Scalar projection / Component)

    comp_v(u) = (u·v) / ‖v‖

    向量 u 在向量 v 方向上的分量（帶正負號）
    The signed length of the projection of u onto v
    """  # EN: Execute statement: """.
    dot_uv = dot_product(u, v)  # EN: Assign dot_uv from expression: dot_product(u, v).
    norm_v = vector_norm(v)  # EN: Assign norm_v from expression: vector_norm(v).

    if norm_v == 0:  # EN: Branch on a condition: if norm_v == 0:.
        raise ValueError("無法計算到零向量的投影 (Cannot compute projection onto zero vector)")  # EN: Raise an exception: raise ValueError("無法計算到零向量的投影 (Cannot compute projection onto zero vect….

    return dot_uv / norm_v  # EN: Return a value: return dot_uv / norm_v.


def is_orthogonal(u: Vector, v: Vector, tolerance: float = 1e-10) -> bool:  # EN: Define is_orthogonal and its behavior.
    """
    檢查兩向量是否正交 (Check if two vectors are orthogonal)

    若 u·v = 0，則 u ⊥ v
    """  # EN: Execute statement: """.
    return abs(dot_product(u, v)) < tolerance  # EN: Return a value: return abs(dot_product(u, v)) < tolerance.


def print_vector(name: str, v: Vector) -> None:  # EN: Define print_vector and its behavior.
    """印出向量 (Print vector)"""  # EN: Execute statement: """印出向量 (Print vector)""".
    formatted = ", ".join(f"{x:.4f}" for x in v)  # EN: Assign formatted from expression: ", ".join(f"{x:.4f}" for x in v).
    print(f"{name} = [{formatted}]")  # EN: Print formatted output to the console.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線 (Print separator)"""  # EN: Execute statement: """印出分隔線 (Print separator)""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式：示範所有向量運算 (Main: Demonstrate all vector operations)"""  # EN: Execute statement: """主程式：示範所有向量運算 (Main: Demonstrate all vector operations)""".

    print_separator("向量運算示範 - 手刻版本\nVector Operations Demo - Manual Implementation")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 定義範例向量 (Define example vectors)
    # ========================================
    u = [3.0, 4.0]  # EN: Assign u from expression: [3.0, 4.0].
    v = [1.0, 2.0]  # EN: Assign v from expression: [1.0, 2.0].

    print("\n範例向量 (Example vectors):")  # EN: Print formatted output to the console.
    print_vector("u", u)  # EN: Call print_vector(...) to perform an operation.
    print_vector("v", v)  # EN: Call print_vector(...) to perform an operation.

    # ========================================
    # 1. 向量加法 (Vector Addition)
    # ========================================
    print_separator("1. 向量加法 (Vector Addition)")  # EN: Call print_separator(...) to perform an operation.

    result = vector_add(u, v)  # EN: Assign result from expression: vector_add(u, v).
    print(f"u + v = [{u[0]}, {u[1]}] + [{v[0]}, {v[1]}]")  # EN: Print formatted output to the console.
    print_vector("u + v", result)  # EN: Call print_vector(...) to perform an operation.

    # ========================================
    # 2. 向量減法 (Vector Subtraction)
    # ========================================
    print_separator("2. 向量減法 (Vector Subtraction)")  # EN: Call print_separator(...) to perform an operation.

    result = vector_subtract(u, v)  # EN: Assign result from expression: vector_subtract(u, v).
    print(f"u - v = [{u[0]}, {u[1]}] - [{v[0]}, {v[1]}]")  # EN: Print formatted output to the console.
    print_vector("u - v", result)  # EN: Call print_vector(...) to perform an operation.

    # ========================================
    # 3. 純量乘法 (Scalar Multiplication)
    # ========================================
    print_separator("3. 純量乘法 (Scalar Multiplication)")  # EN: Call print_separator(...) to perform an operation.

    c = 2.5  # EN: Assign c from expression: 2.5.
    result = scalar_multiply(c, u)  # EN: Assign result from expression: scalar_multiply(c, u).
    print(f"{c} × u = {c} × [{u[0]}, {u[1]}]")  # EN: Print formatted output to the console.
    print_vector(f"{c}u", result)  # EN: Call print_vector(...) to perform an operation.

    # 負純量示範 (Negative scalar)
    result_neg = scalar_multiply(-1, u)  # EN: Assign result_neg from expression: scalar_multiply(-1, u).
    print_vector("-u (反向量)", result_neg)  # EN: Call print_vector(...) to perform an operation.

    # ========================================
    # 4. 向量長度 (Vector Norm)
    # ========================================
    print_separator("4. 向量長度 (Vector Norm)")  # EN: Call print_separator(...) to perform an operation.

    norm_u = vector_norm(u)  # EN: Assign norm_u from expression: vector_norm(u).
    print(f"‖u‖ = √({u[0]}² + {u[1]}²) = √({u[0]**2} + {u[1]**2}) = √{u[0]**2 + u[1]**2}")  # EN: Print formatted output to the console.
    print(f"‖u‖ = {norm_u:.4f}")  # EN: Print formatted output to the console.

    # 經典的 3-4-5 直角三角形！
    print("\n這就是經典的 3-4-5 直角三角形！")  # EN: Print formatted output to the console.
    print("This is the classic 3-4-5 right triangle!")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 向量正規化 (Normalization)
    # ========================================
    print_separator("5. 向量正規化 (Normalization)")  # EN: Call print_separator(...) to perform an operation.

    u_hat = normalize(u)  # EN: Assign u_hat from expression: normalize(u).
    print(f"û = u / ‖u‖ = [{u[0]}, {u[1]}] / {norm_u}")  # EN: Print formatted output to the console.
    print_vector("û (單位向量)", u_hat)  # EN: Call print_vector(...) to perform an operation.
    print(f"‖û‖ = {vector_norm(u_hat):.4f} (應該是 1)")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 內積 (Dot Product)
    # ========================================
    print_separator("6. 內積 (Dot Product)")  # EN: Call print_separator(...) to perform an operation.

    dot = dot_product(u, v)  # EN: Assign dot from expression: dot_product(u, v).
    print(f"u·v = {u[0]}×{v[0]} + {u[1]}×{v[1]}")  # EN: Print formatted output to the console.
    print(f"u·v = {u[0]*v[0]} + {u[1]*v[1]} = {dot}")  # EN: Print formatted output to the console.

    # 驗證：‖v‖² = v·v
    print(f"\n驗證 ‖v‖² = v·v:")  # EN: Print formatted output to the console.
    print(f"‖v‖² = {vector_norm(v)**2:.4f}")  # EN: Print formatted output to the console.
    print(f"v·v  = {dot_product(v, v):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 夾角計算 (Angle Calculation)
    # ========================================
    print_separator("7. 夾角計算 (Angle Between Vectors)")  # EN: Call print_separator(...) to perform an operation.

    angle_rad = angle_between(u, v)  # EN: Assign angle_rad from expression: angle_between(u, v).
    angle_deg = math.degrees(angle_rad)  # EN: Assign angle_deg from expression: math.degrees(angle_rad).

    print(f"θ = arccos((u·v) / (‖u‖×‖v‖))")  # EN: Print formatted output to the console.
    print(f"θ = arccos({dot} / ({norm_u:.4f} × {vector_norm(v):.4f}))")  # EN: Print formatted output to the console.
    print(f"θ = {angle_rad:.4f} 弧度 (radians)")  # EN: Print formatted output to the console.
    print(f"θ = {angle_deg:.2f}° 度 (degrees)")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 正交檢驗 (Orthogonality Check)
    # ========================================
    print_separator("8. 正交檢驗 (Orthogonality Check)")  # EN: Call print_separator(...) to perform an operation.

    # 兩個正交向量
    a = [1.0, 0.0]  # EN: Assign a from expression: [1.0, 0.0].
    b = [0.0, 1.0]  # EN: Assign b from expression: [0.0, 1.0].

    print_vector("a", a)  # EN: Call print_vector(...) to perform an operation.
    print_vector("b", b)  # EN: Call print_vector(...) to perform an operation.
    print(f"a·b = {dot_product(a, b)}")  # EN: Print formatted output to the console.
    print(f"a 和 b 是否正交？ {is_orthogonal(a, b)}")  # EN: Print formatted output to the console.

    print()  # EN: Print formatted output to the console.
    print_vector("u", u)  # EN: Call print_vector(...) to perform an operation.
    print_vector("v", v)  # EN: Call print_vector(...) to perform an operation.
    print(f"u·v = {dot_product(u, v)}")  # EN: Print formatted output to the console.
    print(f"u 和 v 是否正交？ {is_orthogonal(u, v)}")  # EN: Print formatted output to the console.

    # ========================================
    # 9. 向量投影 (Vector Projection)
    # ========================================
    print_separator("9. 向量投影 (Vector Projection)")  # EN: Call print_separator(...) to perform an operation.

    # 將 u 投影到 v 上
    proj = project(u, v)  # EN: Assign proj from expression: project(u, v).
    comp = scalar_projection(u, v)  # EN: Assign comp from expression: scalar_projection(u, v).

    print(f"將 u 投影到 v 上 (Project u onto v):")  # EN: Print formatted output to the console.
    print_vector("proj_v(u)", proj)  # EN: Call print_vector(...) to perform an operation.
    print(f"comp_v(u) = {comp:.4f} (投影長度/純量投影)")  # EN: Print formatted output to the console.

    # 驗證：投影向量與 (u - 投影) 正交
    perpendicular = vector_subtract(u, proj)  # EN: Assign perpendicular from expression: vector_subtract(u, proj).
    print()  # EN: Print formatted output to the console.
    print("驗證：u = proj + perp，其中 perp ⊥ v")  # EN: Print formatted output to the console.
    print_vector("perp = u - proj", perpendicular)  # EN: Call print_vector(...) to perform an operation.
    print(f"perp · v = {dot_product(perpendicular, v):.10f} (應該接近 0)")  # EN: Print formatted output to the console.

    # ========================================
    # 10. 3D 向量示範 (3D Vector Demo)
    # ========================================
    print_separator("10. 3D 向量示範 (3D Vector Demo)")  # EN: Call print_separator(...) to perform an operation.

    p = [1.0, 2.0, 3.0]  # EN: Assign p from expression: [1.0, 2.0, 3.0].
    q = [4.0, 5.0, 6.0]  # EN: Assign q from expression: [4.0, 5.0, 6.0].

    print_vector("p", p)  # EN: Call print_vector(...) to perform an operation.
    print_vector("q", q)  # EN: Call print_vector(...) to perform an operation.
    print_vector("p + q", vector_add(p, q))  # EN: Call print_vector(...) to perform an operation.
    print(f"p · q = {dot_product(p, q)}")  # EN: Print formatted output to the console.
    print(f"‖p‖ = {vector_norm(p):.4f}")  # EN: Print formatted output to the console.
    print(f"夾角 = {math.degrees(angle_between(p, q)):.2f}°")  # EN: Print formatted output to the console.

    # ========================================
    # 11. 線性組合示範 (Linear Combination Demo)
    # ========================================
    print_separator("11. 線性組合示範 (Linear Combination)")  # EN: Call print_separator(...) to perform an operation.

    # e1 和 e2 是標準基底向量
    e1 = [1.0, 0.0]  # EN: Assign e1 from expression: [1.0, 0.0].
    e2 = [0.0, 1.0]  # EN: Assign e2 from expression: [0.0, 1.0].

    print("標準基底向量 (Standard basis vectors):")  # EN: Print formatted output to the console.
    print_vector("e₁", e1)  # EN: Call print_vector(...) to perform an operation.
    print_vector("e₂", e2)  # EN: Call print_vector(...) to perform an operation.

    # 任何 2D 向量都可以表示為 e1 和 e2 的線性組合
    target = [3.0, 4.0]  # EN: Assign target from expression: [3.0, 4.0].
    print(f"\n向量 [3, 4] = 3·e₁ + 4·e₂")  # EN: Print formatted output to the console.

    combination = vector_add(  # EN: Assign combination from expression: vector_add(.
        scalar_multiply(3, e1),  # EN: Call scalar_multiply(...) to perform an operation.
        scalar_multiply(4, e2)  # EN: Call scalar_multiply(...) to perform an operation.
    )  # EN: Execute statement: ).
    print_vector("3·e₁ + 4·e₂", combination)  # EN: Call print_vector(...) to perform an operation.

    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print("所有向量運算示範完成！")  # EN: Print formatted output to the console.
    print("All vector operations demonstrated!")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
