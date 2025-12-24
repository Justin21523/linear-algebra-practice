"""
內積與正交性 - 手刻版本 (Inner Product and Orthogonality - Manual Implementation)

本程式示範：
1. 向量內積計算
2. 向量長度（範數）
3. 向量夾角
4. 正交性判斷
5. Cauchy-Schwarz 不等式
6. 正交矩陣驗證

不使用 NumPy，純手刻實作以理解底層計算。
"""

from typing import List
import math


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ========================================
# 基本向量運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:
    """
    計算兩向量的內積 (Dot Product)

    x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ

    時間複雜度：O(n)
    """
    if len(x) != len(y):
        raise ValueError("向量維度必須相同")

    result = 0.0
    for i in range(len(x)):
        result += x[i] * y[i]
    return result


def vector_norm(x: List[float]) -> float:
    """
    計算向量的長度（L2 範數）

    ‖x‖ = √(x · x) = √(x₁² + x₂² + ... + xₙ²)
    """
    return math.sqrt(dot_product(x, x))


def normalize(x: List[float]) -> List[float]:
    """
    正規化向量為單位向量

    û = x / ‖x‖
    """
    norm = vector_norm(x)
    if norm < 1e-10:
        raise ValueError("零向量無法正規化")
    return [xi / norm for xi in x]


def vector_angle(x: List[float], y: List[float]) -> float:
    """
    計算兩向量的夾角（弧度）

    cos θ = (x · y) / (‖x‖ ‖y‖)
    """
    dot = dot_product(x, y)
    norm_x = vector_norm(x)
    norm_y = vector_norm(y)

    if norm_x < 1e-10 or norm_y < 1e-10:
        raise ValueError("零向量沒有定義夾角")

    cos_theta = dot / (norm_x * norm_y)
    # 處理浮點數誤差
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def is_orthogonal(x: List[float], y: List[float], tol: float = 1e-10) -> bool:
    """
    判斷兩向量是否正交

    x ⊥ y ⟺ x · y = 0
    """
    return abs(dot_product(x, y)) < tol


# ========================================
# 正交相關運算
# ========================================

def is_orthogonal_set(vectors: List[List[float]], tol: float = 1e-10) -> bool:
    """
    判斷向量組是否為正交組

    對所有 i ≠ j，vᵢ · vⱼ = 0
    """
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            if not is_orthogonal(vectors[i], vectors[j], tol):
                return False
    return True


def is_orthonormal_set(vectors: List[List[float]], tol: float = 1e-10) -> bool:
    """
    判斷向量組是否為標準正交組

    正交 + 每個向量長度為 1
    """
    # 檢查每個向量是否為單位向量
    for v in vectors:
        if abs(vector_norm(v) - 1.0) > tol:
            return False

    # 檢查是否兩兩正交
    return is_orthogonal_set(vectors, tol)


# ========================================
# 矩陣運算（用於正交矩陣）
# ========================================

def matrix_transpose(A: List[List[float]]) -> List[List[float]]:
    """矩陣轉置"""
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩陣乘法"""
    m = len(A)
    n = len(B[0])
    k = len(B)

    if len(A[0]) != k:
        raise ValueError("矩陣維度不匹配")

    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for p in range(k):
                result[i][j] += A[i][p] * B[p][j]
    return result


def is_identity(A: List[List[float]], tol: float = 1e-10) -> bool:
    """判斷是否為單位矩陣"""
    n = len(A)
    if len(A[0]) != n:
        return False

    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(A[i][j] - expected) > tol:
                return False
    return True


def is_orthogonal_matrix(Q: List[List[float]], tol: float = 1e-10) -> bool:
    """
    判斷矩陣是否為正交矩陣

    QᵀQ = I
    """
    Q_T = matrix_transpose(Q)
    product = matrix_multiply(Q_T, Q)
    return is_identity(product, tol)


# ========================================
# 不等式驗證
# ========================================

def verify_cauchy_schwarz(x: List[float], y: List[float]) -> dict:
    """
    驗證 Cauchy-Schwarz 不等式

    |x · y| ≤ ‖x‖ ‖y‖
    """
    dot = abs(dot_product(x, y))
    product_of_norms = vector_norm(x) * vector_norm(y)

    return {
        'left_side': dot,
        'right_side': product_of_norms,
        'satisfied': dot <= product_of_norms + 1e-10,
        'equality': abs(dot - product_of_norms) < 1e-10
    }


def verify_triangle_inequality(x: List[float], y: List[float]) -> dict:
    """
    驗證三角不等式

    ‖x + y‖ ≤ ‖x‖ + ‖y‖
    """
    x_plus_y = [x[i] + y[i] for i in range(len(x))]
    norm_sum = vector_norm(x_plus_y)
    sum_of_norms = vector_norm(x) + vector_norm(y)

    return {
        'left_side': norm_sum,
        'right_side': sum_of_norms,
        'satisfied': norm_sum <= sum_of_norms + 1e-10
    }


# ========================================
# 輔助函數
# ========================================

def print_vector(name: str, v: List[float]) -> None:
    """印出向量"""
    formatted = [f"{x:.4f}" for x in v]
    print(f"{name} = [{', '.join(formatted)}]")


def print_matrix(name: str, M: List[List[float]]) -> None:
    """印出矩陣"""
    print(f"{name} =")
    for row in M:
        formatted = [f"{x:8.4f}" for x in row]
        print(f"  [{', '.join(formatted)}]")


def main():
    """主程式"""

    print_separator("內積與正交性示範（手刻版）\nInner Product & Orthogonality Demo (Manual)")

    # ========================================
    # 1. 內積計算
    # ========================================
    print_separator("1. 內積計算 (Dot Product)")

    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]

    print_vector("x", x)
    print_vector("y", y)
    print(f"\nx · y = {dot_product(x, y)}")
    print("計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32")

    # ========================================
    # 2. 向量長度
    # ========================================
    print_separator("2. 向量長度 (Vector Norm)")

    v = [3.0, 4.0]
    print_vector("v", v)
    print(f"‖v‖ = {vector_norm(v)}")
    print("計算：√(3² + 4²) = √(9 + 16) = √25 = 5")

    # 正規化
    v_normalized = normalize(v)
    print(f"\n單位向量：")
    print_vector("v̂ = v/‖v‖", v_normalized)
    print(f"‖v̂‖ = {vector_norm(v_normalized)}")

    # ========================================
    # 3. 向量夾角
    # ========================================
    print_separator("3. 向量夾角 (Vector Angle)")

    a = [1.0, 0.0]
    b = [1.0, 1.0]

    print_vector("a", a)
    print_vector("b", b)

    theta = vector_angle(a, b)
    print(f"\n夾角 θ = {theta:.4f} rad = {math.degrees(theta):.2f}°")
    print(f"cos θ = {math.cos(theta):.4f}")
    print("預期：cos 45° = 1/√2 ≈ 0.7071")

    # ========================================
    # 4. 正交性判斷
    # ========================================
    print_separator("4. 正交性判斷 (Orthogonality Check)")

    u1 = [1.0, 2.0]
    u2 = [-2.0, 1.0]

    print_vector("u₁", u1)
    print_vector("u₂", u2)
    print(f"\nu₁ · u₂ = {dot_product(u1, u2)}")
    print(f"u₁ ⊥ u₂？ {is_orthogonal(u1, u2)}")

    # 非正交的例子
    w1 = [1.0, 1.0]
    w2 = [1.0, 2.0]

    print(f"\n另一組：")
    print_vector("w₁", w1)
    print_vector("w₂", w2)
    print(f"w₁ · w₂ = {dot_product(w1, w2)}")
    print(f"w₁ ⊥ w₂？ {is_orthogonal(w1, w2)}")

    # ========================================
    # 5. 正交組與標準正交組
    # ========================================
    print_separator("5. 正交組與標準正交組")

    # 標準基底（標準正交）
    e1 = [1.0, 0.0, 0.0]
    e2 = [0.0, 1.0, 0.0]
    e3 = [0.0, 0.0, 1.0]
    standard_basis = [e1, e2, e3]

    print("標準基底 {e₁, e₂, e₃}：")
    for i, e in enumerate(standard_basis, 1):
        print_vector(f"e{i}", e)

    print(f"\n正交組？ {is_orthogonal_set(standard_basis)}")
    print(f"標準正交組？ {is_orthonormal_set(standard_basis)}")

    # 正交但非標準正交
    v1 = [1.0, 1.0]
    v2 = [-1.0, 1.0]
    orthogonal_set = [v1, v2]

    print(f"\n另一組：")
    print_vector("v₁", v1)
    print_vector("v₂", v2)
    print(f"‖v₁‖ = {vector_norm(v1):.4f}")
    print(f"‖v₂‖ = {vector_norm(v2):.4f}")
    print(f"正交組？ {is_orthogonal_set(orthogonal_set)}")
    print(f"標準正交組？ {is_orthonormal_set(orthogonal_set)}")

    # ========================================
    # 6. 正交矩陣
    # ========================================
    print_separator("6. 正交矩陣 (Orthogonal Matrix)")

    # 旋轉矩陣（45度）
    theta = math.pi / 4
    Q = [
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ]

    print(f"旋轉矩陣（θ = 45°）：")
    print_matrix("Q", Q)

    Q_T = matrix_transpose(Q)
    print_matrix("\nQᵀ", Q_T)

    QTQ = matrix_multiply(Q_T, Q)
    print_matrix("\nQᵀQ", QTQ)

    print(f"\nQ 是正交矩陣？ {is_orthogonal_matrix(Q)}")

    # 驗證保長度
    x = [3.0, 4.0]
    Qx = [Q[0][0]*x[0] + Q[0][1]*x[1], Q[1][0]*x[0] + Q[1][1]*x[1]]

    print(f"\n保長度驗證：")
    print_vector("x", x)
    print_vector("Qx", Qx)
    print(f"‖x‖ = {vector_norm(x):.4f}")
    print(f"‖Qx‖ = {vector_norm(Qx):.4f}")

    # ========================================
    # 7. Cauchy-Schwarz 不等式
    # ========================================
    print_separator("7. Cauchy-Schwarz 不等式")

    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]

    print_vector("x", x)
    print_vector("y", y)

    result = verify_cauchy_schwarz(x, y)
    print(f"\n|x · y| = {result['left_side']:.4f}")
    print(f"‖x‖ ‖y‖ = {result['right_side']:.4f}")
    print(f"|x · y| ≤ ‖x‖ ‖y‖？ {result['satisfied']}")
    print(f"等號成立？ {result['equality']}（等號成立 ⟺ 平行）")

    # 平行向量的情況
    print("\n平行向量的情況：")
    p = [1.0, 2.0]
    q = [2.0, 4.0]  # q = 2p

    print_vector("p", p)
    print_vector("q = 2p", q)

    result_parallel = verify_cauchy_schwarz(p, q)
    print(f"|p · q| = {result_parallel['left_side']:.4f}")
    print(f"‖p‖ ‖q‖ = {result_parallel['right_side']:.4f}")
    print(f"等號成立？ {result_parallel['equality']}")

    # ========================================
    # 8. 三角不等式
    # ========================================
    print_separator("8. 三角不等式")

    x = [3.0, 0.0]
    y = [0.0, 4.0]

    print_vector("x", x)
    print_vector("y", y)

    result = verify_triangle_inequality(x, y)
    print(f"\n‖x + y‖ = {result['left_side']:.4f}")
    print(f"‖x‖ + ‖y‖ = {result['right_side']:.4f}")
    print(f"‖x + y‖ ≤ ‖x‖ + ‖y‖？ {result['satisfied']}")
    print("\n幾何意義：三角形兩邊之和 ≥ 第三邊")

    # 總結
    print_separator("總結")
    print("""
內積與正交性的核心公式：

1. 內積：x · y = Σ xᵢyᵢ

2. 長度：‖x‖ = √(x · x)

3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖)

4. 正交：x ⊥ y ⟺ x · y = 0

5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ

6. Cauchy-Schwarz：|x · y| ≤ ‖x‖ ‖y‖

7. 三角不等式：‖x + y‖ ≤ ‖x‖ + ‖y‖
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
