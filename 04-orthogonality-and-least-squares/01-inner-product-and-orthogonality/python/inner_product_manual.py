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
"""  # EN: Execute statement: """.

from typing import List  # EN: Import symbol(s) from a module: from typing import List.
import math  # EN: Import module(s): import math.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


# ========================================
# 基本向量運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:  # EN: Define dot_product and its behavior.
    """
    計算兩向量的內積 (Dot Product)

    x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ

    時間複雜度：O(n)
    """  # EN: Execute statement: """.
    if len(x) != len(y):  # EN: Branch on a condition: if len(x) != len(y):.
        raise ValueError("向量維度必須相同")  # EN: Raise an exception: raise ValueError("向量維度必須相同").

    result = 0.0  # EN: Assign result from expression: 0.0.
    for i in range(len(x)):  # EN: Iterate with a for-loop: for i in range(len(x)):.
        result += x[i] * y[i]  # EN: Update result via += using: x[i] * y[i].
    return result  # EN: Return a value: return result.


def vector_norm(x: List[float]) -> float:  # EN: Define vector_norm and its behavior.
    """
    計算向量的長度（L2 範數）

    ‖x‖ = √(x · x) = √(x₁² + x₂² + ... + xₙ²)
    """  # EN: Execute statement: """.
    return math.sqrt(dot_product(x, x))  # EN: Return a value: return math.sqrt(dot_product(x, x)).


def normalize(x: List[float]) -> List[float]:  # EN: Define normalize and its behavior.
    """
    正規化向量為單位向量

    û = x / ‖x‖
    """  # EN: Execute statement: """.
    norm = vector_norm(x)  # EN: Assign norm from expression: vector_norm(x).
    if norm < 1e-10:  # EN: Branch on a condition: if norm < 1e-10:.
        raise ValueError("零向量無法正規化")  # EN: Raise an exception: raise ValueError("零向量無法正規化").
    return [xi / norm for xi in x]  # EN: Return a value: return [xi / norm for xi in x].


def vector_angle(x: List[float], y: List[float]) -> float:  # EN: Define vector_angle and its behavior.
    """
    計算兩向量的夾角（弧度）

    cos θ = (x · y) / (‖x‖ ‖y‖)
    """  # EN: Execute statement: """.
    dot = dot_product(x, y)  # EN: Assign dot from expression: dot_product(x, y).
    norm_x = vector_norm(x)  # EN: Assign norm_x from expression: vector_norm(x).
    norm_y = vector_norm(y)  # EN: Assign norm_y from expression: vector_norm(y).

    if norm_x < 1e-10 or norm_y < 1e-10:  # EN: Branch on a condition: if norm_x < 1e-10 or norm_y < 1e-10:.
        raise ValueError("零向量沒有定義夾角")  # EN: Raise an exception: raise ValueError("零向量沒有定義夾角").

    cos_theta = dot / (norm_x * norm_y)  # EN: Assign cos_theta from expression: dot / (norm_x * norm_y).
    # 處理浮點數誤差
    cos_theta = max(-1.0, min(1.0, cos_theta))  # EN: Assign cos_theta from expression: max(-1.0, min(1.0, cos_theta)).
    return math.acos(cos_theta)  # EN: Return a value: return math.acos(cos_theta).


def is_orthogonal(x: List[float], y: List[float], tol: float = 1e-10) -> bool:  # EN: Define is_orthogonal and its behavior.
    """
    判斷兩向量是否正交

    x ⊥ y ⟺ x · y = 0
    """  # EN: Execute statement: """.
    return abs(dot_product(x, y)) < tol  # EN: Return a value: return abs(dot_product(x, y)) < tol.


# ========================================
# 正交相關運算
# ========================================

def is_orthogonal_set(vectors: List[List[float]], tol: float = 1e-10) -> bool:  # EN: Define is_orthogonal_set and its behavior.
    """
    判斷向量組是否為正交組

    對所有 i ≠ j，vᵢ · vⱼ = 0
    """  # EN: Execute statement: """.
    n = len(vectors)  # EN: Assign n from expression: len(vectors).
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        for j in range(i + 1, n):  # EN: Iterate with a for-loop: for j in range(i + 1, n):.
            if not is_orthogonal(vectors[i], vectors[j], tol):  # EN: Branch on a condition: if not is_orthogonal(vectors[i], vectors[j], tol):.
                return False  # EN: Return a value: return False.
    return True  # EN: Return a value: return True.


def is_orthonormal_set(vectors: List[List[float]], tol: float = 1e-10) -> bool:  # EN: Define is_orthonormal_set and its behavior.
    """
    判斷向量組是否為標準正交組

    正交 + 每個向量長度為 1
    """  # EN: Execute statement: """.
    # 檢查每個向量是否為單位向量
    for v in vectors:  # EN: Iterate with a for-loop: for v in vectors:.
        if abs(vector_norm(v) - 1.0) > tol:  # EN: Branch on a condition: if abs(vector_norm(v) - 1.0) > tol:.
            return False  # EN: Return a value: return False.

    # 檢查是否兩兩正交
    return is_orthogonal_set(vectors, tol)  # EN: Return a value: return is_orthogonal_set(vectors, tol).


# ========================================
# 矩陣運算（用於正交矩陣）
# ========================================

def matrix_transpose(A: List[List[float]]) -> List[List[float]]:  # EN: Define matrix_transpose and its behavior.
    """矩陣轉置"""  # EN: Execute statement: """矩陣轉置""".
    m = len(A)  # EN: Assign m from expression: len(A).
    n = len(A[0])  # EN: Assign n from expression: len(A[0]).
    return [[A[i][j] for i in range(m)] for j in range(n)]  # EN: Return a value: return [[A[i][j] for i in range(m)] for j in range(n)].


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:  # EN: Define matrix_multiply and its behavior.
    """矩陣乘法"""  # EN: Execute statement: """矩陣乘法""".
    m = len(A)  # EN: Assign m from expression: len(A).
    n = len(B[0])  # EN: Assign n from expression: len(B[0]).
    k = len(B)  # EN: Assign k from expression: len(B).

    if len(A[0]) != k:  # EN: Branch on a condition: if len(A[0]) != k:.
        raise ValueError("矩陣維度不匹配")  # EN: Raise an exception: raise ValueError("矩陣維度不匹配").

    result = [[0.0] * n for _ in range(m)]  # EN: Assign result from expression: [[0.0] * n for _ in range(m)].
    for i in range(m):  # EN: Iterate with a for-loop: for i in range(m):.
        for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
            for p in range(k):  # EN: Iterate with a for-loop: for p in range(k):.
                result[i][j] += A[i][p] * B[p][j]  # EN: Execute statement: result[i][j] += A[i][p] * B[p][j].
    return result  # EN: Return a value: return result.


def is_identity(A: List[List[float]], tol: float = 1e-10) -> bool:  # EN: Define is_identity and its behavior.
    """判斷是否為單位矩陣"""  # EN: Execute statement: """判斷是否為單位矩陣""".
    n = len(A)  # EN: Assign n from expression: len(A).
    if len(A[0]) != n:  # EN: Branch on a condition: if len(A[0]) != n:.
        return False  # EN: Return a value: return False.

    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
            expected = 1.0 if i == j else 0.0  # EN: Assign expected from expression: 1.0 if i == j else 0.0.
            if abs(A[i][j] - expected) > tol:  # EN: Branch on a condition: if abs(A[i][j] - expected) > tol:.
                return False  # EN: Return a value: return False.
    return True  # EN: Return a value: return True.


def is_orthogonal_matrix(Q: List[List[float]], tol: float = 1e-10) -> bool:  # EN: Define is_orthogonal_matrix and its behavior.
    """
    判斷矩陣是否為正交矩陣

    QᵀQ = I
    """  # EN: Execute statement: """.
    Q_T = matrix_transpose(Q)  # EN: Assign Q_T from expression: matrix_transpose(Q).
    product = matrix_multiply(Q_T, Q)  # EN: Assign product from expression: matrix_multiply(Q_T, Q).
    return is_identity(product, tol)  # EN: Return a value: return is_identity(product, tol).


# ========================================
# 不等式驗證
# ========================================

def verify_cauchy_schwarz(x: List[float], y: List[float]) -> dict:  # EN: Define verify_cauchy_schwarz and its behavior.
    """
    驗證 Cauchy-Schwarz 不等式

    |x · y| ≤ ‖x‖ ‖y‖
    """  # EN: Execute statement: """.
    dot = abs(dot_product(x, y))  # EN: Assign dot from expression: abs(dot_product(x, y)).
    product_of_norms = vector_norm(x) * vector_norm(y)  # EN: Assign product_of_norms from expression: vector_norm(x) * vector_norm(y).

    return {  # EN: Return a value: return {.
        'left_side': dot,  # EN: Execute statement: 'left_side': dot,.
        'right_side': product_of_norms,  # EN: Execute statement: 'right_side': product_of_norms,.
        'satisfied': dot <= product_of_norms + 1e-10,  # EN: Execute statement: 'satisfied': dot <= product_of_norms + 1e-10,.
        'equality': abs(dot - product_of_norms) < 1e-10  # EN: Execute statement: 'equality': abs(dot - product_of_norms) < 1e-10.
    }  # EN: Execute statement: }.


def verify_triangle_inequality(x: List[float], y: List[float]) -> dict:  # EN: Define verify_triangle_inequality and its behavior.
    """
    驗證三角不等式

    ‖x + y‖ ≤ ‖x‖ + ‖y‖
    """  # EN: Execute statement: """.
    x_plus_y = [x[i] + y[i] for i in range(len(x))]  # EN: Assign x_plus_y from expression: [x[i] + y[i] for i in range(len(x))].
    norm_sum = vector_norm(x_plus_y)  # EN: Assign norm_sum from expression: vector_norm(x_plus_y).
    sum_of_norms = vector_norm(x) + vector_norm(y)  # EN: Assign sum_of_norms from expression: vector_norm(x) + vector_norm(y).

    return {  # EN: Return a value: return {.
        'left_side': norm_sum,  # EN: Execute statement: 'left_side': norm_sum,.
        'right_side': sum_of_norms,  # EN: Execute statement: 'right_side': sum_of_norms,.
        'satisfied': norm_sum <= sum_of_norms + 1e-10  # EN: Execute statement: 'satisfied': norm_sum <= sum_of_norms + 1e-10.
    }  # EN: Execute statement: }.


# ========================================
# 輔助函數
# ========================================

def print_vector(name: str, v: List[float]) -> None:  # EN: Define print_vector and its behavior.
    """印出向量"""  # EN: Execute statement: """印出向量""".
    formatted = [f"{x:.4f}" for x in v]  # EN: Assign formatted from expression: [f"{x:.4f}" for x in v].
    print(f"{name} = [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


def print_matrix(name: str, M: List[List[float]]) -> None:  # EN: Define print_matrix and its behavior.
    """印出矩陣"""  # EN: Execute statement: """印出矩陣""".
    print(f"{name} =")  # EN: Print formatted output to the console.
    for row in M:  # EN: Iterate with a for-loop: for row in M:.
        formatted = [f"{x:8.4f}" for x in row]  # EN: Assign formatted from expression: [f"{x:8.4f}" for x in row].
        print(f"  [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("內積與正交性示範（手刻版）\nInner Product & Orthogonality Demo (Manual)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 內積計算
    # ========================================
    print_separator("1. 內積計算 (Dot Product)")  # EN: Call print_separator(...) to perform an operation.

    x = [1.0, 2.0, 3.0]  # EN: Assign x from expression: [1.0, 2.0, 3.0].
    y = [4.0, 5.0, 6.0]  # EN: Assign y from expression: [4.0, 5.0, 6.0].

    print_vector("x", x)  # EN: Call print_vector(...) to perform an operation.
    print_vector("y", y)  # EN: Call print_vector(...) to perform an operation.
    print(f"\nx · y = {dot_product(x, y)}")  # EN: Print formatted output to the console.
    print("計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 向量長度
    # ========================================
    print_separator("2. 向量長度 (Vector Norm)")  # EN: Call print_separator(...) to perform an operation.

    v = [3.0, 4.0]  # EN: Assign v from expression: [3.0, 4.0].
    print_vector("v", v)  # EN: Call print_vector(...) to perform an operation.
    print(f"‖v‖ = {vector_norm(v)}")  # EN: Print formatted output to the console.
    print("計算：√(3² + 4²) = √(9 + 16) = √25 = 5")  # EN: Print formatted output to the console.

    # 正規化
    v_normalized = normalize(v)  # EN: Assign v_normalized from expression: normalize(v).
    print(f"\n單位向量：")  # EN: Print formatted output to the console.
    print_vector("v̂ = v/‖v‖", v_normalized)  # EN: Call print_vector(...) to perform an operation.
    print(f"‖v̂‖ = {vector_norm(v_normalized)}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 向量夾角
    # ========================================
    print_separator("3. 向量夾角 (Vector Angle)")  # EN: Call print_separator(...) to perform an operation.

    a = [1.0, 0.0]  # EN: Assign a from expression: [1.0, 0.0].
    b = [1.0, 1.0]  # EN: Assign b from expression: [1.0, 1.0].

    print_vector("a", a)  # EN: Call print_vector(...) to perform an operation.
    print_vector("b", b)  # EN: Call print_vector(...) to perform an operation.

    theta = vector_angle(a, b)  # EN: Assign theta from expression: vector_angle(a, b).
    print(f"\n夾角 θ = {theta:.4f} rad = {math.degrees(theta):.2f}°")  # EN: Print formatted output to the console.
    print(f"cos θ = {math.cos(theta):.4f}")  # EN: Print formatted output to the console.
    print("預期：cos 45° = 1/√2 ≈ 0.7071")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 正交性判斷
    # ========================================
    print_separator("4. 正交性判斷 (Orthogonality Check)")  # EN: Call print_separator(...) to perform an operation.

    u1 = [1.0, 2.0]  # EN: Assign u1 from expression: [1.0, 2.0].
    u2 = [-2.0, 1.0]  # EN: Assign u2 from expression: [-2.0, 1.0].

    print_vector("u₁", u1)  # EN: Call print_vector(...) to perform an operation.
    print_vector("u₂", u2)  # EN: Call print_vector(...) to perform an operation.
    print(f"\nu₁ · u₂ = {dot_product(u1, u2)}")  # EN: Print formatted output to the console.
    print(f"u₁ ⊥ u₂？ {is_orthogonal(u1, u2)}")  # EN: Print formatted output to the console.

    # 非正交的例子
    w1 = [1.0, 1.0]  # EN: Assign w1 from expression: [1.0, 1.0].
    w2 = [1.0, 2.0]  # EN: Assign w2 from expression: [1.0, 2.0].

    print(f"\n另一組：")  # EN: Print formatted output to the console.
    print_vector("w₁", w1)  # EN: Call print_vector(...) to perform an operation.
    print_vector("w₂", w2)  # EN: Call print_vector(...) to perform an operation.
    print(f"w₁ · w₂ = {dot_product(w1, w2)}")  # EN: Print formatted output to the console.
    print(f"w₁ ⊥ w₂？ {is_orthogonal(w1, w2)}")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 正交組與標準正交組
    # ========================================
    print_separator("5. 正交組與標準正交組")  # EN: Call print_separator(...) to perform an operation.

    # 標準基底（標準正交）
    e1 = [1.0, 0.0, 0.0]  # EN: Assign e1 from expression: [1.0, 0.0, 0.0].
    e2 = [0.0, 1.0, 0.0]  # EN: Assign e2 from expression: [0.0, 1.0, 0.0].
    e3 = [0.0, 0.0, 1.0]  # EN: Assign e3 from expression: [0.0, 0.0, 1.0].
    standard_basis = [e1, e2, e3]  # EN: Assign standard_basis from expression: [e1, e2, e3].

    print("標準基底 {e₁, e₂, e₃}：")  # EN: Print formatted output to the console.
    for i, e in enumerate(standard_basis, 1):  # EN: Iterate with a for-loop: for i, e in enumerate(standard_basis, 1):.
        print_vector(f"e{i}", e)  # EN: Call print_vector(...) to perform an operation.

    print(f"\n正交組？ {is_orthogonal_set(standard_basis)}")  # EN: Print formatted output to the console.
    print(f"標準正交組？ {is_orthonormal_set(standard_basis)}")  # EN: Print formatted output to the console.

    # 正交但非標準正交
    v1 = [1.0, 1.0]  # EN: Assign v1 from expression: [1.0, 1.0].
    v2 = [-1.0, 1.0]  # EN: Assign v2 from expression: [-1.0, 1.0].
    orthogonal_set = [v1, v2]  # EN: Assign orthogonal_set from expression: [v1, v2].

    print(f"\n另一組：")  # EN: Print formatted output to the console.
    print_vector("v₁", v1)  # EN: Call print_vector(...) to perform an operation.
    print_vector("v₂", v2)  # EN: Call print_vector(...) to perform an operation.
    print(f"‖v₁‖ = {vector_norm(v1):.4f}")  # EN: Print formatted output to the console.
    print(f"‖v₂‖ = {vector_norm(v2):.4f}")  # EN: Print formatted output to the console.
    print(f"正交組？ {is_orthogonal_set(orthogonal_set)}")  # EN: Print formatted output to the console.
    print(f"標準正交組？ {is_orthonormal_set(orthogonal_set)}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 正交矩陣
    # ========================================
    print_separator("6. 正交矩陣 (Orthogonal Matrix)")  # EN: Call print_separator(...) to perform an operation.

    # 旋轉矩陣（45度）
    theta = math.pi / 4  # EN: Assign theta from expression: math.pi / 4.
    Q = [  # EN: Assign Q from expression: [.
        [math.cos(theta), -math.sin(theta)],  # EN: Execute statement: [math.cos(theta), -math.sin(theta)],.
        [math.sin(theta), math.cos(theta)]  # EN: Execute statement: [math.sin(theta), math.cos(theta)].
    ]  # EN: Execute statement: ].

    print(f"旋轉矩陣（θ = 45°）：")  # EN: Print formatted output to the console.
    print_matrix("Q", Q)  # EN: Call print_matrix(...) to perform an operation.

    Q_T = matrix_transpose(Q)  # EN: Assign Q_T from expression: matrix_transpose(Q).
    print_matrix("\nQᵀ", Q_T)  # EN: Call print_matrix(...) to perform an operation.

    QTQ = matrix_multiply(Q_T, Q)  # EN: Assign QTQ from expression: matrix_multiply(Q_T, Q).
    print_matrix("\nQᵀQ", QTQ)  # EN: Call print_matrix(...) to perform an operation.

    print(f"\nQ 是正交矩陣？ {is_orthogonal_matrix(Q)}")  # EN: Print formatted output to the console.

    # 驗證保長度
    x = [3.0, 4.0]  # EN: Assign x from expression: [3.0, 4.0].
    Qx = [Q[0][0]*x[0] + Q[0][1]*x[1], Q[1][0]*x[0] + Q[1][1]*x[1]]  # EN: Assign Qx from expression: [Q[0][0]*x[0] + Q[0][1]*x[1], Q[1][0]*x[0] + Q[1][1]*x[1]].

    print(f"\n保長度驗證：")  # EN: Print formatted output to the console.
    print_vector("x", x)  # EN: Call print_vector(...) to perform an operation.
    print_vector("Qx", Qx)  # EN: Call print_vector(...) to perform an operation.
    print(f"‖x‖ = {vector_norm(x):.4f}")  # EN: Print formatted output to the console.
    print(f"‖Qx‖ = {vector_norm(Qx):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 7. Cauchy-Schwarz 不等式
    # ========================================
    print_separator("7. Cauchy-Schwarz 不等式")  # EN: Call print_separator(...) to perform an operation.

    x = [1.0, 2.0, 3.0]  # EN: Assign x from expression: [1.0, 2.0, 3.0].
    y = [4.0, 5.0, 6.0]  # EN: Assign y from expression: [4.0, 5.0, 6.0].

    print_vector("x", x)  # EN: Call print_vector(...) to perform an operation.
    print_vector("y", y)  # EN: Call print_vector(...) to perform an operation.

    result = verify_cauchy_schwarz(x, y)  # EN: Assign result from expression: verify_cauchy_schwarz(x, y).
    print(f"\n|x · y| = {result['left_side']:.4f}")  # EN: Print formatted output to the console.
    print(f"‖x‖ ‖y‖ = {result['right_side']:.4f}")  # EN: Print formatted output to the console.
    print(f"|x · y| ≤ ‖x‖ ‖y‖？ {result['satisfied']}")  # EN: Print formatted output to the console.
    print(f"等號成立？ {result['equality']}（等號成立 ⟺ 平行）")  # EN: Print formatted output to the console.

    # 平行向量的情況
    print("\n平行向量的情況：")  # EN: Print formatted output to the console.
    p = [1.0, 2.0]  # EN: Assign p from expression: [1.0, 2.0].
    q = [2.0, 4.0]  # q = 2p  # EN: Assign q from expression: [2.0, 4.0] # q = 2p.

    print_vector("p", p)  # EN: Call print_vector(...) to perform an operation.
    print_vector("q = 2p", q)  # EN: Call print_vector(...) to perform an operation.

    result_parallel = verify_cauchy_schwarz(p, q)  # EN: Assign result_parallel from expression: verify_cauchy_schwarz(p, q).
    print(f"|p · q| = {result_parallel['left_side']:.4f}")  # EN: Print formatted output to the console.
    print(f"‖p‖ ‖q‖ = {result_parallel['right_side']:.4f}")  # EN: Print formatted output to the console.
    print(f"等號成立？ {result_parallel['equality']}")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 三角不等式
    # ========================================
    print_separator("8. 三角不等式")  # EN: Call print_separator(...) to perform an operation.

    x = [3.0, 0.0]  # EN: Assign x from expression: [3.0, 0.0].
    y = [0.0, 4.0]  # EN: Assign y from expression: [0.0, 4.0].

    print_vector("x", x)  # EN: Call print_vector(...) to perform an operation.
    print_vector("y", y)  # EN: Call print_vector(...) to perform an operation.

    result = verify_triangle_inequality(x, y)  # EN: Assign result from expression: verify_triangle_inequality(x, y).
    print(f"\n‖x + y‖ = {result['left_side']:.4f}")  # EN: Print formatted output to the console.
    print(f"‖x‖ + ‖y‖ = {result['right_side']:.4f}")  # EN: Print formatted output to the console.
    print(f"‖x + y‖ ≤ ‖x‖ + ‖y‖？ {result['satisfied']}")  # EN: Print formatted output to the console.
    print("\n幾何意義：三角形兩邊之和 ≥ 第三邊")  # EN: Print formatted output to the console.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
    print("""
內積與正交性的核心公式：

1. 內積：x · y = Σ xᵢyᵢ

2. 長度：‖x‖ = √(x · x)

3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖)

4. 正交：x ⊥ y ⟺ x · y = 0

5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ

6. Cauchy-Schwarz：|x · y| ≤ ‖x‖ ‖y‖

7. 三角不等式：‖x + y‖ ≤ ‖x‖ + ‖y‖
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
