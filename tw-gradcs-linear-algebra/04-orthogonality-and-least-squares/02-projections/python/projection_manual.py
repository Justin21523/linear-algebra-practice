"""
投影 - 手刻版本 (Projections - Manual Implementation)

本程式示範：
1. 投影到直線
2. 投影矩陣及其性質
3. 投影到子空間
4. 誤差向量的正交性驗證

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
# 基本向量和矩陣運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:
    """內積"""
    return sum(xi * yi for xi, yi in zip(x, y))


def vector_norm(x: List[float]) -> float:
    """向量長度"""
    return math.sqrt(dot_product(x, x))


def scalar_multiply(c: float, x: List[float]) -> List[float]:
    """純量乘向量"""
    return [c * xi for xi in x]


def vector_add(x: List[float], y: List[float]) -> List[float]:
    """向量加法"""
    return [xi + yi for xi, yi in zip(x, y)]


def vector_subtract(x: List[float], y: List[float]) -> List[float]:
    """向量減法"""
    return [xi - yi for xi, yi in zip(x, y)]


def outer_product(x: List[float], y: List[float]) -> List[List[float]]:
    """外積（產生矩陣）"""
    m, n = len(x), len(y)
    return [[x[i] * y[j] for j in range(n)] for i in range(m)]


def matrix_transpose(A: List[List[float]]) -> List[List[float]]:
    """矩陣轉置"""
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩陣乘法"""
    m, k, n = len(A), len(B), len(B[0])
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for p in range(k):
                result[i][j] += A[i][p] * B[p][j]
    return result


def matrix_vector_multiply(A: List[List[float]], x: List[float]) -> List[float]:
    """矩陣乘向量"""
    m = len(A)
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(m)]


def matrix_scalar_multiply(c: float, A: List[List[float]]) -> List[List[float]]:
    """純量乘矩陣"""
    return [[c * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def identity_matrix(n: int) -> List[List[float]]:
    """單位矩陣"""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matrix_subtract(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩陣減法"""
    m, n = len(A), len(A[0])
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)]


def inverse_2x2(A: List[List[float]]) -> List[List[float]]:
    """2x2 矩陣的逆"""
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(det) < 1e-10:
        raise ValueError("矩陣不可逆")
    return [
        [A[1][1] / det, -A[0][1] / det],
        [-A[1][0] / det, A[0][0] / det]
    ]


# ========================================
# 投影函數
# ========================================

def project_onto_line(b: List[float], a: List[float]) -> dict:
    """
    計算向量 b 到直線（方向為 a）的投影

    公式：p = (aᵀb / aᵀa) * a
    """
    aTb = dot_product(a, b)
    aTa = dot_product(a, a)

    if aTa < 1e-10:
        raise ValueError("方向向量不能為零向量")

    # 投影係數
    x_hat = aTb / aTa

    # 投影向量
    p = scalar_multiply(x_hat, a)

    # 誤差向量
    e = vector_subtract(b, p)

    return {
        'x_hat': x_hat,
        'projection': p,
        'error': e,
        'error_norm': vector_norm(e)
    }


def projection_matrix_line(a: List[float]) -> List[List[float]]:
    """
    計算投影到直線的投影矩陣

    公式：P = aaᵀ / (aᵀa)
    """
    aTa = dot_product(a, a)
    if aTa < 1e-10:
        raise ValueError("方向向量不能為零向量")

    # aaᵀ 是外積（矩陣）
    aaT = outer_product(a, a)

    # P = aaᵀ / (aᵀa)
    P = matrix_scalar_multiply(1.0 / aTa, aaT)

    return P


def project_onto_subspace(b: List[float], A: List[List[float]]) -> dict:
    """
    計算向量 b 到子空間 C(A) 的投影

    公式：
        x̂ = (AᵀA)⁻¹ Aᵀb
        p = Ax̂ = A(AᵀA)⁻¹Aᵀb
    """
    m = len(A)
    n = len(A[0])

    AT = matrix_transpose(A)

    # AᵀA
    ATA = matrix_multiply(AT, A)

    # Aᵀb
    ATb = matrix_vector_multiply(AT, b)

    # 解 AᵀA x̂ = Aᵀb
    # 這裡簡化處理，假設是 2x2 或 1x1
    if n == 1:
        x_hat = [ATb[0] / ATA[0][0]]
    elif n == 2:
        ATA_inv = inverse_2x2(ATA)
        x_hat = matrix_vector_multiply(ATA_inv, ATb)
    else:
        raise ValueError("此簡化版本只支援 n <= 2")

    # 投影 p = Ax̂
    p = matrix_vector_multiply(A, x_hat)

    # 誤差
    e = vector_subtract(b, p)

    return {
        'x_hat': x_hat,
        'projection': p,
        'error': e,
        'error_norm': vector_norm(e)
    }


# ========================================
# 驗證函數
# ========================================

def verify_projection_matrix_properties(P: List[List[float]], name: str = "P") -> None:
    """驗證投影矩陣的性質：對稱性和冪等性"""
    n = len(P)

    print(f"\n驗證 {name} 的性質：")

    # 對稱性：Pᵀ = P
    PT = matrix_transpose(P)
    is_symmetric = all(
        abs(P[i][j] - PT[i][j]) < 1e-10
        for i in range(n) for j in range(n)
    )
    print(f"  對稱性 ({name}ᵀ = {name})：{is_symmetric}")

    # 冪等性：P² = P
    P2 = matrix_multiply(P, P)
    is_idempotent = all(
        abs(P[i][j] - P2[i][j]) < 1e-10
        for i in range(n) for j in range(n)
    )
    print(f"  冪等性 ({name}² = {name})：{is_idempotent}")


def verify_orthogonality(e: List[float], A: List[List[float]]) -> bool:
    """驗證誤差向量 e 垂直於 C(A) 的所有向量"""
    n_cols = len(A[0])
    for j in range(n_cols):
        col = [A[i][j] for i in range(len(A))]
        dot = dot_product(e, col)
        if abs(dot) > 1e-10:
            return False
    return True


# ========================================
# 輔助顯示函數
# ========================================

def print_vector(name: str, v: List[float]) -> None:
    formatted = [f"{x:.4f}" for x in v]
    print(f"{name} = [{', '.join(formatted)}]")


def print_matrix(name: str, M: List[List[float]]) -> None:
    print(f"{name} =")
    for row in M:
        formatted = [f"{x:8.4f}" for x in row]
        print(f"  [{', '.join(formatted)}]")


def main():
    """主程式"""

    print_separator("投影示範（手刻版）\nProjection Demo (Manual)")

    # ========================================
    # 1. 投影到直線
    # ========================================
    print_separator("1. 投影到直線")

    a = [1.0, 1.0]
    b = [2.0, 0.0]

    print_vector("方向 a", a)
    print_vector("向量 b", b)

    result = project_onto_line(b, a)

    print(f"\n投影係數 x̂ = (aᵀb)/(aᵀa) = {result['x_hat']:.4f}")
    print_vector("投影 p = x̂a", result['projection'])
    print_vector("誤差 e = b - p", result['error'])

    # 驗證正交性
    e_dot_a = dot_product(result['error'], a)
    print(f"\n驗證 e ⊥ a：e · a = {e_dot_a:.6f}")
    print(f"正交？ {abs(e_dot_a) < 1e-10}")

    # ========================================
    # 2. 投影矩陣（到直線）
    # ========================================
    print_separator("2. 投影矩陣（到直線）")

    a = [1.0, 2.0]
    print_vector("方向 a", a)

    P = projection_matrix_line(a)
    print_matrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P)

    verify_projection_matrix_properties(P)

    # 用投影矩陣計算投影
    b = [3.0, 4.0]
    print_vector("\n向量 b", b)

    p = matrix_vector_multiply(P, b)
    print_vector("投影 p = Pb", p)

    # ========================================
    # 3. 投影到平面（子空間）
    # ========================================
    print_separator("3. 投影到平面（子空間）")

    # 平面由兩個向量張成
    A = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ]
    b = [1.0, 2.0, 3.0]

    print("矩陣 A（平面的基底）：")
    print_matrix("A", A)
    print_vector("\n向量 b", b)

    result = project_onto_subspace(b, A)

    print_vector("\n係數 x̂ = (AᵀA)⁻¹Aᵀb", result['x_hat'])
    print_vector("投影 p = Ax̂", result['projection'])
    print_vector("誤差 e = b - p", result['error'])
    print(f"誤差長度 ‖e‖ = {result['error_norm']:.4f}")

    # 驗證誤差正交於子空間
    is_orthogonal = verify_orthogonality(result['error'], A)
    print(f"\ne 垂直於 C(A)？ {is_orthogonal}")

    # ========================================
    # 4. 另一個子空間例子
    # ========================================
    print_separator("4. 投影到斜平面")

    A = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ]
    b = [2.0, 3.0, 4.0]

    print("矩陣 A：")
    print_matrix("A", A)
    print_vector("\n向量 b", b)

    result = project_onto_subspace(b, A)

    print_vector("\n係數 x̂", result['x_hat'])
    print_vector("投影 p", result['projection'])
    print_vector("誤差 e", result['error'])
    print(f"誤差長度 ‖e‖ = {result['error_norm']:.4f}")

    # 驗證
    is_orthogonal = verify_orthogonality(result['error'], A)
    print(f"\ne 垂直於 C(A)？ {is_orthogonal}")

    # ========================================
    # 5. I - P 也是投影矩陣
    # ========================================
    print_separator("5. 補空間投影矩陣 I - P")

    a = [1.0, 0.0]
    P = projection_matrix_line(a)

    print_vector("方向 a", a)
    print_matrix("P = aaᵀ/(aᵀa)", P)

    I = identity_matrix(2)
    I_minus_P = matrix_subtract(I, P)
    print_matrix("\nI - P", I_minus_P)

    verify_projection_matrix_properties(I_minus_P, "I-P")

    # I - P 投影到 a 的正交補
    b = [3.0, 4.0]
    print_vector("\n向量 b", b)

    p = matrix_vector_multiply(P, b)
    e = matrix_vector_multiply(I_minus_P, b)

    print_vector("Pb（投影到 a）", p)
    print_vector("(I-P)b（投影到 a⊥）", e)
    print_vector("Pb + (I-P)b", vector_add(p, e))

    # 總結
    print_separator("總結")
    print("""
投影公式：

1. 投影到直線：
   p = (aᵀb / aᵀa) a
   P = aaᵀ / (aᵀa)

2. 投影到子空間：
   p = A(AᵀA)⁻¹Aᵀb
   P = A(AᵀA)⁻¹Aᵀ

3. 投影矩陣性質：
   Pᵀ = P（對稱）
   P² = P（冪等）

4. 正交分解：
   b = Pb + (I-P)b
     = p + e
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
