"""
投影 - 手刻版本 (Projections - Manual Implementation)

本程式示範：
1. 投影到直線
2. 投影矩陣及其性質
3. 投影到子空間
4. 誤差向量的正交性驗證

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
# 基本向量和矩陣運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:  # EN: Define dot_product and its behavior.
    """內積"""  # EN: Execute statement: """內積""".
    return sum(xi * yi for xi, yi in zip(x, y))  # EN: Return a value: return sum(xi * yi for xi, yi in zip(x, y)).


def vector_norm(x: List[float]) -> float:  # EN: Define vector_norm and its behavior.
    """向量長度"""  # EN: Execute statement: """向量長度""".
    return math.sqrt(dot_product(x, x))  # EN: Return a value: return math.sqrt(dot_product(x, x)).


def scalar_multiply(c: float, x: List[float]) -> List[float]:  # EN: Define scalar_multiply and its behavior.
    """純量乘向量"""  # EN: Execute statement: """純量乘向量""".
    return [c * xi for xi in x]  # EN: Return a value: return [c * xi for xi in x].


def vector_add(x: List[float], y: List[float]) -> List[float]:  # EN: Define vector_add and its behavior.
    """向量加法"""  # EN: Execute statement: """向量加法""".
    return [xi + yi for xi, yi in zip(x, y)]  # EN: Return a value: return [xi + yi for xi, yi in zip(x, y)].


def vector_subtract(x: List[float], y: List[float]) -> List[float]:  # EN: Define vector_subtract and its behavior.
    """向量減法"""  # EN: Execute statement: """向量減法""".
    return [xi - yi for xi, yi in zip(x, y)]  # EN: Return a value: return [xi - yi for xi, yi in zip(x, y)].


def outer_product(x: List[float], y: List[float]) -> List[List[float]]:  # EN: Define outer_product and its behavior.
    """外積（產生矩陣）"""  # EN: Execute statement: """外積（產生矩陣）""".
    m, n = len(x), len(y)  # EN: Execute statement: m, n = len(x), len(y).
    return [[x[i] * y[j] for j in range(n)] for i in range(m)]  # EN: Return a value: return [[x[i] * y[j] for j in range(n)] for i in range(m)].


def matrix_transpose(A: List[List[float]]) -> List[List[float]]:  # EN: Define matrix_transpose and its behavior.
    """矩陣轉置"""  # EN: Execute statement: """矩陣轉置""".
    m, n = len(A), len(A[0])  # EN: Execute statement: m, n = len(A), len(A[0]).
    return [[A[i][j] for i in range(m)] for j in range(n)]  # EN: Return a value: return [[A[i][j] for i in range(m)] for j in range(n)].


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:  # EN: Define matrix_multiply and its behavior.
    """矩陣乘法"""  # EN: Execute statement: """矩陣乘法""".
    m, k, n = len(A), len(B), len(B[0])  # EN: Execute statement: m, k, n = len(A), len(B), len(B[0]).
    result = [[0.0] * n for _ in range(m)]  # EN: Assign result from expression: [[0.0] * n for _ in range(m)].
    for i in range(m):  # EN: Iterate with a for-loop: for i in range(m):.
        for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
            for p in range(k):  # EN: Iterate with a for-loop: for p in range(k):.
                result[i][j] += A[i][p] * B[p][j]  # EN: Execute statement: result[i][j] += A[i][p] * B[p][j].
    return result  # EN: Return a value: return result.


def matrix_vector_multiply(A: List[List[float]], x: List[float]) -> List[float]:  # EN: Define matrix_vector_multiply and its behavior.
    """矩陣乘向量"""  # EN: Execute statement: """矩陣乘向量""".
    m = len(A)  # EN: Assign m from expression: len(A).
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(m)]  # EN: Return a value: return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(m)].


def matrix_scalar_multiply(c: float, A: List[List[float]]) -> List[List[float]]:  # EN: Define matrix_scalar_multiply and its behavior.
    """純量乘矩陣"""  # EN: Execute statement: """純量乘矩陣""".
    return [[c * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]  # EN: Return a value: return [[c * A[i][j] for j in range(len(A[0]))] for i in range(len(A))].


def identity_matrix(n: int) -> List[List[float]]:  # EN: Define identity_matrix and its behavior.
    """單位矩陣"""  # EN: Execute statement: """單位矩陣""".
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]  # EN: Return a value: return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)].


def matrix_subtract(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:  # EN: Define matrix_subtract and its behavior.
    """矩陣減法"""  # EN: Execute statement: """矩陣減法""".
    m, n = len(A), len(A[0])  # EN: Execute statement: m, n = len(A), len(A[0]).
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)]  # EN: Return a value: return [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)].


def inverse_2x2(A: List[List[float]]) -> List[List[float]]:  # EN: Define inverse_2x2 and its behavior.
    """2x2 矩陣的逆"""  # EN: Execute statement: """2x2 矩陣的逆""".
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]  # EN: Assign det from expression: A[0][0] * A[1][1] - A[0][1] * A[1][0].
    if abs(det) < 1e-10:  # EN: Branch on a condition: if abs(det) < 1e-10:.
        raise ValueError("矩陣不可逆")  # EN: Raise an exception: raise ValueError("矩陣不可逆").
    return [  # EN: Return a value: return [.
        [A[1][1] / det, -A[0][1] / det],  # EN: Execute statement: [A[1][1] / det, -A[0][1] / det],.
        [-A[1][0] / det, A[0][0] / det]  # EN: Execute statement: [-A[1][0] / det, A[0][0] / det].
    ]  # EN: Execute statement: ].


# ========================================
# 投影函數
# ========================================

def project_onto_line(b: List[float], a: List[float]) -> dict:  # EN: Define project_onto_line and its behavior.
    """
    計算向量 b 到直線（方向為 a）的投影

    公式：p = (aᵀb / aᵀa) * a
    """  # EN: Execute statement: """.
    aTb = dot_product(a, b)  # EN: Assign aTb from expression: dot_product(a, b).
    aTa = dot_product(a, a)  # EN: Assign aTa from expression: dot_product(a, a).

    if aTa < 1e-10:  # EN: Branch on a condition: if aTa < 1e-10:.
        raise ValueError("方向向量不能為零向量")  # EN: Raise an exception: raise ValueError("方向向量不能為零向量").

    # 投影係數
    x_hat = aTb / aTa  # EN: Assign x_hat from expression: aTb / aTa.

    # 投影向量
    p = scalar_multiply(x_hat, a)  # EN: Assign p from expression: scalar_multiply(x_hat, a).

    # 誤差向量
    e = vector_subtract(b, p)  # EN: Assign e from expression: vector_subtract(b, p).

    return {  # EN: Return a value: return {.
        'x_hat': x_hat,  # EN: Execute statement: 'x_hat': x_hat,.
        'projection': p,  # EN: Execute statement: 'projection': p,.
        'error': e,  # EN: Execute statement: 'error': e,.
        'error_norm': vector_norm(e)  # EN: Execute statement: 'error_norm': vector_norm(e).
    }  # EN: Execute statement: }.


def projection_matrix_line(a: List[float]) -> List[List[float]]:  # EN: Define projection_matrix_line and its behavior.
    """
    計算投影到直線的投影矩陣

    公式：P = aaᵀ / (aᵀa)
    """  # EN: Execute statement: """.
    aTa = dot_product(a, a)  # EN: Assign aTa from expression: dot_product(a, a).
    if aTa < 1e-10:  # EN: Branch on a condition: if aTa < 1e-10:.
        raise ValueError("方向向量不能為零向量")  # EN: Raise an exception: raise ValueError("方向向量不能為零向量").

    # aaᵀ 是外積（矩陣）
    aaT = outer_product(a, a)  # EN: Assign aaT from expression: outer_product(a, a).

    # P = aaᵀ / (aᵀa)
    P = matrix_scalar_multiply(1.0 / aTa, aaT)  # EN: Assign P from expression: matrix_scalar_multiply(1.0 / aTa, aaT).

    return P  # EN: Return a value: return P.


def project_onto_subspace(b: List[float], A: List[List[float]]) -> dict:  # EN: Define project_onto_subspace and its behavior.
    """
    計算向量 b 到子空間 C(A) 的投影

    公式：
        x̂ = (AᵀA)⁻¹ Aᵀb
        p = Ax̂ = A(AᵀA)⁻¹Aᵀb
    """  # EN: Execute statement: """.
    m = len(A)  # EN: Assign m from expression: len(A).
    n = len(A[0])  # EN: Assign n from expression: len(A[0]).

    AT = matrix_transpose(A)  # EN: Assign AT from expression: matrix_transpose(A).

    # AᵀA
    ATA = matrix_multiply(AT, A)  # EN: Assign ATA from expression: matrix_multiply(AT, A).

    # Aᵀb
    ATb = matrix_vector_multiply(AT, b)  # EN: Assign ATb from expression: matrix_vector_multiply(AT, b).

    # 解 AᵀA x̂ = Aᵀb
    # 這裡簡化處理，假設是 2x2 或 1x1
    if n == 1:  # EN: Branch on a condition: if n == 1:.
        x_hat = [ATb[0] / ATA[0][0]]  # EN: Assign x_hat from expression: [ATb[0] / ATA[0][0]].
    elif n == 2:  # EN: Branch on a condition: elif n == 2:.
        ATA_inv = inverse_2x2(ATA)  # EN: Assign ATA_inv from expression: inverse_2x2(ATA).
        x_hat = matrix_vector_multiply(ATA_inv, ATb)  # EN: Assign x_hat from expression: matrix_vector_multiply(ATA_inv, ATb).
    else:  # EN: Execute the fallback branch when prior conditions are false.
        raise ValueError("此簡化版本只支援 n <= 2")  # EN: Raise an exception: raise ValueError("此簡化版本只支援 n <= 2").

    # 投影 p = Ax̂
    p = matrix_vector_multiply(A, x_hat)  # EN: Assign p from expression: matrix_vector_multiply(A, x_hat).

    # 誤差
    e = vector_subtract(b, p)  # EN: Assign e from expression: vector_subtract(b, p).

    return {  # EN: Return a value: return {.
        'x_hat': x_hat,  # EN: Execute statement: 'x_hat': x_hat,.
        'projection': p,  # EN: Execute statement: 'projection': p,.
        'error': e,  # EN: Execute statement: 'error': e,.
        'error_norm': vector_norm(e)  # EN: Execute statement: 'error_norm': vector_norm(e).
    }  # EN: Execute statement: }.


# ========================================
# 驗證函數
# ========================================

def verify_projection_matrix_properties(P: List[List[float]], name: str = "P") -> None:  # EN: Define verify_projection_matrix_properties and its behavior.
    """驗證投影矩陣的性質：對稱性和冪等性"""  # EN: Execute statement: """驗證投影矩陣的性質：對稱性和冪等性""".
    n = len(P)  # EN: Assign n from expression: len(P).

    print(f"\n驗證 {name} 的性質：")  # EN: Print formatted output to the console.

    # 對稱性：Pᵀ = P
    PT = matrix_transpose(P)  # EN: Assign PT from expression: matrix_transpose(P).
    is_symmetric = all(  # EN: Assign is_symmetric from expression: all(.
        abs(P[i][j] - PT[i][j]) < 1e-10  # EN: Call abs(...) to perform an operation.
        for i in range(n) for j in range(n)  # EN: Iterate with a for-loop: for i in range(n) for j in range(n).
    )  # EN: Execute statement: ).
    print(f"  對稱性 ({name}ᵀ = {name})：{is_symmetric}")  # EN: Print formatted output to the console.

    # 冪等性：P² = P
    P2 = matrix_multiply(P, P)  # EN: Assign P2 from expression: matrix_multiply(P, P).
    is_idempotent = all(  # EN: Assign is_idempotent from expression: all(.
        abs(P[i][j] - P2[i][j]) < 1e-10  # EN: Call abs(...) to perform an operation.
        for i in range(n) for j in range(n)  # EN: Iterate with a for-loop: for i in range(n) for j in range(n).
    )  # EN: Execute statement: ).
    print(f"  冪等性 ({name}² = {name})：{is_idempotent}")  # EN: Print formatted output to the console.


def verify_orthogonality(e: List[float], A: List[List[float]]) -> bool:  # EN: Define verify_orthogonality and its behavior.
    """驗證誤差向量 e 垂直於 C(A) 的所有向量"""  # EN: Execute statement: """驗證誤差向量 e 垂直於 C(A) 的所有向量""".
    n_cols = len(A[0])  # EN: Assign n_cols from expression: len(A[0]).
    for j in range(n_cols):  # EN: Iterate with a for-loop: for j in range(n_cols):.
        col = [A[i][j] for i in range(len(A))]  # EN: Assign col from expression: [A[i][j] for i in range(len(A))].
        dot = dot_product(e, col)  # EN: Assign dot from expression: dot_product(e, col).
        if abs(dot) > 1e-10:  # EN: Branch on a condition: if abs(dot) > 1e-10:.
            return False  # EN: Return a value: return False.
    return True  # EN: Return a value: return True.


# ========================================
# 輔助顯示函數
# ========================================

def print_vector(name: str, v: List[float]) -> None:  # EN: Define print_vector and its behavior.
    formatted = [f"{x:.4f}" for x in v]  # EN: Assign formatted from expression: [f"{x:.4f}" for x in v].
    print(f"{name} = [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


def print_matrix(name: str, M: List[List[float]]) -> None:  # EN: Define print_matrix and its behavior.
    print(f"{name} =")  # EN: Print formatted output to the console.
    for row in M:  # EN: Iterate with a for-loop: for row in M:.
        formatted = [f"{x:8.4f}" for x in row]  # EN: Assign formatted from expression: [f"{x:8.4f}" for x in row].
        print(f"  [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("投影示範（手刻版）\nProjection Demo (Manual)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 投影到直線
    # ========================================
    print_separator("1. 投影到直線")  # EN: Call print_separator(...) to perform an operation.

    a = [1.0, 1.0]  # EN: Assign a from expression: [1.0, 1.0].
    b = [2.0, 0.0]  # EN: Assign b from expression: [2.0, 0.0].

    print_vector("方向 a", a)  # EN: Call print_vector(...) to perform an operation.
    print_vector("向量 b", b)  # EN: Call print_vector(...) to perform an operation.

    result = project_onto_line(b, a)  # EN: Assign result from expression: project_onto_line(b, a).

    print(f"\n投影係數 x̂ = (aᵀb)/(aᵀa) = {result['x_hat']:.4f}")  # EN: Print formatted output to the console.
    print_vector("投影 p = x̂a", result['projection'])  # EN: Call print_vector(...) to perform an operation.
    print_vector("誤差 e = b - p", result['error'])  # EN: Call print_vector(...) to perform an operation.

    # 驗證正交性
    e_dot_a = dot_product(result['error'], a)  # EN: Assign e_dot_a from expression: dot_product(result['error'], a).
    print(f"\n驗證 e ⊥ a：e · a = {e_dot_a:.6f}")  # EN: Print formatted output to the console.
    print(f"正交？ {abs(e_dot_a) < 1e-10}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 投影矩陣（到直線）
    # ========================================
    print_separator("2. 投影矩陣（到直線）")  # EN: Call print_separator(...) to perform an operation.

    a = [1.0, 2.0]  # EN: Assign a from expression: [1.0, 2.0].
    print_vector("方向 a", a)  # EN: Call print_vector(...) to perform an operation.

    P = projection_matrix_line(a)  # EN: Assign P from expression: projection_matrix_line(a).
    print_matrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P)  # EN: Call print_matrix(...) to perform an operation.

    verify_projection_matrix_properties(P)  # EN: Call verify_projection_matrix_properties(...) to perform an operation.

    # 用投影矩陣計算投影
    b = [3.0, 4.0]  # EN: Assign b from expression: [3.0, 4.0].
    print_vector("\n向量 b", b)  # EN: Call print_vector(...) to perform an operation.

    p = matrix_vector_multiply(P, b)  # EN: Assign p from expression: matrix_vector_multiply(P, b).
    print_vector("投影 p = Pb", p)  # EN: Call print_vector(...) to perform an operation.

    # ========================================
    # 3. 投影到平面（子空間）
    # ========================================
    print_separator("3. 投影到平面（子空間）")  # EN: Call print_separator(...) to perform an operation.

    # 平面由兩個向量張成
    A = [  # EN: Assign A from expression: [.
        [1.0, 0.0],  # EN: Execute statement: [1.0, 0.0],.
        [0.0, 1.0],  # EN: Execute statement: [0.0, 1.0],.
        [0.0, 0.0]  # EN: Execute statement: [0.0, 0.0].
    ]  # EN: Execute statement: ].
    b = [1.0, 2.0, 3.0]  # EN: Assign b from expression: [1.0, 2.0, 3.0].

    print("矩陣 A（平面的基底）：")  # EN: Print formatted output to the console.
    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print_vector("\n向量 b", b)  # EN: Call print_vector(...) to perform an operation.

    result = project_onto_subspace(b, A)  # EN: Assign result from expression: project_onto_subspace(b, A).

    print_vector("\n係數 x̂ = (AᵀA)⁻¹Aᵀb", result['x_hat'])  # EN: Call print_vector(...) to perform an operation.
    print_vector("投影 p = Ax̂", result['projection'])  # EN: Call print_vector(...) to perform an operation.
    print_vector("誤差 e = b - p", result['error'])  # EN: Call print_vector(...) to perform an operation.
    print(f"誤差長度 ‖e‖ = {result['error_norm']:.4f}")  # EN: Print formatted output to the console.

    # 驗證誤差正交於子空間
    is_orthogonal = verify_orthogonality(result['error'], A)  # EN: Assign is_orthogonal from expression: verify_orthogonality(result['error'], A).
    print(f"\ne 垂直於 C(A)？ {is_orthogonal}")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 另一個子空間例子
    # ========================================
    print_separator("4. 投影到斜平面")  # EN: Call print_separator(...) to perform an operation.

    A = [  # EN: Assign A from expression: [.
        [1.0, 1.0],  # EN: Execute statement: [1.0, 1.0],.
        [1.0, 0.0],  # EN: Execute statement: [1.0, 0.0],.
        [0.0, 1.0]  # EN: Execute statement: [0.0, 1.0].
    ]  # EN: Execute statement: ].
    b = [2.0, 3.0, 4.0]  # EN: Assign b from expression: [2.0, 3.0, 4.0].

    print("矩陣 A：")  # EN: Print formatted output to the console.
    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print_vector("\n向量 b", b)  # EN: Call print_vector(...) to perform an operation.

    result = project_onto_subspace(b, A)  # EN: Assign result from expression: project_onto_subspace(b, A).

    print_vector("\n係數 x̂", result['x_hat'])  # EN: Call print_vector(...) to perform an operation.
    print_vector("投影 p", result['projection'])  # EN: Call print_vector(...) to perform an operation.
    print_vector("誤差 e", result['error'])  # EN: Call print_vector(...) to perform an operation.
    print(f"誤差長度 ‖e‖ = {result['error_norm']:.4f}")  # EN: Print formatted output to the console.

    # 驗證
    is_orthogonal = verify_orthogonality(result['error'], A)  # EN: Assign is_orthogonal from expression: verify_orthogonality(result['error'], A).
    print(f"\ne 垂直於 C(A)？ {is_orthogonal}")  # EN: Print formatted output to the console.

    # ========================================
    # 5. I - P 也是投影矩陣
    # ========================================
    print_separator("5. 補空間投影矩陣 I - P")  # EN: Call print_separator(...) to perform an operation.

    a = [1.0, 0.0]  # EN: Assign a from expression: [1.0, 0.0].
    P = projection_matrix_line(a)  # EN: Assign P from expression: projection_matrix_line(a).

    print_vector("方向 a", a)  # EN: Call print_vector(...) to perform an operation.
    print_matrix("P = aaᵀ/(aᵀa)", P)  # EN: Call print_matrix(...) to perform an operation.

    I = identity_matrix(2)  # EN: Assign I from expression: identity_matrix(2).
    I_minus_P = matrix_subtract(I, P)  # EN: Assign I_minus_P from expression: matrix_subtract(I, P).
    print_matrix("\nI - P", I_minus_P)  # EN: Call print_matrix(...) to perform an operation.

    verify_projection_matrix_properties(I_minus_P, "I-P")  # EN: Call verify_projection_matrix_properties(...) to perform an operation.

    # I - P 投影到 a 的正交補
    b = [3.0, 4.0]  # EN: Assign b from expression: [3.0, 4.0].
    print_vector("\n向量 b", b)  # EN: Call print_vector(...) to perform an operation.

    p = matrix_vector_multiply(P, b)  # EN: Assign p from expression: matrix_vector_multiply(P, b).
    e = matrix_vector_multiply(I_minus_P, b)  # EN: Assign e from expression: matrix_vector_multiply(I_minus_P, b).

    print_vector("Pb（投影到 a）", p)  # EN: Call print_vector(...) to perform an operation.
    print_vector("(I-P)b（投影到 a⊥）", e)  # EN: Call print_vector(...) to perform an operation.
    print_vector("Pb + (I-P)b", vector_add(p, e))  # EN: Call print_vector(...) to perform an operation.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
