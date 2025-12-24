"""
Gram-Schmidt 正交化 - 手刻版本 (Gram-Schmidt Process - Manual Implementation)

本程式示範：
1. Classical Gram-Schmidt (CGS)
2. Modified Gram-Schmidt (MGS)
3. 驗證正交性
4. QR 分解

不使用 NumPy 的線性代數函數，手刻實作以理解底層計算。
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
    """內積"""  # EN: Execute statement: """內積""".
    return sum(xi * yi for xi, yi in zip(x, y))  # EN: Return a value: return sum(xi * yi for xi, yi in zip(x, y)).


def vector_norm(x: List[float]) -> float:  # EN: Define vector_norm and its behavior.
    """向量長度"""  # EN: Execute statement: """向量長度""".
    return math.sqrt(dot_product(x, x))  # EN: Return a value: return math.sqrt(dot_product(x, x)).


def scalar_multiply(c: float, x: List[float]) -> List[float]:  # EN: Define scalar_multiply and its behavior.
    """純量乘向量"""  # EN: Execute statement: """純量乘向量""".
    return [c * xi for xi in x]  # EN: Return a value: return [c * xi for xi in x].


def vector_subtract(x: List[float], y: List[float]) -> List[float]:  # EN: Define vector_subtract and its behavior.
    """向量減法"""  # EN: Execute statement: """向量減法""".
    return [xi - yi for xi, yi in zip(x, y)]  # EN: Return a value: return [xi - yi for xi, yi in zip(x, y)].


def vector_copy(x: List[float]) -> List[float]:  # EN: Define vector_copy and its behavior.
    """複製向量"""  # EN: Execute statement: """複製向量""".
    return x[:]  # EN: Return a value: return x[:].


def normalize(x: List[float]) -> List[float]:  # EN: Define normalize and its behavior.
    """正規化為單位向量"""  # EN: Execute statement: """正規化為單位向量""".
    norm = vector_norm(x)  # EN: Assign norm from expression: vector_norm(x).
    if norm < 1e-10:  # EN: Branch on a condition: if norm < 1e-10:.
        raise ValueError("零向量無法正規化")  # EN: Raise an exception: raise ValueError("零向量無法正規化").
    return [xi / norm for xi in x]  # EN: Return a value: return [xi / norm for xi in x].


# ========================================
# Gram-Schmidt 演算法
# ========================================

def classical_gram_schmidt(A: List[List[float]]) -> List[List[float]]:  # EN: Define classical_gram_schmidt and its behavior.
    """
    Classical Gram-Schmidt (CGS)

    輸入：向量組 A（每行是一個向量）
    輸出：正交向量組 Q
    """  # EN: Execute statement: """.
    n = len(A)  # EN: Assign n from expression: len(A).
    m = len(A[0]) if n > 0 else 0  # EN: Assign m from expression: len(A[0]) if n > 0 else 0.

    Q = []  # EN: Assign Q from expression: [].

    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        # 從 aᵢ 開始
        qi = vector_copy(A[i])  # EN: Assign qi from expression: vector_copy(A[i]).

        # 減去在前面所有向量上的投影
        for j in range(i):  # EN: Iterate with a for-loop: for j in range(i):.
            qj = Q[j]  # EN: Assign qj from expression: Q[j].
            # proj_qj(qi) = (qj · qi / qj · qj) * qj
            coeff = dot_product(qj, qi) / dot_product(qj, qj)  # EN: Assign coeff from expression: dot_product(qj, qi) / dot_product(qj, qj).
            proj = scalar_multiply(coeff, qj)  # EN: Assign proj from expression: scalar_multiply(coeff, qj).
            qi = vector_subtract(qi, proj)  # EN: Assign qi from expression: vector_subtract(qi, proj).

        Q.append(qi)  # EN: Execute statement: Q.append(qi).

    return Q  # EN: Return a value: return Q.


def modified_gram_schmidt(A: List[List[float]]) -> List[List[float]]:  # EN: Define modified_gram_schmidt and its behavior.
    """
    Modified Gram-Schmidt (MGS) - 數值更穩定

    輸入：向量組 A（每行是一個向量）
    輸出：正交向量組 Q
    """  # EN: Execute statement: """.
    n = len(A)  # EN: Assign n from expression: len(A).

    # 複製輸入
    Q = [vector_copy(A[i]) for i in range(n)]  # EN: Assign Q from expression: [vector_copy(A[i]) for i in range(n)].

    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        # 先減去前面所有向量的投影（用更新後的 Q[i]）
        for j in range(i):  # EN: Iterate with a for-loop: for j in range(i):.
            # proj_Qj(Qi) = (Qj · Qi / Qj · Qj) * Qj
            coeff = dot_product(Q[j], Q[i]) / dot_product(Q[j], Q[j])  # EN: Assign coeff from expression: dot_product(Q[j], Q[i]) / dot_product(Q[j], Q[j]).
            proj = scalar_multiply(coeff, Q[j])  # EN: Assign proj from expression: scalar_multiply(coeff, Q[j]).
            Q[i] = vector_subtract(Q[i], proj)  # EN: Execute statement: Q[i] = vector_subtract(Q[i], proj).

    return Q  # EN: Return a value: return Q.


def gram_schmidt_normalized(A: List[List[float]], modified: bool = True) -> List[List[float]]:  # EN: Define gram_schmidt_normalized and its behavior.
    """
    Gram-Schmidt 正交化並標準化

    輸入：向量組 A
    輸出：標準正交向量組 E
    """  # EN: Execute statement: """.
    if modified:  # EN: Branch on a condition: if modified:.
        Q = modified_gram_schmidt(A)  # EN: Assign Q from expression: modified_gram_schmidt(A).
    else:  # EN: Execute the fallback branch when prior conditions are false.
        Q = classical_gram_schmidt(A)  # EN: Assign Q from expression: classical_gram_schmidt(A).

    # 標準化
    E = [normalize(q) for q in Q]  # EN: Assign E from expression: [normalize(q) for q in Q].
    return E  # EN: Return a value: return E.


# ========================================
# QR 分解
# ========================================

def qr_decomposition(A: List[List[float]]) -> tuple:  # EN: Define qr_decomposition and its behavior.
    """
    QR 分解：A = QR

    輸入：矩陣 A（行向量作為列）
    輸出：Q（標準正交矩陣）, R（上三角矩陣）
    """  # EN: Execute statement: """.
    n = len(A)  # EN: Assign n from expression: len(A).
    m = len(A[0]) if n > 0 else 0  # EN: Assign m from expression: len(A[0]) if n > 0 else 0.

    # Gram-Schmidt 得到正交向量
    Q_orthogonal = modified_gram_schmidt(A)  # EN: Assign Q_orthogonal from expression: modified_gram_schmidt(A).

    # 計算 R（在標準化之前）
    R = [[0.0] * n for _ in range(n)]  # EN: Assign R from expression: [[0.0] * n for _ in range(n)].

    # Q 標準化，同時記錄係數到 R
    Q = []  # EN: Assign Q from expression: [].
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        # R[i][i] = ‖qᵢ‖
        R[i][i] = vector_norm(Q_orthogonal[i])  # EN: Execute statement: R[i][i] = vector_norm(Q_orthogonal[i]).
        # 標準化
        qi_normalized = normalize(Q_orthogonal[i])  # EN: Assign qi_normalized from expression: normalize(Q_orthogonal[i]).
        Q.append(qi_normalized)  # EN: Execute statement: Q.append(qi_normalized).

        # R[i][j] = eᵢ · aⱼ（i < j）
        for j in range(i + 1, n):  # EN: Iterate with a for-loop: for j in range(i + 1, n):.
            R[i][j] = dot_product(qi_normalized, A[j])  # EN: Execute statement: R[i][j] = dot_product(qi_normalized, A[j]).

    return Q, R  # EN: Return a value: return Q, R.


# ========================================
# 驗證函數
# ========================================

def verify_orthogonality(Q: List[List[float]], tol: float = 1e-10) -> bool:  # EN: Define verify_orthogonality and its behavior.
    """驗證向量組是否正交"""  # EN: Execute statement: """驗證向量組是否正交""".
    n = len(Q)  # EN: Assign n from expression: len(Q).
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        for j in range(i + 1, n):  # EN: Iterate with a for-loop: for j in range(i + 1, n):.
            dot = dot_product(Q[i], Q[j])  # EN: Assign dot from expression: dot_product(Q[i], Q[j]).
            if abs(dot) > tol:  # EN: Branch on a condition: if abs(dot) > tol:.
                return False  # EN: Return a value: return False.
    return True  # EN: Return a value: return True.


def verify_orthonormality(Q: List[List[float]], tol: float = 1e-10) -> bool:  # EN: Define verify_orthonormality and its behavior.
    """驗證向量組是否標準正交"""  # EN: Execute statement: """驗證向量組是否標準正交""".
    n = len(Q)  # EN: Assign n from expression: len(Q).
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        # 檢查長度
        if abs(vector_norm(Q[i]) - 1.0) > tol:  # EN: Branch on a condition: if abs(vector_norm(Q[i]) - 1.0) > tol:.
            return False  # EN: Return a value: return False.
        # 檢查正交
        for j in range(i + 1, n):  # EN: Iterate with a for-loop: for j in range(i + 1, n):.
            if abs(dot_product(Q[i], Q[j])) > tol:  # EN: Branch on a condition: if abs(dot_product(Q[i], Q[j])) > tol:.
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

    print_separator("Gram-Schmidt 正交化示範（手刻版）\nGram-Schmidt Process Demo (Manual)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 基本 Gram-Schmidt
    # ========================================
    print_separator("1. Classical Gram-Schmidt")  # EN: Call print_separator(...) to perform an operation.

    # 輸入向量組
    A = [  # EN: Assign A from expression: [.
        [1.0, 1.0, 0.0],  # EN: Execute statement: [1.0, 1.0, 0.0],.
        [1.0, 0.0, 1.0],  # EN: Execute statement: [1.0, 0.0, 1.0],.
        [0.0, 1.0, 1.0]  # EN: Execute statement: [0.0, 1.0, 1.0].
    ]  # EN: Execute statement: ].

    print("輸入向量組 A：")  # EN: Print formatted output to the console.
    for i, a in enumerate(A, 1):  # EN: Iterate with a for-loop: for i, a in enumerate(A, 1):.
        print_vector(f"a{i}", a)  # EN: Call print_vector(...) to perform an operation.

    # CGS
    Q_cgs = classical_gram_schmidt(A)  # EN: Assign Q_cgs from expression: classical_gram_schmidt(A).

    print("\n正交化結果（CGS）：")  # EN: Print formatted output to the console.
    for i, q in enumerate(Q_cgs, 1):  # EN: Iterate with a for-loop: for i, q in enumerate(Q_cgs, 1):.
        print_vector(f"q{i}", q)  # EN: Call print_vector(...) to perform an operation.
        print(f"    ‖q{i}‖ = {vector_norm(q):.4f}")  # EN: Print formatted output to the console.

    # 驗證正交性
    print(f"\n正交？ {verify_orthogonality(Q_cgs)}")  # EN: Print formatted output to the console.

    # 驗證內積
    print("\n內積驗證：")  # EN: Print formatted output to the console.
    print(f"q₁ · q₂ = {dot_product(Q_cgs[0], Q_cgs[1]):.6f}")  # EN: Print formatted output to the console.
    print(f"q₁ · q₃ = {dot_product(Q_cgs[0], Q_cgs[2]):.6f}")  # EN: Print formatted output to the console.
    print(f"q₂ · q₃ = {dot_product(Q_cgs[1], Q_cgs[2]):.6f}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. Modified Gram-Schmidt
    # ========================================
    print_separator("2. Modified Gram-Schmidt (MGS)")  # EN: Call print_separator(...) to perform an operation.

    Q_mgs = modified_gram_schmidt(A)  # EN: Assign Q_mgs from expression: modified_gram_schmidt(A).

    print("正交化結果（MGS）：")  # EN: Print formatted output to the console.
    for i, q in enumerate(Q_mgs, 1):  # EN: Iterate with a for-loop: for i, q in enumerate(Q_mgs, 1):.
        print_vector(f"q{i}", q)  # EN: Call print_vector(...) to perform an operation.

    print(f"\n正交？ {verify_orthogonality(Q_mgs)}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 標準正交化
    # ========================================
    print_separator("3. 標準正交化")  # EN: Call print_separator(...) to perform an operation.

    E = gram_schmidt_normalized(A)  # EN: Assign E from expression: gram_schmidt_normalized(A).

    print("標準正交向量組 E：")  # EN: Print formatted output to the console.
    for i, e in enumerate(E, 1):  # EN: Iterate with a for-loop: for i, e in enumerate(E, 1):.
        print_vector(f"e{i}", e)  # EN: Call print_vector(...) to perform an operation.
        print(f"    ‖e{i}‖ = {vector_norm(e):.4f}")  # EN: Print formatted output to the console.

    print(f"\n標準正交？ {verify_orthonormality(E)}")  # EN: Print formatted output to the console.

    # ========================================
    # 4. QR 分解
    # ========================================
    print_separator("4. QR 分解")  # EN: Call print_separator(...) to perform an operation.

    Q, R = qr_decomposition(A)  # EN: Execute statement: Q, R = qr_decomposition(A).

    print("Q（標準正交矩陣）：")  # EN: Print formatted output to the console.
    print_matrix("Q", Q)  # EN: Call print_matrix(...) to perform an operation.

    print("\nR（上三角矩陣）：")  # EN: Print formatted output to the console.
    print_matrix("R", R)  # EN: Call print_matrix(...) to perform an operation.

    # 驗證 A = QR
    print("\n驗證 A = QR：")  # EN: Print formatted output to the console.
    # 重建 A
    for i in range(len(A)):  # EN: Iterate with a for-loop: for i in range(len(A)):.
        reconstructed = [0.0] * len(A[0])  # EN: Assign reconstructed from expression: [0.0] * len(A[0]).
        for j in range(len(Q)):  # EN: Iterate with a for-loop: for j in range(len(Q)):.
            for k in range(len(A[0])):  # EN: Iterate with a for-loop: for k in range(len(A[0])):.
                reconstructed[k] += Q[j][k] * R[j][i]  # EN: Execute statement: reconstructed[k] += Q[j][k] * R[j][i].
        print_vector(f"QR 第 {i+1} 行", reconstructed)  # EN: Call print_vector(...) to perform an operation.

    # ========================================
    # 5. 2D 範例
    # ========================================
    print_separator("5. 2D 範例")  # EN: Call print_separator(...) to perform an operation.

    A2 = [  # EN: Assign A2 from expression: [.
        [3.0, 1.0],  # EN: Execute statement: [3.0, 1.0],.
        [2.0, 2.0]  # EN: Execute statement: [2.0, 2.0].
    ]  # EN: Execute statement: ].

    print("輸入向量組：")  # EN: Print formatted output to the console.
    for i, a in enumerate(A2, 1):  # EN: Iterate with a for-loop: for i, a in enumerate(A2, 1):.
        print_vector(f"a{i}", a)  # EN: Call print_vector(...) to perform an operation.

    Q2 = classical_gram_schmidt(A2)  # EN: Assign Q2 from expression: classical_gram_schmidt(A2).

    print("\n正交化結果：")  # EN: Print formatted output to the console.
    for i, q in enumerate(Q2, 1):  # EN: Iterate with a for-loop: for i, q in enumerate(Q2, 1):.
        print_vector(f"q{i}", q)  # EN: Call print_vector(...) to perform an operation.

    print("\n計算過程：")  # EN: Print formatted output to the console.
    print("q₁ = a₁")  # EN: Print formatted output to the console.
    proj = scalar_multiply(  # EN: Assign proj from expression: scalar_multiply(.
        dot_product(Q2[0], A2[1]) / dot_product(Q2[0], Q2[0]),  # EN: Call dot_product(...) to perform an operation.
        Q2[0]  # EN: Execute statement: Q2[0].
    )  # EN: Execute statement: ).
    print_vector("proj_q₁(a₂)", proj)  # EN: Call print_vector(...) to perform an operation.
    print("q₂ = a₂ - proj_q₁(a₂)")  # EN: Print formatted output to the console.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
    print("""
Gram-Schmidt 正交化：

1. 投影公式：
   proj_q(a) = (qᵀa / qᵀq) q

2. 正交化：
   q₁ = a₁
   qₖ = aₖ - Σᵢ₌₁ᵏ⁻¹ proj_{qᵢ}(aₖ)

3. 標準化：
   eᵢ = qᵢ / ‖qᵢ‖

4. QR 分解：
   A = QR
   Q: 標準正交矩陣
   R: 上三角矩陣

5. CGS vs MGS：
   MGS 數值更穩定，因為每步都用更新後的向量
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
