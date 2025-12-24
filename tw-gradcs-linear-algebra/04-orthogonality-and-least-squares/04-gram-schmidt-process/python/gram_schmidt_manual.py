"""
Gram-Schmidt 正交化 - 手刻版本 (Gram-Schmidt Process - Manual Implementation)

本程式示範：
1. Classical Gram-Schmidt (CGS)
2. Modified Gram-Schmidt (MGS)
3. 驗證正交性
4. QR 分解

不使用 NumPy 的線性代數函數，手刻實作以理解底層計算。
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
    """內積"""
    return sum(xi * yi for xi, yi in zip(x, y))


def vector_norm(x: List[float]) -> float:
    """向量長度"""
    return math.sqrt(dot_product(x, x))


def scalar_multiply(c: float, x: List[float]) -> List[float]:
    """純量乘向量"""
    return [c * xi for xi in x]


def vector_subtract(x: List[float], y: List[float]) -> List[float]:
    """向量減法"""
    return [xi - yi for xi, yi in zip(x, y)]


def vector_copy(x: List[float]) -> List[float]:
    """複製向量"""
    return x[:]


def normalize(x: List[float]) -> List[float]:
    """正規化為單位向量"""
    norm = vector_norm(x)
    if norm < 1e-10:
        raise ValueError("零向量無法正規化")
    return [xi / norm for xi in x]


# ========================================
# Gram-Schmidt 演算法
# ========================================

def classical_gram_schmidt(A: List[List[float]]) -> List[List[float]]:
    """
    Classical Gram-Schmidt (CGS)

    輸入：向量組 A（每行是一個向量）
    輸出：正交向量組 Q
    """
    n = len(A)
    m = len(A[0]) if n > 0 else 0

    Q = []

    for i in range(n):
        # 從 aᵢ 開始
        qi = vector_copy(A[i])

        # 減去在前面所有向量上的投影
        for j in range(i):
            qj = Q[j]
            # proj_qj(qi) = (qj · qi / qj · qj) * qj
            coeff = dot_product(qj, qi) / dot_product(qj, qj)
            proj = scalar_multiply(coeff, qj)
            qi = vector_subtract(qi, proj)

        Q.append(qi)

    return Q


def modified_gram_schmidt(A: List[List[float]]) -> List[List[float]]:
    """
    Modified Gram-Schmidt (MGS) - 數值更穩定

    輸入：向量組 A（每行是一個向量）
    輸出：正交向量組 Q
    """
    n = len(A)

    # 複製輸入
    Q = [vector_copy(A[i]) for i in range(n)]

    for i in range(n):
        # 先減去前面所有向量的投影（用更新後的 Q[i]）
        for j in range(i):
            # proj_Qj(Qi) = (Qj · Qi / Qj · Qj) * Qj
            coeff = dot_product(Q[j], Q[i]) / dot_product(Q[j], Q[j])
            proj = scalar_multiply(coeff, Q[j])
            Q[i] = vector_subtract(Q[i], proj)

    return Q


def gram_schmidt_normalized(A: List[List[float]], modified: bool = True) -> List[List[float]]:
    """
    Gram-Schmidt 正交化並標準化

    輸入：向量組 A
    輸出：標準正交向量組 E
    """
    if modified:
        Q = modified_gram_schmidt(A)
    else:
        Q = classical_gram_schmidt(A)

    # 標準化
    E = [normalize(q) for q in Q]
    return E


# ========================================
# QR 分解
# ========================================

def qr_decomposition(A: List[List[float]]) -> tuple:
    """
    QR 分解：A = QR

    輸入：矩陣 A（行向量作為列）
    輸出：Q（標準正交矩陣）, R（上三角矩陣）
    """
    n = len(A)
    m = len(A[0]) if n > 0 else 0

    # Gram-Schmidt 得到正交向量
    Q_orthogonal = modified_gram_schmidt(A)

    # 計算 R（在標準化之前）
    R = [[0.0] * n for _ in range(n)]

    # Q 標準化，同時記錄係數到 R
    Q = []
    for i in range(n):
        # R[i][i] = ‖qᵢ‖
        R[i][i] = vector_norm(Q_orthogonal[i])
        # 標準化
        qi_normalized = normalize(Q_orthogonal[i])
        Q.append(qi_normalized)

        # R[i][j] = eᵢ · aⱼ（i < j）
        for j in range(i + 1, n):
            R[i][j] = dot_product(qi_normalized, A[j])

    return Q, R


# ========================================
# 驗證函數
# ========================================

def verify_orthogonality(Q: List[List[float]], tol: float = 1e-10) -> bool:
    """驗證向量組是否正交"""
    n = len(Q)
    for i in range(n):
        for j in range(i + 1, n):
            dot = dot_product(Q[i], Q[j])
            if abs(dot) > tol:
                return False
    return True


def verify_orthonormality(Q: List[List[float]], tol: float = 1e-10) -> bool:
    """驗證向量組是否標準正交"""
    n = len(Q)
    for i in range(n):
        # 檢查長度
        if abs(vector_norm(Q[i]) - 1.0) > tol:
            return False
        # 檢查正交
        for j in range(i + 1, n):
            if abs(dot_product(Q[i], Q[j])) > tol:
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

    print_separator("Gram-Schmidt 正交化示範（手刻版）\nGram-Schmidt Process Demo (Manual)")

    # ========================================
    # 1. 基本 Gram-Schmidt
    # ========================================
    print_separator("1. Classical Gram-Schmidt")

    # 輸入向量組
    A = [
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ]

    print("輸入向量組 A：")
    for i, a in enumerate(A, 1):
        print_vector(f"a{i}", a)

    # CGS
    Q_cgs = classical_gram_schmidt(A)

    print("\n正交化結果（CGS）：")
    for i, q in enumerate(Q_cgs, 1):
        print_vector(f"q{i}", q)
        print(f"    ‖q{i}‖ = {vector_norm(q):.4f}")

    # 驗證正交性
    print(f"\n正交？ {verify_orthogonality(Q_cgs)}")

    # 驗證內積
    print("\n內積驗證：")
    print(f"q₁ · q₂ = {dot_product(Q_cgs[0], Q_cgs[1]):.6f}")
    print(f"q₁ · q₃ = {dot_product(Q_cgs[0], Q_cgs[2]):.6f}")
    print(f"q₂ · q₃ = {dot_product(Q_cgs[1], Q_cgs[2]):.6f}")

    # ========================================
    # 2. Modified Gram-Schmidt
    # ========================================
    print_separator("2. Modified Gram-Schmidt (MGS)")

    Q_mgs = modified_gram_schmidt(A)

    print("正交化結果（MGS）：")
    for i, q in enumerate(Q_mgs, 1):
        print_vector(f"q{i}", q)

    print(f"\n正交？ {verify_orthogonality(Q_mgs)}")

    # ========================================
    # 3. 標準正交化
    # ========================================
    print_separator("3. 標準正交化")

    E = gram_schmidt_normalized(A)

    print("標準正交向量組 E：")
    for i, e in enumerate(E, 1):
        print_vector(f"e{i}", e)
        print(f"    ‖e{i}‖ = {vector_norm(e):.4f}")

    print(f"\n標準正交？ {verify_orthonormality(E)}")

    # ========================================
    # 4. QR 分解
    # ========================================
    print_separator("4. QR 分解")

    Q, R = qr_decomposition(A)

    print("Q（標準正交矩陣）：")
    print_matrix("Q", Q)

    print("\nR（上三角矩陣）：")
    print_matrix("R", R)

    # 驗證 A = QR
    print("\n驗證 A = QR：")
    # 重建 A
    for i in range(len(A)):
        reconstructed = [0.0] * len(A[0])
        for j in range(len(Q)):
            for k in range(len(A[0])):
                reconstructed[k] += Q[j][k] * R[j][i]
        print_vector(f"QR 第 {i+1} 行", reconstructed)

    # ========================================
    # 5. 2D 範例
    # ========================================
    print_separator("5. 2D 範例")

    A2 = [
        [3.0, 1.0],
        [2.0, 2.0]
    ]

    print("輸入向量組：")
    for i, a in enumerate(A2, 1):
        print_vector(f"a{i}", a)

    Q2 = classical_gram_schmidt(A2)

    print("\n正交化結果：")
    for i, q in enumerate(Q2, 1):
        print_vector(f"q{i}", q)

    print("\n計算過程：")
    print("q₁ = a₁")
    proj = scalar_multiply(
        dot_product(Q2[0], A2[1]) / dot_product(Q2[0], Q2[0]),
        Q2[0]
    )
    print_vector("proj_q₁(a₂)", proj)
    print("q₂ = a₂ - proj_q₁(a₂)")

    # 總結
    print_separator("總結")
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
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
