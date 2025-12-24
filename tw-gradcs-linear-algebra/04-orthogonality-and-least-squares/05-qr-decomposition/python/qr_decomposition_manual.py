"""
QR 分解 - 手刻版本 (QR Decomposition - Manual Implementation)

本程式示範：
1. Gram-Schmidt QR 分解
2. 用 QR 解最小平方問題
3. 驗證 QR 分解的性質
"""

from typing import List, Tuple
import math


def print_separator(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ========================================
# 基本向量運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:
    return sum(xi * yi for xi, yi in zip(x, y))


def vector_norm(x: List[float]) -> float:
    return math.sqrt(dot_product(x, x))


def scalar_multiply(c: float, x: List[float]) -> List[float]:
    return [c * xi for xi in x]


def vector_subtract(x: List[float], y: List[float]) -> List[float]:
    return [xi - yi for xi, yi in zip(x, y)]


def get_column(A: List[List[float]], j: int) -> List[float]:
    """取得矩陣的第 j 行"""
    return [A[i][j] for i in range(len(A))]


# ========================================
# QR 分解
# ========================================

def qr_decomposition(A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Gram-Schmidt QR 分解

    A = QR
    Q: m×n 標準正交矩陣
    R: n×n 上三角矩陣
    """
    m = len(A)
    n = len(A[0])

    # Q 以行向量形式存儲
    Q = [[0.0] * n for _ in range(m)]
    R = [[0.0] * n for _ in range(n)]

    for j in range(n):
        # 取得 A 的第 j 行
        v = get_column(A, j)

        # 減去前面所有 q 向量的投影
        for i in range(j):
            qi = get_column(Q, i)
            R[i][j] = dot_product(qi, get_column(A, j))
            proj = scalar_multiply(R[i][j], qi)
            v = vector_subtract(v, proj)

        # 標準化
        R[j][j] = vector_norm(v)

        if R[j][j] > 1e-10:
            for i in range(m):
                Q[i][j] = v[i] / R[j][j]

    return Q, R


def solve_upper_triangular(R: List[List[float]], b: List[float]) -> List[float]:
    """回代法解上三角方程組 Rx = b"""
    n = len(b)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= R[i][j] * x[j]
        x[i] /= R[i][i]

    return x


def qr_least_squares(A: List[List[float]], b: List[float]) -> List[float]:
    """用 QR 分解解最小平方問題"""
    Q, R = qr_decomposition(A)

    # Qᵀb
    m, n = len(Q), len(Q[0])
    Qt_b = []
    for j in range(n):
        qj = get_column(Q, j)
        Qt_b.append(dot_product(qj, b))

    # 解 Rx = Qᵀb
    x = solve_upper_triangular(R, Qt_b)

    return x


# ========================================
# 驗證函數
# ========================================

def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩陣乘法"""
    m, k, n = len(A), len(B), len(B[0])
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for p in range(k):
                result[i][j] += A[i][p] * B[p][j]
    return result


def transpose(A: List[List[float]]) -> List[List[float]]:
    """矩陣轉置"""
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def print_vector(name: str, v: List[float]) -> None:
    formatted = [f"{x:.4f}" for x in v]
    print(f"{name} = [{', '.join(formatted)}]")


def print_matrix(name: str, M: List[List[float]]) -> None:
    print(f"{name} =")
    for row in M:
        formatted = [f"{x:8.4f}" for x in row]
        print(f"  [{', '.join(formatted)}]")


def main():
    print_separator("QR 分解示範（手刻版）\nQR Decomposition Demo (Manual)")

    # ========================================
    # 1. 基本 QR 分解
    # ========================================
    print_separator("1. 基本 QR 分解")

    A = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ]

    print("輸入矩陣 A：")
    print_matrix("A", A)

    Q, R = qr_decomposition(A)

    print("\nQR 分解結果：")
    print_matrix("Q", Q)
    print_matrix("\nR", R)

    # 驗證 QᵀQ = I
    QT = transpose(Q)
    QTQ = matrix_multiply(QT, Q)
    print("\n驗證 QᵀQ = I：")
    print_matrix("QᵀQ", QTQ)

    # 驗證 A = QR
    QR = matrix_multiply(Q, R)
    print("\n驗證 A = QR：")
    print_matrix("QR", QR)

    # ========================================
    # 2. 用 QR 解最小平方
    # ========================================
    print_separator("2. 用 QR 解最小平方")

    # 數據
    t = [0.0, 1.0, 2.0]
    b = [1.0, 3.0, 4.0]

    print("數據點：")
    for ti, bi in zip(t, b):
        print(f"  ({ti}, {bi})")

    # 設計矩陣
    A_ls = [[1.0, ti] for ti in t]
    print("\n設計矩陣 A：")
    print_matrix("A", A_ls)
    print_vector("觀測值 b", b)

    # QR 分解
    Q_ls, R_ls = qr_decomposition(A_ls)
    print_matrix("\nQ", Q_ls)
    print_matrix("R", R_ls)

    # 解最小平方
    x = qr_least_squares(A_ls, b)
    print_vector("\n解 x", x)
    print(f"\n最佳直線：y = {x[0]:.4f} + {x[1]:.4f}t")

    # ========================================
    # 3. 另一個例子
    # ========================================
    print_separator("3. 3×3 矩陣的 QR 分解")

    A2 = [
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ]

    print("輸入矩陣 A：")
    print_matrix("A", A2)

    Q2, R2 = qr_decomposition(A2)

    print("\nQR 分解結果：")
    print_matrix("Q", Q2)
    print_matrix("\nR", R2)

    # 總結
    print_separator("總結")
    print("""
QR 分解核心：

1. A = QR
   - Q: 標準正交矩陣 (QᵀQ = I)
   - R: 上三角矩陣

2. Gram-Schmidt 演算法：
   - 對 A 的行向量正交化得到 Q
   - R 的元素是投影係數

3. 用 QR 解最小平方：
   min ‖Ax - b‖²
   → Rx = Qᵀb

4. 優勢：
   - 比正規方程更穩定
   - 避免計算 AᵀA
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
