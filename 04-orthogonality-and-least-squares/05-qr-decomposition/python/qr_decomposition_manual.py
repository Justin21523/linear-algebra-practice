"""
QR 分解 - 手刻版本 (QR Decomposition - Manual Implementation)

本程式示範：
1. Gram-Schmidt QR 分解
2. 用 QR 解最小平方問題
3. 驗證 QR 分解的性質
"""  # EN: Execute statement: """.

from typing import List, Tuple  # EN: Import symbol(s) from a module: from typing import List, Tuple.
import math  # EN: Import module(s): import math.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


# ========================================
# 基本向量運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:  # EN: Define dot_product and its behavior.
    return sum(xi * yi for xi, yi in zip(x, y))  # EN: Return a value: return sum(xi * yi for xi, yi in zip(x, y)).


def vector_norm(x: List[float]) -> float:  # EN: Define vector_norm and its behavior.
    return math.sqrt(dot_product(x, x))  # EN: Return a value: return math.sqrt(dot_product(x, x)).


def scalar_multiply(c: float, x: List[float]) -> List[float]:  # EN: Define scalar_multiply and its behavior.
    return [c * xi for xi in x]  # EN: Return a value: return [c * xi for xi in x].


def vector_subtract(x: List[float], y: List[float]) -> List[float]:  # EN: Define vector_subtract and its behavior.
    return [xi - yi for xi, yi in zip(x, y)]  # EN: Return a value: return [xi - yi for xi, yi in zip(x, y)].


def get_column(A: List[List[float]], j: int) -> List[float]:  # EN: Define get_column and its behavior.
    """取得矩陣的第 j 行"""  # EN: Execute statement: """取得矩陣的第 j 行""".
    return [A[i][j] for i in range(len(A))]  # EN: Return a value: return [A[i][j] for i in range(len(A))].


# ========================================
# QR 分解
# ========================================

def qr_decomposition(A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:  # EN: Define qr_decomposition and its behavior.
    """
    Gram-Schmidt QR 分解

    A = QR
    Q: m×n 標準正交矩陣
    R: n×n 上三角矩陣
    """  # EN: Execute statement: """.
    m = len(A)  # EN: Assign m from expression: len(A).
    n = len(A[0])  # EN: Assign n from expression: len(A[0]).

    # Q 以行向量形式存儲
    Q = [[0.0] * n for _ in range(m)]  # EN: Assign Q from expression: [[0.0] * n for _ in range(m)].
    R = [[0.0] * n for _ in range(n)]  # EN: Assign R from expression: [[0.0] * n for _ in range(n)].

    for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
        # 取得 A 的第 j 行
        v = get_column(A, j)  # EN: Assign v from expression: get_column(A, j).

        # 減去前面所有 q 向量的投影
        for i in range(j):  # EN: Iterate with a for-loop: for i in range(j):.
            qi = get_column(Q, i)  # EN: Assign qi from expression: get_column(Q, i).
            R[i][j] = dot_product(qi, get_column(A, j))  # EN: Execute statement: R[i][j] = dot_product(qi, get_column(A, j)).
            proj = scalar_multiply(R[i][j], qi)  # EN: Assign proj from expression: scalar_multiply(R[i][j], qi).
            v = vector_subtract(v, proj)  # EN: Assign v from expression: vector_subtract(v, proj).

        # 標準化
        R[j][j] = vector_norm(v)  # EN: Execute statement: R[j][j] = vector_norm(v).

        if R[j][j] > 1e-10:  # EN: Branch on a condition: if R[j][j] > 1e-10:.
            for i in range(m):  # EN: Iterate with a for-loop: for i in range(m):.
                Q[i][j] = v[i] / R[j][j]  # EN: Execute statement: Q[i][j] = v[i] / R[j][j].

    return Q, R  # EN: Return a value: return Q, R.


def solve_upper_triangular(R: List[List[float]], b: List[float]) -> List[float]:  # EN: Define solve_upper_triangular and its behavior.
    """回代法解上三角方程組 Rx = b"""  # EN: Execute statement: """回代法解上三角方程組 Rx = b""".
    n = len(b)  # EN: Assign n from expression: len(b).
    x = [0.0] * n  # EN: Assign x from expression: [0.0] * n.

    for i in range(n - 1, -1, -1):  # EN: Iterate with a for-loop: for i in range(n - 1, -1, -1):.
        x[i] = b[i]  # EN: Execute statement: x[i] = b[i].
        for j in range(i + 1, n):  # EN: Iterate with a for-loop: for j in range(i + 1, n):.
            x[i] -= R[i][j] * x[j]  # EN: Execute statement: x[i] -= R[i][j] * x[j].
        x[i] /= R[i][i]  # EN: Execute statement: x[i] /= R[i][i].

    return x  # EN: Return a value: return x.


def qr_least_squares(A: List[List[float]], b: List[float]) -> List[float]:  # EN: Define qr_least_squares and its behavior.
    """用 QR 分解解最小平方問題"""  # EN: Execute statement: """用 QR 分解解最小平方問題""".
    Q, R = qr_decomposition(A)  # EN: Execute statement: Q, R = qr_decomposition(A).

    # Qᵀb
    m, n = len(Q), len(Q[0])  # EN: Execute statement: m, n = len(Q), len(Q[0]).
    Qt_b = []  # EN: Assign Qt_b from expression: [].
    for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
        qj = get_column(Q, j)  # EN: Assign qj from expression: get_column(Q, j).
        Qt_b.append(dot_product(qj, b))  # EN: Execute statement: Qt_b.append(dot_product(qj, b)).

    # 解 Rx = Qᵀb
    x = solve_upper_triangular(R, Qt_b)  # EN: Assign x from expression: solve_upper_triangular(R, Qt_b).

    return x  # EN: Return a value: return x.


# ========================================
# 驗證函數
# ========================================

def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:  # EN: Define matrix_multiply and its behavior.
    """矩陣乘法"""  # EN: Execute statement: """矩陣乘法""".
    m, k, n = len(A), len(B), len(B[0])  # EN: Execute statement: m, k, n = len(A), len(B), len(B[0]).
    result = [[0.0] * n for _ in range(m)]  # EN: Assign result from expression: [[0.0] * n for _ in range(m)].
    for i in range(m):  # EN: Iterate with a for-loop: for i in range(m):.
        for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
            for p in range(k):  # EN: Iterate with a for-loop: for p in range(k):.
                result[i][j] += A[i][p] * B[p][j]  # EN: Execute statement: result[i][j] += A[i][p] * B[p][j].
    return result  # EN: Return a value: return result.


def transpose(A: List[List[float]]) -> List[List[float]]:  # EN: Define transpose and its behavior.
    """矩陣轉置"""  # EN: Execute statement: """矩陣轉置""".
    m, n = len(A), len(A[0])  # EN: Execute statement: m, n = len(A), len(A[0]).
    return [[A[i][j] for i in range(m)] for j in range(n)]  # EN: Return a value: return [[A[i][j] for i in range(m)] for j in range(n)].


def print_vector(name: str, v: List[float]) -> None:  # EN: Define print_vector and its behavior.
    formatted = [f"{x:.4f}" for x in v]  # EN: Assign formatted from expression: [f"{x:.4f}" for x in v].
    print(f"{name} = [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


def print_matrix(name: str, M: List[List[float]]) -> None:  # EN: Define print_matrix and its behavior.
    print(f"{name} =")  # EN: Print formatted output to the console.
    for row in M:  # EN: Iterate with a for-loop: for row in M:.
        formatted = [f"{x:8.4f}" for x in row]  # EN: Assign formatted from expression: [f"{x:8.4f}" for x in row].
        print(f"  [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    print_separator("QR 分解示範（手刻版）\nQR Decomposition Demo (Manual)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 基本 QR 分解
    # ========================================
    print_separator("1. 基本 QR 分解")  # EN: Call print_separator(...) to perform an operation.

    A = [  # EN: Assign A from expression: [.
        [1.0, 1.0],  # EN: Execute statement: [1.0, 1.0],.
        [1.0, 0.0],  # EN: Execute statement: [1.0, 0.0],.
        [0.0, 1.0]  # EN: Execute statement: [0.0, 1.0].
    ]  # EN: Execute statement: ].

    print("輸入矩陣 A：")  # EN: Print formatted output to the console.
    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.

    Q, R = qr_decomposition(A)  # EN: Execute statement: Q, R = qr_decomposition(A).

    print("\nQR 分解結果：")  # EN: Print formatted output to the console.
    print_matrix("Q", Q)  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("\nR", R)  # EN: Call print_matrix(...) to perform an operation.

    # 驗證 QᵀQ = I
    QT = transpose(Q)  # EN: Assign QT from expression: transpose(Q).
    QTQ = matrix_multiply(QT, Q)  # EN: Assign QTQ from expression: matrix_multiply(QT, Q).
    print("\n驗證 QᵀQ = I：")  # EN: Print formatted output to the console.
    print_matrix("QᵀQ", QTQ)  # EN: Call print_matrix(...) to perform an operation.

    # 驗證 A = QR
    QR = matrix_multiply(Q, R)  # EN: Assign QR from expression: matrix_multiply(Q, R).
    print("\n驗證 A = QR：")  # EN: Print formatted output to the console.
    print_matrix("QR", QR)  # EN: Call print_matrix(...) to perform an operation.

    # ========================================
    # 2. 用 QR 解最小平方
    # ========================================
    print_separator("2. 用 QR 解最小平方")  # EN: Call print_separator(...) to perform an operation.

    # 數據
    t = [0.0, 1.0, 2.0]  # EN: Assign t from expression: [0.0, 1.0, 2.0].
    b = [1.0, 3.0, 4.0]  # EN: Assign b from expression: [1.0, 3.0, 4.0].

    print("數據點：")  # EN: Print formatted output to the console.
    for ti, bi in zip(t, b):  # EN: Iterate with a for-loop: for ti, bi in zip(t, b):.
        print(f"  ({ti}, {bi})")  # EN: Print formatted output to the console.

    # 設計矩陣
    A_ls = [[1.0, ti] for ti in t]  # EN: Assign A_ls from expression: [[1.0, ti] for ti in t].
    print("\n設計矩陣 A：")  # EN: Print formatted output to the console.
    print_matrix("A", A_ls)  # EN: Call print_matrix(...) to perform an operation.
    print_vector("觀測值 b", b)  # EN: Call print_vector(...) to perform an operation.

    # QR 分解
    Q_ls, R_ls = qr_decomposition(A_ls)  # EN: Execute statement: Q_ls, R_ls = qr_decomposition(A_ls).
    print_matrix("\nQ", Q_ls)  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("R", R_ls)  # EN: Call print_matrix(...) to perform an operation.

    # 解最小平方
    x = qr_least_squares(A_ls, b)  # EN: Assign x from expression: qr_least_squares(A_ls, b).
    print_vector("\n解 x", x)  # EN: Call print_vector(...) to perform an operation.
    print(f"\n最佳直線：y = {x[0]:.4f} + {x[1]:.4f}t")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 另一個例子
    # ========================================
    print_separator("3. 3×3 矩陣的 QR 分解")  # EN: Call print_separator(...) to perform an operation.

    A2 = [  # EN: Assign A2 from expression: [.
        [1.0, 1.0, 0.0],  # EN: Execute statement: [1.0, 1.0, 0.0],.
        [1.0, 0.0, 1.0],  # EN: Execute statement: [1.0, 0.0, 1.0],.
        [0.0, 1.0, 1.0]  # EN: Execute statement: [0.0, 1.0, 1.0].
    ]  # EN: Execute statement: ].

    print("輸入矩陣 A：")  # EN: Print formatted output to the console.
    print_matrix("A", A2)  # EN: Call print_matrix(...) to perform an operation.

    Q2, R2 = qr_decomposition(A2)  # EN: Execute statement: Q2, R2 = qr_decomposition(A2).

    print("\nQR 分解結果：")  # EN: Print formatted output to the console.
    print_matrix("Q", Q2)  # EN: Call print_matrix(...) to perform an operation.
    print_matrix("\nR", R2)  # EN: Call print_matrix(...) to perform an operation.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
