"""
最小平方回歸 - 手刻版本 (Least Squares Regression - Manual Implementation)

本程式示範：
1. 正規方程求解最小平方問題
2. 簡單線性迴歸
3. 多項式擬合
4. 殘差分析

不使用 NumPy 的線性代數函數，手刻實作以理解底層計算。
"""  # EN: Execute statement: """.

from typing import List, Tuple  # EN: Import symbol(s) from a module: from typing import List, Tuple.
import math  # EN: Import module(s): import math.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


# ========================================
# 基本運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:  # EN: Define dot_product and its behavior.
    """內積"""  # EN: Execute statement: """內積""".
    return sum(xi * yi for xi, yi in zip(x, y))  # EN: Return a value: return sum(xi * yi for xi, yi in zip(x, y)).


def vector_norm(x: List[float]) -> float:  # EN: Define vector_norm and its behavior.
    """向量長度"""  # EN: Execute statement: """向量長度""".
    return math.sqrt(dot_product(x, x))  # EN: Return a value: return math.sqrt(dot_product(x, x)).


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
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]  # EN: Return a value: return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A….


def vector_subtract(x: List[float], y: List[float]) -> List[float]:  # EN: Define vector_subtract and its behavior.
    """向量減法"""  # EN: Execute statement: """向量減法""".
    return [xi - yi for xi, yi in zip(x, y)]  # EN: Return a value: return [xi - yi for xi, yi in zip(x, y)].


def solve_2x2(A: List[List[float]], b: List[float]) -> List[float]:  # EN: Define solve_2x2 and its behavior.
    """解 2×2 線性方程組"""  # EN: Execute statement: """解 2×2 線性方程組""".
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]  # EN: Assign det from expression: A[0][0] * A[1][1] - A[0][1] * A[1][0].
    if abs(det) < 1e-10:  # EN: Branch on a condition: if abs(det) < 1e-10:.
        raise ValueError("矩陣接近奇異")  # EN: Raise an exception: raise ValueError("矩陣接近奇異").
    x0 = (A[1][1] * b[0] - A[0][1] * b[1]) / det  # EN: Assign x0 from expression: (A[1][1] * b[0] - A[0][1] * b[1]) / det.
    x1 = (-A[1][0] * b[0] + A[0][0] * b[1]) / det  # EN: Assign x1 from expression: (-A[1][0] * b[0] + A[0][0] * b[1]) / det.
    return [x0, x1]  # EN: Return a value: return [x0, x1].


def gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:  # EN: Define gaussian_elimination and its behavior.
    """高斯消去法解線性方程組"""  # EN: Execute statement: """高斯消去法解線性方程組""".
    n = len(b)  # EN: Assign n from expression: len(b).
    # 增廣矩陣
    aug = [A[i][:] + [b[i]] for i in range(n)]  # EN: Assign aug from expression: [A[i][:] + [b[i]] for i in range(n)].

    # 前向消去
    for col in range(n):  # EN: Iterate with a for-loop: for col in range(n):.
        # 找主元
        max_row = col  # EN: Assign max_row from expression: col.
        for row in range(col + 1, n):  # EN: Iterate with a for-loop: for row in range(col + 1, n):.
            if abs(aug[row][col]) > abs(aug[max_row][col]):  # EN: Branch on a condition: if abs(aug[row][col]) > abs(aug[max_row][col]):.
                max_row = row  # EN: Assign max_row from expression: row.
        aug[col], aug[max_row] = aug[max_row], aug[col]  # EN: Execute statement: aug[col], aug[max_row] = aug[max_row], aug[col].

        if abs(aug[col][col]) < 1e-10:  # EN: Branch on a condition: if abs(aug[col][col]) < 1e-10:.
            raise ValueError("矩陣接近奇異")  # EN: Raise an exception: raise ValueError("矩陣接近奇異").

        # 消去
        for row in range(col + 1, n):  # EN: Iterate with a for-loop: for row in range(col + 1, n):.
            factor = aug[row][col] / aug[col][col]  # EN: Assign factor from expression: aug[row][col] / aug[col][col].
            for j in range(col, n + 1):  # EN: Iterate with a for-loop: for j in range(col, n + 1):.
                aug[row][j] -= factor * aug[col][j]  # EN: Execute statement: aug[row][j] -= factor * aug[col][j].

    # 回代
    x = [0.0] * n  # EN: Assign x from expression: [0.0] * n.
    for i in range(n - 1, -1, -1):  # EN: Iterate with a for-loop: for i in range(n - 1, -1, -1):.
        x[i] = aug[i][n]  # EN: Execute statement: x[i] = aug[i][n].
        for j in range(i + 1, n):  # EN: Iterate with a for-loop: for j in range(i + 1, n):.
            x[i] -= aug[i][j] * x[j]  # EN: Execute statement: x[i] -= aug[i][j] * x[j].
        x[i] /= aug[i][i]  # EN: Execute statement: x[i] /= aug[i][i].

    return x  # EN: Return a value: return x.


# ========================================
# 最小平方核心函數
# ========================================

def least_squares_solve(A: List[List[float]], b: List[float]) -> dict:  # EN: Define least_squares_solve and its behavior.
    """
    解最小平方問題：min ‖Ax - b‖²

    使用正規方程：AᵀA x̂ = Aᵀb
    """  # EN: Execute statement: """.
    m = len(A)  # EN: Assign m from expression: len(A).
    n = len(A[0])  # EN: Assign n from expression: len(A[0]).

    # 計算 AᵀA
    AT = matrix_transpose(A)  # EN: Assign AT from expression: matrix_transpose(A).
    ATA = matrix_multiply(AT, A)  # EN: Assign ATA from expression: matrix_multiply(AT, A).

    # 計算 Aᵀb
    ATb = matrix_vector_multiply(AT, b)  # EN: Assign ATb from expression: matrix_vector_multiply(AT, b).

    # 解正規方程
    if n == 2:  # EN: Branch on a condition: if n == 2:.
        x_hat = solve_2x2(ATA, ATb)  # EN: Assign x_hat from expression: solve_2x2(ATA, ATb).
    else:  # EN: Execute the fallback branch when prior conditions are false.
        x_hat = gaussian_elimination(ATA, ATb)  # EN: Assign x_hat from expression: gaussian_elimination(ATA, ATb).

    # 計算擬合值和殘差
    y_hat = matrix_vector_multiply(A, x_hat)  # EN: Assign y_hat from expression: matrix_vector_multiply(A, x_hat).
    residual = vector_subtract(b, y_hat)  # EN: Assign residual from expression: vector_subtract(b, y_hat).
    residual_norm = vector_norm(residual)  # EN: Assign residual_norm from expression: vector_norm(residual).

    # 計算 R²
    b_mean = sum(b) / len(b)  # EN: Assign b_mean from expression: sum(b) / len(b).
    tss = sum((bi - b_mean) ** 2 for bi in b)  # EN: Assign tss from expression: sum((bi - b_mean) ** 2 for bi in b).
    rss = residual_norm ** 2  # EN: Assign rss from expression: residual_norm ** 2.
    r_squared = 1 - rss / tss if tss > 0 else 0  # EN: Assign r_squared from expression: 1 - rss / tss if tss > 0 else 0.

    return {  # EN: Return a value: return {.
        'coefficients': x_hat,  # EN: Execute statement: 'coefficients': x_hat,.
        'fitted': y_hat,  # EN: Execute statement: 'fitted': y_hat,.
        'residual': residual,  # EN: Execute statement: 'residual': residual,.
        'residual_norm': residual_norm,  # EN: Execute statement: 'residual_norm': residual_norm,.
        'r_squared': r_squared,  # EN: Execute statement: 'r_squared': r_squared,.
        'ATA': ATA,  # EN: Execute statement: 'ATA': ATA,.
        'ATb': ATb  # EN: Execute statement: 'ATb': ATb.
    }  # EN: Execute statement: }.


def create_design_matrix_linear(t: List[float]) -> List[List[float]]:  # EN: Define create_design_matrix_linear and its behavior.
    """建立線性迴歸的設計矩陣 [1, t]"""  # EN: Execute statement: """建立線性迴歸的設計矩陣 [1, t]""".
    return [[1.0, ti] for ti in t]  # EN: Return a value: return [[1.0, ti] for ti in t].


def create_design_matrix_polynomial(t: List[float], degree: int) -> List[List[float]]:  # EN: Define create_design_matrix_polynomial and its behavior.
    """建立多項式迴歸的設計矩陣"""  # EN: Execute statement: """建立多項式迴歸的設計矩陣""".
    return [[ti ** k for k in range(degree + 1)] for ti in t]  # EN: Return a value: return [[ti ** k for k in range(degree + 1)] for ti in t].


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

    print_separator("最小平方回歸示範（手刻版）\nLeast Squares Regression Demo (Manual)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 簡單線性迴歸
    # ========================================
    print_separator("1. 簡單線性迴歸：y = C + Dt")  # EN: Call print_separator(...) to perform an operation.

    # 數據點
    t = [0.0, 1.0, 2.0]  # EN: Assign t from expression: [0.0, 1.0, 2.0].
    b = [1.0, 3.0, 4.0]  # EN: Assign b from expression: [1.0, 3.0, 4.0].

    print("數據點：")  # EN: Print formatted output to the console.
    for ti, bi in zip(t, b):  # EN: Iterate with a for-loop: for ti, bi in zip(t, b):.
        print(f"  t = {ti}, b = {bi}")  # EN: Print formatted output to the console.

    # 建立設計矩陣
    A = create_design_matrix_linear(t)  # EN: Assign A from expression: create_design_matrix_linear(t).
    print("\n設計矩陣 A [1, t]：")  # EN: Print formatted output to the console.
    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.

    print_vector("觀測值 b", b)  # EN: Call print_vector(...) to perform an operation.

    # 解最小平方
    result = least_squares_solve(A, b)  # EN: Assign result from expression: least_squares_solve(A, b).

    print("\n【正規方程】")  # EN: Print formatted output to the console.
    print_matrix("AᵀA", result['ATA'])  # EN: Call print_matrix(...) to perform an operation.
    print_vector("Aᵀb", result['ATb'])  # EN: Call print_vector(...) to perform an operation.

    print("\n【解】")  # EN: Print formatted output to the console.
    C, D = result['coefficients']  # EN: Execute statement: C, D = result['coefficients'].
    print(f"C（截距）= {C:.4f}")  # EN: Print formatted output to the console.
    print(f"D（斜率）= {D:.4f}")  # EN: Print formatted output to the console.
    print(f"\n最佳直線：y = {C:.4f} + {D:.4f}t")  # EN: Print formatted output to the console.

    print("\n【擬合結果】")  # EN: Print formatted output to the console.
    print_vector("擬合值 ŷ = Ax̂", result['fitted'])  # EN: Call print_vector(...) to perform an operation.
    print_vector("殘差 e = b - ŷ", result['residual'])  # EN: Call print_vector(...) to perform an operation.
    print(f"殘差範數 ‖e‖ = {result['residual_norm']:.4f}")  # EN: Print formatted output to the console.
    print(f"R² = {result['r_squared']:.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 驗證正交性
    # ========================================
    print_separator("2. 驗證殘差正交於行空間")  # EN: Call print_separator(...) to perform an operation.

    AT = matrix_transpose(A)  # EN: Assign AT from expression: matrix_transpose(A).
    ATe = matrix_vector_multiply(AT, result['residual'])  # EN: Assign ATe from expression: matrix_vector_multiply(AT, result['residual']).
    print_vector("Aᵀe", ATe)  # EN: Call print_vector(...) to perform an operation.
    print("（應為零向量，表示 e ⊥ C(A)）")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 更多數據點的線性迴歸
    # ========================================
    print_separator("3. 更多數據點的線性迴歸")  # EN: Call print_separator(...) to perform an operation.

    t2 = [0.0, 1.0, 2.0, 3.0, 4.0]  # EN: Assign t2 from expression: [0.0, 1.0, 2.0, 3.0, 4.0].
    b2 = [1.0, 2.5, 3.5, 5.0, 6.5]  # EN: Assign b2 from expression: [1.0, 2.5, 3.5, 5.0, 6.5].

    print("數據點：")  # EN: Print formatted output to the console.
    for ti, bi in zip(t2, b2):  # EN: Iterate with a for-loop: for ti, bi in zip(t2, b2):.
        print(f"  ({ti}, {bi})")  # EN: Print formatted output to the console.

    A2 = create_design_matrix_linear(t2)  # EN: Assign A2 from expression: create_design_matrix_linear(t2).
    result2 = least_squares_solve(A2, b2)  # EN: Assign result2 from expression: least_squares_solve(A2, b2).

    C2, D2 = result2['coefficients']  # EN: Execute statement: C2, D2 = result2['coefficients'].
    print(f"\n最佳直線：y = {C2:.4f} + {D2:.4f}t")  # EN: Print formatted output to the console.
    print(f"R² = {result2['r_squared']:.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 多項式擬合
    # ========================================
    print_separator("4. 二次多項式擬合：y = c₀ + c₁t + c₂t²")  # EN: Call print_separator(...) to perform an operation.

    t3 = [0.0, 1.0, 2.0, 3.0, 4.0]  # EN: Assign t3 from expression: [0.0, 1.0, 2.0, 3.0, 4.0].
    b3 = [1.0, 2.0, 5.0, 10.0, 17.0]  # EN: Assign b3 from expression: [1.0, 2.0, 5.0, 10.0, 17.0].

    print("數據點：")  # EN: Print formatted output to the console.
    for ti, bi in zip(t3, b3):  # EN: Iterate with a for-loop: for ti, bi in zip(t3, b3):.
        print(f"  ({ti}, {bi})")  # EN: Print formatted output to the console.

    A3 = create_design_matrix_polynomial(t3, degree=2)  # EN: Assign A3 from expression: create_design_matrix_polynomial(t3, degree=2).
    print("\n設計矩陣 A [1, t, t²]：")  # EN: Print formatted output to the console.
    print_matrix("A", A3)  # EN: Call print_matrix(...) to perform an operation.

    result3 = least_squares_solve(A3, b3)  # EN: Assign result3 from expression: least_squares_solve(A3, b3).

    c0, c1, c2 = result3['coefficients']  # EN: Execute statement: c0, c1, c2 = result3['coefficients'].
    print(f"\n最佳多項式：y = {c0:.4f} + {c1:.4f}t + {c2:.4f}t²")  # EN: Print formatted output to the console.
    print(f"R² = {result3['r_squared']:.4f}")  # EN: Print formatted output to the console.

    print("\n【擬合結果】")  # EN: Print formatted output to the console.
    print_vector("擬合值 ŷ", result3['fitted'])  # EN: Call print_vector(...) to perform an operation.
    print_vector("殘差 e", result3['residual'])  # EN: Call print_vector(...) to perform an operation.
    print(f"殘差範數 ‖e‖ = {result3['residual_norm']:.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 無解系統的最佳近似
    # ========================================
    print_separator("5. 無解系統的最佳近似")  # EN: Call print_separator(...) to perform an operation.

    # 三條直線不交於一點
    A4 = [  # EN: Assign A4 from expression: [.
        [1.0, 1.0],  # EN: Execute statement: [1.0, 1.0],.
        [1.0, -1.0],  # EN: Execute statement: [1.0, -1.0],.
        [2.0, 1.0]  # EN: Execute statement: [2.0, 1.0].
    ]  # EN: Execute statement: ].
    b4 = [1.0, 1.0, 3.0]  # EN: Assign b4 from expression: [1.0, 1.0, 3.0].

    print("方程組：")  # EN: Print formatted output to the console.
    print("  x + y = 1")  # EN: Print formatted output to the console.
    print("  x - y = 1")  # EN: Print formatted output to the console.
    print("  2x + y = 3")  # EN: Print formatted output to the console.

    print_matrix("\nA", A4)  # EN: Call print_matrix(...) to perform an operation.
    print_vector("b", b4)  # EN: Call print_vector(...) to perform an operation.

    result4 = least_squares_solve(A4, b4)  # EN: Assign result4 from expression: least_squares_solve(A4, b4).

    x_hat = result4['coefficients']  # EN: Assign x_hat from expression: result4['coefficients'].
    print(f"\n最小平方解：x = {x_hat[0]:.4f}, y = {x_hat[1]:.4f}")  # EN: Print formatted output to the console.
    print_vector("Ax̂（最接近 b）", result4['fitted'])  # EN: Call print_vector(...) to perform an operation.
    print_vector("殘差 e", result4['residual'])  # EN: Call print_vector(...) to perform an operation.
    print(f"殘差範數 ‖e‖ = {result4['residual_norm']:.4f}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
    print("""
最小平方法核心公式：

1. 問題：最小化 ‖Ax - b‖²

2. 正規方程：AᵀA x̂ = Aᵀb

3. 解：x̂ = (AᵀA)⁻¹Aᵀb

4. 幾何意義：
   - Ax̂ 是 b 在 C(A) 上的投影
   - e = b - Ax̂ 垂直於 C(A)

5. 線性迴歸設計矩陣：
   - 線性：[1, t]
   - 多項式：[1, t, t², ...]

6. R² = 1 - RSS/TSS（越接近 1 越好）
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
