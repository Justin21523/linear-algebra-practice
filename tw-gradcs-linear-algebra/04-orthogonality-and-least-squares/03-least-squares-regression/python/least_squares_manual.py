"""
最小平方回歸 - 手刻版本 (Least Squares Regression - Manual Implementation)

本程式示範：
1. 正規方程求解最小平方問題
2. 簡單線性迴歸
3. 多項式擬合
4. 殘差分析

不使用 NumPy 的線性代數函數，手刻實作以理解底層計算。
"""

from typing import List, Tuple
import math


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ========================================
# 基本運算
# ========================================

def dot_product(x: List[float], y: List[float]) -> float:
    """內積"""
    return sum(xi * yi for xi, yi in zip(x, y))


def vector_norm(x: List[float]) -> float:
    """向量長度"""
    return math.sqrt(dot_product(x, x))


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
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def vector_subtract(x: List[float], y: List[float]) -> List[float]:
    """向量減法"""
    return [xi - yi for xi, yi in zip(x, y)]


def solve_2x2(A: List[List[float]], b: List[float]) -> List[float]:
    """解 2×2 線性方程組"""
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(det) < 1e-10:
        raise ValueError("矩陣接近奇異")
    x0 = (A[1][1] * b[0] - A[0][1] * b[1]) / det
    x1 = (-A[1][0] * b[0] + A[0][0] * b[1]) / det
    return [x0, x1]


def gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    """高斯消去法解線性方程組"""
    n = len(b)
    # 增廣矩陣
    aug = [A[i][:] + [b[i]] for i in range(n)]

    # 前向消去
    for col in range(n):
        # 找主元
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        if abs(aug[col][col]) < 1e-10:
            raise ValueError("矩陣接近奇異")

        # 消去
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # 回代
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x


# ========================================
# 最小平方核心函數
# ========================================

def least_squares_solve(A: List[List[float]], b: List[float]) -> dict:
    """
    解最小平方問題：min ‖Ax - b‖²

    使用正規方程：AᵀA x̂ = Aᵀb
    """
    m = len(A)
    n = len(A[0])

    # 計算 AᵀA
    AT = matrix_transpose(A)
    ATA = matrix_multiply(AT, A)

    # 計算 Aᵀb
    ATb = matrix_vector_multiply(AT, b)

    # 解正規方程
    if n == 2:
        x_hat = solve_2x2(ATA, ATb)
    else:
        x_hat = gaussian_elimination(ATA, ATb)

    # 計算擬合值和殘差
    y_hat = matrix_vector_multiply(A, x_hat)
    residual = vector_subtract(b, y_hat)
    residual_norm = vector_norm(residual)

    # 計算 R²
    b_mean = sum(b) / len(b)
    tss = sum((bi - b_mean) ** 2 for bi in b)
    rss = residual_norm ** 2
    r_squared = 1 - rss / tss if tss > 0 else 0

    return {
        'coefficients': x_hat,
        'fitted': y_hat,
        'residual': residual,
        'residual_norm': residual_norm,
        'r_squared': r_squared,
        'ATA': ATA,
        'ATb': ATb
    }


def create_design_matrix_linear(t: List[float]) -> List[List[float]]:
    """建立線性迴歸的設計矩陣 [1, t]"""
    return [[1.0, ti] for ti in t]


def create_design_matrix_polynomial(t: List[float], degree: int) -> List[List[float]]:
    """建立多項式迴歸的設計矩陣"""
    return [[ti ** k for k in range(degree + 1)] for ti in t]


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

    print_separator("最小平方回歸示範（手刻版）\nLeast Squares Regression Demo (Manual)")

    # ========================================
    # 1. 簡單線性迴歸
    # ========================================
    print_separator("1. 簡單線性迴歸：y = C + Dt")

    # 數據點
    t = [0.0, 1.0, 2.0]
    b = [1.0, 3.0, 4.0]

    print("數據點：")
    for ti, bi in zip(t, b):
        print(f"  t = {ti}, b = {bi}")

    # 建立設計矩陣
    A = create_design_matrix_linear(t)
    print("\n設計矩陣 A [1, t]：")
    print_matrix("A", A)

    print_vector("觀測值 b", b)

    # 解最小平方
    result = least_squares_solve(A, b)

    print("\n【正規方程】")
    print_matrix("AᵀA", result['ATA'])
    print_vector("Aᵀb", result['ATb'])

    print("\n【解】")
    C, D = result['coefficients']
    print(f"C（截距）= {C:.4f}")
    print(f"D（斜率）= {D:.4f}")
    print(f"\n最佳直線：y = {C:.4f} + {D:.4f}t")

    print("\n【擬合結果】")
    print_vector("擬合值 ŷ = Ax̂", result['fitted'])
    print_vector("殘差 e = b - ŷ", result['residual'])
    print(f"殘差範數 ‖e‖ = {result['residual_norm']:.4f}")
    print(f"R² = {result['r_squared']:.4f}")

    # ========================================
    # 2. 驗證正交性
    # ========================================
    print_separator("2. 驗證殘差正交於行空間")

    AT = matrix_transpose(A)
    ATe = matrix_vector_multiply(AT, result['residual'])
    print_vector("Aᵀe", ATe)
    print("（應為零向量，表示 e ⊥ C(A)）")

    # ========================================
    # 3. 更多數據點的線性迴歸
    # ========================================
    print_separator("3. 更多數據點的線性迴歸")

    t2 = [0.0, 1.0, 2.0, 3.0, 4.0]
    b2 = [1.0, 2.5, 3.5, 5.0, 6.5]

    print("數據點：")
    for ti, bi in zip(t2, b2):
        print(f"  ({ti}, {bi})")

    A2 = create_design_matrix_linear(t2)
    result2 = least_squares_solve(A2, b2)

    C2, D2 = result2['coefficients']
    print(f"\n最佳直線：y = {C2:.4f} + {D2:.4f}t")
    print(f"R² = {result2['r_squared']:.4f}")

    # ========================================
    # 4. 多項式擬合
    # ========================================
    print_separator("4. 二次多項式擬合：y = c₀ + c₁t + c₂t²")

    t3 = [0.0, 1.0, 2.0, 3.0, 4.0]
    b3 = [1.0, 2.0, 5.0, 10.0, 17.0]

    print("數據點：")
    for ti, bi in zip(t3, b3):
        print(f"  ({ti}, {bi})")

    A3 = create_design_matrix_polynomial(t3, degree=2)
    print("\n設計矩陣 A [1, t, t²]：")
    print_matrix("A", A3)

    result3 = least_squares_solve(A3, b3)

    c0, c1, c2 = result3['coefficients']
    print(f"\n最佳多項式：y = {c0:.4f} + {c1:.4f}t + {c2:.4f}t²")
    print(f"R² = {result3['r_squared']:.4f}")

    print("\n【擬合結果】")
    print_vector("擬合值 ŷ", result3['fitted'])
    print_vector("殘差 e", result3['residual'])
    print(f"殘差範數 ‖e‖ = {result3['residual_norm']:.4f}")

    # ========================================
    # 5. 無解系統的最佳近似
    # ========================================
    print_separator("5. 無解系統的最佳近似")

    # 三條直線不交於一點
    A4 = [
        [1.0, 1.0],
        [1.0, -1.0],
        [2.0, 1.0]
    ]
    b4 = [1.0, 1.0, 3.0]

    print("方程組：")
    print("  x + y = 1")
    print("  x - y = 1")
    print("  2x + y = 3")

    print_matrix("\nA", A4)
    print_vector("b", b4)

    result4 = least_squares_solve(A4, b4)

    x_hat = result4['coefficients']
    print(f"\n最小平方解：x = {x_hat[0]:.4f}, y = {x_hat[1]:.4f}")
    print_vector("Ax̂（最接近 b）", result4['fitted'])
    print_vector("殘差 e", result4['residual'])
    print(f"殘差範數 ‖e‖ = {result4['residual_norm']:.4f}")

    # 總結
    print_separator("總結")
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
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
