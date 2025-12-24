"""
克萊姆法則 - 手刻版本 (Cramer's Rule - Manual Implementation)

本程式示範：
1. 克萊姆法則的實作
2. 2×2 和 3×3 系統求解
3. 與消去法比較
"""

from typing import List, Optional


def print_separator(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_matrix(name: str, M: List[List[float]]) -> None:
    print(f"{name} =")
    for row in M:
        formatted = [f"{x:8.4f}" for x in row]
        print(f"  [{', '.join(formatted)}]")


def print_vector(name: str, v: List[float]) -> None:
    formatted = [f"{x:.4f}" for x in v]
    print(f"{name} = [{', '.join(formatted)}]")


# ========================================
# 行列式計算
# ========================================

def det_2x2(A: List[List[float]]) -> float:
    """2×2 行列式"""
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def det_3x3(A: List[List[float]]) -> float:
    """3×3 行列式"""
    a, b, c = A[0]
    d, e, f = A[1]
    g, h, i = A[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def determinant(A: List[List[float]]) -> float:
    """計算行列式（支援 2×2 和 3×3）"""
    n = len(A)
    if n == 2:
        return det_2x2(A)
    elif n == 3:
        return det_3x3(A)
    else:
        raise ValueError("僅支援 2×2 和 3×3 矩陣")


# ========================================
# 克萊姆法則
# ========================================

def replace_column(A: List[List[float]], b: List[float], j: int) -> List[List[float]]:
    """建立 Aⱼ：把 A 的第 j 行換成 b"""
    n = len(A)
    Aj = [row[:] for row in A]  # 複製
    for i in range(n):
        Aj[i][j] = b[i]
    return Aj


def cramers_rule(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """
    用克萊姆法則解 Ax = b

    xⱼ = det(Aⱼ) / det(A)
    """
    n = len(A)
    det_A = determinant(A)

    if abs(det_A) < 1e-10:
        return None  # 矩陣奇異

    x = []
    for j in range(n):
        Aj = replace_column(A, b, j)
        det_Aj = determinant(Aj)
        x.append(det_Aj / det_A)

    return x


def cramers_rule_verbose(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """克萊姆法則（詳細過程）"""
    n = len(A)
    det_A = determinant(A)

    print(f"det(A) = {det_A:.4f}")

    if abs(det_A) < 1e-10:
        print("det(A) = 0，矩陣奇異，無法使用克萊姆法則")
        return None

    x = []
    for j in range(n):
        Aj = replace_column(A, b, j)
        det_Aj = determinant(Aj)
        xj = det_Aj / det_A

        print(f"\nA{j+1}（第 {j+1} 行換成 b）：")
        print_matrix(f"A{j+1}", Aj)
        print(f"det(A{j+1}) = {det_Aj:.4f}")
        print(f"x{j+1} = det(A{j+1})/det(A) = {det_Aj:.4f}/{det_A:.4f} = {xj:.4f}")

        x.append(xj)

    return x


# ========================================
# 高斯消去法（用於比較）
# ========================================

def gaussian_elimination(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """高斯消去法解 Ax = b"""
    n = len(A)

    # 建立增廣矩陣
    Aug = [A[i][:] + [b[i]] for i in range(n)]

    # 前向消去
    for col in range(n):
        # 找主元
        max_row = col
        for row in range(col + 1, n):
            if abs(Aug[row][col]) > abs(Aug[max_row][col]):
                max_row = row

        Aug[col], Aug[max_row] = Aug[max_row], Aug[col]

        if abs(Aug[col][col]) < 1e-10:
            return None

        # 消去
        for row in range(col + 1, n):
            factor = Aug[row][col] / Aug[col][col]
            for j in range(col, n + 1):
                Aug[row][j] -= factor * Aug[col][j]

    # 回代
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = Aug[i][n]
        for j in range(i + 1, n):
            x[i] -= Aug[i][j] * x[j]
        x[i] /= Aug[i][i]

    return x


def main():
    print_separator("克萊姆法則示範（手刻版）\nCramer's Rule Demo (Manual)")

    # ========================================
    # 1. 2×2 系統
    # ========================================
    print_separator("1. 2×2 系統")

    A = [[2, 3], [4, 5]]
    b = [8, 14]

    print("方程組：")
    print("  2x + 3y = 8")
    print("  4x + 5y = 14")

    print_matrix("\nA", A)
    print_vector("b", b)

    print("\n克萊姆法則求解：")
    x = cramers_rule_verbose(A, b)

    print(f"\n解：x = {x[0]:.4f}, y = {x[1]:.4f}")

    # 驗證
    print("\n驗證：")
    print(f"  2({x[0]:.4f}) + 3({x[1]:.4f}) = {2*x[0] + 3*x[1]:.4f}")
    print(f"  4({x[0]:.4f}) + 5({x[1]:.4f}) = {4*x[0] + 5*x[1]:.4f}")

    # ========================================
    # 2. 3×3 系統
    # ========================================
    print_separator("2. 3×3 系統")

    A = [
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ]
    b = [8, -11, -3]

    print("方程組：")
    print("   2x +  y -  z =  8")
    print("  -3x -  y + 2z = -11")
    print("  -2x +  y + 2z = -3")

    print_matrix("\nA", A)
    print_vector("b", b)

    print("\n克萊姆法則求解：")
    x = cramers_rule_verbose(A, b)

    print(f"\n解：x = {x[0]:.4f}, y = {x[1]:.4f}, z = {x[2]:.4f}")

    # 驗證
    print("\n驗證：")
    print(f"   2({x[0]}) + ({x[1]}) - ({x[2]}) = {2*x[0] + x[1] - x[2]:.4f}")
    print(f"  -3({x[0]}) - ({x[1]}) + 2({x[2]}) = {-3*x[0] - x[1] + 2*x[2]:.4f}")
    print(f"  -2({x[0]}) + ({x[1]}) + 2({x[2]}) = {-2*x[0] + x[1] + 2*x[2]:.4f}")

    # ========================================
    # 3. 與高斯消去比較
    # ========================================
    print_separator("3. 與高斯消去比較")

    A = [[1, 2, 1], [2, 1, 1], [1, 1, 2]]
    b = [6, 5, 6]

    print_matrix("A", A)
    print_vector("b", b)

    x_cramer = cramers_rule(A, b)
    x_gauss = gaussian_elimination(A, b)

    print(f"\n克萊姆法則：x = [{x_cramer[0]:.4f}, {x_cramer[1]:.4f}, {x_cramer[2]:.4f}]")
    print(f"高斯消去：  x = [{x_gauss[0]:.4f}, {x_gauss[1]:.4f}, {x_gauss[2]:.4f}]")

    # ========================================
    # 4. 只求一個未知數
    # ========================================
    print_separator("4. 只求特定未知數")

    A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    b = [14, 32, 53]

    print("假設只需要求 x₂：")
    print_matrix("A", A)
    print_vector("b", b)

    det_A = determinant(A)
    A2 = replace_column(A, b, 1)
    det_A2 = determinant(A2)
    x2 = det_A2 / det_A

    print(f"\ndet(A) = {det_A:.4f}")
    print_matrix("A₂", A2)
    print(f"det(A₂) = {det_A2:.4f}")
    print(f"\nx₂ = det(A₂)/det(A) = {x2:.4f}")

    # 完整解驗證
    x_full = cramers_rule(A, b)
    print(f"\n（完整解：x = [{x_full[0]:.4f}, {x_full[1]:.4f}, {x_full[2]:.4f}]）")

    # ========================================
    # 5. 奇異矩陣情況
    # ========================================
    print_separator("5. 奇異矩陣情況")

    A_singular = [[1, 2], [2, 4]]
    b_singular = [3, 6]

    print_matrix("A（奇異）", A_singular)
    print_vector("b", b_singular)

    det_A = determinant(A_singular)
    print(f"\ndet(A) = {det_A}")

    result = cramers_rule(A_singular, b_singular)
    if result is None:
        print("克萊姆法則不適用（det(A) = 0）")

    # 總結
    print_separator("總結")
    print("""
克萊姆法則：
  xⱼ = det(Aⱼ) / det(A)
  Aⱼ = A 的第 j 行換成 b

適用條件：
  - det(A) ≠ 0（A 可逆）
  - 方陣系統

優點：
  - 公式簡潔
  - 只需一個未知數時效率高
  - 適合符號計算

缺點：
  - 時間複雜度 O(n!)
  - 大矩陣數值不穩定
  - 只適用小系統
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
