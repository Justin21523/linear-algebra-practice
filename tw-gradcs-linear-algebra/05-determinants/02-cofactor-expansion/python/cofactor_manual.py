"""
餘因子展開 - 手刻版本 (Cofactor Expansion - Manual Implementation)

本程式示範：
1. 子行列式與餘因子計算
2. 餘因子展開求行列式
3. 餘因子矩陣與伴隨矩陣
4. 用伴隨矩陣求逆矩陣
"""

from typing import List


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


# ========================================
# 子行列式與餘因子
# ========================================

def get_minor_matrix(A: List[List[float]], row: int, col: int) -> List[List[float]]:
    """取得去掉第 row 列、第 col 行後的子矩陣"""
    n = len(A)
    return [[A[i][j] for j in range(n) if j != col]
            for i in range(n) if i != row]


def det_2x2(A: List[List[float]]) -> float:
    """2×2 行列式"""
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def minor(A: List[List[float]], i: int, j: int) -> float:
    """計算子行列式 Mᵢⱼ"""
    sub_matrix = get_minor_matrix(A, i, j)
    return determinant(sub_matrix)


def cofactor(A: List[List[float]], i: int, j: int) -> float:
    """計算餘因子 Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ"""
    sign = (-1) ** (i + j)
    return sign * minor(A, i, j)


def determinant(A: List[List[float]]) -> float:
    """用餘因子展開計算行列式（遞迴）"""
    n = len(A)

    # 基本情況
    if n == 1:
        return A[0][0]
    if n == 2:
        return det_2x2(A)

    # 沿第一列展開
    det = 0.0
    for j in range(n):
        det += A[0][j] * cofactor(A, 0, j)

    return det


def determinant_by_row(A: List[List[float]], row: int) -> float:
    """沿指定列展開"""
    n = len(A)
    det = 0.0
    for j in range(n):
        det += A[row][j] * cofactor(A, row, j)
    return det


def determinant_by_col(A: List[List[float]], col: int) -> float:
    """沿指定行展開"""
    n = len(A)
    det = 0.0
    for i in range(n):
        det += A[i][col] * cofactor(A, i, col)
    return det


# ========================================
# 餘因子矩陣與伴隨矩陣
# ========================================

def cofactor_matrix(A: List[List[float]]) -> List[List[float]]:
    """計算餘因子矩陣"""
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = cofactor(A, i, j)
    return C


def transpose(A: List[List[float]]) -> List[List[float]]:
    """矩陣轉置"""
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def adjugate(A: List[List[float]]) -> List[List[float]]:
    """計算伴隨矩陣 adj(A) = Cᵀ"""
    return transpose(cofactor_matrix(A))


def scalar_multiply_matrix(c: float, A: List[List[float]]) -> List[List[float]]:
    """純量乘矩陣"""
    return [[c * x for x in row] for row in A]


def inverse_by_adjugate(A: List[List[float]]) -> List[List[float]]:
    """用伴隨矩陣計算逆矩陣：A⁻¹ = adj(A) / det(A)"""
    det = determinant(A)
    if abs(det) < 1e-10:
        raise ValueError("矩陣不可逆（det = 0）")
    adj = adjugate(A)
    return scalar_multiply_matrix(1.0 / det, adj)


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


def main():
    print_separator("餘因子展開示範（手刻版）\nCofactor Expansion Demo (Manual)")

    # ========================================
    # 1. 基本概念：子行列式與餘因子
    # ========================================
    print_separator("1. 子行列式與餘因子")

    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print_matrix("A", A)

    # 計算 M₁₂（去掉第1列第2行）
    sub = get_minor_matrix(A, 0, 1)
    print("\n去掉第 1 列、第 2 行：")
    print_matrix("子矩陣", sub)
    print(f"M₁₂ = det(子矩陣) = {det_2x2(sub)}")

    # 計算餘因子
    print(f"\nC₁₂ = (-1)^(1+2) × M₁₂ = -1 × {det_2x2(sub)} = {cofactor(A, 0, 1)}")

    # ========================================
    # 2. 餘因子展開計算行列式
    # ========================================
    print_separator("2. 餘因子展開計算行列式")

    print_matrix("A", A)

    print("\n沿第一列展開：")
    print(f"det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃")
    print(f"       = {A[0][0]} × {cofactor(A, 0, 0)} + {A[0][1]} × {cofactor(A, 0, 1)} + {A[0][2]} × {cofactor(A, 0, 2)}")
    print(f"       = {A[0][0] * cofactor(A, 0, 0)} + {A[0][1] * cofactor(A, 0, 1)} + {A[0][2] * cofactor(A, 0, 2)}")
    print(f"       = {determinant(A)}")

    print(f"\n沿第一行展開：det(A) = {determinant_by_col(A, 0)}")
    print(f"沿第二列展開：det(A) = {determinant_by_row(A, 1)}")

    # ========================================
    # 3. 選擇最佳展開位置
    # ========================================
    print_separator("3. 選擇有零元素的列展開")

    B = [
        [1, 0, 0],
        [2, 3, 0],
        [4, 5, 6]
    ]

    print_matrix("B（下三角）", B)
    print("\n沿第一列展開（兩個零）：")
    print(f"det(B) = {B[0][0]} × C₁₁ + 0 + 0")
    print(f"       = 1 × det([[3, 0], [5, 6]])")
    print(f"       = 1 × (3×6 - 0×5)")
    print(f"       = {determinant(B)}")

    # ========================================
    # 4. 餘因子矩陣
    # ========================================
    print_separator("4. 餘因子矩陣")

    A = [
        [2, 1, 3],
        [1, 0, 2],
        [4, 1, 5]
    ]

    print_matrix("A", A)
    print(f"\ndet(A) = {determinant(A)}")

    C = cofactor_matrix(A)
    print("\n餘因子矩陣 C：")
    print_matrix("C", C)

    print("\n各餘因子計算過程：")
    for i in range(3):
        for j in range(3):
            sign = "+" if (i + j) % 2 == 0 else "-"
            print(f"C_{i+1}{j+1} = {sign}M_{i+1}{j+1} = {cofactor(A, i, j):.1f}")

    # ========================================
    # 5. 伴隨矩陣
    # ========================================
    print_separator("5. 伴隨矩陣 adj(A) = Cᵀ")

    adj_A = adjugate(A)
    print_matrix("adj(A)", adj_A)

    # ========================================
    # 6. 用伴隨矩陣求逆矩陣
    # ========================================
    print_separator("6. 用伴隨矩陣求逆矩陣")

    print("A⁻¹ = adj(A) / det(A)")
    print(f"    = adj(A) / {determinant(A)}")

    A_inv = inverse_by_adjugate(A)
    print_matrix("\nA⁻¹", A_inv)

    # 驗證
    I = matrix_multiply(A, A_inv)
    print("\n驗證 A × A⁻¹：")
    print_matrix("A × A⁻¹", I)

    # ========================================
    # 7. 4×4 行列式範例
    # ========================================
    print_separator("7. 4×4 行列式範例")

    D = [
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [0, 0, 5, 6],
        [0, 0, 7, 8]
    ]

    print_matrix("D（塊對角）", D)
    print(f"\ndet(D) = {determinant(D)}")
    print("（= det([[1,2],[3,4]]) × det([[5,6],[7,8]])）")
    print(f"（= {det_2x2([[1,2],[3,4]])} × {det_2x2([[5,6],[7,8]])} = {det_2x2([[1,2],[3,4]]) * det_2x2([[5,6],[7,8]])}）")

    # 總結
    print_separator("總結")
    print("""
餘因子展開公式：
  Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ
  det(A) = Σⱼ aᵢⱼ Cᵢⱼ（沿第 i 列）

伴隨矩陣：
  adj(A) = Cᵀ

逆矩陣公式：
  A⁻¹ = adj(A) / det(A)

計算技巧：
  - 選擇零最多的列/行展開
  - 三角矩陣 → det = 對角線乘積

時間複雜度：O(n!)，只適合小矩陣
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
