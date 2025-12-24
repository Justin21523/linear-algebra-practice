"""
餘因子展開 - NumPy 版本 (Cofactor Expansion - NumPy Implementation)

本程式示範：
1. NumPy 實作餘因子展開
2. 餘因子矩陣與伴隨矩陣
3. 用伴隨矩陣求逆矩陣
4. 與 np.linalg.inv 比較
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def get_minor_matrix(A: np.ndarray, row: int, col: int) -> np.ndarray:
    """取得去掉第 row 列、第 col 行後的子矩陣"""
    return np.delete(np.delete(A, row, axis=0), col, axis=1)


def minor(A: np.ndarray, i: int, j: int) -> float:
    """計算子行列式 Mᵢⱼ"""
    sub = get_minor_matrix(A, i, j)
    return np.linalg.det(sub)


def cofactor(A: np.ndarray, i: int, j: int) -> float:
    """計算餘因子 Cᵢⱼ"""
    sign = (-1) ** (i + j)
    return sign * minor(A, i, j)


def determinant_by_cofactor(A: np.ndarray, row: int = 0) -> float:
    """用餘因子展開計算行列式（沿指定列）"""
    n = A.shape[0]
    det = 0.0
    for j in range(n):
        det += A[row, j] * cofactor(A, row, j)
    return det


def cofactor_matrix(A: np.ndarray) -> np.ndarray:
    """計算餘因子矩陣"""
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = cofactor(A, i, j)
    return C


def adjugate(A: np.ndarray) -> np.ndarray:
    """計算伴隨矩陣 adj(A) = Cᵀ"""
    return cofactor_matrix(A).T


def inverse_by_adjugate(A: np.ndarray) -> np.ndarray:
    """用伴隨矩陣計算逆矩陣"""
    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        raise ValueError("矩陣不可逆")
    return adjugate(A) / det


def main():
    print_separator("餘因子展開示範（NumPy 版）\nCofactor Expansion Demo (NumPy)")

    # ========================================
    # 1. 子行列式與餘因子
    # ========================================
    print_separator("1. 子行列式與餘因子")

    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    print(f"A:\n{A}")

    print("\n所有子行列式 Mᵢⱼ：")
    for i in range(3):
        for j in range(3):
            print(f"  M_{i+1}{j+1} = {minor(A, i, j):8.4f}", end="")
        print()

    print("\n所有餘因子 Cᵢⱼ：")
    for i in range(3):
        for j in range(3):
            print(f"  C_{i+1}{j+1} = {cofactor(A, i, j):8.4f}", end="")
        print()

    # ========================================
    # 2. 餘因子展開
    # ========================================
    print_separator("2. 餘因子展開計算行列式")

    print(f"A:\n{A}")

    print("\n沿各列展開：")
    for row in range(3):
        det = determinant_by_cofactor(A, row)
        print(f"  沿第 {row+1} 列：det(A) = {det:.4f}")

    print(f"\nnp.linalg.det(A) = {np.linalg.det(A):.4f}")

    # ========================================
    # 3. 餘因子矩陣與伴隨矩陣
    # ========================================
    print_separator("3. 餘因子矩陣與伴隨矩陣")

    A = np.array([
        [2, 1, 3],
        [1, 0, 2],
        [4, 1, 5]
    ], dtype=float)

    print(f"A:\n{A}")
    print(f"\ndet(A) = {np.linalg.det(A):.4f}")

    C = cofactor_matrix(A)
    print(f"\n餘因子矩陣 C:\n{C}")

    adj_A = adjugate(A)
    print(f"\n伴隨矩陣 adj(A) = Cᵀ:\n{adj_A}")

    # ========================================
    # 4. 用伴隨矩陣求逆矩陣
    # ========================================
    print_separator("4. 用伴隨矩陣求逆矩陣")

    det_A = np.linalg.det(A)
    print(f"A⁻¹ = adj(A) / det(A)")
    print(f"    = adj(A) / {det_A:.4f}")

    A_inv = inverse_by_adjugate(A)
    print(f"\nA⁻¹（伴隨矩陣法）:\n{A_inv}")

    A_inv_np = np.linalg.inv(A)
    print(f"\nA⁻¹（NumPy）:\n{A_inv_np}")

    print(f"\n差異（Frobenius 範數）：{np.linalg.norm(A_inv - A_inv_np):.2e}")

    # 驗證
    print(f"\n驗證 A @ A⁻¹:\n{A @ A_inv}")

    # ========================================
    # 5. 特殊情況
    # ========================================
    print_separator("5. 特殊情況")

    # 上三角矩陣
    U = np.array([
        [2, 3, 1],
        [0, 4, 5],
        [0, 0, 6]
    ], dtype=float)

    print(f"上三角矩陣 U:\n{U}")
    print(f"det(U) = {np.linalg.det(U):.4f}（= 2 × 4 × 6 = 48）")

    adj_U = adjugate(U)
    print(f"\nadj(U):\n{adj_U}")

    # ========================================
    # 6. 2×2 公式驗證
    # ========================================
    print_separator("6. 2×2 伴隨矩陣公式")

    A2 = np.array([[3, 4], [5, 6]], dtype=float)
    print(f"A:\n{A2}")
    print(f"\n對於 2×2 矩陣 [[a, b], [c, d]]：")
    print(f"adj(A) = [[d, -b], [-c, a]]")
    print(f"       = [[{A2[1,1]}, {-A2[0,1]}], [{-A2[1,0]}, {A2[0,0]}]]")

    adj_A2 = adjugate(A2)
    print(f"\n計算得到的 adj(A):\n{adj_A2}")

    det_A2 = np.linalg.det(A2)
    print(f"\ndet(A) = ad - bc = {A2[0,0]}×{A2[1,1]} - {A2[0,1]}×{A2[1,0]} = {det_A2:.4f}")

    A2_inv = adj_A2 / det_A2
    print(f"\nA⁻¹ = adj(A)/det(A):\n{A2_inv}")

    # ========================================
    # 7. Cramer's Rule 預覽
    # ========================================
    print_separator("7. Cramer's Rule 預覽")

    print("用餘因子展開可以推導 Cramer's Rule：")
    print("解 Ax = b 時，xⱼ = det(Aⱼ) / det(A)")
    print("其中 Aⱼ 是把 A 的第 j 行換成 b")

    A = np.array([[2, 1], [5, 3]], dtype=float)
    b = np.array([8, 21], dtype=float)

    print(f"\n例：Ax = b")
    print(f"A:\n{A}")
    print(f"b: {b}")

    det_A = np.linalg.det(A)
    A1 = A.copy()
    A1[:, 0] = b
    A2 = A.copy()
    A2[:, 1] = b

    x1 = np.linalg.det(A1) / det_A
    x2 = np.linalg.det(A2) / det_A

    print(f"\nx₁ = det(A₁)/det(A) = {np.linalg.det(A1)}/{det_A} = {x1:.4f}")
    print(f"x₂ = det(A₂)/det(A) = {np.linalg.det(A2)}/{det_A} = {x2:.4f}")

    x_direct = np.linalg.solve(A, b)
    print(f"\n直接解：x = {x_direct}")

    # 總結
    print_separator("NumPy 餘因子展開總結")
    print("""
基本函數：
  np.delete(A, i, axis=0)  # 刪除第 i 列
  np.delete(A, j, axis=1)  # 刪除第 j 行

公式：
  Cᵢⱼ = (-1)^(i+j) × det(Mᵢⱼ)
  adj(A) = Cᵀ
  A⁻¹ = adj(A) / det(A)

2×2 特例：
  [[a, b], [c, d]]
  adj = [[d, -b], [-c, a]]
  det = ad - bc

注意：餘因子展開時間複雜度 O(n!)，
      實際應用使用 np.linalg.det/inv
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
