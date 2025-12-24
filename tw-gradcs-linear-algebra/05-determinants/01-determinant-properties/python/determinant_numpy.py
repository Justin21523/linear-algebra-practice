"""
行列式的性質 - NumPy 版本 (Determinant Properties - NumPy Implementation)

本程式示範：
1. np.linalg.det 的使用
2. 驗證所有行列式性質
3. 特殊矩陣的行列式
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    print_separator("行列式性質示範（NumPy 版）\nDeterminant Properties Demo (NumPy)")

    # ========================================
    # 1. 基本計算
    # ========================================
    print_separator("1. 基本行列式計算")

    # 2×2
    A2 = np.array([[3, 8], [4, 6]], dtype=float)
    print(f"A (2×2):\n{A2}")
    print(f"det(A) = {np.linalg.det(A2):.4f}")
    print(f"驗證：3×6 - 8×4 = {3*6 - 8*4}")

    # 3×3
    A3 = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ], dtype=float)
    print(f"\nA (3×3):\n{A3}")
    print(f"det(A) = {np.linalg.det(A3):.4f}")

    # 4×4
    A4 = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 17]
    ], dtype=float)
    print(f"\nA (4×4):\n{A4}")
    print(f"det(A) = {np.linalg.det(A4):.4f}")

    # ========================================
    # 2. 性質 1：det(I) = 1
    # ========================================
    print_separator("2. 性質 1：det(I) = 1")

    for n in [2, 3, 4, 5]:
        I_n = np.eye(n)
        print(f"det(I_{n}) = {np.linalg.det(I_n):.4f}")

    # ========================================
    # 3. 性質 2：列交換變號
    # ========================================
    print_separator("3. 性質 2：列交換變號")

    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ], dtype=float)

    print(f"原矩陣 A:\n{A}")
    print(f"det(A) = {np.linalg.det(A):.4f}")

    # 交換第 1, 2 列
    A_swap = A[[1, 0, 2], :]
    print(f"\n交換第 1, 2 列:\n{A_swap}")
    print(f"det(交換後) = {np.linalg.det(A_swap):.4f}")
    print("驗證：變號 ✓")

    # ========================================
    # 4. 性質 3：列加法不變
    # ========================================
    print_separator("4. 性質 3：rᵢ ← rᵢ + c·rⱼ 不變")

    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ], dtype=float)

    print(f"原矩陣 A:\n{A}")
    print(f"det(A) = {np.linalg.det(A):.4f}")

    # r₂ ← r₂ - 4·r₁
    A_add = A.copy()
    A_add[1] = A_add[1] - 4 * A_add[0]
    print(f"\nr₂ ← r₂ - 4r₁:\n{A_add}")
    print(f"det(列運算後) = {np.linalg.det(A_add):.4f}")
    print("驗證：不變 ✓")

    # ========================================
    # 5. 乘積公式
    # ========================================
    print_separator("5. 乘積公式：det(AB) = det(A)·det(B)")

    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)

    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print(f"AB:\n{A @ B}")

    det_A = np.linalg.det(A)
    det_B = np.linalg.det(B)
    det_AB = np.linalg.det(A @ B)

    print(f"\ndet(A) = {det_A:.4f}")
    print(f"det(B) = {det_B:.4f}")
    print(f"det(A)·det(B) = {det_A * det_B:.4f}")
    print(f"det(AB) = {det_AB:.4f}")
    print(f"差異 = {abs(det_A * det_B - det_AB):.2e}")

    # ========================================
    # 6. 轉置公式
    # ========================================
    print_separator("6. 轉置公式：det(Aᵀ) = det(A)")

    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ], dtype=float)

    print(f"A:\n{A}")
    print(f"Aᵀ:\n{A.T}")
    print(f"\ndet(A) = {np.linalg.det(A):.4f}")
    print(f"det(Aᵀ) = {np.linalg.det(A.T):.4f}")

    # ========================================
    # 7. 純量乘法
    # ========================================
    print_separator("7. 純量乘法：det(cA) = cⁿ·det(A)")

    A = np.array([[1, 2], [3, 4]], dtype=float)
    c = 3
    n = 2

    print(f"A (2×2):\n{A}")
    print(f"c = {c}")
    print(f"{c}A:\n{c * A}")

    det_A = np.linalg.det(A)
    det_cA = np.linalg.det(c * A)

    print(f"\ndet(A) = {det_A:.4f}")
    print(f"cⁿ·det(A) = {c}² × {det_A:.4f} = {c**n * det_A:.4f}")
    print(f"det(cA) = {det_cA:.4f}")

    # 3×3 例子
    A3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
    n = 3
    print(f"\n3×3 矩陣：cⁿ·det(A) = {c}³ × {np.linalg.det(A3):.4f} = {c**n * np.linalg.det(A3):.4f}")
    print(f"det(cA) = {np.linalg.det(c * A3):.4f}")

    # ========================================
    # 8. 逆矩陣公式
    # ========================================
    print_separator("8. 逆矩陣公式：det(A⁻¹) = 1/det(A)")

    A = np.array([[1, 2], [3, 4]], dtype=float)
    A_inv = np.linalg.inv(A)

    print(f"A:\n{A}")
    print(f"A⁻¹:\n{A_inv}")

    det_A = np.linalg.det(A)
    det_A_inv = np.linalg.det(A_inv)

    print(f"\ndet(A) = {det_A:.4f}")
    print(f"1/det(A) = {1/det_A:.4f}")
    print(f"det(A⁻¹) = {det_A_inv:.4f}")

    # ========================================
    # 9. 特殊矩陣
    # ========================================
    print_separator("9. 特殊矩陣的行列式")

    # 對角矩陣
    D = np.diag([2, 3, 5])
    print(f"對角矩陣 D:\n{D}")
    print(f"det(D) = {np.linalg.det(D):.4f}（= 2×3×5 = 30）")

    # 上三角矩陣
    U = np.array([[2, 1, 3], [0, 4, 5], [0, 0, 6]], dtype=float)
    print(f"\n上三角矩陣 U:\n{U}")
    print(f"det(U) = {np.linalg.det(U):.4f}（= 2×4×6 = 48）")

    # 正交矩陣
    theta = np.pi / 4
    Q = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    print(f"\n旋轉矩陣（45°）:\n{Q}")
    print(f"det(Q) = {np.linalg.det(Q):.4f}（旋轉 → det = 1）")

    # 反射矩陣
    R = np.array([[1, 0], [0, -1]], dtype=float)
    print(f"\n反射矩陣:\n{R}")
    print(f"det(R) = {np.linalg.det(R):.4f}（反射 → det = -1）")

    # ========================================
    # 10. 奇異矩陣
    # ========================================
    print_separator("10. 奇異矩陣：det(A) = 0")

    # 列相依
    A_singular = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [5, 7, 9]  # = 第一列 + 第二列
    ], dtype=float)
    print(f"列相依矩陣:\n{A_singular}")
    print(f"det = {np.linalg.det(A_singular):.4e}")
    print(f"rank = {np.linalg.matrix_rank(A_singular)}")

    # ========================================
    # 11. 隨機驗證
    # ========================================
    print_separator("11. 隨機矩陣驗證")

    np.random.seed(42)
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 4)

    det_A = np.linalg.det(A)
    det_B = np.linalg.det(B)

    print(f"隨機 4×4 矩陣 A, B")
    print(f"\n驗證 det(AB) = det(A)·det(B):")
    print(f"  det(A)·det(B) = {det_A * det_B:.6f}")
    print(f"  det(AB) = {np.linalg.det(A @ B):.6f}")

    print(f"\n驗證 det(Aᵀ) = det(A):")
    print(f"  det(A) = {det_A:.6f}")
    print(f"  det(Aᵀ) = {np.linalg.det(A.T):.6f}")

    # 總結
    print_separator("NumPy 行列式函數總結")
    print("""
基本用法：
  det = np.linalg.det(A)

重要性質驗證：
  det(I) = 1
  det(AB) = det(A)·det(B)
  det(Aᵀ) = det(A)
  det(A⁻¹) = 1/det(A)
  det(cA) = cⁿ·det(A)

可逆性：
  A 可逆 ⟺ det(A) ≠ 0

注意：
  det(A+B) ≠ det(A) + det(B)（行列式不是線性函數！）
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
