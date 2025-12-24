"""
行列式的性質 - NumPy 版本 (Determinant Properties - NumPy Implementation)

本程式示範：
1. np.linalg.det 的使用
2. 驗證所有行列式性質
3. 特殊矩陣的行列式
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    print_separator("行列式性質示範（NumPy 版）\nDeterminant Properties Demo (NumPy)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 基本計算
    # ========================================
    print_separator("1. 基本行列式計算")  # EN: Call print_separator(...) to perform an operation.

    # 2×2
    A2 = np.array([[3, 8], [4, 6]], dtype=float)  # EN: Assign A2 from expression: np.array([[3, 8], [4, 6]], dtype=float).
    print(f"A (2×2):\n{A2}")  # EN: Print formatted output to the console.
    print(f"det(A) = {np.linalg.det(A2):.4f}")  # EN: Print formatted output to the console.
    print(f"驗證：3×6 - 8×4 = {3*6 - 8*4}")  # EN: Print formatted output to the console.

    # 3×3
    A3 = np.array([  # EN: Assign A3 from expression: np.array([.
        [1, 2, 3],  # EN: Execute statement: [1, 2, 3],.
        [4, 5, 6],  # EN: Execute statement: [4, 5, 6],.
        [7, 8, 10]  # EN: Execute statement: [7, 8, 10].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).
    print(f"\nA (3×3):\n{A3}")  # EN: Print formatted output to the console.
    print(f"det(A) = {np.linalg.det(A3):.4f}")  # EN: Print formatted output to the console.

    # 4×4
    A4 = np.array([  # EN: Assign A4 from expression: np.array([.
        [1, 2, 3, 4],  # EN: Execute statement: [1, 2, 3, 4],.
        [5, 6, 7, 8],  # EN: Execute statement: [5, 6, 7, 8],.
        [9, 10, 11, 12],  # EN: Execute statement: [9, 10, 11, 12],.
        [13, 14, 15, 17]  # EN: Execute statement: [13, 14, 15, 17].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).
    print(f"\nA (4×4):\n{A4}")  # EN: Print formatted output to the console.
    print(f"det(A) = {np.linalg.det(A4):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 性質 1：det(I) = 1
    # ========================================
    print_separator("2. 性質 1：det(I) = 1")  # EN: Call print_separator(...) to perform an operation.

    for n in [2, 3, 4, 5]:  # EN: Iterate with a for-loop: for n in [2, 3, 4, 5]:.
        I_n = np.eye(n)  # EN: Assign I_n from expression: np.eye(n).
        print(f"det(I_{n}) = {np.linalg.det(I_n):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 性質 2：列交換變號
    # ========================================
    print_separator("3. 性質 2：列交換變號")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 2, 3],  # EN: Execute statement: [1, 2, 3],.
        [4, 5, 6],  # EN: Execute statement: [4, 5, 6],.
        [7, 8, 10]  # EN: Execute statement: [7, 8, 10].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"原矩陣 A:\n{A}")  # EN: Print formatted output to the console.
    print(f"det(A) = {np.linalg.det(A):.4f}")  # EN: Print formatted output to the console.

    # 交換第 1, 2 列
    A_swap = A[[1, 0, 2], :]  # EN: Assign A_swap from expression: A[[1, 0, 2], :].
    print(f"\n交換第 1, 2 列:\n{A_swap}")  # EN: Print formatted output to the console.
    print(f"det(交換後) = {np.linalg.det(A_swap):.4f}")  # EN: Print formatted output to the console.
    print("驗證：變號 ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 性質 3：列加法不變
    # ========================================
    print_separator("4. 性質 3：rᵢ ← rᵢ + c·rⱼ 不變")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 2, 3],  # EN: Execute statement: [1, 2, 3],.
        [4, 5, 6],  # EN: Execute statement: [4, 5, 6],.
        [7, 8, 10]  # EN: Execute statement: [7, 8, 10].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"原矩陣 A:\n{A}")  # EN: Print formatted output to the console.
    print(f"det(A) = {np.linalg.det(A):.4f}")  # EN: Print formatted output to the console.

    # r₂ ← r₂ - 4·r₁
    A_add = A.copy()  # EN: Assign A_add from expression: A.copy().
    A_add[1] = A_add[1] - 4 * A_add[0]  # EN: Execute statement: A_add[1] = A_add[1] - 4 * A_add[0].
    print(f"\nr₂ ← r₂ - 4r₁:\n{A_add}")  # EN: Print formatted output to the console.
    print(f"det(列運算後) = {np.linalg.det(A_add):.4f}")  # EN: Print formatted output to the console.
    print("驗證：不變 ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 乘積公式
    # ========================================
    print_separator("5. 乘積公式：det(AB) = det(A)·det(B)")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([[1, 2], [3, 4]], dtype=float)  # EN: Assign A from expression: np.array([[1, 2], [3, 4]], dtype=float).
    B = np.array([[5, 6], [7, 8]], dtype=float)  # EN: Assign B from expression: np.array([[5, 6], [7, 8]], dtype=float).

    print(f"A:\n{A}")  # EN: Print formatted output to the console.
    print(f"B:\n{B}")  # EN: Print formatted output to the console.
    print(f"AB:\n{A @ B}")  # EN: Print formatted output to the console.

    det_A = np.linalg.det(A)  # EN: Assign det_A from expression: np.linalg.det(A).
    det_B = np.linalg.det(B)  # EN: Assign det_B from expression: np.linalg.det(B).
    det_AB = np.linalg.det(A @ B)  # EN: Assign det_AB from expression: np.linalg.det(A @ B).

    print(f"\ndet(A) = {det_A:.4f}")  # EN: Print formatted output to the console.
    print(f"det(B) = {det_B:.4f}")  # EN: Print formatted output to the console.
    print(f"det(A)·det(B) = {det_A * det_B:.4f}")  # EN: Print formatted output to the console.
    print(f"det(AB) = {det_AB:.4f}")  # EN: Print formatted output to the console.
    print(f"差異 = {abs(det_A * det_B - det_AB):.2e}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 轉置公式
    # ========================================
    print_separator("6. 轉置公式：det(Aᵀ) = det(A)")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 2, 3],  # EN: Execute statement: [1, 2, 3],.
        [4, 5, 6],  # EN: Execute statement: [4, 5, 6],.
        [7, 8, 10]  # EN: Execute statement: [7, 8, 10].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A:\n{A}")  # EN: Print formatted output to the console.
    print(f"Aᵀ:\n{A.T}")  # EN: Print formatted output to the console.
    print(f"\ndet(A) = {np.linalg.det(A):.4f}")  # EN: Print formatted output to the console.
    print(f"det(Aᵀ) = {np.linalg.det(A.T):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 純量乘法
    # ========================================
    print_separator("7. 純量乘法：det(cA) = cⁿ·det(A)")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([[1, 2], [3, 4]], dtype=float)  # EN: Assign A from expression: np.array([[1, 2], [3, 4]], dtype=float).
    c = 3  # EN: Assign c from expression: 3.
    n = 2  # EN: Assign n from expression: 2.

    print(f"A (2×2):\n{A}")  # EN: Print formatted output to the console.
    print(f"c = {c}")  # EN: Print formatted output to the console.
    print(f"{c}A:\n{c * A}")  # EN: Print formatted output to the console.

    det_A = np.linalg.det(A)  # EN: Assign det_A from expression: np.linalg.det(A).
    det_cA = np.linalg.det(c * A)  # EN: Assign det_cA from expression: np.linalg.det(c * A).

    print(f"\ndet(A) = {det_A:.4f}")  # EN: Print formatted output to the console.
    print(f"cⁿ·det(A) = {c}² × {det_A:.4f} = {c**n * det_A:.4f}")  # EN: Print formatted output to the console.
    print(f"det(cA) = {det_cA:.4f}")  # EN: Print formatted output to the console.

    # 3×3 例子
    A3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)  # EN: Assign A3 from expression: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float).
    n = 3  # EN: Assign n from expression: 3.
    print(f"\n3×3 矩陣：cⁿ·det(A) = {c}³ × {np.linalg.det(A3):.4f} = {c**n * np.linalg.det(A3):.4f}")  # EN: Print formatted output to the console.
    print(f"det(cA) = {np.linalg.det(c * A3):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 逆矩陣公式
    # ========================================
    print_separator("8. 逆矩陣公式：det(A⁻¹) = 1/det(A)")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([[1, 2], [3, 4]], dtype=float)  # EN: Assign A from expression: np.array([[1, 2], [3, 4]], dtype=float).
    A_inv = np.linalg.inv(A)  # EN: Assign A_inv from expression: np.linalg.inv(A).

    print(f"A:\n{A}")  # EN: Print formatted output to the console.
    print(f"A⁻¹:\n{A_inv}")  # EN: Print formatted output to the console.

    det_A = np.linalg.det(A)  # EN: Assign det_A from expression: np.linalg.det(A).
    det_A_inv = np.linalg.det(A_inv)  # EN: Assign det_A_inv from expression: np.linalg.det(A_inv).

    print(f"\ndet(A) = {det_A:.4f}")  # EN: Print formatted output to the console.
    print(f"1/det(A) = {1/det_A:.4f}")  # EN: Print formatted output to the console.
    print(f"det(A⁻¹) = {det_A_inv:.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 9. 特殊矩陣
    # ========================================
    print_separator("9. 特殊矩陣的行列式")  # EN: Call print_separator(...) to perform an operation.

    # 對角矩陣
    D = np.diag([2, 3, 5])  # EN: Assign D from expression: np.diag([2, 3, 5]).
    print(f"對角矩陣 D:\n{D}")  # EN: Print formatted output to the console.
    print(f"det(D) = {np.linalg.det(D):.4f}（= 2×3×5 = 30）")  # EN: Print formatted output to the console.

    # 上三角矩陣
    U = np.array([[2, 1, 3], [0, 4, 5], [0, 0, 6]], dtype=float)  # EN: Assign U from expression: np.array([[2, 1, 3], [0, 4, 5], [0, 0, 6]], dtype=float).
    print(f"\n上三角矩陣 U:\n{U}")  # EN: Print formatted output to the console.
    print(f"det(U) = {np.linalg.det(U):.4f}（= 2×4×6 = 48）")  # EN: Print formatted output to the console.

    # 正交矩陣
    theta = np.pi / 4  # EN: Assign theta from expression: np.pi / 4.
    Q = np.array([  # EN: Assign Q from expression: np.array([.
        [np.cos(theta), -np.sin(theta)],  # EN: Execute statement: [np.cos(theta), -np.sin(theta)],.
        [np.sin(theta), np.cos(theta)]  # EN: Execute statement: [np.sin(theta), np.cos(theta)].
    ])  # EN: Execute statement: ]).
    print(f"\n旋轉矩陣（45°）:\n{Q}")  # EN: Print formatted output to the console.
    print(f"det(Q) = {np.linalg.det(Q):.4f}（旋轉 → det = 1）")  # EN: Print formatted output to the console.

    # 反射矩陣
    R = np.array([[1, 0], [0, -1]], dtype=float)  # EN: Assign R from expression: np.array([[1, 0], [0, -1]], dtype=float).
    print(f"\n反射矩陣:\n{R}")  # EN: Print formatted output to the console.
    print(f"det(R) = {np.linalg.det(R):.4f}（反射 → det = -1）")  # EN: Print formatted output to the console.

    # ========================================
    # 10. 奇異矩陣
    # ========================================
    print_separator("10. 奇異矩陣：det(A) = 0")  # EN: Call print_separator(...) to perform an operation.

    # 列相依
    A_singular = np.array([  # EN: Assign A_singular from expression: np.array([.
        [1, 2, 3],  # EN: Execute statement: [1, 2, 3],.
        [4, 5, 6],  # EN: Execute statement: [4, 5, 6],.
        [5, 7, 9]  # = 第一列 + 第二列  # EN: Execute statement: [5, 7, 9] # = 第一列 + 第二列.
    ], dtype=float)  # EN: Execute statement: ], dtype=float).
    print(f"列相依矩陣:\n{A_singular}")  # EN: Print formatted output to the console.
    print(f"det = {np.linalg.det(A_singular):.4e}")  # EN: Print formatted output to the console.
    print(f"rank = {np.linalg.matrix_rank(A_singular)}")  # EN: Print formatted output to the console.

    # ========================================
    # 11. 隨機驗證
    # ========================================
    print_separator("11. 隨機矩陣驗證")  # EN: Call print_separator(...) to perform an operation.

    np.random.seed(42)  # EN: Execute statement: np.random.seed(42).
    A = np.random.randn(4, 4)  # EN: Assign A from expression: np.random.randn(4, 4).
    B = np.random.randn(4, 4)  # EN: Assign B from expression: np.random.randn(4, 4).

    det_A = np.linalg.det(A)  # EN: Assign det_A from expression: np.linalg.det(A).
    det_B = np.linalg.det(B)  # EN: Assign det_B from expression: np.linalg.det(B).

    print(f"隨機 4×4 矩陣 A, B")  # EN: Print formatted output to the console.
    print(f"\n驗證 det(AB) = det(A)·det(B):")  # EN: Print formatted output to the console.
    print(f"  det(A)·det(B) = {det_A * det_B:.6f}")  # EN: Print formatted output to the console.
    print(f"  det(AB) = {np.linalg.det(A @ B):.6f}")  # EN: Print formatted output to the console.

    print(f"\n驗證 det(Aᵀ) = det(A):")  # EN: Print formatted output to the console.
    print(f"  det(A) = {det_A:.6f}")  # EN: Print formatted output to the console.
    print(f"  det(Aᵀ) = {np.linalg.det(A.T):.6f}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("NumPy 行列式函數總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
