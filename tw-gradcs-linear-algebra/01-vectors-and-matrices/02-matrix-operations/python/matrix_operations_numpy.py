"""
矩陣運算：NumPy 版本 (Matrix Operations: NumPy Implementation)

本程式示範：
1. NumPy 陣列與矩陣的建立
2. 矩陣加減與純量乘法
3. 轉置與對稱性
4. NumPy 的便利功能

This program demonstrates matrix operations using NumPy.
"""

import numpy as np

# 設定輸出格式 (Set output format)
np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線 (Print separator)"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    """主程式 (Main program)"""

    print_separator("矩陣運算示範 - NumPy 版本\nMatrix Operations Demo - NumPy Implementation")

    # ========================================
    # 1. 建立矩陣 (Creating Matrices)
    # ========================================
    print_separator("1. 建立矩陣 (Creating Matrices)")

    # 從巢狀 list 建立
    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=float)

    B = np.array([
        [7, 8, 9],
        [10, 11, 12]
    ], dtype=float)

    print(f"A (shape: {A.shape}):\n{A}\n")
    print(f"B (shape: {B.shape}):\n{B}\n")

    # 其他建立方式
    print("其他建立方式：")
    print(f"np.zeros((2, 3)):\n{np.zeros((2, 3))}\n")
    print(f"np.ones((2, 3)):\n{np.ones((2, 3))}\n")
    print(f"np.eye(3) (單位矩陣):\n{np.eye(3)}\n")
    print(f"np.diag([1, 2, 3]) (對角矩陣):\n{np.diag([1, 2, 3])}\n")

    # arange + reshape
    print(f"np.arange(1, 7).reshape(2, 3):\n{np.arange(1, 7).reshape(2, 3)}\n")

    # ========================================
    # 2. 矩陣加法與減法 (Addition & Subtraction)
    # ========================================
    print_separator("2. 矩陣加法與減法 (Addition & Subtraction)")

    print(f"A:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"A + B:\n{A + B}\n")
    print(f"B - A:\n{B - A}\n")

    # ========================================
    # 3. 純量乘法 (Scalar Multiplication)
    # ========================================
    print_separator("3. 純量乘法 (Scalar Multiplication)")

    c = 2.0
    print(f"{c} * A:\n{c * A}\n")
    print(f"-A:\n{-A}\n")
    print(f"A / 2:\n{A / 2}\n")

    # ========================================
    # 4. 矩陣轉置 (Transpose)
    # ========================================
    print_separator("4. 矩陣轉置 (Transpose)")

    print(f"A (shape: {A.shape}):\n{A}\n")

    # 方法一：.T 屬性
    print(f"A.T (shape: {A.T.shape}):\n{A.T}\n")

    # 方法二：np.transpose()
    print(f"np.transpose(A):\n{np.transpose(A)}\n")

    # 驗證 (Aᵀ)ᵀ = A
    print(f"A.T.T == A ? {np.allclose(A.T.T, A)}")

    # ========================================
    # 5. 轉置的性質 (Transpose Properties)
    # ========================================
    print_separator("5. 轉置的性質 (Transpose Properties)")

    # (A + B)ᵀ = Aᵀ + Bᵀ
    print("驗證 (A + B).T == A.T + B.T:")
    print(f"  (A + B).T:\n{(A + B).T}\n")
    print(f"  A.T + B.T:\n{A.T + B.T}\n")
    print(f"  相等？ {np.allclose((A + B).T, A.T + B.T)}\n")

    # (cA)ᵀ = cAᵀ
    print("驗證 (2*A).T == 2*A.T:")
    print(f"  相等？ {np.allclose((2*A).T, 2*A.T)}")

    # ========================================
    # 6. 對稱矩陣 (Symmetric Matrix)
    # ========================================
    print_separator("6. 對稱矩陣 (Symmetric Matrix)")

    S = np.array([
        [1, 2, 3],
        [2, 5, 6],
        [3, 6, 9]
    ], dtype=float)

    print(f"S:\n{S}\n")
    print(f"S.T:\n{S.T}\n")

    # 檢查對稱性
    is_symmetric = np.allclose(S, S.T)
    print(f"S 是對稱矩陣？ {is_symmetric}")

    # 非對稱矩陣
    N = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    print(f"\nN:\n{N}\n")
    print(f"N 是對稱矩陣？ {np.allclose(N, N.T)}")

    # ========================================
    # 7. 列與行的操作 (Row and Column Operations)
    # ========================================
    print_separator("7. 列與行的操作 (Row and Column Operations)")

    print(f"A:\n{A}\n")

    # 取出列
    print(f"第 0 列 A[0, :]: {A[0, :]}")
    print(f"第 1 列 A[1, :]: {A[1, :]}")

    # 取出行
    print(f"\n第 0 行 A[:, 0]: {A[:, 0]}")
    print(f"第 1 行 A[:, 1]: {A[:, 1]}")
    print(f"第 2 行 A[:, 2]: {A[:, 2]}")

    # 取出子矩陣
    print(f"\n子矩陣 A[0:2, 1:3]:\n{A[0:2, 1:3]}")

    # ========================================
    # 8. 元素級運算 (Element-wise Operations)
    # ========================================
    print_separator("8. 元素級運算 (Element-wise Operations)")

    print(f"A:\n{A}\n")
    print(f"A * A (元素相乘，不是矩陣乘法!):\n{A * A}\n")
    print(f"A ** 2 (元素平方):\n{A ** 2}\n")
    print(f"np.sqrt(A):\n{np.sqrt(A)}\n")

    # ========================================
    # 9. 構造對稱矩陣 (Constructing Symmetric Matrix)
    # ========================================
    print_separator("9. 構造對稱矩陣：AᵀA 與 AAᵀ")

    M = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=float)

    print(f"M (3×2):\n{M}\n")

    # MᵀM: 2×2 對稱矩陣
    MtM = M.T @ M
    print(f"MᵀM (2×2):\n{MtM}")
    print(f"MᵀM 是對稱的？ {np.allclose(MtM, MtM.T)}\n")

    # MMᵀ: 3×3 對稱矩陣
    MMt = M @ M.T
    print(f"MMᵀ (3×3):\n{MMt}")
    print(f"MMᵀ 是對稱的？ {np.allclose(MMt, MMt.T)}")

    # ========================================
    # 10. 有用的矩陣函數 (Useful Matrix Functions)
    # ========================================
    print_separator("10. 有用的 NumPy 函數")

    C = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    print(f"C:\n{C}\n")

    print(f"np.sum(C) (全部元素): {np.sum(C)}")
    print(f"np.sum(C, axis=0) (每行總和): {np.sum(C, axis=0)}")
    print(f"np.sum(C, axis=1) (每列總和): {np.sum(C, axis=1)}")

    print(f"\nnp.max(C): {np.max(C)}")
    print(f"np.min(C): {np.min(C)}")
    print(f"np.mean(C): {np.mean(C)}")

    print(f"\nnp.trace(C) (跡，對角線元素和): {np.trace(C)}")
    print(f"np.diag(C) (取出對角線): {np.diag(C)}")

    # ========================================
    # 11. 矩陣的形狀操作 (Reshaping)
    # ========================================
    print_separator("11. 矩陣形狀操作 (Reshaping)")

    D = np.arange(1, 13)
    print(f"原始 1D 陣列 D: {D}")

    D_2x6 = D.reshape(2, 6)
    print(f"\nD.reshape(2, 6):\n{D_2x6}")

    D_3x4 = D.reshape(3, 4)
    print(f"\nD.reshape(3, 4):\n{D_3x4}")

    D_4x3 = D.reshape(4, 3)
    print(f"\nD.reshape(4, 3):\n{D_4x3}")

    # flatten：攤平回 1D
    print(f"\nD_3x4.flatten(): {D_3x4.flatten()}")

    print()
    print("=" * 60)
    print("所有 NumPy 矩陣運算示範完成！")
    print("All NumPy matrix operations demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
