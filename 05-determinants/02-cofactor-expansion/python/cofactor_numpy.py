"""
餘因子展開 - NumPy 版本 (Cofactor Expansion - NumPy Implementation)

本程式示範：
1. NumPy 實作餘因子展開
2. 餘因子矩陣與伴隨矩陣
3. 用伴隨矩陣求逆矩陣
4. 與 np.linalg.inv 比較
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def get_minor_matrix(A: np.ndarray, row: int, col: int) -> np.ndarray:  # EN: Define get_minor_matrix and its behavior.
    """取得去掉第 row 列、第 col 行後的子矩陣"""  # EN: Execute statement: """取得去掉第 row 列、第 col 行後的子矩陣""".
    return np.delete(np.delete(A, row, axis=0), col, axis=1)  # EN: Return a value: return np.delete(np.delete(A, row, axis=0), col, axis=1).


def minor(A: np.ndarray, i: int, j: int) -> float:  # EN: Define minor and its behavior.
    """計算子行列式 Mᵢⱼ"""  # EN: Execute statement: """計算子行列式 Mᵢⱼ""".
    sub = get_minor_matrix(A, i, j)  # EN: Assign sub from expression: get_minor_matrix(A, i, j).
    return np.linalg.det(sub)  # EN: Return a value: return np.linalg.det(sub).


def cofactor(A: np.ndarray, i: int, j: int) -> float:  # EN: Define cofactor and its behavior.
    """計算餘因子 Cᵢⱼ"""  # EN: Execute statement: """計算餘因子 Cᵢⱼ""".
    sign = (-1) ** (i + j)  # EN: Assign sign from expression: (-1) ** (i + j).
    return sign * minor(A, i, j)  # EN: Return a value: return sign * minor(A, i, j).


def determinant_by_cofactor(A: np.ndarray, row: int = 0) -> float:  # EN: Define determinant_by_cofactor and its behavior.
    """用餘因子展開計算行列式（沿指定列）"""  # EN: Execute statement: """用餘因子展開計算行列式（沿指定列）""".
    n = A.shape[0]  # EN: Assign n from expression: A.shape[0].
    det = 0.0  # EN: Assign det from expression: 0.0.
    for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
        det += A[row, j] * cofactor(A, row, j)  # EN: Update det via += using: A[row, j] * cofactor(A, row, j).
    return det  # EN: Return a value: return det.


def cofactor_matrix(A: np.ndarray) -> np.ndarray:  # EN: Define cofactor_matrix and its behavior.
    """計算餘因子矩陣"""  # EN: Execute statement: """計算餘因子矩陣""".
    n = A.shape[0]  # EN: Assign n from expression: A.shape[0].
    C = np.zeros((n, n))  # EN: Assign C from expression: np.zeros((n, n)).
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
            C[i, j] = cofactor(A, i, j)  # EN: Execute statement: C[i, j] = cofactor(A, i, j).
    return C  # EN: Return a value: return C.


def adjugate(A: np.ndarray) -> np.ndarray:  # EN: Define adjugate and its behavior.
    """計算伴隨矩陣 adj(A) = Cᵀ"""  # EN: Execute statement: """計算伴隨矩陣 adj(A) = Cᵀ""".
    return cofactor_matrix(A).T  # EN: Return a value: return cofactor_matrix(A).T.


def inverse_by_adjugate(A: np.ndarray) -> np.ndarray:  # EN: Define inverse_by_adjugate and its behavior.
    """用伴隨矩陣計算逆矩陣"""  # EN: Execute statement: """用伴隨矩陣計算逆矩陣""".
    det = np.linalg.det(A)  # EN: Assign det from expression: np.linalg.det(A).
    if abs(det) < 1e-10:  # EN: Branch on a condition: if abs(det) < 1e-10:.
        raise ValueError("矩陣不可逆")  # EN: Raise an exception: raise ValueError("矩陣不可逆").
    return adjugate(A) / det  # EN: Return a value: return adjugate(A) / det.


def main():  # EN: Define main and its behavior.
    print_separator("餘因子展開示範（NumPy 版）\nCofactor Expansion Demo (NumPy)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 子行列式與餘因子
    # ========================================
    print_separator("1. 子行列式與餘因子")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 2, 3],  # EN: Execute statement: [1, 2, 3],.
        [4, 5, 6],  # EN: Execute statement: [4, 5, 6],.
        [7, 8, 9]  # EN: Execute statement: [7, 8, 9].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A:\n{A}")  # EN: Print formatted output to the console.

    print("\n所有子行列式 Mᵢⱼ：")  # EN: Print formatted output to the console.
    for i in range(3):  # EN: Iterate with a for-loop: for i in range(3):.
        for j in range(3):  # EN: Iterate with a for-loop: for j in range(3):.
            print(f"  M_{i+1}{j+1} = {minor(A, i, j):8.4f}", end="")  # EN: Print formatted output to the console.
        print()  # EN: Print formatted output to the console.

    print("\n所有餘因子 Cᵢⱼ：")  # EN: Print formatted output to the console.
    for i in range(3):  # EN: Iterate with a for-loop: for i in range(3):.
        for j in range(3):  # EN: Iterate with a for-loop: for j in range(3):.
            print(f"  C_{i+1}{j+1} = {cofactor(A, i, j):8.4f}", end="")  # EN: Print formatted output to the console.
        print()  # EN: Print formatted output to the console.

    # ========================================
    # 2. 餘因子展開
    # ========================================
    print_separator("2. 餘因子展開計算行列式")  # EN: Call print_separator(...) to perform an operation.

    print(f"A:\n{A}")  # EN: Print formatted output to the console.

    print("\n沿各列展開：")  # EN: Print formatted output to the console.
    for row in range(3):  # EN: Iterate with a for-loop: for row in range(3):.
        det = determinant_by_cofactor(A, row)  # EN: Assign det from expression: determinant_by_cofactor(A, row).
        print(f"  沿第 {row+1} 列：det(A) = {det:.4f}")  # EN: Print formatted output to the console.

    print(f"\nnp.linalg.det(A) = {np.linalg.det(A):.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 餘因子矩陣與伴隨矩陣
    # ========================================
    print_separator("3. 餘因子矩陣與伴隨矩陣")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [2, 1, 3],  # EN: Execute statement: [2, 1, 3],.
        [1, 0, 2],  # EN: Execute statement: [1, 0, 2],.
        [4, 1, 5]  # EN: Execute statement: [4, 1, 5].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A:\n{A}")  # EN: Print formatted output to the console.
    print(f"\ndet(A) = {np.linalg.det(A):.4f}")  # EN: Print formatted output to the console.

    C = cofactor_matrix(A)  # EN: Assign C from expression: cofactor_matrix(A).
    print(f"\n餘因子矩陣 C:\n{C}")  # EN: Print formatted output to the console.

    adj_A = adjugate(A)  # EN: Assign adj_A from expression: adjugate(A).
    print(f"\n伴隨矩陣 adj(A) = Cᵀ:\n{adj_A}")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 用伴隨矩陣求逆矩陣
    # ========================================
    print_separator("4. 用伴隨矩陣求逆矩陣")  # EN: Call print_separator(...) to perform an operation.

    det_A = np.linalg.det(A)  # EN: Assign det_A from expression: np.linalg.det(A).
    print(f"A⁻¹ = adj(A) / det(A)")  # EN: Print formatted output to the console.
    print(f"    = adj(A) / {det_A:.4f}")  # EN: Print formatted output to the console.

    A_inv = inverse_by_adjugate(A)  # EN: Assign A_inv from expression: inverse_by_adjugate(A).
    print(f"\nA⁻¹（伴隨矩陣法）:\n{A_inv}")  # EN: Print formatted output to the console.

    A_inv_np = np.linalg.inv(A)  # EN: Assign A_inv_np from expression: np.linalg.inv(A).
    print(f"\nA⁻¹（NumPy）:\n{A_inv_np}")  # EN: Print formatted output to the console.

    print(f"\n差異（Frobenius 範數）：{np.linalg.norm(A_inv - A_inv_np):.2e}")  # EN: Print formatted output to the console.

    # 驗證
    print(f"\n驗證 A @ A⁻¹:\n{A @ A_inv}")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 特殊情況
    # ========================================
    print_separator("5. 特殊情況")  # EN: Call print_separator(...) to perform an operation.

    # 上三角矩陣
    U = np.array([  # EN: Assign U from expression: np.array([.
        [2, 3, 1],  # EN: Execute statement: [2, 3, 1],.
        [0, 4, 5],  # EN: Execute statement: [0, 4, 5],.
        [0, 0, 6]  # EN: Execute statement: [0, 0, 6].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"上三角矩陣 U:\n{U}")  # EN: Print formatted output to the console.
    print(f"det(U) = {np.linalg.det(U):.4f}（= 2 × 4 × 6 = 48）")  # EN: Print formatted output to the console.

    adj_U = adjugate(U)  # EN: Assign adj_U from expression: adjugate(U).
    print(f"\nadj(U):\n{adj_U}")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 2×2 公式驗證
    # ========================================
    print_separator("6. 2×2 伴隨矩陣公式")  # EN: Call print_separator(...) to perform an operation.

    A2 = np.array([[3, 4], [5, 6]], dtype=float)  # EN: Assign A2 from expression: np.array([[3, 4], [5, 6]], dtype=float).
    print(f"A:\n{A2}")  # EN: Print formatted output to the console.
    print(f"\n對於 2×2 矩陣 [[a, b], [c, d]]：")  # EN: Print formatted output to the console.
    print(f"adj(A) = [[d, -b], [-c, a]]")  # EN: Print formatted output to the console.
    print(f"       = [[{A2[1,1]}, {-A2[0,1]}], [{-A2[1,0]}, {A2[0,0]}]]")  # EN: Print formatted output to the console.

    adj_A2 = adjugate(A2)  # EN: Assign adj_A2 from expression: adjugate(A2).
    print(f"\n計算得到的 adj(A):\n{adj_A2}")  # EN: Print formatted output to the console.

    det_A2 = np.linalg.det(A2)  # EN: Assign det_A2 from expression: np.linalg.det(A2).
    print(f"\ndet(A) = ad - bc = {A2[0,0]}×{A2[1,1]} - {A2[0,1]}×{A2[1,0]} = {det_A2:.4f}")  # EN: Print formatted output to the console.

    A2_inv = adj_A2 / det_A2  # EN: Assign A2_inv from expression: adj_A2 / det_A2.
    print(f"\nA⁻¹ = adj(A)/det(A):\n{A2_inv}")  # EN: Print formatted output to the console.

    # ========================================
    # 7. Cramer's Rule 預覽
    # ========================================
    print_separator("7. Cramer's Rule 預覽")  # EN: Call print_separator(...) to perform an operation.

    print("用餘因子展開可以推導 Cramer's Rule：")  # EN: Print formatted output to the console.
    print("解 Ax = b 時，xⱼ = det(Aⱼ) / det(A)")  # EN: Print formatted output to the console.
    print("其中 Aⱼ 是把 A 的第 j 行換成 b")  # EN: Print formatted output to the console.

    A = np.array([[2, 1], [5, 3]], dtype=float)  # EN: Assign A from expression: np.array([[2, 1], [5, 3]], dtype=float).
    b = np.array([8, 21], dtype=float)  # EN: Assign b from expression: np.array([8, 21], dtype=float).

    print(f"\n例：Ax = b")  # EN: Print formatted output to the console.
    print(f"A:\n{A}")  # EN: Print formatted output to the console.
    print(f"b: {b}")  # EN: Print formatted output to the console.

    det_A = np.linalg.det(A)  # EN: Assign det_A from expression: np.linalg.det(A).
    A1 = A.copy()  # EN: Assign A1 from expression: A.copy().
    A1[:, 0] = b  # EN: Execute statement: A1[:, 0] = b.
    A2 = A.copy()  # EN: Assign A2 from expression: A.copy().
    A2[:, 1] = b  # EN: Execute statement: A2[:, 1] = b.

    x1 = np.linalg.det(A1) / det_A  # EN: Assign x1 from expression: np.linalg.det(A1) / det_A.
    x2 = np.linalg.det(A2) / det_A  # EN: Assign x2 from expression: np.linalg.det(A2) / det_A.

    print(f"\nx₁ = det(A₁)/det(A) = {np.linalg.det(A1)}/{det_A} = {x1:.4f}")  # EN: Print formatted output to the console.
    print(f"x₂ = det(A₂)/det(A) = {np.linalg.det(A2)}/{det_A} = {x2:.4f}")  # EN: Print formatted output to the console.

    x_direct = np.linalg.solve(A, b)  # EN: Assign x_direct from expression: np.linalg.solve(A, b).
    print(f"\n直接解：x = {x_direct}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("NumPy 餘因子展開總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
