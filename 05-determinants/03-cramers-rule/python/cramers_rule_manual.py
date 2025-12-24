"""
克萊姆法則 - 手刻版本 (Cramer's Rule - Manual Implementation)

本程式示範：
1. 克萊姆法則的實作
2. 2×2 和 3×3 系統求解
3. 與消去法比較
"""  # EN: Execute statement: """.

from typing import List, Optional  # EN: Import symbol(s) from a module: from typing import List, Optional.


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def print_matrix(name: str, M: List[List[float]]) -> None:  # EN: Define print_matrix and its behavior.
    print(f"{name} =")  # EN: Print formatted output to the console.
    for row in M:  # EN: Iterate with a for-loop: for row in M:.
        formatted = [f"{x:8.4f}" for x in row]  # EN: Assign formatted from expression: [f"{x:8.4f}" for x in row].
        print(f"  [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


def print_vector(name: str, v: List[float]) -> None:  # EN: Define print_vector and its behavior.
    formatted = [f"{x:.4f}" for x in v]  # EN: Assign formatted from expression: [f"{x:.4f}" for x in v].
    print(f"{name} = [{', '.join(formatted)}]")  # EN: Print formatted output to the console.


# ========================================
# 行列式計算
# ========================================

def det_2x2(A: List[List[float]]) -> float:  # EN: Define det_2x2 and its behavior.
    """2×2 行列式"""  # EN: Execute statement: """2×2 行列式""".
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]  # EN: Return a value: return A[0][0] * A[1][1] - A[0][1] * A[1][0].


def det_3x3(A: List[List[float]]) -> float:  # EN: Define det_3x3 and its behavior.
    """3×3 行列式"""  # EN: Execute statement: """3×3 行列式""".
    a, b, c = A[0]  # EN: Execute statement: a, b, c = A[0].
    d, e, f = A[1]  # EN: Execute statement: d, e, f = A[1].
    g, h, i = A[2]  # EN: Execute statement: g, h, i = A[2].
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)  # EN: Return a value: return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g).


def determinant(A: List[List[float]]) -> float:  # EN: Define determinant and its behavior.
    """計算行列式（支援 2×2 和 3×3）"""  # EN: Execute statement: """計算行列式（支援 2×2 和 3×3）""".
    n = len(A)  # EN: Assign n from expression: len(A).
    if n == 2:  # EN: Branch on a condition: if n == 2:.
        return det_2x2(A)  # EN: Return a value: return det_2x2(A).
    elif n == 3:  # EN: Branch on a condition: elif n == 3:.
        return det_3x3(A)  # EN: Return a value: return det_3x3(A).
    else:  # EN: Execute the fallback branch when prior conditions are false.
        raise ValueError("僅支援 2×2 和 3×3 矩陣")  # EN: Raise an exception: raise ValueError("僅支援 2×2 和 3×3 矩陣").


# ========================================
# 克萊姆法則
# ========================================

def replace_column(A: List[List[float]], b: List[float], j: int) -> List[List[float]]:  # EN: Define replace_column and its behavior.
    """建立 Aⱼ：把 A 的第 j 行換成 b"""  # EN: Execute statement: """建立 Aⱼ：把 A 的第 j 行換成 b""".
    n = len(A)  # EN: Assign n from expression: len(A).
    Aj = [row[:] for row in A]  # 複製  # EN: Assign Aj from expression: [row[:] for row in A] # 複製.
    for i in range(n):  # EN: Iterate with a for-loop: for i in range(n):.
        Aj[i][j] = b[i]  # EN: Execute statement: Aj[i][j] = b[i].
    return Aj  # EN: Return a value: return Aj.


def cramers_rule(A: List[List[float]], b: List[float]) -> Optional[List[float]]:  # EN: Define cramers_rule and its behavior.
    """
    用克萊姆法則解 Ax = b

    xⱼ = det(Aⱼ) / det(A)
    """  # EN: Execute statement: """.
    n = len(A)  # EN: Assign n from expression: len(A).
    det_A = determinant(A)  # EN: Assign det_A from expression: determinant(A).

    if abs(det_A) < 1e-10:  # EN: Branch on a condition: if abs(det_A) < 1e-10:.
        return None  # 矩陣奇異  # EN: Return a value: return None # 矩陣奇異.

    x = []  # EN: Assign x from expression: [].
    for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
        Aj = replace_column(A, b, j)  # EN: Assign Aj from expression: replace_column(A, b, j).
        det_Aj = determinant(Aj)  # EN: Assign det_Aj from expression: determinant(Aj).
        x.append(det_Aj / det_A)  # EN: Execute statement: x.append(det_Aj / det_A).

    return x  # EN: Return a value: return x.


def cramers_rule_verbose(A: List[List[float]], b: List[float]) -> Optional[List[float]]:  # EN: Define cramers_rule_verbose and its behavior.
    """克萊姆法則（詳細過程）"""  # EN: Execute statement: """克萊姆法則（詳細過程）""".
    n = len(A)  # EN: Assign n from expression: len(A).
    det_A = determinant(A)  # EN: Assign det_A from expression: determinant(A).

    print(f"det(A) = {det_A:.4f}")  # EN: Print formatted output to the console.

    if abs(det_A) < 1e-10:  # EN: Branch on a condition: if abs(det_A) < 1e-10:.
        print("det(A) = 0，矩陣奇異，無法使用克萊姆法則")  # EN: Print formatted output to the console.
        return None  # EN: Return a value: return None.

    x = []  # EN: Assign x from expression: [].
    for j in range(n):  # EN: Iterate with a for-loop: for j in range(n):.
        Aj = replace_column(A, b, j)  # EN: Assign Aj from expression: replace_column(A, b, j).
        det_Aj = determinant(Aj)  # EN: Assign det_Aj from expression: determinant(Aj).
        xj = det_Aj / det_A  # EN: Assign xj from expression: det_Aj / det_A.

        print(f"\nA{j+1}（第 {j+1} 行換成 b）：")  # EN: Print formatted output to the console.
        print_matrix(f"A{j+1}", Aj)  # EN: Call print_matrix(...) to perform an operation.
        print(f"det(A{j+1}) = {det_Aj:.4f}")  # EN: Print formatted output to the console.
        print(f"x{j+1} = det(A{j+1})/det(A) = {det_Aj:.4f}/{det_A:.4f} = {xj:.4f}")  # EN: Print formatted output to the console.

        x.append(xj)  # EN: Execute statement: x.append(xj).

    return x  # EN: Return a value: return x.


# ========================================
# 高斯消去法（用於比較）
# ========================================

def gaussian_elimination(A: List[List[float]], b: List[float]) -> Optional[List[float]]:  # EN: Define gaussian_elimination and its behavior.
    """高斯消去法解 Ax = b"""  # EN: Execute statement: """高斯消去法解 Ax = b""".
    n = len(A)  # EN: Assign n from expression: len(A).

    # 建立增廣矩陣
    Aug = [A[i][:] + [b[i]] for i in range(n)]  # EN: Assign Aug from expression: [A[i][:] + [b[i]] for i in range(n)].

    # 前向消去
    for col in range(n):  # EN: Iterate with a for-loop: for col in range(n):.
        # 找主元
        max_row = col  # EN: Assign max_row from expression: col.
        for row in range(col + 1, n):  # EN: Iterate with a for-loop: for row in range(col + 1, n):.
            if abs(Aug[row][col]) > abs(Aug[max_row][col]):  # EN: Branch on a condition: if abs(Aug[row][col]) > abs(Aug[max_row][col]):.
                max_row = row  # EN: Assign max_row from expression: row.

        Aug[col], Aug[max_row] = Aug[max_row], Aug[col]  # EN: Execute statement: Aug[col], Aug[max_row] = Aug[max_row], Aug[col].

        if abs(Aug[col][col]) < 1e-10:  # EN: Branch on a condition: if abs(Aug[col][col]) < 1e-10:.
            return None  # EN: Return a value: return None.

        # 消去
        for row in range(col + 1, n):  # EN: Iterate with a for-loop: for row in range(col + 1, n):.
            factor = Aug[row][col] / Aug[col][col]  # EN: Assign factor from expression: Aug[row][col] / Aug[col][col].
            for j in range(col, n + 1):  # EN: Iterate with a for-loop: for j in range(col, n + 1):.
                Aug[row][j] -= factor * Aug[col][j]  # EN: Execute statement: Aug[row][j] -= factor * Aug[col][j].

    # 回代
    x = [0.0] * n  # EN: Assign x from expression: [0.0] * n.
    for i in range(n - 1, -1, -1):  # EN: Iterate with a for-loop: for i in range(n - 1, -1, -1):.
        x[i] = Aug[i][n]  # EN: Execute statement: x[i] = Aug[i][n].
        for j in range(i + 1, n):  # EN: Iterate with a for-loop: for j in range(i + 1, n):.
            x[i] -= Aug[i][j] * x[j]  # EN: Execute statement: x[i] -= Aug[i][j] * x[j].
        x[i] /= Aug[i][i]  # EN: Execute statement: x[i] /= Aug[i][i].

    return x  # EN: Return a value: return x.


def main():  # EN: Define main and its behavior.
    print_separator("克萊姆法則示範（手刻版）\nCramer's Rule Demo (Manual)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 2×2 系統
    # ========================================
    print_separator("1. 2×2 系統")  # EN: Call print_separator(...) to perform an operation.

    A = [[2, 3], [4, 5]]  # EN: Assign A from expression: [[2, 3], [4, 5]].
    b = [8, 14]  # EN: Assign b from expression: [8, 14].

    print("方程組：")  # EN: Print formatted output to the console.
    print("  2x + 3y = 8")  # EN: Print formatted output to the console.
    print("  4x + 5y = 14")  # EN: Print formatted output to the console.

    print_matrix("\nA", A)  # EN: Call print_matrix(...) to perform an operation.
    print_vector("b", b)  # EN: Call print_vector(...) to perform an operation.

    print("\n克萊姆法則求解：")  # EN: Print formatted output to the console.
    x = cramers_rule_verbose(A, b)  # EN: Assign x from expression: cramers_rule_verbose(A, b).

    print(f"\n解：x = {x[0]:.4f}, y = {x[1]:.4f}")  # EN: Print formatted output to the console.

    # 驗證
    print("\n驗證：")  # EN: Print formatted output to the console.
    print(f"  2({x[0]:.4f}) + 3({x[1]:.4f}) = {2*x[0] + 3*x[1]:.4f}")  # EN: Print formatted output to the console.
    print(f"  4({x[0]:.4f}) + 5({x[1]:.4f}) = {4*x[0] + 5*x[1]:.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 3×3 系統
    # ========================================
    print_separator("2. 3×3 系統")  # EN: Call print_separator(...) to perform an operation.

    A = [  # EN: Assign A from expression: [.
        [2, 1, -1],  # EN: Execute statement: [2, 1, -1],.
        [-3, -1, 2],  # EN: Execute statement: [-3, -1, 2],.
        [-2, 1, 2]  # EN: Execute statement: [-2, 1, 2].
    ]  # EN: Execute statement: ].
    b = [8, -11, -3]  # EN: Assign b from expression: [8, -11, -3].

    print("方程組：")  # EN: Print formatted output to the console.
    print("   2x +  y -  z =  8")  # EN: Print formatted output to the console.
    print("  -3x -  y + 2z = -11")  # EN: Print formatted output to the console.
    print("  -2x +  y + 2z = -3")  # EN: Print formatted output to the console.

    print_matrix("\nA", A)  # EN: Call print_matrix(...) to perform an operation.
    print_vector("b", b)  # EN: Call print_vector(...) to perform an operation.

    print("\n克萊姆法則求解：")  # EN: Print formatted output to the console.
    x = cramers_rule_verbose(A, b)  # EN: Assign x from expression: cramers_rule_verbose(A, b).

    print(f"\n解：x = {x[0]:.4f}, y = {x[1]:.4f}, z = {x[2]:.4f}")  # EN: Print formatted output to the console.

    # 驗證
    print("\n驗證：")  # EN: Print formatted output to the console.
    print(f"   2({x[0]}) + ({x[1]}) - ({x[2]}) = {2*x[0] + x[1] - x[2]:.4f}")  # EN: Print formatted output to the console.
    print(f"  -3({x[0]}) - ({x[1]}) + 2({x[2]}) = {-3*x[0] - x[1] + 2*x[2]:.4f}")  # EN: Print formatted output to the console.
    print(f"  -2({x[0]}) + ({x[1]}) + 2({x[2]}) = {-2*x[0] + x[1] + 2*x[2]:.4f}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 與高斯消去比較
    # ========================================
    print_separator("3. 與高斯消去比較")  # EN: Call print_separator(...) to perform an operation.

    A = [[1, 2, 1], [2, 1, 1], [1, 1, 2]]  # EN: Assign A from expression: [[1, 2, 1], [2, 1, 1], [1, 1, 2]].
    b = [6, 5, 6]  # EN: Assign b from expression: [6, 5, 6].

    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print_vector("b", b)  # EN: Call print_vector(...) to perform an operation.

    x_cramer = cramers_rule(A, b)  # EN: Assign x_cramer from expression: cramers_rule(A, b).
    x_gauss = gaussian_elimination(A, b)  # EN: Assign x_gauss from expression: gaussian_elimination(A, b).

    print(f"\n克萊姆法則：x = [{x_cramer[0]:.4f}, {x_cramer[1]:.4f}, {x_cramer[2]:.4f}]")  # EN: Print formatted output to the console.
    print(f"高斯消去：  x = [{x_gauss[0]:.4f}, {x_gauss[1]:.4f}, {x_gauss[2]:.4f}]")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 只求一個未知數
    # ========================================
    print_separator("4. 只求特定未知數")  # EN: Call print_separator(...) to perform an operation.

    A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]  # EN: Assign A from expression: [[1, 2, 3], [4, 5, 6], [7, 8, 10]].
    b = [14, 32, 53]  # EN: Assign b from expression: [14, 32, 53].

    print("假設只需要求 x₂：")  # EN: Print formatted output to the console.
    print_matrix("A", A)  # EN: Call print_matrix(...) to perform an operation.
    print_vector("b", b)  # EN: Call print_vector(...) to perform an operation.

    det_A = determinant(A)  # EN: Assign det_A from expression: determinant(A).
    A2 = replace_column(A, b, 1)  # EN: Assign A2 from expression: replace_column(A, b, 1).
    det_A2 = determinant(A2)  # EN: Assign det_A2 from expression: determinant(A2).
    x2 = det_A2 / det_A  # EN: Assign x2 from expression: det_A2 / det_A.

    print(f"\ndet(A) = {det_A:.4f}")  # EN: Print formatted output to the console.
    print_matrix("A₂", A2)  # EN: Call print_matrix(...) to perform an operation.
    print(f"det(A₂) = {det_A2:.4f}")  # EN: Print formatted output to the console.
    print(f"\nx₂ = det(A₂)/det(A) = {x2:.4f}")  # EN: Print formatted output to the console.

    # 完整解驗證
    x_full = cramers_rule(A, b)  # EN: Assign x_full from expression: cramers_rule(A, b).
    print(f"\n（完整解：x = [{x_full[0]:.4f}, {x_full[1]:.4f}, {x_full[2]:.4f}]）")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 奇異矩陣情況
    # ========================================
    print_separator("5. 奇異矩陣情況")  # EN: Call print_separator(...) to perform an operation.

    A_singular = [[1, 2], [2, 4]]  # EN: Assign A_singular from expression: [[1, 2], [2, 4]].
    b_singular = [3, 6]  # EN: Assign b_singular from expression: [3, 6].

    print_matrix("A（奇異）", A_singular)  # EN: Call print_matrix(...) to perform an operation.
    print_vector("b", b_singular)  # EN: Call print_vector(...) to perform an operation.

    det_A = determinant(A_singular)  # EN: Assign det_A from expression: determinant(A_singular).
    print(f"\ndet(A) = {det_A}")  # EN: Print formatted output to the console.

    result = cramers_rule(A_singular, b_singular)  # EN: Assign result from expression: cramers_rule(A_singular, b_singular).
    if result is None:  # EN: Branch on a condition: if result is None:.
        print("克萊姆法則不適用（det(A) = 0）")  # EN: Print formatted output to the console.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
