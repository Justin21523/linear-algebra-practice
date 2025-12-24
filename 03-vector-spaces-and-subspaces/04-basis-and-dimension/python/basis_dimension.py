"""
基底與維度 (Basis and Dimension)

本程式示範：
1. 找向量空間的基底
2. 計算維度
3. 座標變換
4. 驗證基底的性質

This program demonstrates finding bases, computing dimensions,
and coordinate transformations.
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.
from typing import List, Tuple  # EN: Import symbol(s) from a module: from typing import List, Tuple.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def find_basis(vectors: List[np.ndarray]) -> Tuple[List[np.ndarray], List[int]]:  # EN: Define find_basis and its behavior.
    """
    從向量組中找出一組基底（最大獨立子集）

    Returns:
        (基底向量列表, 基底向量的原索引)
    """  # EN: Execute statement: """.
    if len(vectors) == 0:  # EN: Branch on a condition: if len(vectors) == 0:.
        return [], []  # EN: Return a value: return [], [].

    A = np.column_stack(vectors)  # EN: Assign A from expression: np.column_stack(vectors).
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.

    # RREF 找主元行
    A_work = A.astype(float).copy()  # EN: Assign A_work from expression: A.astype(float).copy().
    pivot_cols = []  # EN: Assign pivot_cols from expression: [].
    row = 0  # EN: Assign row from expression: 0.

    for col in range(n):  # EN: Iterate with a for-loop: for col in range(n):.
        if row >= m:  # EN: Branch on a condition: if row >= m:.
            break  # EN: Control flow statement: break.

        max_row = row + np.argmax(np.abs(A_work[row:, col]))  # EN: Assign max_row from expression: row + np.argmax(np.abs(A_work[row:, col])).

        if np.abs(A_work[max_row, col]) < 1e-10:  # EN: Branch on a condition: if np.abs(A_work[max_row, col]) < 1e-10:.
            continue  # EN: Control flow statement: continue.

        A_work[[row, max_row]] = A_work[[max_row, row]]  # EN: Execute statement: A_work[[row, max_row]] = A_work[[max_row, row]].

        for i in range(m):  # EN: Iterate with a for-loop: for i in range(m):.
            if i != row and np.abs(A_work[i, col]) > 1e-10:  # EN: Branch on a condition: if i != row and np.abs(A_work[i, col]) > 1e-10:.
                A_work[i] -= A_work[i, col] / A_work[row, col] * A_work[row]  # EN: Execute statement: A_work[i] -= A_work[i, col] / A_work[row, col] * A_work[row].

        pivot_cols.append(col)  # EN: Execute statement: pivot_cols.append(col).
        row += 1  # EN: Update row via += using: 1.

    basis = [vectors[i] for i in pivot_cols]  # EN: Assign basis from expression: [vectors[i] for i in pivot_cols].
    return basis, pivot_cols  # EN: Return a value: return basis, pivot_cols.


def is_basis(vectors: List[np.ndarray], space_dim: int) -> Tuple[bool, str]:  # EN: Define is_basis and its behavior.
    """
    檢查向量組是否為 ℝⁿ 的基底

    Returns:
        (是否為基底, 原因)
    """  # EN: Execute statement: """.
    if len(vectors) != space_dim:  # EN: Branch on a condition: if len(vectors) != space_dim:.
        return False, f"元素數 {len(vectors)} ≠ 空間維度 {space_dim}"  # EN: Return a value: return False, f"元素數 {len(vectors)} ≠ 空間維度 {space_dim}".

    A = np.column_stack(vectors)  # EN: Assign A from expression: np.column_stack(vectors).
    rank = np.linalg.matrix_rank(A)  # EN: Assign rank from expression: np.linalg.matrix_rank(A).

    if rank < space_dim:  # EN: Branch on a condition: if rank < space_dim:.
        return False, f"向量組相依（秩 = {rank} < {space_dim}）"  # EN: Return a value: return False, f"向量組相依（秩 = {rank} < {space_dim}）".

    return True, "獨立且生成空間"  # EN: Return a value: return True, "獨立且生成空間".


def coordinates_in_basis(x: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:  # EN: Define coordinates_in_basis and its behavior.
    """
    計算向量 x 在給定基底下的座標

    x = c₁b₁ + c₂b₂ + ... + cₙbₙ
    返回 [c₁, c₂, ..., cₙ]
    """  # EN: Execute statement: """.
    P = np.column_stack(basis)  # 基底矩陣  # EN: Assign P from expression: np.column_stack(basis) # 基底矩陣.
    coords = np.linalg.solve(P, x)  # [x]_B = P⁻¹ x  # EN: Assign coords from expression: np.linalg.solve(P, x) # [x]_B = P⁻¹ x.
    return coords  # EN: Return a value: return coords.


def vector_from_coordinates(coords: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:  # EN: Define vector_from_coordinates and its behavior.
    """
    從座標和基底還原向量

    x = c₁b₁ + c₂b₂ + ... + cₙbₙ
    """  # EN: Execute statement: """.
    P = np.column_stack(basis)  # EN: Assign P from expression: np.column_stack(basis).
    return P @ coords  # EN: Return a value: return P @ coords.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("基底與維度示範\nBasis and Dimension Demo")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. ℝ² 的標準基底
    # ========================================
    print_separator("1. ℝ² 的標準基底")  # EN: Call print_separator(...) to perform an operation.

    e1 = np.array([1.0, 0.0])  # EN: Assign e1 from expression: np.array([1.0, 0.0]).
    e2 = np.array([0.0, 1.0])  # EN: Assign e2 from expression: np.array([0.0, 1.0]).
    std_basis_2d = [e1, e2]  # EN: Assign std_basis_2d from expression: [e1, e2].

    print(f"e₁ = {e1}")  # EN: Print formatted output to the console.
    print(f"e₂ = {e2}")  # EN: Print formatted output to the console.

    is_b, reason = is_basis(std_basis_2d, 2)  # EN: Execute statement: is_b, reason = is_basis(std_basis_2d, 2).
    print(f"\n是 ℝ² 的基底？ {is_b}（{reason}）")  # EN: Print formatted output to the console.

    # 任意向量的表示
    x = np.array([3.0, 4.0])  # EN: Assign x from expression: np.array([3.0, 4.0]).
    print(f"\n向量 x = {x}")  # EN: Print formatted output to the console.
    print(f"在標準基底下：x = {x[0]}·e₁ + {x[1]}·e₂")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 非標準基底
    # ========================================
    print_separator("2. ℝ² 的非標準基底")  # EN: Call print_separator(...) to perform an operation.

    b1 = np.array([1.0, 1.0])  # EN: Assign b1 from expression: np.array([1.0, 1.0]).
    b2 = np.array([1.0, -1.0])  # EN: Assign b2 from expression: np.array([1.0, -1.0]).
    custom_basis = [b1, b2]  # EN: Assign custom_basis from expression: [b1, b2].

    print(f"b₁ = {b1}")  # EN: Print formatted output to the console.
    print(f"b₂ = {b2}")  # EN: Print formatted output to the console.

    is_b, reason = is_basis(custom_basis, 2)  # EN: Execute statement: is_b, reason = is_basis(custom_basis, 2).
    print(f"\n是 ℝ² 的基底？ {is_b}（{reason}）")  # EN: Print formatted output to the console.

    # 行列式驗證
    B = np.column_stack(custom_basis)  # EN: Assign B from expression: np.column_stack(custom_basis).
    det_B = np.linalg.det(B)  # EN: Assign det_B from expression: np.linalg.det(B).
    print(f"det([b₁|b₂]) = {det_B} ≠ 0 ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 座標變換
    # ========================================
    print_separator("3. 座標變換")  # EN: Call print_separator(...) to perform an operation.

    x = np.array([3.0, 1.0])  # EN: Assign x from expression: np.array([3.0, 1.0]).
    print(f"向量 x = {x}（標準座標）")  # EN: Print formatted output to the console.

    # 在非標準基底下的座標
    coords_B = coordinates_in_basis(x, custom_basis)  # EN: Assign coords_B from expression: coordinates_in_basis(x, custom_basis).
    print(f"\n在基底 B = {{b₁, b₂}} 下的座標：")  # EN: Print formatted output to the console.
    print(f"[x]_B = {coords_B}")  # EN: Print formatted output to the console.

    # 驗證
    x_reconstructed = vector_from_coordinates(coords_B, custom_basis)  # EN: Assign x_reconstructed from expression: vector_from_coordinates(coords_B, custom_basis).
    print(f"\n驗證：{coords_B[0]:.4f}·b₁ + {coords_B[1]:.4f}·b₂ = {x_reconstructed}")  # EN: Print formatted output to the console.

    print(f"\n解釋：")  # EN: Print formatted output to the console.
    print(f"  {coords_B[0]:.4f}·[1,1] + {coords_B[1]:.4f}·[1,-1]")  # EN: Print formatted output to the console.
    print(f"= [{coords_B[0]:.4f} + {coords_B[1]:.4f}, {coords_B[0]:.4f} - {coords_B[1]:.4f}]")  # EN: Print formatted output to the console.
    print(f"= {x}")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 從向量組找基底
    # ========================================
    print_separator("4. 從向量組找基底")  # EN: Call print_separator(...) to perform an operation.

    vectors = [  # EN: Assign vectors from expression: [.
        np.array([1.0, 2.0, 3.0]),  # EN: Execute statement: np.array([1.0, 2.0, 3.0]),.
        np.array([2.0, 4.0, 6.0]),  # = 2 * v1  # EN: Execute statement: np.array([2.0, 4.0, 6.0]), # = 2 * v1.
        np.array([1.0, 1.0, 1.0]),  # EN: Execute statement: np.array([1.0, 1.0, 1.0]),.
        np.array([0.0, 1.0, 2.0]),  # EN: Execute statement: np.array([0.0, 1.0, 2.0]),.
    ]  # EN: Execute statement: ].

    print("向量組：")  # EN: Print formatted output to the console.
    for i, v in enumerate(vectors):  # EN: Iterate with a for-loop: for i, v in enumerate(vectors):.
        print(f"  v{i+1} = {v}")  # EN: Print formatted output to the console.

    basis, indices = find_basis(vectors)  # EN: Execute statement: basis, indices = find_basis(vectors).

    print(f"\n基底（最大獨立子集）：")  # EN: Print formatted output to the console.
    for i, idx in enumerate(indices):  # EN: Iterate with a for-loop: for i, idx in enumerate(indices):.
        print(f"  v{idx+1} = {basis[i]}")  # EN: Print formatted output to the console.

    print(f"\nspan{{v₁, v₂, v₃, v₄}} 的維度 = {len(basis)}")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 維度計算
    # ========================================
    print_separator("5. 維度計算")  # EN: Call print_separator(...) to perform an operation.

    print("常見向量空間的維度：")  # EN: Print formatted output to the console.
    print(f"  dim(ℝⁿ) = n")  # EN: Print formatted output to the console.
    print(f"  dim(ℝ²) = 2")  # EN: Print formatted output to the console.
    print(f"  dim(ℝ³) = 3")  # EN: Print formatted output to the console.
    print(f"  dim(Pₙ) = n + 1（次數 ≤ n 的多項式）")  # EN: Print formatted output to the console.
    print(f"  dim(M₂ₓ₃) = 2 × 3 = 6（2×3 矩陣）")  # EN: Print formatted output to the console.
    print(f"  dim({{0}}) = 0（零空間）")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 秩-零度定理應用
    # ========================================
    print_separator("6. 秩-零度定理應用")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 2, 1, 0],  # EN: Execute statement: [1, 2, 1, 0],.
        [2, 4, 0, 2],  # EN: Execute statement: [2, 4, 0, 2],.
        [3, 6, 2, 1]  # EN: Execute statement: [3, 6, 2, 1].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A =\n{A}\n")  # EN: Print formatted output to the console.

    m, n = A.shape  # EN: Execute statement: m, n = A.shape.
    rank_A = np.linalg.matrix_rank(A)  # EN: Assign rank_A from expression: np.linalg.matrix_rank(A).
    nullity_A = n - rank_A  # EN: Assign nullity_A from expression: n - rank_A.

    print(f"A 是 {m}×{n} 矩陣")  # EN: Print formatted output to the console.
    print(f"rank(A) = dim C(A) = {rank_A}")  # EN: Print formatted output to the console.
    print(f"nullity(A) = dim N(A) = {nullity_A}")  # EN: Print formatted output to the console.
    print(f"\nrank + nullity = {rank_A} + {nullity_A} = {n} = 行數 ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 子空間維度的比較
    # ========================================
    print_separator("7. 子空間維度")  # EN: Call print_separator(...) to perform an operation.

    print("若 W 是 V 的子空間，則 dim(W) ≤ dim(V)")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    print("例：ℝ³ 的子空間")  # EN: Print formatted output to the console.
    print("  • 0 維：原點 {0}")  # EN: Print formatted output to the console.
    print("  • 1 維：通過原點的直線")  # EN: Print formatted output to the console.
    print("  • 2 維：通過原點的平面")  # EN: Print formatted output to the console.
    print("  • 3 維：整個 ℝ³")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 基底變換矩陣
    # ========================================
    print_separator("8. 基底變換矩陣")  # EN: Call print_separator(...) to perform an operation.

    print("設 B₁ 和 B₂ 是 ℝⁿ 的兩組基底")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    B1 = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]  # 標準基底  # EN: Assign B1 from expression: [np.array([1.0, 0.0]), np.array([0.0, 1.0])] # 標準基底.
    B2 = [np.array([1.0, 1.0]), np.array([1.0, -1.0])]  # 非標準基底  # EN: Assign B2 from expression: [np.array([1.0, 1.0]), np.array([1.0, -1.0])] # 非標準基底.

    print("B₁ = {[1,0], [0,1]}（標準基底）")  # EN: Print formatted output to the console.
    print("B₂ = {[1,1], [1,-1]}")  # EN: Print formatted output to the console.

    # 從 B₁ 到 B₂ 的變換矩陣
    P_B1 = np.column_stack(B1)  # EN: Assign P_B1 from expression: np.column_stack(B1).
    P_B2 = np.column_stack(B2)  # EN: Assign P_B2 from expression: np.column_stack(B2).

    # [x]_B2 = P_B2⁻¹ @ P_B1 @ [x]_B1
    # 但 P_B1 = I，所以 [x]_B2 = P_B2⁻¹ @ x
    change_matrix = np.linalg.inv(P_B2) @ P_B1  # EN: Assign change_matrix from expression: np.linalg.inv(P_B2) @ P_B1.
    print(f"\n從 B₁ 到 B₂ 的變換矩陣：")  # EN: Print formatted output to the console.
    print(f"P_{{B₁→B₂}} =\n{change_matrix}")  # EN: Print formatted output to the console.

    # 範例
    x_B1 = np.array([3.0, 1.0])  # 標準座標  # EN: Assign x_B1 from expression: np.array([3.0, 1.0]) # 標準座標.
    x_B2 = change_matrix @ x_B1  # EN: Assign x_B2 from expression: change_matrix @ x_B1.
    print(f"\n[x]_B₁ = {x_B1}")  # EN: Print formatted output to the console.
    print(f"[x]_B₂ = P @ [x]_B₁ = {x_B2}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
    print("""
基底與維度的關鍵概念：

1. 基底 = 獨立 + 生成
2. 維度 = 基底的元素個數（與選擇的基底無關）
3. 座標 = 在特定基底下的係數

重要公式：
- x = P · [x]_B（從 B-座標還原標準座標）
- [x]_B = P⁻¹ · x（從標準座標求 B-座標）
- rank(A) + nullity(A) = n

常見錯誤：
- 混淆「基底元素個數」和「向量的分量數」
- 座標變換矩陣的方向
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("基底與維度示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
