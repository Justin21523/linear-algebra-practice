"""
線性獨立 (Linear Independence)

本程式示範：
1. 判斷向量組是否線性獨立
2. 找出線性相依關係（係數）
3. 找最大線性獨立子集
4. 幾何意義視覺化

This program demonstrates how to determine linear independence
and find dependency relations.
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.
from typing import List, Tuple, Optional  # EN: Import symbol(s) from a module: from typing import List, Tuple, Optional.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def is_linearly_independent(vectors: List[np.ndarray]) -> bool:  # EN: Define is_linearly_independent and its behavior.
    """
    判斷向量組是否線性獨立

    方法：將向量排成矩陣的行，檢查 rank 是否等於行數
    """  # EN: Execute statement: """.
    if len(vectors) == 0:  # EN: Branch on a condition: if len(vectors) == 0:.
        return True  # EN: Return a value: return True.

    A = np.column_stack(vectors)  # EN: Assign A from expression: np.column_stack(vectors).
    rank = np.linalg.matrix_rank(A)  # EN: Assign rank from expression: np.linalg.matrix_rank(A).

    return rank == len(vectors)  # EN: Return a value: return rank == len(vectors).


def find_dependency_relation(vectors: List[np.ndarray]) -> Optional[np.ndarray]:  # EN: Define find_dependency_relation and its behavior.
    """
    找出線性相依關係

    若相依，返回係數 c 使得 c₁v₁ + c₂v₂ + ... = 0
    若獨立，返回 None
    """  # EN: Execute statement: """.
    if len(vectors) == 0:  # EN: Branch on a condition: if len(vectors) == 0:.
        return None  # EN: Return a value: return None.

    A = np.column_stack(vectors)  # EN: Assign A from expression: np.column_stack(vectors).

    # 用 SVD 找零空間
    U, S, Vh = np.linalg.svd(A)  # EN: Execute statement: U, S, Vh = np.linalg.svd(A).

    # 找接近零的奇異值
    tol = max(A.shape) * np.finfo(float).eps * S[0] if len(S) > 0 else 1e-10  # EN: Assign tol from expression: max(A.shape) * np.finfo(float).eps * S[0] if len(S) > 0 else 1e-10.
    null_mask = S < tol  # EN: Assign null_mask from expression: S < tol.

    if not np.any(null_mask) and len(S) == len(vectors):  # EN: Branch on a condition: if not np.any(null_mask) and len(S) == len(vectors):.
        return None  # 獨立  # EN: Return a value: return None # 獨立.

    # 取零空間的一個向量
    if len(S) < len(vectors):  # EN: Branch on a condition: if len(S) < len(vectors):.
        # 行數 > 秩，取 Vh 的後面幾列
        null_space = Vh[len(S):, :].T  # EN: Assign null_space from expression: Vh[len(S):, :].T.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        null_space = Vh[null_mask, :].T  # EN: Assign null_space from expression: Vh[null_mask, :].T.

    if null_space.shape[1] > 0:  # EN: Branch on a condition: if null_space.shape[1] > 0:.
        return null_space[:, 0]  # EN: Return a value: return null_space[:, 0].
    return None  # EN: Return a value: return None.


def find_maximal_independent_subset(vectors: List[np.ndarray]) -> Tuple[List[int], int]:  # EN: Define find_maximal_independent_subset and its behavior.
    """
    找最大線性獨立子集

    使用 RREF 方法：主元行對應的原向量是獨立的

    Returns:
        (獨立向量的索引列表, 秩)
    """  # EN: Execute statement: """.
    if len(vectors) == 0:  # EN: Branch on a condition: if len(vectors) == 0:.
        return [], 0  # EN: Return a value: return [], 0.

    A = np.column_stack(vectors)  # EN: Assign A from expression: np.column_stack(vectors).
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.

    # 簡化版 RREF（只需要找主元行）
    A_work = A.astype(float).copy()  # EN: Assign A_work from expression: A.astype(float).copy().
    pivot_cols = []  # EN: Assign pivot_cols from expression: [].
    row = 0  # EN: Assign row from expression: 0.

    for col in range(n):  # EN: Iterate with a for-loop: for col in range(n):.
        if row >= m:  # EN: Branch on a condition: if row >= m:.
            break  # EN: Control flow statement: break.

        # 找主元
        max_row = row + np.argmax(np.abs(A_work[row:, col]))  # EN: Assign max_row from expression: row + np.argmax(np.abs(A_work[row:, col])).

        if np.abs(A_work[max_row, col]) < 1e-10:  # EN: Branch on a condition: if np.abs(A_work[max_row, col]) < 1e-10:.
            continue  # EN: Control flow statement: continue.

        # 換列
        A_work[[row, max_row]] = A_work[[max_row, row]]  # EN: Execute statement: A_work[[row, max_row]] = A_work[[max_row, row]].

        # 消去
        for i in range(m):  # EN: Iterate with a for-loop: for i in range(m):.
            if i != row and np.abs(A_work[i, col]) > 1e-10:  # EN: Branch on a condition: if i != row and np.abs(A_work[i, col]) > 1e-10:.
                A_work[i] -= A_work[i, col] / A_work[row, col] * A_work[row]  # EN: Execute statement: A_work[i] -= A_work[i, col] / A_work[row, col] * A_work[row].

        pivot_cols.append(col)  # EN: Execute statement: pivot_cols.append(col).
        row += 1  # EN: Update row via += using: 1.

    return pivot_cols, len(pivot_cols)  # EN: Return a value: return pivot_cols, len(pivot_cols).


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("線性獨立示範\nLinear Independence Demo")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 範例 1：兩個 2D 向量
    # ========================================
    print_separator("1. 兩個 2D 向量")  # EN: Call print_separator(...) to perform an operation.

    v1 = np.array([1.0, 2.0])  # EN: Assign v1 from expression: np.array([1.0, 2.0]).
    v2 = np.array([3.0, 4.0])  # EN: Assign v2 from expression: np.array([3.0, 4.0]).

    print(f"v₁ = {v1}")  # EN: Print formatted output to the console.
    print(f"v₂ = {v2}")  # EN: Print formatted output to the console.

    independent = is_linearly_independent([v1, v2])  # EN: Assign independent from expression: is_linearly_independent([v1, v2]).
    print(f"\n線性獨立？ {independent}")  # EN: Print formatted output to the console.

    if independent:  # EN: Branch on a condition: if independent:.
        # 用行列式驗證
        det = v1[0] * v2[1] - v1[1] * v2[0]  # EN: Assign det from expression: v1[0] * v2[1] - v1[1] * v2[0].
        print(f"det([v₁|v₂]) = {det} ≠ 0 ✓")  # EN: Print formatted output to the console.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        coeffs = find_dependency_relation([v1, v2])  # EN: Assign coeffs from expression: find_dependency_relation([v1, v2]).
        print(f"相依關係：{coeffs[0]:.4f}·v₁ + {coeffs[1]:.4f}·v₂ = 0")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 2：平行向量（相依）
    # ========================================
    print_separator("2. 平行向量（相依）")  # EN: Call print_separator(...) to perform an operation.

    v1 = np.array([1.0, 2.0])  # EN: Assign v1 from expression: np.array([1.0, 2.0]).
    v2 = np.array([2.0, 4.0])  # v2 = 2*v1  # EN: Assign v2 from expression: np.array([2.0, 4.0]) # v2 = 2*v1.

    print(f"v₁ = {v1}")  # EN: Print formatted output to the console.
    print(f"v₂ = {v2}")  # EN: Print formatted output to the console.
    print("注意：v₂ = 2·v₁")  # EN: Print formatted output to the console.

    independent = is_linearly_independent([v1, v2])  # EN: Assign independent from expression: is_linearly_independent([v1, v2]).
    print(f"\n線性獨立？ {independent}")  # EN: Print formatted output to the console.

    if not independent:  # EN: Branch on a condition: if not independent:.
        coeffs = find_dependency_relation([v1, v2])  # EN: Assign coeffs from expression: find_dependency_relation([v1, v2]).
        if coeffs is not None:  # EN: Branch on a condition: if coeffs is not None:.
            # 正規化係數
            coeffs = coeffs / np.abs(coeffs).max()  # EN: Assign coeffs from expression: coeffs / np.abs(coeffs).max().
            print(f"相依關係：{coeffs[0]:.4f}·v₁ + {coeffs[1]:.4f}·v₂ = 0")  # EN: Print formatted output to the console.
            print("（即 2·v₁ - 1·v₂ = 0）")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 3：三個 3D 向量（獨立）
    # ========================================
    print_separator("3. 三個獨立的 3D 向量")  # EN: Call print_separator(...) to perform an operation.

    v1 = np.array([1.0, 0.0, 0.0])  # EN: Assign v1 from expression: np.array([1.0, 0.0, 0.0]).
    v2 = np.array([0.0, 1.0, 0.0])  # EN: Assign v2 from expression: np.array([0.0, 1.0, 0.0]).
    v3 = np.array([0.0, 0.0, 1.0])  # EN: Assign v3 from expression: np.array([0.0, 0.0, 1.0]).

    print(f"v₁ = {v1}")  # EN: Print formatted output to the console.
    print(f"v₂ = {v2}")  # EN: Print formatted output to the console.
    print(f"v₃ = {v3}")  # EN: Print formatted output to the console.
    print("（標準基底向量）")  # EN: Print formatted output to the console.

    independent = is_linearly_independent([v1, v2, v3])  # EN: Assign independent from expression: is_linearly_independent([v1, v2, v3]).
    print(f"\n線性獨立？ {independent}")  # EN: Print formatted output to the console.

    A = np.column_stack([v1, v2, v3])  # EN: Assign A from expression: np.column_stack([v1, v2, v3]).
    det = np.linalg.det(A)  # EN: Assign det from expression: np.linalg.det(A).
    print(f"det([v₁|v₂|v₃]) = {det} ≠ 0 ✓")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 4：三個共面向量（相依）
    # ========================================
    print_separator("4. 三個共面向量（相依）")  # EN: Call print_separator(...) to perform an operation.

    v1 = np.array([1.0, 0.0, 0.0])  # EN: Assign v1 from expression: np.array([1.0, 0.0, 0.0]).
    v2 = np.array([0.0, 1.0, 0.0])  # EN: Assign v2 from expression: np.array([0.0, 1.0, 0.0]).
    v3 = np.array([1.0, 1.0, 0.0])  # v3 = v1 + v2  # EN: Assign v3 from expression: np.array([1.0, 1.0, 0.0]) # v3 = v1 + v2.

    print(f"v₁ = {v1}")  # EN: Print formatted output to the console.
    print(f"v₂ = {v2}")  # EN: Print formatted output to the console.
    print(f"v₃ = {v3}")  # EN: Print formatted output to the console.
    print("注意：v₃ = v₁ + v₂，三向量共面（都在 z=0 平面）")  # EN: Print formatted output to the console.

    independent = is_linearly_independent([v1, v2, v3])  # EN: Assign independent from expression: is_linearly_independent([v1, v2, v3]).
    print(f"\n線性獨立？ {independent}")  # EN: Print formatted output to the console.

    coeffs = find_dependency_relation([v1, v2, v3])  # EN: Assign coeffs from expression: find_dependency_relation([v1, v2, v3]).
    if coeffs is not None:  # EN: Branch on a condition: if coeffs is not None:.
        coeffs = coeffs / np.abs(coeffs).max()  # EN: Assign coeffs from expression: coeffs / np.abs(coeffs).max().
        print(f"相依關係：{coeffs[0]:.4f}·v₁ + {coeffs[1]:.4f}·v₂ + {coeffs[2]:.4f}·v₃ = 0")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 5：找最大獨立子集
    # ========================================
    print_separator("5. 找最大線性獨立子集")  # EN: Call print_separator(...) to perform an operation.

    vectors = [  # EN: Assign vectors from expression: [.
        np.array([1.0, 2.0, 3.0]),  # EN: Execute statement: np.array([1.0, 2.0, 3.0]),.
        np.array([2.0, 4.0, 6.0]),  # = 2 * v1  # EN: Execute statement: np.array([2.0, 4.0, 6.0]), # = 2 * v1.
        np.array([1.0, 1.0, 1.0]),  # EN: Execute statement: np.array([1.0, 1.0, 1.0]),.
        np.array([0.0, 1.0, 2.0]),  # EN: Execute statement: np.array([0.0, 1.0, 2.0]),.
    ]  # EN: Execute statement: ].

    print("向量組：")  # EN: Print formatted output to the console.
    for i, v in enumerate(vectors):  # EN: Iterate with a for-loop: for i, v in enumerate(vectors):.
        print(f"  v{i+1} = {v}")  # EN: Print formatted output to the console.

    print("\n注意：v₂ = 2·v₁")  # EN: Print formatted output to the console.

    independent_indices, rank = find_maximal_independent_subset(vectors)  # EN: Execute statement: independent_indices, rank = find_maximal_independent_subset(vectors).

    print(f"\n最大獨立子集的索引：{independent_indices}")  # EN: Print formatted output to the console.
    print(f"秩 = {rank}")  # EN: Print formatted output to the console.

    print("\n獨立向量：")  # EN: Print formatted output to the console.
    for idx in independent_indices:  # EN: Iterate with a for-loop: for idx in independent_indices:.
        print(f"  v{idx+1} = {vectors[idx]}")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 6：向量數超過維度（必定相依）
    # ========================================
    print_separator("6. 向量數 > 維度（必定相依）")  # EN: Call print_separator(...) to perform an operation.

    # 4 個 3D 向量
    vectors_4 = [  # EN: Assign vectors_4 from expression: [.
        np.array([1.0, 0.0, 0.0]),  # EN: Execute statement: np.array([1.0, 0.0, 0.0]),.
        np.array([0.0, 1.0, 0.0]),  # EN: Execute statement: np.array([0.0, 1.0, 0.0]),.
        np.array([0.0, 0.0, 1.0]),  # EN: Execute statement: np.array([0.0, 0.0, 1.0]),.
        np.array([1.0, 1.0, 1.0]),  # EN: Execute statement: np.array([1.0, 1.0, 1.0]),.
    ]  # EN: Execute statement: ].

    print("4 個 3D 向量：")  # EN: Print formatted output to the console.
    for i, v in enumerate(vectors_4):  # EN: Iterate with a for-loop: for i, v in enumerate(vectors_4):.
        print(f"  v{i+1} = {v}")  # EN: Print formatted output to the console.

    print(f"\n在 ℝ³ 中，最多 3 個向量可以獨立")  # EN: Print formatted output to the console.

    independent = is_linearly_independent(vectors_4)  # EN: Assign independent from expression: is_linearly_independent(vectors_4).
    print(f"這 4 個向量線性獨立？ {independent}")  # EN: Print formatted output to the console.

    coeffs = find_dependency_relation(vectors_4)  # EN: Assign coeffs from expression: find_dependency_relation(vectors_4).
    if coeffs is not None:  # EN: Branch on a condition: if coeffs is not None:.
        print(f"相依關係係數：{coeffs}")  # EN: Print formatted output to the console.
        # 驗證
        result = sum(c * v for c, v in zip(coeffs, vectors_4))  # EN: Assign result from expression: sum(c * v for c, v in zip(coeffs, vectors_4)).
        print(f"驗證 Σcᵢvᵢ = {result}")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 7：包含零向量
    # ========================================
    print_separator("7. 包含零向量（必定相依）")  # EN: Call print_separator(...) to perform an operation.

    v1 = np.array([1.0, 2.0])  # EN: Assign v1 from expression: np.array([1.0, 2.0]).
    v2 = np.array([0.0, 0.0])  # 零向量  # EN: Assign v2 from expression: np.array([0.0, 0.0]) # 零向量.

    print(f"v₁ = {v1}")  # EN: Print formatted output to the console.
    print(f"v₂ = {v2}（零向量）")  # EN: Print formatted output to the console.

    independent = is_linearly_independent([v1, v2])  # EN: Assign independent from expression: is_linearly_independent([v1, v2]).
    print(f"\n線性獨立？ {independent}")  # EN: Print formatted output to the console.
    print("原因：0·v₁ + 1·v₂ = 0，係數 (0, 1) 不全為零")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 8：秩與獨立性
    # ========================================
    print_separator("8. 秩與獨立性的關係")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 2, 3],  # EN: Execute statement: [1, 2, 3],.
        [4, 5, 6],  # EN: Execute statement: [4, 5, 6],.
        [7, 8, 9]  # EN: Execute statement: [7, 8, 9].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A =\n{A}\n")  # EN: Print formatted output to the console.

    rank = np.linalg.matrix_rank(A)  # EN: Assign rank from expression: np.linalg.matrix_rank(A).
    print(f"rank(A) = {rank}")  # EN: Print formatted output to the console.
    print(f"行數 n = {A.shape[1]}")  # EN: Print formatted output to the console.
    print(f"\n行向量獨立？ rank == n ? {rank == A.shape[1]}")  # EN: Print formatted output to the console.

    if rank < A.shape[1]:  # EN: Branch on a condition: if rank < A.shape[1]:.
        print(f"\n行向量相依！（秩 {rank} < 行數 {A.shape[1]}）")  # EN: Print formatted output to the console.

        # 找相依關係
        col_vectors = [A[:, j] for j in range(A.shape[1])]  # EN: Assign col_vectors from expression: [A[:, j] for j in range(A.shape[1])].
        coeffs = find_dependency_relation(col_vectors)  # EN: Assign coeffs from expression: find_dependency_relation(col_vectors).
        if coeffs is not None:  # EN: Branch on a condition: if coeffs is not None:.
            print(f"相依關係：{coeffs}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
    print("""
線性獨立判斷方法：

1. 定義法：c₁v₁ + ... + cₖvₖ = 0 是否只有零解
2. 矩陣法：rank([v₁|...|vₖ]) == k ?
3. 行列式法（方陣）：det ≠ 0 ?

快速判斷：
- 包含零向量 → 相依
- ℝⁿ 中 >n 個向量 → 相依
- 2 個向量平行 → 相依
- n 個向量在 ℝⁿ 中，det ≠ 0 → 獨立
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("線性獨立示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
