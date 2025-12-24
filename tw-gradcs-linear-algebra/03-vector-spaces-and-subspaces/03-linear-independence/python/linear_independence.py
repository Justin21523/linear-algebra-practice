"""
線性獨立 (Linear Independence)

本程式示範：
1. 判斷向量組是否線性獨立
2. 找出線性相依關係（係數）
3. 找最大線性獨立子集
4. 幾何意義視覺化

This program demonstrates how to determine linear independence
and find dependency relations.
"""

import numpy as np
from typing import List, Tuple, Optional

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def is_linearly_independent(vectors: List[np.ndarray]) -> bool:
    """
    判斷向量組是否線性獨立

    方法：將向量排成矩陣的行，檢查 rank 是否等於行數
    """
    if len(vectors) == 0:
        return True

    A = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(A)

    return rank == len(vectors)


def find_dependency_relation(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    找出線性相依關係

    若相依，返回係數 c 使得 c₁v₁ + c₂v₂ + ... = 0
    若獨立，返回 None
    """
    if len(vectors) == 0:
        return None

    A = np.column_stack(vectors)

    # 用 SVD 找零空間
    U, S, Vh = np.linalg.svd(A)

    # 找接近零的奇異值
    tol = max(A.shape) * np.finfo(float).eps * S[0] if len(S) > 0 else 1e-10
    null_mask = S < tol

    if not np.any(null_mask) and len(S) == len(vectors):
        return None  # 獨立

    # 取零空間的一個向量
    if len(S) < len(vectors):
        # 行數 > 秩，取 Vh 的後面幾列
        null_space = Vh[len(S):, :].T
    else:
        null_space = Vh[null_mask, :].T

    if null_space.shape[1] > 0:
        return null_space[:, 0]
    return None


def find_maximal_independent_subset(vectors: List[np.ndarray]) -> Tuple[List[int], int]:
    """
    找最大線性獨立子集

    使用 RREF 方法：主元行對應的原向量是獨立的

    Returns:
        (獨立向量的索引列表, 秩)
    """
    if len(vectors) == 0:
        return [], 0

    A = np.column_stack(vectors)
    m, n = A.shape

    # 簡化版 RREF（只需要找主元行）
    A_work = A.astype(float).copy()
    pivot_cols = []
    row = 0

    for col in range(n):
        if row >= m:
            break

        # 找主元
        max_row = row + np.argmax(np.abs(A_work[row:, col]))

        if np.abs(A_work[max_row, col]) < 1e-10:
            continue

        # 換列
        A_work[[row, max_row]] = A_work[[max_row, row]]

        # 消去
        for i in range(m):
            if i != row and np.abs(A_work[i, col]) > 1e-10:
                A_work[i] -= A_work[i, col] / A_work[row, col] * A_work[row]

        pivot_cols.append(col)
        row += 1

    return pivot_cols, len(pivot_cols)


def main():
    """主程式"""

    print_separator("線性獨立示範\nLinear Independence Demo")

    # ========================================
    # 範例 1：兩個 2D 向量
    # ========================================
    print_separator("1. 兩個 2D 向量")

    v1 = np.array([1.0, 2.0])
    v2 = np.array([3.0, 4.0])

    print(f"v₁ = {v1}")
    print(f"v₂ = {v2}")

    independent = is_linearly_independent([v1, v2])
    print(f"\n線性獨立？ {independent}")

    if independent:
        # 用行列式驗證
        det = v1[0] * v2[1] - v1[1] * v2[0]
        print(f"det([v₁|v₂]) = {det} ≠ 0 ✓")
    else:
        coeffs = find_dependency_relation([v1, v2])
        print(f"相依關係：{coeffs[0]:.4f}·v₁ + {coeffs[1]:.4f}·v₂ = 0")

    # ========================================
    # 範例 2：平行向量（相依）
    # ========================================
    print_separator("2. 平行向量（相依）")

    v1 = np.array([1.0, 2.0])
    v2 = np.array([2.0, 4.0])  # v2 = 2*v1

    print(f"v₁ = {v1}")
    print(f"v₂ = {v2}")
    print("注意：v₂ = 2·v₁")

    independent = is_linearly_independent([v1, v2])
    print(f"\n線性獨立？ {independent}")

    if not independent:
        coeffs = find_dependency_relation([v1, v2])
        if coeffs is not None:
            # 正規化係數
            coeffs = coeffs / np.abs(coeffs).max()
            print(f"相依關係：{coeffs[0]:.4f}·v₁ + {coeffs[1]:.4f}·v₂ = 0")
            print("（即 2·v₁ - 1·v₂ = 0）")

    # ========================================
    # 範例 3：三個 3D 向量（獨立）
    # ========================================
    print_separator("3. 三個獨立的 3D 向量")

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    v3 = np.array([0.0, 0.0, 1.0])

    print(f"v₁ = {v1}")
    print(f"v₂ = {v2}")
    print(f"v₃ = {v3}")
    print("（標準基底向量）")

    independent = is_linearly_independent([v1, v2, v3])
    print(f"\n線性獨立？ {independent}")

    A = np.column_stack([v1, v2, v3])
    det = np.linalg.det(A)
    print(f"det([v₁|v₂|v₃]) = {det} ≠ 0 ✓")

    # ========================================
    # 範例 4：三個共面向量（相依）
    # ========================================
    print_separator("4. 三個共面向量（相依）")

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    v3 = np.array([1.0, 1.0, 0.0])  # v3 = v1 + v2

    print(f"v₁ = {v1}")
    print(f"v₂ = {v2}")
    print(f"v₃ = {v3}")
    print("注意：v₃ = v₁ + v₂，三向量共面（都在 z=0 平面）")

    independent = is_linearly_independent([v1, v2, v3])
    print(f"\n線性獨立？ {independent}")

    coeffs = find_dependency_relation([v1, v2, v3])
    if coeffs is not None:
        coeffs = coeffs / np.abs(coeffs).max()
        print(f"相依關係：{coeffs[0]:.4f}·v₁ + {coeffs[1]:.4f}·v₂ + {coeffs[2]:.4f}·v₃ = 0")

    # ========================================
    # 範例 5：找最大獨立子集
    # ========================================
    print_separator("5. 找最大線性獨立子集")

    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),  # = 2 * v1
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 2.0]),
    ]

    print("向量組：")
    for i, v in enumerate(vectors):
        print(f"  v{i+1} = {v}")

    print("\n注意：v₂ = 2·v₁")

    independent_indices, rank = find_maximal_independent_subset(vectors)

    print(f"\n最大獨立子集的索引：{independent_indices}")
    print(f"秩 = {rank}")

    print("\n獨立向量：")
    for idx in independent_indices:
        print(f"  v{idx+1} = {vectors[idx]}")

    # ========================================
    # 範例 6：向量數超過維度（必定相依）
    # ========================================
    print_separator("6. 向量數 > 維度（必定相依）")

    # 4 個 3D 向量
    vectors_4 = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
    ]

    print("4 個 3D 向量：")
    for i, v in enumerate(vectors_4):
        print(f"  v{i+1} = {v}")

    print(f"\n在 ℝ³ 中，最多 3 個向量可以獨立")

    independent = is_linearly_independent(vectors_4)
    print(f"這 4 個向量線性獨立？ {independent}")

    coeffs = find_dependency_relation(vectors_4)
    if coeffs is not None:
        print(f"相依關係係數：{coeffs}")
        # 驗證
        result = sum(c * v for c, v in zip(coeffs, vectors_4))
        print(f"驗證 Σcᵢvᵢ = {result}")

    # ========================================
    # 範例 7：包含零向量
    # ========================================
    print_separator("7. 包含零向量（必定相依）")

    v1 = np.array([1.0, 2.0])
    v2 = np.array([0.0, 0.0])  # 零向量

    print(f"v₁ = {v1}")
    print(f"v₂ = {v2}（零向量）")

    independent = is_linearly_independent([v1, v2])
    print(f"\n線性獨立？ {independent}")
    print("原因：0·v₁ + 1·v₂ = 0，係數 (0, 1) 不全為零")

    # ========================================
    # 範例 8：秩與獨立性
    # ========================================
    print_separator("8. 秩與獨立性的關係")

    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    print(f"A =\n{A}\n")

    rank = np.linalg.matrix_rank(A)
    print(f"rank(A) = {rank}")
    print(f"行數 n = {A.shape[1]}")
    print(f"\n行向量獨立？ rank == n ? {rank == A.shape[1]}")

    if rank < A.shape[1]:
        print(f"\n行向量相依！（秩 {rank} < 行數 {A.shape[1]}）")

        # 找相依關係
        col_vectors = [A[:, j] for j in range(A.shape[1])]
        coeffs = find_dependency_relation(col_vectors)
        if coeffs is not None:
            print(f"相依關係：{coeffs}")

    # 總結
    print_separator("總結")
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
""")

    print("=" * 60)
    print("線性獨立示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
