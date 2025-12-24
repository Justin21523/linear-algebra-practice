"""
四大基本子空間 (The Four Fundamental Subspaces)

本程式示範：
1. 計算矩陣的四大基本子空間
2. 驗證正交關係
3. 驗證維度公式
4. Strang 的「大圖景」

This program demonstrates the four fundamental subspaces of a matrix
as emphasized by Gilbert Strang.
"""

import numpy as np
from scipy.linalg import null_space
from typing import Dict, List

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def rref_with_pivots(A: np.ndarray):
    """計算 RREF 和主元行索引"""
    A = A.astype(float).copy()
    m, n = A.shape
    pivot_cols = []
    row = 0

    for col in range(n):
        if row >= m:
            break

        max_row = row + np.argmax(np.abs(A[row:, col]))
        if np.abs(A[max_row, col]) < 1e-10:
            continue

        A[[row, max_row]] = A[[max_row, row]]
        A[row] = A[row] / A[row, col]

        for i in range(m):
            if i != row:
                A[i] = A[i] - A[i, col] * A[row]

        pivot_cols.append(col)
        row += 1

    return A, pivot_cols


def compute_four_subspaces(A: np.ndarray) -> Dict:
    """
    計算矩陣 A 的四大基本子空間

    Returns:
        包含四個子空間基底的字典
    """
    m, n = A.shape
    rank = np.linalg.matrix_rank(A)

    result = {
        'm': m,
        'n': n,
        'rank': rank,
    }

    # 1. 行空間 C(A) - 在 ℝᵐ 中
    R, pivot_cols = rref_with_pivots(A)
    if len(pivot_cols) > 0:
        result['C(A)'] = A[:, pivot_cols]
    else:
        result['C(A)'] = np.zeros((m, 0))
    result['dim_C(A)'] = rank

    # 2. 零空間 N(A) - 在 ℝⁿ 中
    N_A = null_space(A)
    result['N(A)'] = N_A
    result['dim_N(A)'] = N_A.shape[1]

    # 3. 列空間 C(Aᵀ) - 在 ℝⁿ 中
    # RREF 的非零列，或 Aᵀ 的行空間
    R_T, pivot_cols_T = rref_with_pivots(A.T)
    if len(pivot_cols_T) > 0:
        result['C(A^T)'] = A.T[:, pivot_cols_T]
    else:
        result['C(A^T)'] = np.zeros((n, 0))
    result['dim_C(A^T)'] = rank

    # 4. 左零空間 N(Aᵀ) - 在 ℝᵐ 中
    N_AT = null_space(A.T)
    result['N(A^T)'] = N_AT
    result['dim_N(A^T)'] = N_AT.shape[1]

    return result


def verify_orthogonality(subspaces: Dict) -> None:
    """驗證子空間的正交關係"""
    print("\n【正交關係驗證】")

    # C(A) ⊥ N(Aᵀ) 在 ℝᵐ 中
    C_A = subspaces['C(A)']
    N_AT = subspaces['N(A^T)']

    if C_A.shape[1] > 0 and N_AT.shape[1] > 0:
        dot_products = C_A.T @ N_AT
        print(f"C(A)ᵀ @ N(Aᵀ) =\n{dot_products}")
        print(f"C(A) ⊥ N(Aᵀ)？ {np.allclose(dot_products, 0)}")
    else:
        print("C(A) 或 N(Aᵀ) 為空，自動滿足正交")

    # C(Aᵀ) ⊥ N(A) 在 ℝⁿ 中
    C_AT = subspaces['C(A^T)']
    N_A = subspaces['N(A)']

    if C_AT.shape[1] > 0 and N_A.shape[1] > 0:
        dot_products = C_AT.T @ N_A
        print(f"\nC(Aᵀ)ᵀ @ N(A) =\n{dot_products}")
        print(f"C(Aᵀ) ⊥ N(A)？ {np.allclose(dot_products, 0)}")
    else:
        print("\nC(Aᵀ) 或 N(A) 為空，自動滿足正交")


def verify_dimensions(subspaces: Dict) -> None:
    """驗證維度公式"""
    print("\n【維度公式驗證】")

    m = subspaces['m']
    n = subspaces['n']
    r = subspaces['rank']

    print(f"矩陣大小：{m}×{n}")
    print(f"秩 r = {r}")
    print()

    # ℝᵐ 中
    dim_CA = subspaces['dim_C(A)']
    dim_NAT = subspaces['dim_N(A^T)']
    print(f"在 ℝᵐ 中：")
    print(f"  dim C(A) = {dim_CA}")
    print(f"  dim N(Aᵀ) = {dim_NAT}")
    print(f"  dim C(A) + dim N(Aᵀ) = {dim_CA + dim_NAT} = m = {m}？ {dim_CA + dim_NAT == m}")

    # ℝⁿ 中
    dim_CAT = subspaces['dim_C(A^T)']
    dim_NA = subspaces['dim_N(A)']
    print(f"\n在 ℝⁿ 中：")
    print(f"  dim C(Aᵀ) = {dim_CAT}")
    print(f"  dim N(A) = {dim_NA}")
    print(f"  dim C(Aᵀ) + dim N(A) = {dim_CAT + dim_NA} = n = {n}？ {dim_CAT + dim_NA == n}")


def print_subspace_bases(subspaces: Dict) -> None:
    """印出各子空間的基底"""
    print("\n【四大子空間】")

    print(f"\n1. 行空間 C(A)（在 ℝᵐ 中）")
    print(f"   維度 = {subspaces['dim_C(A)']}")
    if subspaces['C(A)'].shape[1] > 0:
        print(f"   基底向量：")
        for j in range(subspaces['C(A)'].shape[1]):
            print(f"     {subspaces['C(A)'][:, j]}")
    else:
        print("   只有零向量")

    print(f"\n2. 零空間 N(A)（在 ℝⁿ 中）")
    print(f"   維度 = {subspaces['dim_N(A)']}")
    if subspaces['N(A)'].shape[1] > 0:
        print(f"   基底向量：")
        for j in range(subspaces['N(A)'].shape[1]):
            print(f"     {subspaces['N(A)'][:, j]}")
    else:
        print("   只有零向量")

    print(f"\n3. 列空間 C(Aᵀ)（在 ℝⁿ 中）")
    print(f"   維度 = {subspaces['dim_C(A^T)']}")
    if subspaces['C(A^T)'].shape[1] > 0:
        print(f"   基底向量：")
        for j in range(subspaces['C(A^T)'].shape[1]):
            print(f"     {subspaces['C(A^T)'][:, j]}")
    else:
        print("   只有零向量")

    print(f"\n4. 左零空間 N(Aᵀ)（在 ℝᵐ 中）")
    print(f"   維度 = {subspaces['dim_N(A^T)']}")
    if subspaces['N(A^T)'].shape[1] > 0:
        print(f"   基底向量：")
        for j in range(subspaces['N(A^T)'].shape[1]):
            print(f"     {subspaces['N(A^T)'][:, j]}")
    else:
        print("   只有零向量")


def main():
    """主程式"""

    print_separator("四大基本子空間示範\nThe Four Fundamental Subspaces Demo")

    # ========================================
    # 範例 1：典型矩陣
    # ========================================
    print_separator("範例 1：2×3 秩虧矩陣")

    A1 = np.array([
        [1, 3, 5],
        [2, 6, 10]
    ], dtype=float)

    print(f"A =\n{A1}")

    subspaces1 = compute_four_subspaces(A1)
    print_subspace_bases(subspaces1)
    verify_dimensions(subspaces1)
    verify_orthogonality(subspaces1)

    # ========================================
    # 範例 2：滿秩方陣
    # ========================================
    print_separator("範例 2：3×3 滿秩方陣")

    A2 = np.array([
        [1, 2, 1],
        [0, 1, 1],
        [1, 0, 1]
    ], dtype=float)

    print(f"A =\n{A2}")

    subspaces2 = compute_four_subspaces(A2)
    print_subspace_bases(subspaces2)
    verify_dimensions(subspaces2)

    print("\n對於滿秩方陣：")
    print("  N(A) = {0}（只有零解）")
    print("  N(Aᵀ) = {0}（只有零解）")
    print("  C(A) = ℝᵐ（可以到達任何向量）")
    print("  C(Aᵀ) = ℝⁿ（列向量生成整個空間）")

    # ========================================
    # 範例 3：高矩陣
    # ========================================
    print_separator("範例 3：4×2 矩陣（m > n）")

    A3 = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 1]
    ], dtype=float)

    print(f"A =\n{A3}")

    subspaces3 = compute_four_subspaces(A3)
    print_subspace_bases(subspaces3)
    verify_dimensions(subspaces3)
    verify_orthogonality(subspaces3)

    # ========================================
    # 範例 4：寬矩陣
    # ========================================
    print_separator("範例 4：2×4 矩陣（m < n）")

    A4 = np.array([
        [1, 2, 1, 0],
        [2, 4, 0, 2]
    ], dtype=float)

    print(f"A =\n{A4}")

    subspaces4 = compute_four_subspaces(A4)
    print_subspace_bases(subspaces4)
    verify_dimensions(subspaces4)
    verify_orthogonality(subspaces4)

    # ========================================
    # 大圖景
    # ========================================
    print_separator("Strang 的大圖景 (The Big Picture)")

    print("""
    對於 m×n 矩陣 A，秩 = r：

            ℝⁿ                              ℝᵐ
        ┌─────────┐                     ┌─────────┐
        │         │                     │         │
        │  C(Aᵀ)  │  ───── A ─────→     │  C(A)   │
        │  dim=r  │      1-to-1         │  dim=r  │
        │         │                     │         │
        ├─────────┤                     ├─────────┤
        │         │                     │         │
        │   N(A)  │  ───── A ─────→     │  N(Aᵀ)  │
        │ dim=n-r │        0            │ dim=m-r │
        │         │                     │         │
        └─────────┘                     └─────────┘

    關鍵觀察：
    1. A 把 C(Aᵀ) 一對一映射到 C(A)
    2. A 把 N(A) 全部壓縮到 0
    3. C(Aᵀ) ⊥ N(A) 在 ℝⁿ 中
    4. C(A) ⊥ N(Aᵀ) 在 ℝᵐ 中
    5. ℝⁿ = C(Aᵀ) ⊕ N(A)
    6. ℝᵐ = C(A) ⊕ N(Aᵀ)
    """)

    # ========================================
    # 應用：Ax = b 的解
    # ========================================
    print_separator("應用：Ax = b 的解")

    A = np.array([[1, 2], [2, 4]], dtype=float)
    print(f"A =\n{A}")

    b1 = np.array([3, 6])
    b2 = np.array([1, 1])

    print(f"\nb₁ = {b1}")
    subspaces = compute_four_subspaces(A)

    # 檢查 b 是否在 C(A) 中
    _, res1, _, _ = np.linalg.lstsq(A, b1, rcond=None)
    if len(res1) == 0 or np.linalg.norm(A @ np.linalg.lstsq(A, b1, rcond=None)[0] - b1) < 1e-10:
        print("b₁ ∈ C(A)，Ax = b₁ 有解")
    else:
        print("b₁ ∉ C(A)，Ax = b₁ 無解")

    print(f"\nb₂ = {b2}")
    residual = np.linalg.norm(A @ np.linalg.lstsq(A, b2, rcond=None)[0] - b2)
    if residual < 1e-10:
        print("b₂ ∈ C(A)，Ax = b₂ 有解")
    else:
        print(f"b₂ ∉ C(A)，Ax = b₂ 無解（殘差 = {residual:.4f}）")

    # 等價條件
    N_AT = subspaces['N(A^T)']
    if N_AT.shape[1] > 0:
        print(f"\n等價檢驗：b ⊥ N(Aᵀ)")
        print(f"N(Aᵀ) 的基底：{N_AT[:, 0]}")
        print(f"b₁ · N(Aᵀ) = {np.dot(b1, N_AT[:, 0]):.4f}")
        print(f"b₂ · N(Aᵀ) = {np.dot(b2, N_AT[:, 0]):.4f}")

    # 總結
    print_separator("總結")
    print("""
四大基本子空間的關鍵：

維度：
  dim C(A) = dim C(Aᵀ) = r（秩）
  dim N(A) = n - r
  dim N(Aᵀ) = m - r

正交：
  C(A) ⊥ N(Aᵀ) 在 ℝᵐ 中
  C(Aᵀ) ⊥ N(A) 在 ℝⁿ 中

直和分解：
  ℝᵐ = C(A) ⊕ N(Aᵀ)
  ℝⁿ = C(Aᵀ) ⊕ N(A)
""")

    print("=" * 60)
    print("四大基本子空間示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
