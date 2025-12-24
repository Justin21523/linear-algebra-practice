"""
四大基本子空間 (The Four Fundamental Subspaces)

本程式示範：
1. 計算矩陣的四大基本子空間
2. 驗證正交關係
3. 驗證維度公式
4. Strang 的「大圖景」

This program demonstrates the four fundamental subspaces of a matrix
as emphasized by Gilbert Strang.
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.
from scipy.linalg import null_space  # EN: Import symbol(s) from a module: from scipy.linalg import null_space.
from typing import Dict, List  # EN: Import symbol(s) from a module: from typing import Dict, List.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def rref_with_pivots(A: np.ndarray):  # EN: Define rref_with_pivots and its behavior.
    """計算 RREF 和主元行索引"""  # EN: Execute statement: """計算 RREF 和主元行索引""".
    A = A.astype(float).copy()  # EN: Assign A from expression: A.astype(float).copy().
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.
    pivot_cols = []  # EN: Assign pivot_cols from expression: [].
    row = 0  # EN: Assign row from expression: 0.

    for col in range(n):  # EN: Iterate with a for-loop: for col in range(n):.
        if row >= m:  # EN: Branch on a condition: if row >= m:.
            break  # EN: Control flow statement: break.

        max_row = row + np.argmax(np.abs(A[row:, col]))  # EN: Assign max_row from expression: row + np.argmax(np.abs(A[row:, col])).
        if np.abs(A[max_row, col]) < 1e-10:  # EN: Branch on a condition: if np.abs(A[max_row, col]) < 1e-10:.
            continue  # EN: Control flow statement: continue.

        A[[row, max_row]] = A[[max_row, row]]  # EN: Execute statement: A[[row, max_row]] = A[[max_row, row]].
        A[row] = A[row] / A[row, col]  # EN: Execute statement: A[row] = A[row] / A[row, col].

        for i in range(m):  # EN: Iterate with a for-loop: for i in range(m):.
            if i != row:  # EN: Branch on a condition: if i != row:.
                A[i] = A[i] - A[i, col] * A[row]  # EN: Execute statement: A[i] = A[i] - A[i, col] * A[row].

        pivot_cols.append(col)  # EN: Execute statement: pivot_cols.append(col).
        row += 1  # EN: Update row via += using: 1.

    return A, pivot_cols  # EN: Return a value: return A, pivot_cols.


def compute_four_subspaces(A: np.ndarray) -> Dict:  # EN: Define compute_four_subspaces and its behavior.
    """
    計算矩陣 A 的四大基本子空間

    Returns:
        包含四個子空間基底的字典
    """  # EN: Execute statement: """.
    m, n = A.shape  # EN: Execute statement: m, n = A.shape.
    rank = np.linalg.matrix_rank(A)  # EN: Assign rank from expression: np.linalg.matrix_rank(A).

    result = {  # EN: Assign result from expression: {.
        'm': m,  # EN: Execute statement: 'm': m,.
        'n': n,  # EN: Execute statement: 'n': n,.
        'rank': rank,  # EN: Execute statement: 'rank': rank,.
    }  # EN: Execute statement: }.

    # 1. 行空間 C(A) - 在 ℝᵐ 中
    R, pivot_cols = rref_with_pivots(A)  # EN: Execute statement: R, pivot_cols = rref_with_pivots(A).
    if len(pivot_cols) > 0:  # EN: Branch on a condition: if len(pivot_cols) > 0:.
        result['C(A)'] = A[:, pivot_cols]  # EN: Execute statement: result['C(A)'] = A[:, pivot_cols].
    else:  # EN: Execute the fallback branch when prior conditions are false.
        result['C(A)'] = np.zeros((m, 0))  # EN: Execute statement: result['C(A)'] = np.zeros((m, 0)).
    result['dim_C(A)'] = rank  # EN: Execute statement: result['dim_C(A)'] = rank.

    # 2. 零空間 N(A) - 在 ℝⁿ 中
    N_A = null_space(A)  # EN: Assign N_A from expression: null_space(A).
    result['N(A)'] = N_A  # EN: Execute statement: result['N(A)'] = N_A.
    result['dim_N(A)'] = N_A.shape[1]  # EN: Execute statement: result['dim_N(A)'] = N_A.shape[1].

    # 3. 列空間 C(Aᵀ) - 在 ℝⁿ 中
    # RREF 的非零列，或 Aᵀ 的行空間
    R_T, pivot_cols_T = rref_with_pivots(A.T)  # EN: Execute statement: R_T, pivot_cols_T = rref_with_pivots(A.T).
    if len(pivot_cols_T) > 0:  # EN: Branch on a condition: if len(pivot_cols_T) > 0:.
        result['C(A^T)'] = A.T[:, pivot_cols_T]  # EN: Execute statement: result['C(A^T)'] = A.T[:, pivot_cols_T].
    else:  # EN: Execute the fallback branch when prior conditions are false.
        result['C(A^T)'] = np.zeros((n, 0))  # EN: Execute statement: result['C(A^T)'] = np.zeros((n, 0)).
    result['dim_C(A^T)'] = rank  # EN: Execute statement: result['dim_C(A^T)'] = rank.

    # 4. 左零空間 N(Aᵀ) - 在 ℝᵐ 中
    N_AT = null_space(A.T)  # EN: Assign N_AT from expression: null_space(A.T).
    result['N(A^T)'] = N_AT  # EN: Execute statement: result['N(A^T)'] = N_AT.
    result['dim_N(A^T)'] = N_AT.shape[1]  # EN: Execute statement: result['dim_N(A^T)'] = N_AT.shape[1].

    return result  # EN: Return a value: return result.


def verify_orthogonality(subspaces: Dict) -> None:  # EN: Define verify_orthogonality and its behavior.
    """驗證子空間的正交關係"""  # EN: Execute statement: """驗證子空間的正交關係""".
    print("\n【正交關係驗證】")  # EN: Print formatted output to the console.

    # C(A) ⊥ N(Aᵀ) 在 ℝᵐ 中
    C_A = subspaces['C(A)']  # EN: Assign C_A from expression: subspaces['C(A)'].
    N_AT = subspaces['N(A^T)']  # EN: Assign N_AT from expression: subspaces['N(A^T)'].

    if C_A.shape[1] > 0 and N_AT.shape[1] > 0:  # EN: Branch on a condition: if C_A.shape[1] > 0 and N_AT.shape[1] > 0:.
        dot_products = C_A.T @ N_AT  # EN: Assign dot_products from expression: C_A.T @ N_AT.
        print(f"C(A)ᵀ @ N(Aᵀ) =\n{dot_products}")  # EN: Print formatted output to the console.
        print(f"C(A) ⊥ N(Aᵀ)？ {np.allclose(dot_products, 0)}")  # EN: Print formatted output to the console.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        print("C(A) 或 N(Aᵀ) 為空，自動滿足正交")  # EN: Print formatted output to the console.

    # C(Aᵀ) ⊥ N(A) 在 ℝⁿ 中
    C_AT = subspaces['C(A^T)']  # EN: Assign C_AT from expression: subspaces['C(A^T)'].
    N_A = subspaces['N(A)']  # EN: Assign N_A from expression: subspaces['N(A)'].

    if C_AT.shape[1] > 0 and N_A.shape[1] > 0:  # EN: Branch on a condition: if C_AT.shape[1] > 0 and N_A.shape[1] > 0:.
        dot_products = C_AT.T @ N_A  # EN: Assign dot_products from expression: C_AT.T @ N_A.
        print(f"\nC(Aᵀ)ᵀ @ N(A) =\n{dot_products}")  # EN: Print formatted output to the console.
        print(f"C(Aᵀ) ⊥ N(A)？ {np.allclose(dot_products, 0)}")  # EN: Print formatted output to the console.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        print("\nC(Aᵀ) 或 N(A) 為空，自動滿足正交")  # EN: Print formatted output to the console.


def verify_dimensions(subspaces: Dict) -> None:  # EN: Define verify_dimensions and its behavior.
    """驗證維度公式"""  # EN: Execute statement: """驗證維度公式""".
    print("\n【維度公式驗證】")  # EN: Print formatted output to the console.

    m = subspaces['m']  # EN: Assign m from expression: subspaces['m'].
    n = subspaces['n']  # EN: Assign n from expression: subspaces['n'].
    r = subspaces['rank']  # EN: Assign r from expression: subspaces['rank'].

    print(f"矩陣大小：{m}×{n}")  # EN: Print formatted output to the console.
    print(f"秩 r = {r}")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    # ℝᵐ 中
    dim_CA = subspaces['dim_C(A)']  # EN: Assign dim_CA from expression: subspaces['dim_C(A)'].
    dim_NAT = subspaces['dim_N(A^T)']  # EN: Assign dim_NAT from expression: subspaces['dim_N(A^T)'].
    print(f"在 ℝᵐ 中：")  # EN: Print formatted output to the console.
    print(f"  dim C(A) = {dim_CA}")  # EN: Print formatted output to the console.
    print(f"  dim N(Aᵀ) = {dim_NAT}")  # EN: Print formatted output to the console.
    print(f"  dim C(A) + dim N(Aᵀ) = {dim_CA + dim_NAT} = m = {m}？ {dim_CA + dim_NAT == m}")  # EN: Print formatted output to the console.

    # ℝⁿ 中
    dim_CAT = subspaces['dim_C(A^T)']  # EN: Assign dim_CAT from expression: subspaces['dim_C(A^T)'].
    dim_NA = subspaces['dim_N(A)']  # EN: Assign dim_NA from expression: subspaces['dim_N(A)'].
    print(f"\n在 ℝⁿ 中：")  # EN: Print formatted output to the console.
    print(f"  dim C(Aᵀ) = {dim_CAT}")  # EN: Print formatted output to the console.
    print(f"  dim N(A) = {dim_NA}")  # EN: Print formatted output to the console.
    print(f"  dim C(Aᵀ) + dim N(A) = {dim_CAT + dim_NA} = n = {n}？ {dim_CAT + dim_NA == n}")  # EN: Print formatted output to the console.


def print_subspace_bases(subspaces: Dict) -> None:  # EN: Define print_subspace_bases and its behavior.
    """印出各子空間的基底"""  # EN: Execute statement: """印出各子空間的基底""".
    print("\n【四大子空間】")  # EN: Print formatted output to the console.

    print(f"\n1. 行空間 C(A)（在 ℝᵐ 中）")  # EN: Print formatted output to the console.
    print(f"   維度 = {subspaces['dim_C(A)']}")  # EN: Print formatted output to the console.
    if subspaces['C(A)'].shape[1] > 0:  # EN: Branch on a condition: if subspaces['C(A)'].shape[1] > 0:.
        print(f"   基底向量：")  # EN: Print formatted output to the console.
        for j in range(subspaces['C(A)'].shape[1]):  # EN: Iterate with a for-loop: for j in range(subspaces['C(A)'].shape[1]):.
            print(f"     {subspaces['C(A)'][:, j]}")  # EN: Print formatted output to the console.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        print("   只有零向量")  # EN: Print formatted output to the console.

    print(f"\n2. 零空間 N(A)（在 ℝⁿ 中）")  # EN: Print formatted output to the console.
    print(f"   維度 = {subspaces['dim_N(A)']}")  # EN: Print formatted output to the console.
    if subspaces['N(A)'].shape[1] > 0:  # EN: Branch on a condition: if subspaces['N(A)'].shape[1] > 0:.
        print(f"   基底向量：")  # EN: Print formatted output to the console.
        for j in range(subspaces['N(A)'].shape[1]):  # EN: Iterate with a for-loop: for j in range(subspaces['N(A)'].shape[1]):.
            print(f"     {subspaces['N(A)'][:, j]}")  # EN: Print formatted output to the console.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        print("   只有零向量")  # EN: Print formatted output to the console.

    print(f"\n3. 列空間 C(Aᵀ)（在 ℝⁿ 中）")  # EN: Print formatted output to the console.
    print(f"   維度 = {subspaces['dim_C(A^T)']}")  # EN: Print formatted output to the console.
    if subspaces['C(A^T)'].shape[1] > 0:  # EN: Branch on a condition: if subspaces['C(A^T)'].shape[1] > 0:.
        print(f"   基底向量：")  # EN: Print formatted output to the console.
        for j in range(subspaces['C(A^T)'].shape[1]):  # EN: Iterate with a for-loop: for j in range(subspaces['C(A^T)'].shape[1]):.
            print(f"     {subspaces['C(A^T)'][:, j]}")  # EN: Print formatted output to the console.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        print("   只有零向量")  # EN: Print formatted output to the console.

    print(f"\n4. 左零空間 N(Aᵀ)（在 ℝᵐ 中）")  # EN: Print formatted output to the console.
    print(f"   維度 = {subspaces['dim_N(A^T)']}")  # EN: Print formatted output to the console.
    if subspaces['N(A^T)'].shape[1] > 0:  # EN: Branch on a condition: if subspaces['N(A^T)'].shape[1] > 0:.
        print(f"   基底向量：")  # EN: Print formatted output to the console.
        for j in range(subspaces['N(A^T)'].shape[1]):  # EN: Iterate with a for-loop: for j in range(subspaces['N(A^T)'].shape[1]):.
            print(f"     {subspaces['N(A^T)'][:, j]}")  # EN: Print formatted output to the console.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        print("   只有零向量")  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("四大基本子空間示範\nThe Four Fundamental Subspaces Demo")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 範例 1：典型矩陣
    # ========================================
    print_separator("範例 1：2×3 秩虧矩陣")  # EN: Call print_separator(...) to perform an operation.

    A1 = np.array([  # EN: Assign A1 from expression: np.array([.
        [1, 3, 5],  # EN: Execute statement: [1, 3, 5],.
        [2, 6, 10]  # EN: Execute statement: [2, 6, 10].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A =\n{A1}")  # EN: Print formatted output to the console.

    subspaces1 = compute_four_subspaces(A1)  # EN: Assign subspaces1 from expression: compute_four_subspaces(A1).
    print_subspace_bases(subspaces1)  # EN: Call print_subspace_bases(...) to perform an operation.
    verify_dimensions(subspaces1)  # EN: Call verify_dimensions(...) to perform an operation.
    verify_orthogonality(subspaces1)  # EN: Call verify_orthogonality(...) to perform an operation.

    # ========================================
    # 範例 2：滿秩方陣
    # ========================================
    print_separator("範例 2：3×3 滿秩方陣")  # EN: Call print_separator(...) to perform an operation.

    A2 = np.array([  # EN: Assign A2 from expression: np.array([.
        [1, 2, 1],  # EN: Execute statement: [1, 2, 1],.
        [0, 1, 1],  # EN: Execute statement: [0, 1, 1],.
        [1, 0, 1]  # EN: Execute statement: [1, 0, 1].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A =\n{A2}")  # EN: Print formatted output to the console.

    subspaces2 = compute_four_subspaces(A2)  # EN: Assign subspaces2 from expression: compute_four_subspaces(A2).
    print_subspace_bases(subspaces2)  # EN: Call print_subspace_bases(...) to perform an operation.
    verify_dimensions(subspaces2)  # EN: Call verify_dimensions(...) to perform an operation.

    print("\n對於滿秩方陣：")  # EN: Print formatted output to the console.
    print("  N(A) = {0}（只有零解）")  # EN: Print formatted output to the console.
    print("  N(Aᵀ) = {0}（只有零解）")  # EN: Print formatted output to the console.
    print("  C(A) = ℝᵐ（可以到達任何向量）")  # EN: Print formatted output to the console.
    print("  C(Aᵀ) = ℝⁿ（列向量生成整個空間）")  # EN: Print formatted output to the console.

    # ========================================
    # 範例 3：高矩陣
    # ========================================
    print_separator("範例 3：4×2 矩陣（m > n）")  # EN: Call print_separator(...) to perform an operation.

    A3 = np.array([  # EN: Assign A3 from expression: np.array([.
        [1, 0],  # EN: Execute statement: [1, 0],.
        [0, 1],  # EN: Execute statement: [0, 1],.
        [1, 1],  # EN: Execute statement: [1, 1],.
        [2, 1]  # EN: Execute statement: [2, 1].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A =\n{A3}")  # EN: Print formatted output to the console.

    subspaces3 = compute_four_subspaces(A3)  # EN: Assign subspaces3 from expression: compute_four_subspaces(A3).
    print_subspace_bases(subspaces3)  # EN: Call print_subspace_bases(...) to perform an operation.
    verify_dimensions(subspaces3)  # EN: Call verify_dimensions(...) to perform an operation.
    verify_orthogonality(subspaces3)  # EN: Call verify_orthogonality(...) to perform an operation.

    # ========================================
    # 範例 4：寬矩陣
    # ========================================
    print_separator("範例 4：2×4 矩陣（m < n）")  # EN: Call print_separator(...) to perform an operation.

    A4 = np.array([  # EN: Assign A4 from expression: np.array([.
        [1, 2, 1, 0],  # EN: Execute statement: [1, 2, 1, 0],.
        [2, 4, 0, 2]  # EN: Execute statement: [2, 4, 0, 2].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"A =\n{A4}")  # EN: Print formatted output to the console.

    subspaces4 = compute_four_subspaces(A4)  # EN: Assign subspaces4 from expression: compute_four_subspaces(A4).
    print_subspace_bases(subspaces4)  # EN: Call print_subspace_bases(...) to perform an operation.
    verify_dimensions(subspaces4)  # EN: Call verify_dimensions(...) to perform an operation.
    verify_orthogonality(subspaces4)  # EN: Call verify_orthogonality(...) to perform an operation.

    # ========================================
    # 大圖景
    # ========================================
    print_separator("Strang 的大圖景 (The Big Picture)")  # EN: Call print_separator(...) to perform an operation.

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
    """)  # EN: Execute statement: """).

    # ========================================
    # 應用：Ax = b 的解
    # ========================================
    print_separator("應用：Ax = b 的解")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([[1, 2], [2, 4]], dtype=float)  # EN: Assign A from expression: np.array([[1, 2], [2, 4]], dtype=float).
    print(f"A =\n{A}")  # EN: Print formatted output to the console.

    b1 = np.array([3, 6])  # EN: Assign b1 from expression: np.array([3, 6]).
    b2 = np.array([1, 1])  # EN: Assign b2 from expression: np.array([1, 1]).

    print(f"\nb₁ = {b1}")  # EN: Print formatted output to the console.
    subspaces = compute_four_subspaces(A)  # EN: Assign subspaces from expression: compute_four_subspaces(A).

    # 檢查 b 是否在 C(A) 中
    _, res1, _, _ = np.linalg.lstsq(A, b1, rcond=None)  # EN: Execute statement: _, res1, _, _ = np.linalg.lstsq(A, b1, rcond=None).
    if len(res1) == 0 or np.linalg.norm(A @ np.linalg.lstsq(A, b1, rcond=None)[0] - b1) < 1e-10:  # EN: Branch on a condition: if len(res1) == 0 or np.linalg.norm(A @ np.linalg.lstsq(A, b1, rcond=No….
        print("b₁ ∈ C(A)，Ax = b₁ 有解")  # EN: Print formatted output to the console.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        print("b₁ ∉ C(A)，Ax = b₁ 無解")  # EN: Print formatted output to the console.

    print(f"\nb₂ = {b2}")  # EN: Print formatted output to the console.
    residual = np.linalg.norm(A @ np.linalg.lstsq(A, b2, rcond=None)[0] - b2)  # EN: Assign residual from expression: np.linalg.norm(A @ np.linalg.lstsq(A, b2, rcond=None)[0] - b2).
    if residual < 1e-10:  # EN: Branch on a condition: if residual < 1e-10:.
        print("b₂ ∈ C(A)，Ax = b₂ 有解")  # EN: Print formatted output to the console.
    else:  # EN: Execute the fallback branch when prior conditions are false.
        print(f"b₂ ∉ C(A)，Ax = b₂ 無解（殘差 = {residual:.4f}）")  # EN: Print formatted output to the console.

    # 等價條件
    N_AT = subspaces['N(A^T)']  # EN: Assign N_AT from expression: subspaces['N(A^T)'].
    if N_AT.shape[1] > 0:  # EN: Branch on a condition: if N_AT.shape[1] > 0:.
        print(f"\n等價檢驗：b ⊥ N(Aᵀ)")  # EN: Print formatted output to the console.
        print(f"N(Aᵀ) 的基底：{N_AT[:, 0]}")  # EN: Print formatted output to the console.
        print(f"b₁ · N(Aᵀ) = {np.dot(b1, N_AT[:, 0]):.4f}")  # EN: Print formatted output to the console.
        print(f"b₂ · N(Aᵀ) = {np.dot(b2, N_AT[:, 0]):.4f}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("四大基本子空間示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
