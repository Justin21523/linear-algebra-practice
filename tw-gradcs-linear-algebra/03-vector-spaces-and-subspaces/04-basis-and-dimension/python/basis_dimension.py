"""
基底與維度 (Basis and Dimension)

本程式示範：
1. 找向量空間的基底
2. 計算維度
3. 座標變換
4. 驗證基底的性質

This program demonstrates finding bases, computing dimensions,
and coordinate transformations.
"""

import numpy as np
from typing import List, Tuple

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def find_basis(vectors: List[np.ndarray]) -> Tuple[List[np.ndarray], List[int]]:
    """
    從向量組中找出一組基底（最大獨立子集）

    Returns:
        (基底向量列表, 基底向量的原索引)
    """
    if len(vectors) == 0:
        return [], []

    A = np.column_stack(vectors)
    m, n = A.shape

    # RREF 找主元行
    A_work = A.astype(float).copy()
    pivot_cols = []
    row = 0

    for col in range(n):
        if row >= m:
            break

        max_row = row + np.argmax(np.abs(A_work[row:, col]))

        if np.abs(A_work[max_row, col]) < 1e-10:
            continue

        A_work[[row, max_row]] = A_work[[max_row, row]]

        for i in range(m):
            if i != row and np.abs(A_work[i, col]) > 1e-10:
                A_work[i] -= A_work[i, col] / A_work[row, col] * A_work[row]

        pivot_cols.append(col)
        row += 1

    basis = [vectors[i] for i in pivot_cols]
    return basis, pivot_cols


def is_basis(vectors: List[np.ndarray], space_dim: int) -> Tuple[bool, str]:
    """
    檢查向量組是否為 ℝⁿ 的基底

    Returns:
        (是否為基底, 原因)
    """
    if len(vectors) != space_dim:
        return False, f"元素數 {len(vectors)} ≠ 空間維度 {space_dim}"

    A = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(A)

    if rank < space_dim:
        return False, f"向量組相依（秩 = {rank} < {space_dim}）"

    return True, "獨立且生成空間"


def coordinates_in_basis(x: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:
    """
    計算向量 x 在給定基底下的座標

    x = c₁b₁ + c₂b₂ + ... + cₙbₙ
    返回 [c₁, c₂, ..., cₙ]
    """
    P = np.column_stack(basis)  # 基底矩陣
    coords = np.linalg.solve(P, x)  # [x]_B = P⁻¹ x
    return coords


def vector_from_coordinates(coords: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:
    """
    從座標和基底還原向量

    x = c₁b₁ + c₂b₂ + ... + cₙbₙ
    """
    P = np.column_stack(basis)
    return P @ coords


def main():
    """主程式"""

    print_separator("基底與維度示範\nBasis and Dimension Demo")

    # ========================================
    # 1. ℝ² 的標準基底
    # ========================================
    print_separator("1. ℝ² 的標準基底")

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    std_basis_2d = [e1, e2]

    print(f"e₁ = {e1}")
    print(f"e₂ = {e2}")

    is_b, reason = is_basis(std_basis_2d, 2)
    print(f"\n是 ℝ² 的基底？ {is_b}（{reason}）")

    # 任意向量的表示
    x = np.array([3.0, 4.0])
    print(f"\n向量 x = {x}")
    print(f"在標準基底下：x = {x[0]}·e₁ + {x[1]}·e₂")

    # ========================================
    # 2. 非標準基底
    # ========================================
    print_separator("2. ℝ² 的非標準基底")

    b1 = np.array([1.0, 1.0])
    b2 = np.array([1.0, -1.0])
    custom_basis = [b1, b2]

    print(f"b₁ = {b1}")
    print(f"b₂ = {b2}")

    is_b, reason = is_basis(custom_basis, 2)
    print(f"\n是 ℝ² 的基底？ {is_b}（{reason}）")

    # 行列式驗證
    B = np.column_stack(custom_basis)
    det_B = np.linalg.det(B)
    print(f"det([b₁|b₂]) = {det_B} ≠ 0 ✓")

    # ========================================
    # 3. 座標變換
    # ========================================
    print_separator("3. 座標變換")

    x = np.array([3.0, 1.0])
    print(f"向量 x = {x}（標準座標）")

    # 在非標準基底下的座標
    coords_B = coordinates_in_basis(x, custom_basis)
    print(f"\n在基底 B = {{b₁, b₂}} 下的座標：")
    print(f"[x]_B = {coords_B}")

    # 驗證
    x_reconstructed = vector_from_coordinates(coords_B, custom_basis)
    print(f"\n驗證：{coords_B[0]:.4f}·b₁ + {coords_B[1]:.4f}·b₂ = {x_reconstructed}")

    print(f"\n解釋：")
    print(f"  {coords_B[0]:.4f}·[1,1] + {coords_B[1]:.4f}·[1,-1]")
    print(f"= [{coords_B[0]:.4f} + {coords_B[1]:.4f}, {coords_B[0]:.4f} - {coords_B[1]:.4f}]")
    print(f"= {x}")

    # ========================================
    # 4. 從向量組找基底
    # ========================================
    print_separator("4. 從向量組找基底")

    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),  # = 2 * v1
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 2.0]),
    ]

    print("向量組：")
    for i, v in enumerate(vectors):
        print(f"  v{i+1} = {v}")

    basis, indices = find_basis(vectors)

    print(f"\n基底（最大獨立子集）：")
    for i, idx in enumerate(indices):
        print(f"  v{idx+1} = {basis[i]}")

    print(f"\nspan{{v₁, v₂, v₃, v₄}} 的維度 = {len(basis)}")

    # ========================================
    # 5. 維度計算
    # ========================================
    print_separator("5. 維度計算")

    print("常見向量空間的維度：")
    print(f"  dim(ℝⁿ) = n")
    print(f"  dim(ℝ²) = 2")
    print(f"  dim(ℝ³) = 3")
    print(f"  dim(Pₙ) = n + 1（次數 ≤ n 的多項式）")
    print(f"  dim(M₂ₓ₃) = 2 × 3 = 6（2×3 矩陣）")
    print(f"  dim({{0}}) = 0（零空間）")

    # ========================================
    # 6. 秩-零度定理應用
    # ========================================
    print_separator("6. 秩-零度定理應用")

    A = np.array([
        [1, 2, 1, 0],
        [2, 4, 0, 2],
        [3, 6, 2, 1]
    ], dtype=float)

    print(f"A =\n{A}\n")

    m, n = A.shape
    rank_A = np.linalg.matrix_rank(A)
    nullity_A = n - rank_A

    print(f"A 是 {m}×{n} 矩陣")
    print(f"rank(A) = dim C(A) = {rank_A}")
    print(f"nullity(A) = dim N(A) = {nullity_A}")
    print(f"\nrank + nullity = {rank_A} + {nullity_A} = {n} = 行數 ✓")

    # ========================================
    # 7. 子空間維度的比較
    # ========================================
    print_separator("7. 子空間維度")

    print("若 W 是 V 的子空間，則 dim(W) ≤ dim(V)")
    print()

    print("例：ℝ³ 的子空間")
    print("  • 0 維：原點 {0}")
    print("  • 1 維：通過原點的直線")
    print("  • 2 維：通過原點的平面")
    print("  • 3 維：整個 ℝ³")

    # ========================================
    # 8. 基底變換矩陣
    # ========================================
    print_separator("8. 基底變換矩陣")

    print("設 B₁ 和 B₂ 是 ℝⁿ 的兩組基底")
    print()

    B1 = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]  # 標準基底
    B2 = [np.array([1.0, 1.0]), np.array([1.0, -1.0])]  # 非標準基底

    print("B₁ = {[1,0], [0,1]}（標準基底）")
    print("B₂ = {[1,1], [1,-1]}")

    # 從 B₁ 到 B₂ 的變換矩陣
    P_B1 = np.column_stack(B1)
    P_B2 = np.column_stack(B2)

    # [x]_B2 = P_B2⁻¹ @ P_B1 @ [x]_B1
    # 但 P_B1 = I，所以 [x]_B2 = P_B2⁻¹ @ x
    change_matrix = np.linalg.inv(P_B2) @ P_B1
    print(f"\n從 B₁ 到 B₂ 的變換矩陣：")
    print(f"P_{{B₁→B₂}} =\n{change_matrix}")

    # 範例
    x_B1 = np.array([3.0, 1.0])  # 標準座標
    x_B2 = change_matrix @ x_B1
    print(f"\n[x]_B₁ = {x_B1}")
    print(f"[x]_B₂ = P @ [x]_B₁ = {x_B2}")

    # 總結
    print_separator("總結")
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
""")

    print("=" * 60)
    print("基底與維度示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
