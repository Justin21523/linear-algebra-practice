"""
向量空間示範 (Vector Space Demo)

本程式示範：
1. 向量空間的公理驗證
2. 子空間的判斷
3. 張成 (Span) 的計算
4. 常見的向量空間範例

This program demonstrates vector space concepts including
axiom verification, subspace testing, and span computation.
"""

import numpy as np
from typing import List, Callable, Tuple

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def verify_vector_space_axioms() -> None:
    """
    驗證 ℝ² 滿足向量空間的公理
    Verify that ℝ² satisfies vector space axioms
    """
    print_separator("1. 驗證向量空間公理（以 ℝ² 為例）")

    u = np.array([1.0, 2.0])
    v = np.array([3.0, 4.0])
    w = np.array([5.0, 6.0])
    c, d = 2.0, 3.0

    print(f"u = {u}, v = {v}, w = {w}")
    print(f"c = {c}, d = {d}")

    print("\n【加法公理】")

    # A1: 封閉性
    print(f"A1 封閉性: u + v = {u + v} ∈ ℝ² ✓")

    # A2: 交換律
    print(f"A2 交換律: u + v = {u + v}, v + u = {v + u}")
    print(f"   u + v == v + u ? {np.allclose(u + v, v + u)} ✓")

    # A3: 結合律
    print(f"A3 結合律: (u + v) + w = {(u + v) + w}")
    print(f"           u + (v + w) = {u + (v + w)}")
    print(f"   相等？ {np.allclose((u + v) + w, u + (v + w))} ✓")

    # A4: 零向量
    zero = np.array([0.0, 0.0])
    print(f"A4 零向量: v + 0 = {v + zero}")
    print(f"   v + 0 == v ? {np.allclose(v + zero, v)} ✓")

    # A5: 反向量
    print(f"A5 反向量: v + (-v) = {v + (-v)}")
    print(f"   v + (-v) == 0 ? {np.allclose(v + (-v), zero)} ✓")

    print("\n【純量乘法公理】")

    # M1: 封閉性
    print(f"M1 封閉性: c·v = {c * v} ∈ ℝ² ✓")

    # M2: 分配律（純量對向量）
    print(f"M2 分配律: c(u + v) = {c * (u + v)}")
    print(f"           cu + cv = {c * u + c * v}")
    print(f"   相等？ {np.allclose(c * (u + v), c * u + c * v)} ✓")

    # M3: 分配律（向量對純量）
    print(f"M3 分配律: (c + d)v = {(c + d) * v}")
    print(f"           cv + dv = {c * v + d * v}")
    print(f"   相等？ {np.allclose((c + d) * v, c * v + d * v)} ✓")

    # M4: 結合律
    print(f"M4 結合律: c(dv) = {c * (d * v)}")
    print(f"           (cd)v = {(c * d) * v}")
    print(f"   相等？ {np.allclose(c * (d * v), (c * d) * v)} ✓")

    # M5: 單位元
    print(f"M5 單位元: 1·v = {1 * v}")
    print(f"   1·v == v ? {np.allclose(1 * v, v)} ✓")

    print("\n結論：ℝ² 滿足所有向量空間公理！")


def is_subspace_point_test(points: List[np.ndarray], name: str) -> None:
    """
    用有限點集測試是否「可能」是子空間
    （注意：這只是必要條件，不是充分條件）
    """
    print(f"\n測試 {name}：")

    # 檢查 1：包含零向量？
    zero = np.zeros_like(points[0])
    has_zero = any(np.allclose(p, zero) for p in points)
    print(f"  包含零向量？ {has_zero}")

    if not has_zero:
        print(f"  結論：❌ 不是子空間（不包含零向量）")
        return

    # 檢查 2：加法封閉？（測試所有組合）
    addition_closed = True
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            sum_vec = p1 + p2
            in_set = any(np.allclose(sum_vec, p) for p in points)
            if not in_set:
                print(f"  {p1} + {p2} = {sum_vec} 不在集合中")
                addition_closed = False
                break
        if not addition_closed:
            break

    # 檢查 3：純量乘法封閉？（測試一些純量）
    scalar_closed = True
    test_scalars = [0.5, 2.0, -1.0]
    for p in points:
        for c in test_scalars:
            scaled = c * p
            in_set = any(np.allclose(scaled, q) for q in points)
            # 這裡我們只標記，因為有限點集無法完全測試

    print(f"  （注意：有限點集無法完全驗證，需要數學證明）")


def subspace_examples() -> None:
    """
    子空間的例子與反例
    """
    print_separator("2. 子空間的例子與反例")

    print("\n【✅ 是子空間的例子】")

    print("\n例 1：通過原點的直線（ℝ² 中）")
    print("  W = {t(1, 2) : t ∈ ℝ}")
    print("  • 包含零向量：t=0 時得 (0, 0) ✓")
    print("  • 加法封閉：t₁(1,2) + t₂(1,2) = (t₁+t₂)(1,2) ∈ W ✓")
    print("  • 純量乘法封閉：c·t(1,2) = (ct)(1,2) ∈ W ✓")

    print("\n例 2：xy 平面（ℝ³ 中）")
    print("  W = {(x, y, 0) : x, y ∈ ℝ}")
    print("  • 包含零向量：(0, 0, 0) ∈ W ✓")
    print("  • 加法封閉：(x₁,y₁,0) + (x₂,y₂,0) = (x₁+x₂, y₁+y₂, 0) ∈ W ✓")
    print("  • 純量乘法封閉：c(x,y,0) = (cx, cy, 0) ∈ W ✓")

    print("\n【❌ 不是子空間的例子】")

    print("\n例 3：不通過原點的直線")
    print("  W = {(x, y) : y = x + 1}")
    print("  • 不包含零向量：(0, 0) 不滿足 0 = 0 + 1 ❌")

    print("\n例 4：第一象限")
    print("  W = {(x, y) : x ≥ 0, y ≥ 0}")
    print("  • 包含零向量 ✓")
    print("  • 但純量乘法不封閉：(1, 1) ∈ W，但 -1·(1, 1) = (-1, -1) ∉ W ❌")

    print("\n例 5：單位圓上的點")
    print("  W = {(x, y) : x² + y² = 1}")
    print("  • 不包含零向量：0² + 0² ≠ 1 ❌")

    print("\n例 6：整數向量")
    print("  W = {(x, y) : x, y ∈ ℤ}")
    print("  • 包含零向量 ✓")
    print("  • 加法封閉 ✓")
    print("  • 但純量乘法不封閉：0.5·(1, 0) = (0.5, 0) ∉ W ❌")


def span_demo() -> None:
    """
    張成 (Span) 的示範
    """
    print_separator("3. 張成 (Span) 的示範")

    print("\n【定義】")
    print("span{v₁, ..., vₖ} = 所有線性組合 c₁v₁ + ... + cₖvₖ 形成的集合")

    print("\n【範例 1】單一向量的張成")
    v1 = np.array([1.0, 2.0])
    print(f"v₁ = {v1}")
    print(f"span{{v₁}} = 通過原點、方向為 {v1} 的直線")
    print("這是 ℝ² 的一維子空間")

    # 展示一些線性組合
    print("\n一些線性組合：")
    for t in [-1, 0, 0.5, 1, 2]:
        print(f"  {t}·v₁ = {t * v1}")

    print("\n【範例 2】兩個不平行向量的張成")
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    print(f"v₁ = {v1}, v₂ = {v2}")
    print(f"span{{v₁, v₂}} = 整個 ℝ²")

    # 展示任意向量都可以表示為線性組合
    target = np.array([3.0, 4.0])
    print(f"\n任意向量如 {target} = {target[0]}·v₁ + {target[1]}·v₂")

    print("\n【範例 3】線性相依向量的張成")
    v1 = np.array([1.0, 2.0])
    v2 = np.array([2.0, 4.0])  # v2 = 2·v1
    print(f"v₁ = {v1}, v₂ = {v2}")
    print(f"注意：v₂ = 2·v₁，它們線性相依")
    print(f"span{{v₁, v₂}} = span{{v₁}} = 通過原點、方向為 v₁ 的直線")
    print("加入 v₂ 並沒有增加新的方向！")

    print("\n【範例 4】三維空間中的張成")
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    print(f"v₁ = {v1}")
    print(f"v₂ = {v2}")
    print(f"span{{v₁, v₂}} = xy 平面（ℝ³ 的二維子空間）")


def check_in_span(vectors: List[np.ndarray], target: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    檢查 target 是否在 vectors 的張成中

    使用最小平方法：若 target 可以精確表示為線性組合，殘差為 0
    """
    # 建立矩陣 A = [v₁ | v₂ | ... | vₖ]
    A = np.column_stack(vectors)

    # 解最小平方問題
    coeffs, residuals, rank, s = np.linalg.lstsq(A, target, rcond=None)

    # 檢查殘差
    reconstructed = A @ coeffs
    error = np.linalg.norm(target - reconstructed)

    in_span = error < 1e-10

    return in_span, coeffs


def span_membership_demo() -> None:
    """
    示範如何判斷向量是否在 span 中
    """
    print_separator("4. 判斷向量是否在 Span 中")

    v1 = np.array([1.0, 0.0, 1.0])
    v2 = np.array([0.0, 1.0, 1.0])

    print(f"v₁ = {v1}")
    print(f"v₂ = {v2}")
    print(f"span{{v₁, v₂}} 是 ℝ³ 的某個二維子空間（平面）")

    # 測試向量
    test_vectors = [
        np.array([1.0, 1.0, 2.0]),  # 在 span 中
        np.array([2.0, 3.0, 5.0]),  # 在 span 中
        np.array([1.0, 1.0, 1.0]),  # 不在 span 中
    ]

    print("\n測試各向量：")
    for i, target in enumerate(test_vectors):
        in_span, coeffs = check_in_span([v1, v2], target)

        print(f"\n目標向量：{target}")
        if in_span:
            print(f"  ✓ 在 span 中")
            print(f"  係數：{coeffs[0]:.4f}·v₁ + {coeffs[1]:.4f}·v₂")
            print(f"  驗證：{coeffs[0]*v1 + coeffs[1]*v2}")
        else:
            print(f"  ✗ 不在 span 中")


def polynomial_space_demo() -> None:
    """
    多項式空間示範
    """
    print_separator("5. 多項式空間 P₂")

    print("P₂ = {a₀ + a₁x + a₂x² : aᵢ ∈ ℝ}")
    print("這是一個三維向量空間！")
    print("\n標準基底：{1, x, x²}")
    print("任何多項式都可以寫成：a₀·1 + a₁·x + a₂·x²")

    print("\n【以係數向量表示多項式】")
    print("多項式 p(x) = 2 + 3x - x² 可以表示為係數向量 [2, 3, -1]")

    # 用 NumPy 表示多項式
    p = np.array([2, 3, -1])  # 2 + 3x - x²
    q = np.array([1, -1, 2])  # 1 - x + 2x²

    print(f"\np(x) 係數：{p}")
    print(f"q(x) 係數：{q}")

    # 多項式加法
    print(f"\n(p + q)(x) 係數：{p + q}")
    print("對應：(2+1) + (3-1)x + (-1+2)x² = 3 + 2x + x²")

    # 純量乘法
    c = 2
    print(f"\n{c}·p(x) 係數：{c * p}")
    print(f"對應：{c}(2 + 3x - x²) = 4 + 6x - 2x²")


def main():
    """主程式"""

    print_separator("向量空間示範\nVector Space Demo")

    # 1. 驗證公理
    verify_vector_space_axioms()

    # 2. 子空間例子
    subspace_examples()

    # 3. 張成
    span_demo()

    # 4. 判斷是否在 span 中
    span_membership_demo()

    # 5. 多項式空間
    polynomial_space_demo()

    # 總結
    print_separator("總結")
    print("""
向量空間的關鍵概念：

1. 向量空間是滿足 8 條公理的集合
2. 子空間是「封閉於加法和純量乘法」的子集
3. 判斷子空間：檢查零向量 + 線性組合封閉
4. span{v₁, ..., vₖ} 是所有線性組合形成的子空間

常見陷阱：
- 忘記檢查零向量
- 混淆「子集」和「子空間」
- 以為多個向量一定增加維度（可能線性相依）
""")

    print("=" * 60)
    print("向量空間示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
