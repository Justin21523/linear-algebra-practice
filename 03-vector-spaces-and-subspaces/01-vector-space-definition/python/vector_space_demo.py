"""
向量空間示範 (Vector Space Demo)

本程式示範：
1. 向量空間的公理驗證
2. 子空間的判斷
3. 張成 (Span) 的計算
4. 常見的向量空間範例

This program demonstrates vector space concepts including
axiom verification, subspace testing, and span computation.
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.
from typing import List, Callable, Tuple  # EN: Import symbol(s) from a module: from typing import List, Callable, Tuple.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def verify_vector_space_axioms() -> None:  # EN: Define verify_vector_space_axioms and its behavior.
    """
    驗證 ℝ² 滿足向量空間的公理
    Verify that ℝ² satisfies vector space axioms
    """  # EN: Execute statement: """.
    print_separator("1. 驗證向量空間公理（以 ℝ² 為例）")  # EN: Call print_separator(...) to perform an operation.

    u = np.array([1.0, 2.0])  # EN: Assign u from expression: np.array([1.0, 2.0]).
    v = np.array([3.0, 4.0])  # EN: Assign v from expression: np.array([3.0, 4.0]).
    w = np.array([5.0, 6.0])  # EN: Assign w from expression: np.array([5.0, 6.0]).
    c, d = 2.0, 3.0  # EN: Execute statement: c, d = 2.0, 3.0.

    print(f"u = {u}, v = {v}, w = {w}")  # EN: Print formatted output to the console.
    print(f"c = {c}, d = {d}")  # EN: Print formatted output to the console.

    print("\n【加法公理】")  # EN: Print formatted output to the console.

    # A1: 封閉性
    print(f"A1 封閉性: u + v = {u + v} ∈ ℝ² ✓")  # EN: Print formatted output to the console.

    # A2: 交換律
    print(f"A2 交換律: u + v = {u + v}, v + u = {v + u}")  # EN: Print formatted output to the console.
    print(f"   u + v == v + u ? {np.allclose(u + v, v + u)} ✓")  # EN: Print formatted output to the console.

    # A3: 結合律
    print(f"A3 結合律: (u + v) + w = {(u + v) + w}")  # EN: Print formatted output to the console.
    print(f"           u + (v + w) = {u + (v + w)}")  # EN: Print formatted output to the console.
    print(f"   相等？ {np.allclose((u + v) + w, u + (v + w))} ✓")  # EN: Print formatted output to the console.

    # A4: 零向量
    zero = np.array([0.0, 0.0])  # EN: Assign zero from expression: np.array([0.0, 0.0]).
    print(f"A4 零向量: v + 0 = {v + zero}")  # EN: Print formatted output to the console.
    print(f"   v + 0 == v ? {np.allclose(v + zero, v)} ✓")  # EN: Print formatted output to the console.

    # A5: 反向量
    print(f"A5 反向量: v + (-v) = {v + (-v)}")  # EN: Print formatted output to the console.
    print(f"   v + (-v) == 0 ? {np.allclose(v + (-v), zero)} ✓")  # EN: Print formatted output to the console.

    print("\n【純量乘法公理】")  # EN: Print formatted output to the console.

    # M1: 封閉性
    print(f"M1 封閉性: c·v = {c * v} ∈ ℝ² ✓")  # EN: Print formatted output to the console.

    # M2: 分配律（純量對向量）
    print(f"M2 分配律: c(u + v) = {c * (u + v)}")  # EN: Print formatted output to the console.
    print(f"           cu + cv = {c * u + c * v}")  # EN: Print formatted output to the console.
    print(f"   相等？ {np.allclose(c * (u + v), c * u + c * v)} ✓")  # EN: Print formatted output to the console.

    # M3: 分配律（向量對純量）
    print(f"M3 分配律: (c + d)v = {(c + d) * v}")  # EN: Print formatted output to the console.
    print(f"           cv + dv = {c * v + d * v}")  # EN: Print formatted output to the console.
    print(f"   相等？ {np.allclose((c + d) * v, c * v + d * v)} ✓")  # EN: Print formatted output to the console.

    # M4: 結合律
    print(f"M4 結合律: c(dv) = {c * (d * v)}")  # EN: Print formatted output to the console.
    print(f"           (cd)v = {(c * d) * v}")  # EN: Print formatted output to the console.
    print(f"   相等？ {np.allclose(c * (d * v), (c * d) * v)} ✓")  # EN: Print formatted output to the console.

    # M5: 單位元
    print(f"M5 單位元: 1·v = {1 * v}")  # EN: Print formatted output to the console.
    print(f"   1·v == v ? {np.allclose(1 * v, v)} ✓")  # EN: Print formatted output to the console.

    print("\n結論：ℝ² 滿足所有向量空間公理！")  # EN: Print formatted output to the console.


def is_subspace_point_test(points: List[np.ndarray], name: str) -> None:  # EN: Define is_subspace_point_test and its behavior.
    """
    用有限點集測試是否「可能」是子空間
    （注意：這只是必要條件，不是充分條件）
    """  # EN: Execute statement: """.
    print(f"\n測試 {name}：")  # EN: Print formatted output to the console.

    # 檢查 1：包含零向量？
    zero = np.zeros_like(points[0])  # EN: Assign zero from expression: np.zeros_like(points[0]).
    has_zero = any(np.allclose(p, zero) for p in points)  # EN: Assign has_zero from expression: any(np.allclose(p, zero) for p in points).
    print(f"  包含零向量？ {has_zero}")  # EN: Print formatted output to the console.

    if not has_zero:  # EN: Branch on a condition: if not has_zero:.
        print(f"  結論：❌ 不是子空間（不包含零向量）")  # EN: Print formatted output to the console.
        return  # EN: Return a value: return.

    # 檢查 2：加法封閉？（測試所有組合）
    addition_closed = True  # EN: Assign addition_closed from expression: True.
    for i, p1 in enumerate(points):  # EN: Iterate with a for-loop: for i, p1 in enumerate(points):.
        for j, p2 in enumerate(points):  # EN: Iterate with a for-loop: for j, p2 in enumerate(points):.
            sum_vec = p1 + p2  # EN: Assign sum_vec from expression: p1 + p2.
            in_set = any(np.allclose(sum_vec, p) for p in points)  # EN: Assign in_set from expression: any(np.allclose(sum_vec, p) for p in points).
            if not in_set:  # EN: Branch on a condition: if not in_set:.
                print(f"  {p1} + {p2} = {sum_vec} 不在集合中")  # EN: Print formatted output to the console.
                addition_closed = False  # EN: Assign addition_closed from expression: False.
                break  # EN: Control flow statement: break.
        if not addition_closed:  # EN: Branch on a condition: if not addition_closed:.
            break  # EN: Control flow statement: break.

    # 檢查 3：純量乘法封閉？（測試一些純量）
    scalar_closed = True  # EN: Assign scalar_closed from expression: True.
    test_scalars = [0.5, 2.0, -1.0]  # EN: Assign test_scalars from expression: [0.5, 2.0, -1.0].
    for p in points:  # EN: Iterate with a for-loop: for p in points:.
        for c in test_scalars:  # EN: Iterate with a for-loop: for c in test_scalars:.
            scaled = c * p  # EN: Assign scaled from expression: c * p.
            in_set = any(np.allclose(scaled, q) for q in points)  # EN: Assign in_set from expression: any(np.allclose(scaled, q) for q in points).
            # 這裡我們只標記，因為有限點集無法完全測試

    print(f"  （注意：有限點集無法完全驗證，需要數學證明）")  # EN: Print formatted output to the console.


def subspace_examples() -> None:  # EN: Define subspace_examples and its behavior.
    """
    子空間的例子與反例
    """  # EN: Execute statement: """.
    print_separator("2. 子空間的例子與反例")  # EN: Call print_separator(...) to perform an operation.

    print("\n【✅ 是子空間的例子】")  # EN: Print formatted output to the console.

    print("\n例 1：通過原點的直線（ℝ² 中）")  # EN: Print formatted output to the console.
    print("  W = {t(1, 2) : t ∈ ℝ}")  # EN: Print formatted output to the console.
    print("  • 包含零向量：t=0 時得 (0, 0) ✓")  # EN: Print formatted output to the console.
    print("  • 加法封閉：t₁(1,2) + t₂(1,2) = (t₁+t₂)(1,2) ∈ W ✓")  # EN: Print formatted output to the console.
    print("  • 純量乘法封閉：c·t(1,2) = (ct)(1,2) ∈ W ✓")  # EN: Print formatted output to the console.

    print("\n例 2：xy 平面（ℝ³ 中）")  # EN: Print formatted output to the console.
    print("  W = {(x, y, 0) : x, y ∈ ℝ}")  # EN: Print formatted output to the console.
    print("  • 包含零向量：(0, 0, 0) ∈ W ✓")  # EN: Print formatted output to the console.
    print("  • 加法封閉：(x₁,y₁,0) + (x₂,y₂,0) = (x₁+x₂, y₁+y₂, 0) ∈ W ✓")  # EN: Print formatted output to the console.
    print("  • 純量乘法封閉：c(x,y,0) = (cx, cy, 0) ∈ W ✓")  # EN: Print formatted output to the console.

    print("\n【❌ 不是子空間的例子】")  # EN: Print formatted output to the console.

    print("\n例 3：不通過原點的直線")  # EN: Print formatted output to the console.
    print("  W = {(x, y) : y = x + 1}")  # EN: Print formatted output to the console.
    print("  • 不包含零向量：(0, 0) 不滿足 0 = 0 + 1 ❌")  # EN: Print formatted output to the console.

    print("\n例 4：第一象限")  # EN: Print formatted output to the console.
    print("  W = {(x, y) : x ≥ 0, y ≥ 0}")  # EN: Print formatted output to the console.
    print("  • 包含零向量 ✓")  # EN: Print formatted output to the console.
    print("  • 但純量乘法不封閉：(1, 1) ∈ W，但 -1·(1, 1) = (-1, -1) ∉ W ❌")  # EN: Print formatted output to the console.

    print("\n例 5：單位圓上的點")  # EN: Print formatted output to the console.
    print("  W = {(x, y) : x² + y² = 1}")  # EN: Print formatted output to the console.
    print("  • 不包含零向量：0² + 0² ≠ 1 ❌")  # EN: Print formatted output to the console.

    print("\n例 6：整數向量")  # EN: Print formatted output to the console.
    print("  W = {(x, y) : x, y ∈ ℤ}")  # EN: Print formatted output to the console.
    print("  • 包含零向量 ✓")  # EN: Print formatted output to the console.
    print("  • 加法封閉 ✓")  # EN: Print formatted output to the console.
    print("  • 但純量乘法不封閉：0.5·(1, 0) = (0.5, 0) ∉ W ❌")  # EN: Print formatted output to the console.


def span_demo() -> None:  # EN: Define span_demo and its behavior.
    """
    張成 (Span) 的示範
    """  # EN: Execute statement: """.
    print_separator("3. 張成 (Span) 的示範")  # EN: Call print_separator(...) to perform an operation.

    print("\n【定義】")  # EN: Print formatted output to the console.
    print("span{v₁, ..., vₖ} = 所有線性組合 c₁v₁ + ... + cₖvₖ 形成的集合")  # EN: Print formatted output to the console.

    print("\n【範例 1】單一向量的張成")  # EN: Print formatted output to the console.
    v1 = np.array([1.0, 2.0])  # EN: Assign v1 from expression: np.array([1.0, 2.0]).
    print(f"v₁ = {v1}")  # EN: Print formatted output to the console.
    print(f"span{{v₁}} = 通過原點、方向為 {v1} 的直線")  # EN: Print formatted output to the console.
    print("這是 ℝ² 的一維子空間")  # EN: Print formatted output to the console.

    # 展示一些線性組合
    print("\n一些線性組合：")  # EN: Print formatted output to the console.
    for t in [-1, 0, 0.5, 1, 2]:  # EN: Iterate with a for-loop: for t in [-1, 0, 0.5, 1, 2]:.
        print(f"  {t}·v₁ = {t * v1}")  # EN: Print formatted output to the console.

    print("\n【範例 2】兩個不平行向量的張成")  # EN: Print formatted output to the console.
    v1 = np.array([1.0, 0.0])  # EN: Assign v1 from expression: np.array([1.0, 0.0]).
    v2 = np.array([0.0, 1.0])  # EN: Assign v2 from expression: np.array([0.0, 1.0]).
    print(f"v₁ = {v1}, v₂ = {v2}")  # EN: Print formatted output to the console.
    print(f"span{{v₁, v₂}} = 整個 ℝ²")  # EN: Print formatted output to the console.

    # 展示任意向量都可以表示為線性組合
    target = np.array([3.0, 4.0])  # EN: Assign target from expression: np.array([3.0, 4.0]).
    print(f"\n任意向量如 {target} = {target[0]}·v₁ + {target[1]}·v₂")  # EN: Print formatted output to the console.

    print("\n【範例 3】線性相依向量的張成")  # EN: Print formatted output to the console.
    v1 = np.array([1.0, 2.0])  # EN: Assign v1 from expression: np.array([1.0, 2.0]).
    v2 = np.array([2.0, 4.0])  # v2 = 2·v1  # EN: Assign v2 from expression: np.array([2.0, 4.0]) # v2 = 2·v1.
    print(f"v₁ = {v1}, v₂ = {v2}")  # EN: Print formatted output to the console.
    print(f"注意：v₂ = 2·v₁，它們線性相依")  # EN: Print formatted output to the console.
    print(f"span{{v₁, v₂}} = span{{v₁}} = 通過原點、方向為 v₁ 的直線")  # EN: Print formatted output to the console.
    print("加入 v₂ 並沒有增加新的方向！")  # EN: Print formatted output to the console.

    print("\n【範例 4】三維空間中的張成")  # EN: Print formatted output to the console.
    v1 = np.array([1.0, 0.0, 0.0])  # EN: Assign v1 from expression: np.array([1.0, 0.0, 0.0]).
    v2 = np.array([0.0, 1.0, 0.0])  # EN: Assign v2 from expression: np.array([0.0, 1.0, 0.0]).
    print(f"v₁ = {v1}")  # EN: Print formatted output to the console.
    print(f"v₂ = {v2}")  # EN: Print formatted output to the console.
    print(f"span{{v₁, v₂}} = xy 平面（ℝ³ 的二維子空間）")  # EN: Print formatted output to the console.


def check_in_span(vectors: List[np.ndarray], target: np.ndarray) -> Tuple[bool, np.ndarray]:  # EN: Define check_in_span and its behavior.
    """
    檢查 target 是否在 vectors 的張成中

    使用最小平方法：若 target 可以精確表示為線性組合，殘差為 0
    """  # EN: Execute statement: """.
    # 建立矩陣 A = [v₁ | v₂ | ... | vₖ]
    A = np.column_stack(vectors)  # EN: Assign A from expression: np.column_stack(vectors).

    # 解最小平方問題
    coeffs, residuals, rank, s = np.linalg.lstsq(A, target, rcond=None)  # EN: Execute statement: coeffs, residuals, rank, s = np.linalg.lstsq(A, target, rcond=None).

    # 檢查殘差
    reconstructed = A @ coeffs  # EN: Assign reconstructed from expression: A @ coeffs.
    error = np.linalg.norm(target - reconstructed)  # EN: Assign error from expression: np.linalg.norm(target - reconstructed).

    in_span = error < 1e-10  # EN: Assign in_span from expression: error < 1e-10.

    return in_span, coeffs  # EN: Return a value: return in_span, coeffs.


def span_membership_demo() -> None:  # EN: Define span_membership_demo and its behavior.
    """
    示範如何判斷向量是否在 span 中
    """  # EN: Execute statement: """.
    print_separator("4. 判斷向量是否在 Span 中")  # EN: Call print_separator(...) to perform an operation.

    v1 = np.array([1.0, 0.0, 1.0])  # EN: Assign v1 from expression: np.array([1.0, 0.0, 1.0]).
    v2 = np.array([0.0, 1.0, 1.0])  # EN: Assign v2 from expression: np.array([0.0, 1.0, 1.0]).

    print(f"v₁ = {v1}")  # EN: Print formatted output to the console.
    print(f"v₂ = {v2}")  # EN: Print formatted output to the console.
    print(f"span{{v₁, v₂}} 是 ℝ³ 的某個二維子空間（平面）")  # EN: Print formatted output to the console.

    # 測試向量
    test_vectors = [  # EN: Assign test_vectors from expression: [.
        np.array([1.0, 1.0, 2.0]),  # 在 span 中  # EN: Execute statement: np.array([1.0, 1.0, 2.0]), # 在 span 中.
        np.array([2.0, 3.0, 5.0]),  # 在 span 中  # EN: Execute statement: np.array([2.0, 3.0, 5.0]), # 在 span 中.
        np.array([1.0, 1.0, 1.0]),  # 不在 span 中  # EN: Execute statement: np.array([1.0, 1.0, 1.0]), # 不在 span 中.
    ]  # EN: Execute statement: ].

    print("\n測試各向量：")  # EN: Print formatted output to the console.
    for i, target in enumerate(test_vectors):  # EN: Iterate with a for-loop: for i, target in enumerate(test_vectors):.
        in_span, coeffs = check_in_span([v1, v2], target)  # EN: Execute statement: in_span, coeffs = check_in_span([v1, v2], target).

        print(f"\n目標向量：{target}")  # EN: Print formatted output to the console.
        if in_span:  # EN: Branch on a condition: if in_span:.
            print(f"  ✓ 在 span 中")  # EN: Print formatted output to the console.
            print(f"  係數：{coeffs[0]:.4f}·v₁ + {coeffs[1]:.4f}·v₂")  # EN: Print formatted output to the console.
            print(f"  驗證：{coeffs[0]*v1 + coeffs[1]*v2}")  # EN: Print formatted output to the console.
        else:  # EN: Execute the fallback branch when prior conditions are false.
            print(f"  ✗ 不在 span 中")  # EN: Print formatted output to the console.


def polynomial_space_demo() -> None:  # EN: Define polynomial_space_demo and its behavior.
    """
    多項式空間示範
    """  # EN: Execute statement: """.
    print_separator("5. 多項式空間 P₂")  # EN: Call print_separator(...) to perform an operation.

    print("P₂ = {a₀ + a₁x + a₂x² : aᵢ ∈ ℝ}")  # EN: Print formatted output to the console.
    print("這是一個三維向量空間！")  # EN: Print formatted output to the console.
    print("\n標準基底：{1, x, x²}")  # EN: Print formatted output to the console.
    print("任何多項式都可以寫成：a₀·1 + a₁·x + a₂·x²")  # EN: Print formatted output to the console.

    print("\n【以係數向量表示多項式】")  # EN: Print formatted output to the console.
    print("多項式 p(x) = 2 + 3x - x² 可以表示為係數向量 [2, 3, -1]")  # EN: Print formatted output to the console.

    # 用 NumPy 表示多項式
    p = np.array([2, 3, -1])  # 2 + 3x - x²  # EN: Assign p from expression: np.array([2, 3, -1]) # 2 + 3x - x².
    q = np.array([1, -1, 2])  # 1 - x + 2x²  # EN: Assign q from expression: np.array([1, -1, 2]) # 1 - x + 2x².

    print(f"\np(x) 係數：{p}")  # EN: Print formatted output to the console.
    print(f"q(x) 係數：{q}")  # EN: Print formatted output to the console.

    # 多項式加法
    print(f"\n(p + q)(x) 係數：{p + q}")  # EN: Print formatted output to the console.
    print("對應：(2+1) + (3-1)x + (-1+2)x² = 3 + 2x + x²")  # EN: Print formatted output to the console.

    # 純量乘法
    c = 2  # EN: Assign c from expression: 2.
    print(f"\n{c}·p(x) 係數：{c * p}")  # EN: Print formatted output to the console.
    print(f"對應：{c}(2 + 3x - x²) = 4 + 6x - 2x²")  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("向量空間示範\nVector Space Demo")  # EN: Call print_separator(...) to perform an operation.

    # 1. 驗證公理
    verify_vector_space_axioms()  # EN: Call verify_vector_space_axioms(...) to perform an operation.

    # 2. 子空間例子
    subspace_examples()  # EN: Call subspace_examples(...) to perform an operation.

    # 3. 張成
    span_demo()  # EN: Call span_demo(...) to perform an operation.

    # 4. 判斷是否在 span 中
    span_membership_demo()  # EN: Call span_membership_demo(...) to perform an operation.

    # 5. 多項式空間
    polynomial_space_demo()  # EN: Call polynomial_space_demo(...) to perform an operation.

    # 總結
    print_separator("總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("向量空間示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
