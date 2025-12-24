"""
計算複雜度示範 (Computational Complexity Demo)

本程式示範：
1. 實測各種線性代數運算的時間複雜度
2. 比較不同方法的效率
3. 驗證 O(n²) 和 O(n³) 的差異

This program demonstrates the time complexity of various
linear algebra operations through actual measurements.
"""

import numpy as np
import time
from typing import List, Callable, Tuple

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def measure_time(func: Callable, *args, repeats: int = 3) -> float:
    """測量函數執行時間（取多次平均）"""
    times = []
    for _ in range(repeats):
        start = time.time()
        func(*args)
        times.append(time.time() - start)
    return np.mean(times)


def estimate_complexity(sizes: List[int], times: List[float]) -> Tuple[float, str]:
    """
    估計複雜度指數

    假設 T(n) ∝ n^k，用最小平方法估計 k
    log(T) = k × log(n) + c
    """
    log_n = np.log(sizes)
    log_t = np.log(times)

    # 線性迴歸
    k = np.polyfit(log_n, log_t, 1)[0]

    if k < 1.5:
        complexity = "O(n)"
    elif k < 2.5:
        complexity = "O(n²)"
    elif k < 3.5:
        complexity = "O(n³)"
    else:
        complexity = f"O(n^{k:.1f})"

    return k, complexity


def main():
    """主程式"""

    print_separator("計算複雜度示範\nComputational Complexity Demo")

    # ========================================
    # 1. 向量運算：O(n)
    # ========================================
    print_separator("1. 向量運算：O(n)")

    sizes_vec = [10000, 20000, 40000, 80000, 160000]
    times_add = []
    times_dot = []

    print(f"{'n':>10} {'向量加法':>15} {'內積':>15}")
    print("-" * 45)

    for n in sizes_vec:
        u = np.random.rand(n)
        v = np.random.rand(n)

        t_add = measure_time(lambda: u + v)
        t_dot = measure_time(lambda: np.dot(u, v))

        times_add.append(t_add)
        times_dot.append(t_dot)

        print(f"{n:>10} {t_add*1000:>12.4f} ms {t_dot*1000:>12.4f} ms")

    k_add, comp_add = estimate_complexity(sizes_vec, times_add)
    k_dot, comp_dot = estimate_complexity(sizes_vec, times_dot)

    print(f"\n估計複雜度：")
    print(f"  向量加法：k ≈ {k_add:.2f}，約為 {comp_add}")
    print(f"  內積：k ≈ {k_dot:.2f}，約為 {comp_dot}")

    # ========================================
    # 2. 矩陣-向量乘法：O(n²)
    # ========================================
    print_separator("2. 矩陣-向量乘法：O(n²)")

    sizes_mv = [200, 400, 800, 1600]
    times_mv = []

    print(f"{'n':>10} {'Ax 時間':>15}")
    print("-" * 30)

    for n in sizes_mv:
        A = np.random.rand(n, n)
        x = np.random.rand(n)

        t = measure_time(lambda: A @ x)
        times_mv.append(t)

        print(f"{n:>10} {t*1000:>12.4f} ms")

    k_mv, comp_mv = estimate_complexity(sizes_mv, times_mv)
    print(f"\n估計複雜度：k ≈ {k_mv:.2f}，約為 {comp_mv}")

    # 理論預測
    print("\n理論預測（n 變為 2 倍，時間變為 4 倍）：")
    for i in range(len(sizes_mv) - 1):
        ratio = times_mv[i+1] / times_mv[i]
        print(f"  n: {sizes_mv[i]} → {sizes_mv[i+1]}，時間比 = {ratio:.2f}")

    # ========================================
    # 3. 矩陣乘法：O(n³)
    # ========================================
    print_separator("3. 矩陣乘法：O(n³)")

    sizes_mm = [100, 200, 400, 800]
    times_mm = []

    print(f"{'n':>10} {'AB 時間':>15}")
    print("-" * 30)

    for n in sizes_mm:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        t = measure_time(lambda: A @ B)
        times_mm.append(t)

        print(f"{n:>10} {t*1000:>12.4f} ms")

    k_mm, comp_mm = estimate_complexity(sizes_mm, times_mm)
    print(f"\n估計複雜度：k ≈ {k_mm:.2f}，約為 {comp_mm}")

    print("\n理論預測（n 變為 2 倍，時間變為 8 倍）：")
    for i in range(len(sizes_mm) - 1):
        ratio = times_mm[i+1] / times_mm[i]
        print(f"  n: {sizes_mm[i]} → {sizes_mm[i+1]}，時間比 = {ratio:.2f}")

    # ========================================
    # 4. 解線性方程組：O(n³)
    # ========================================
    print_separator("4. 解線性方程組 (solve)：O(n³)")

    sizes_solve = [100, 200, 400, 800]
    times_solve = []

    print(f"{'n':>10} {'solve 時間':>15}")
    print("-" * 30)

    for n in sizes_solve:
        A = np.random.rand(n, n) + np.eye(n)
        b = np.random.rand(n)

        t = measure_time(lambda: np.linalg.solve(A, b))
        times_solve.append(t)

        print(f"{n:>10} {t*1000:>12.4f} ms")

    k_solve, comp_solve = estimate_complexity(sizes_solve, times_solve)
    print(f"\n估計複雜度：k ≈ {k_solve:.2f}，約為 {comp_solve}")

    # ========================================
    # 5. LU 分解的效率優勢
    # ========================================
    print_separator("5. LU 分解 vs 多次求解")

    from scipy import linalg

    n = 500
    k_values = [1, 10, 50, 100]
    A = np.random.rand(n, n) + np.eye(n)

    print(f"矩陣大小：{n}×{n}")
    print(f"{'k（右手邊數）':>15} {'每次 solve':>15} {'LU + 回代':>15} {'加速比':>10}")
    print("-" * 60)

    # LU 分解時間（只做一次）
    start = time.time()
    lu, piv = linalg.lu_factor(A)
    t_lu = time.time() - start

    for k in k_values:
        B = np.random.rand(n, k)

        # 方法一：每次 solve
        start = time.time()
        for j in range(k):
            np.linalg.solve(A, B[:, j])
        t_solve_k = time.time() - start

        # 方法二：LU + 回代
        start = time.time()
        for j in range(k):
            linalg.lu_solve((lu, piv), B[:, j])
        t_lu_k = time.time() - start + t_lu  # 加上 LU 分解時間

        speedup = t_solve_k / t_lu_k

        print(f"{k:>15} {t_solve_k*1000:>12.2f} ms {t_lu_k*1000:>12.2f} ms {speedup:>10.2f}x")

    # ========================================
    # 6. 反矩陣 vs solve
    # ========================================
    print_separator("6. 反矩陣 vs solve")

    n = 500
    A = np.random.rand(n, n) + np.eye(n)
    b = np.random.rand(n)

    print(f"矩陣大小：{n}×{n}")

    # 方法一：A⁻¹b
    start = time.time()
    A_inv = np.linalg.inv(A)
    x1 = A_inv @ b
    t_inv = time.time() - start

    # 方法二：solve
    start = time.time()
    x2 = np.linalg.solve(A, b)
    t_solve = time.time() - start

    print(f"\n方法 1（A⁻¹b）：{t_inv*1000:.2f} ms")
    print(f"方法 2（solve）：{t_solve*1000:.2f} ms")
    print(f"solve 快 {t_inv/t_solve:.1f} 倍")

    print(f"\n殘差比較：")
    print(f"  A⁻¹b：‖Ax - b‖ = {np.linalg.norm(A @ x1 - b):.2e}")
    print(f"  solve：‖Ax - b‖ = {np.linalg.norm(A @ x2 - b):.2e}")

    # ========================================
    # 7. 計算行列式
    # ========================================
    print_separator("7. 計算行列式：O(n³)")

    sizes_det = [100, 200, 400, 800]
    times_det = []

    print(f"{'n':>10} {'det 時間':>15}")
    print("-" * 30)

    for n in sizes_det:
        A = np.random.rand(n, n)

        t = measure_time(lambda: np.linalg.det(A))
        times_det.append(t)

        print(f"{n:>10} {t*1000:>12.4f} ms")

    k_det, comp_det = estimate_complexity(sizes_det, times_det)
    print(f"\n估計複雜度：k ≈ {k_det:.2f}，約為 {comp_det}")

    # ========================================
    # 8. 三角矩陣求解：O(n²)
    # ========================================
    print_separator("8. 三角矩陣求解：O(n²)")

    from scipy.linalg import solve_triangular

    sizes_tri = [500, 1000, 2000, 4000]
    times_tri = []
    times_dense = []

    print(f"{'n':>10} {'三角求解':>15} {'一般求解':>15}")
    print("-" * 45)

    for n in sizes_tri:
        U = np.triu(np.random.rand(n, n) + np.eye(n))
        b = np.random.rand(n)

        t_tri = measure_time(lambda: solve_triangular(U, b))
        times_tri.append(t_tri)

        # 一般求解（不利用三角結構）
        t_dense = measure_time(lambda: np.linalg.solve(U, b))
        times_dense.append(t_dense)

        print(f"{n:>10} {t_tri*1000:>12.4f} ms {t_dense*1000:>12.4f} ms")

    k_tri, comp_tri = estimate_complexity(sizes_tri, times_tri)
    print(f"\n三角求解複雜度：k ≈ {k_tri:.2f}，約為 {comp_tri}")

    # ========================================
    # 9. 複雜度總結
    # ========================================
    print_separator("9. 複雜度總結")

    print("""
┌─────────────────────────┬───────────┬─────────────────┐
│ 運算                     │ 複雜度    │ n=1000 大約運算 │
├─────────────────────────┼───────────┼─────────────────┤
│ 向量加法 u + v           │ O(n)      │ 10³             │
│ 內積 u·v                 │ O(n)      │ 10³             │
│ 矩陣-向量乘法 Ax         │ O(n²)     │ 10⁶             │
│ 外積 uvᵀ                 │ O(n²)     │ 10⁶             │
│ 矩陣乘法 AB              │ O(n³)     │ 10⁹             │
│ 高斯消去法               │ O(n³)     │ 10⁹             │
│ LU 分解                  │ O(n³)     │ 10⁹             │
│ 求反矩陣                 │ O(n³)     │ 10⁹             │
│ 三角系統求解             │ O(n²)     │ 10⁶             │
│ LU 回代（已分解）        │ O(n²)     │ 10⁶             │
└─────────────────────────┴───────────┴─────────────────┘

實用建議：
1. 優先使用 solve() 而非 inv() @ b
2. 多個 b 時使用 LU 分解
3. 利用矩陣結構（三角、稀疏、對稱）
4. 大型稀疏矩陣使用迭代法
""")

    print("=" * 60)
    print("計算複雜度示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
