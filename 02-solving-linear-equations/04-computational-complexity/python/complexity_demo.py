"""
計算複雜度示範 (Computational Complexity Demo)

本程式示範：
1. 實測各種線性代數運算的時間複雜度
2. 比較不同方法的效率
3. 驗證 O(n²) 和 O(n³) 的差異

This program demonstrates the time complexity of various
linear algebra operations through actual measurements.
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.
import time  # EN: Import module(s): import time.
from typing import List, Callable, Tuple  # EN: Import symbol(s) from a module: from typing import List, Callable, Tuple.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def measure_time(func: Callable, *args, repeats: int = 3) -> float:  # EN: Define measure_time and its behavior.
    """測量函數執行時間（取多次平均）"""  # EN: Execute statement: """測量函數執行時間（取多次平均）""".
    times = []  # EN: Assign times from expression: [].
    for _ in range(repeats):  # EN: Iterate with a for-loop: for _ in range(repeats):.
        start = time.time()  # EN: Assign start from expression: time.time().
        func(*args)  # EN: Call func(...) to perform an operation.
        times.append(time.time() - start)  # EN: Execute statement: times.append(time.time() - start).
    return np.mean(times)  # EN: Return a value: return np.mean(times).


def estimate_complexity(sizes: List[int], times: List[float]) -> Tuple[float, str]:  # EN: Define estimate_complexity and its behavior.
    """
    估計複雜度指數

    假設 T(n) ∝ n^k，用最小平方法估計 k
    log(T) = k × log(n) + c
    """  # EN: Execute statement: """.
    log_n = np.log(sizes)  # EN: Assign log_n from expression: np.log(sizes).
    log_t = np.log(times)  # EN: Assign log_t from expression: np.log(times).

    # 線性迴歸
    k = np.polyfit(log_n, log_t, 1)[0]  # EN: Assign k from expression: np.polyfit(log_n, log_t, 1)[0].

    if k < 1.5:  # EN: Branch on a condition: if k < 1.5:.
        complexity = "O(n)"  # EN: Assign complexity from expression: "O(n)".
    elif k < 2.5:  # EN: Branch on a condition: elif k < 2.5:.
        complexity = "O(n²)"  # EN: Assign complexity from expression: "O(n²)".
    elif k < 3.5:  # EN: Branch on a condition: elif k < 3.5:.
        complexity = "O(n³)"  # EN: Assign complexity from expression: "O(n³)".
    else:  # EN: Execute the fallback branch when prior conditions are false.
        complexity = f"O(n^{k:.1f})"  # EN: Assign complexity from expression: f"O(n^{k:.1f})".

    return k, complexity  # EN: Return a value: return k, complexity.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("計算複雜度示範\nComputational Complexity Demo")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 向量運算：O(n)
    # ========================================
    print_separator("1. 向量運算：O(n)")  # EN: Call print_separator(...) to perform an operation.

    sizes_vec = [10000, 20000, 40000, 80000, 160000]  # EN: Assign sizes_vec from expression: [10000, 20000, 40000, 80000, 160000].
    times_add = []  # EN: Assign times_add from expression: [].
    times_dot = []  # EN: Assign times_dot from expression: [].

    print(f"{'n':>10} {'向量加法':>15} {'內積':>15}")  # EN: Print formatted output to the console.
    print("-" * 45)  # EN: Print formatted output to the console.

    for n in sizes_vec:  # EN: Iterate with a for-loop: for n in sizes_vec:.
        u = np.random.rand(n)  # EN: Assign u from expression: np.random.rand(n).
        v = np.random.rand(n)  # EN: Assign v from expression: np.random.rand(n).

        t_add = measure_time(lambda: u + v)  # EN: Assign t_add from expression: measure_time(lambda: u + v).
        t_dot = measure_time(lambda: np.dot(u, v))  # EN: Assign t_dot from expression: measure_time(lambda: np.dot(u, v)).

        times_add.append(t_add)  # EN: Execute statement: times_add.append(t_add).
        times_dot.append(t_dot)  # EN: Execute statement: times_dot.append(t_dot).

        print(f"{n:>10} {t_add*1000:>12.4f} ms {t_dot*1000:>12.4f} ms")  # EN: Print formatted output to the console.

    k_add, comp_add = estimate_complexity(sizes_vec, times_add)  # EN: Execute statement: k_add, comp_add = estimate_complexity(sizes_vec, times_add).
    k_dot, comp_dot = estimate_complexity(sizes_vec, times_dot)  # EN: Execute statement: k_dot, comp_dot = estimate_complexity(sizes_vec, times_dot).

    print(f"\n估計複雜度：")  # EN: Print formatted output to the console.
    print(f"  向量加法：k ≈ {k_add:.2f}，約為 {comp_add}")  # EN: Print formatted output to the console.
    print(f"  內積：k ≈ {k_dot:.2f}，約為 {comp_dot}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 矩陣-向量乘法：O(n²)
    # ========================================
    print_separator("2. 矩陣-向量乘法：O(n²)")  # EN: Call print_separator(...) to perform an operation.

    sizes_mv = [200, 400, 800, 1600]  # EN: Assign sizes_mv from expression: [200, 400, 800, 1600].
    times_mv = []  # EN: Assign times_mv from expression: [].

    print(f"{'n':>10} {'Ax 時間':>15}")  # EN: Print formatted output to the console.
    print("-" * 30)  # EN: Print formatted output to the console.

    for n in sizes_mv:  # EN: Iterate with a for-loop: for n in sizes_mv:.
        A = np.random.rand(n, n)  # EN: Assign A from expression: np.random.rand(n, n).
        x = np.random.rand(n)  # EN: Assign x from expression: np.random.rand(n).

        t = measure_time(lambda: A @ x)  # EN: Assign t from expression: measure_time(lambda: A @ x).
        times_mv.append(t)  # EN: Execute statement: times_mv.append(t).

        print(f"{n:>10} {t*1000:>12.4f} ms")  # EN: Print formatted output to the console.

    k_mv, comp_mv = estimate_complexity(sizes_mv, times_mv)  # EN: Execute statement: k_mv, comp_mv = estimate_complexity(sizes_mv, times_mv).
    print(f"\n估計複雜度：k ≈ {k_mv:.2f}，約為 {comp_mv}")  # EN: Print formatted output to the console.

    # 理論預測
    print("\n理論預測（n 變為 2 倍，時間變為 4 倍）：")  # EN: Print formatted output to the console.
    for i in range(len(sizes_mv) - 1):  # EN: Iterate with a for-loop: for i in range(len(sizes_mv) - 1):.
        ratio = times_mv[i+1] / times_mv[i]  # EN: Assign ratio from expression: times_mv[i+1] / times_mv[i].
        print(f"  n: {sizes_mv[i]} → {sizes_mv[i+1]}，時間比 = {ratio:.2f}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 矩陣乘法：O(n³)
    # ========================================
    print_separator("3. 矩陣乘法：O(n³)")  # EN: Call print_separator(...) to perform an operation.

    sizes_mm = [100, 200, 400, 800]  # EN: Assign sizes_mm from expression: [100, 200, 400, 800].
    times_mm = []  # EN: Assign times_mm from expression: [].

    print(f"{'n':>10} {'AB 時間':>15}")  # EN: Print formatted output to the console.
    print("-" * 30)  # EN: Print formatted output to the console.

    for n in sizes_mm:  # EN: Iterate with a for-loop: for n in sizes_mm:.
        A = np.random.rand(n, n)  # EN: Assign A from expression: np.random.rand(n, n).
        B = np.random.rand(n, n)  # EN: Assign B from expression: np.random.rand(n, n).

        t = measure_time(lambda: A @ B)  # EN: Assign t from expression: measure_time(lambda: A @ B).
        times_mm.append(t)  # EN: Execute statement: times_mm.append(t).

        print(f"{n:>10} {t*1000:>12.4f} ms")  # EN: Print formatted output to the console.

    k_mm, comp_mm = estimate_complexity(sizes_mm, times_mm)  # EN: Execute statement: k_mm, comp_mm = estimate_complexity(sizes_mm, times_mm).
    print(f"\n估計複雜度：k ≈ {k_mm:.2f}，約為 {comp_mm}")  # EN: Print formatted output to the console.

    print("\n理論預測（n 變為 2 倍，時間變為 8 倍）：")  # EN: Print formatted output to the console.
    for i in range(len(sizes_mm) - 1):  # EN: Iterate with a for-loop: for i in range(len(sizes_mm) - 1):.
        ratio = times_mm[i+1] / times_mm[i]  # EN: Assign ratio from expression: times_mm[i+1] / times_mm[i].
        print(f"  n: {sizes_mm[i]} → {sizes_mm[i+1]}，時間比 = {ratio:.2f}")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 解線性方程組：O(n³)
    # ========================================
    print_separator("4. 解線性方程組 (solve)：O(n³)")  # EN: Call print_separator(...) to perform an operation.

    sizes_solve = [100, 200, 400, 800]  # EN: Assign sizes_solve from expression: [100, 200, 400, 800].
    times_solve = []  # EN: Assign times_solve from expression: [].

    print(f"{'n':>10} {'solve 時間':>15}")  # EN: Print formatted output to the console.
    print("-" * 30)  # EN: Print formatted output to the console.

    for n in sizes_solve:  # EN: Iterate with a for-loop: for n in sizes_solve:.
        A = np.random.rand(n, n) + np.eye(n)  # EN: Assign A from expression: np.random.rand(n, n) + np.eye(n).
        b = np.random.rand(n)  # EN: Assign b from expression: np.random.rand(n).

        t = measure_time(lambda: np.linalg.solve(A, b))  # EN: Assign t from expression: measure_time(lambda: np.linalg.solve(A, b)).
        times_solve.append(t)  # EN: Execute statement: times_solve.append(t).

        print(f"{n:>10} {t*1000:>12.4f} ms")  # EN: Print formatted output to the console.

    k_solve, comp_solve = estimate_complexity(sizes_solve, times_solve)  # EN: Execute statement: k_solve, comp_solve = estimate_complexity(sizes_solve, times_solve).
    print(f"\n估計複雜度：k ≈ {k_solve:.2f}，約為 {comp_solve}")  # EN: Print formatted output to the console.

    # ========================================
    # 5. LU 分解的效率優勢
    # ========================================
    print_separator("5. LU 分解 vs 多次求解")  # EN: Call print_separator(...) to perform an operation.

    from scipy import linalg  # EN: Import symbol(s) from a module: from scipy import linalg.

    n = 500  # EN: Assign n from expression: 500.
    k_values = [1, 10, 50, 100]  # EN: Assign k_values from expression: [1, 10, 50, 100].
    A = np.random.rand(n, n) + np.eye(n)  # EN: Assign A from expression: np.random.rand(n, n) + np.eye(n).

    print(f"矩陣大小：{n}×{n}")  # EN: Print formatted output to the console.
    print(f"{'k（右手邊數）':>15} {'每次 solve':>15} {'LU + 回代':>15} {'加速比':>10}")  # EN: Print formatted output to the console.
    print("-" * 60)  # EN: Print formatted output to the console.

    # LU 分解時間（只做一次）
    start = time.time()  # EN: Assign start from expression: time.time().
    lu, piv = linalg.lu_factor(A)  # EN: Execute statement: lu, piv = linalg.lu_factor(A).
    t_lu = time.time() - start  # EN: Assign t_lu from expression: time.time() - start.

    for k in k_values:  # EN: Iterate with a for-loop: for k in k_values:.
        B = np.random.rand(n, k)  # EN: Assign B from expression: np.random.rand(n, k).

        # 方法一：每次 solve
        start = time.time()  # EN: Assign start from expression: time.time().
        for j in range(k):  # EN: Iterate with a for-loop: for j in range(k):.
            np.linalg.solve(A, B[:, j])  # EN: Execute statement: np.linalg.solve(A, B[:, j]).
        t_solve_k = time.time() - start  # EN: Assign t_solve_k from expression: time.time() - start.

        # 方法二：LU + 回代
        start = time.time()  # EN: Assign start from expression: time.time().
        for j in range(k):  # EN: Iterate with a for-loop: for j in range(k):.
            linalg.lu_solve((lu, piv), B[:, j])  # EN: Execute statement: linalg.lu_solve((lu, piv), B[:, j]).
        t_lu_k = time.time() - start + t_lu  # 加上 LU 分解時間  # EN: Assign t_lu_k from expression: time.time() - start + t_lu # 加上 LU 分解時間.

        speedup = t_solve_k / t_lu_k  # EN: Assign speedup from expression: t_solve_k / t_lu_k.

        print(f"{k:>15} {t_solve_k*1000:>12.2f} ms {t_lu_k*1000:>12.2f} ms {speedup:>10.2f}x")  # EN: Print formatted output to the console.

    # ========================================
    # 6. 反矩陣 vs solve
    # ========================================
    print_separator("6. 反矩陣 vs solve")  # EN: Call print_separator(...) to perform an operation.

    n = 500  # EN: Assign n from expression: 500.
    A = np.random.rand(n, n) + np.eye(n)  # EN: Assign A from expression: np.random.rand(n, n) + np.eye(n).
    b = np.random.rand(n)  # EN: Assign b from expression: np.random.rand(n).

    print(f"矩陣大小：{n}×{n}")  # EN: Print formatted output to the console.

    # 方法一：A⁻¹b
    start = time.time()  # EN: Assign start from expression: time.time().
    A_inv = np.linalg.inv(A)  # EN: Assign A_inv from expression: np.linalg.inv(A).
    x1 = A_inv @ b  # EN: Assign x1 from expression: A_inv @ b.
    t_inv = time.time() - start  # EN: Assign t_inv from expression: time.time() - start.

    # 方法二：solve
    start = time.time()  # EN: Assign start from expression: time.time().
    x2 = np.linalg.solve(A, b)  # EN: Assign x2 from expression: np.linalg.solve(A, b).
    t_solve = time.time() - start  # EN: Assign t_solve from expression: time.time() - start.

    print(f"\n方法 1（A⁻¹b）：{t_inv*1000:.2f} ms")  # EN: Print formatted output to the console.
    print(f"方法 2（solve）：{t_solve*1000:.2f} ms")  # EN: Print formatted output to the console.
    print(f"solve 快 {t_inv/t_solve:.1f} 倍")  # EN: Print formatted output to the console.

    print(f"\n殘差比較：")  # EN: Print formatted output to the console.
    print(f"  A⁻¹b：‖Ax - b‖ = {np.linalg.norm(A @ x1 - b):.2e}")  # EN: Print formatted output to the console.
    print(f"  solve：‖Ax - b‖ = {np.linalg.norm(A @ x2 - b):.2e}")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 計算行列式
    # ========================================
    print_separator("7. 計算行列式：O(n³)")  # EN: Call print_separator(...) to perform an operation.

    sizes_det = [100, 200, 400, 800]  # EN: Assign sizes_det from expression: [100, 200, 400, 800].
    times_det = []  # EN: Assign times_det from expression: [].

    print(f"{'n':>10} {'det 時間':>15}")  # EN: Print formatted output to the console.
    print("-" * 30)  # EN: Print formatted output to the console.

    for n in sizes_det:  # EN: Iterate with a for-loop: for n in sizes_det:.
        A = np.random.rand(n, n)  # EN: Assign A from expression: np.random.rand(n, n).

        t = measure_time(lambda: np.linalg.det(A))  # EN: Assign t from expression: measure_time(lambda: np.linalg.det(A)).
        times_det.append(t)  # EN: Execute statement: times_det.append(t).

        print(f"{n:>10} {t*1000:>12.4f} ms")  # EN: Print formatted output to the console.

    k_det, comp_det = estimate_complexity(sizes_det, times_det)  # EN: Execute statement: k_det, comp_det = estimate_complexity(sizes_det, times_det).
    print(f"\n估計複雜度：k ≈ {k_det:.2f}，約為 {comp_det}")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 三角矩陣求解：O(n²)
    # ========================================
    print_separator("8. 三角矩陣求解：O(n²)")  # EN: Call print_separator(...) to perform an operation.

    from scipy.linalg import solve_triangular  # EN: Import symbol(s) from a module: from scipy.linalg import solve_triangular.

    sizes_tri = [500, 1000, 2000, 4000]  # EN: Assign sizes_tri from expression: [500, 1000, 2000, 4000].
    times_tri = []  # EN: Assign times_tri from expression: [].
    times_dense = []  # EN: Assign times_dense from expression: [].

    print(f"{'n':>10} {'三角求解':>15} {'一般求解':>15}")  # EN: Print formatted output to the console.
    print("-" * 45)  # EN: Print formatted output to the console.

    for n in sizes_tri:  # EN: Iterate with a for-loop: for n in sizes_tri:.
        U = np.triu(np.random.rand(n, n) + np.eye(n))  # EN: Assign U from expression: np.triu(np.random.rand(n, n) + np.eye(n)).
        b = np.random.rand(n)  # EN: Assign b from expression: np.random.rand(n).

        t_tri = measure_time(lambda: solve_triangular(U, b))  # EN: Assign t_tri from expression: measure_time(lambda: solve_triangular(U, b)).
        times_tri.append(t_tri)  # EN: Execute statement: times_tri.append(t_tri).

        # 一般求解（不利用三角結構）
        t_dense = measure_time(lambda: np.linalg.solve(U, b))  # EN: Assign t_dense from expression: measure_time(lambda: np.linalg.solve(U, b)).
        times_dense.append(t_dense)  # EN: Execute statement: times_dense.append(t_dense).

        print(f"{n:>10} {t_tri*1000:>12.4f} ms {t_dense*1000:>12.4f} ms")  # EN: Print formatted output to the console.

    k_tri, comp_tri = estimate_complexity(sizes_tri, times_tri)  # EN: Execute statement: k_tri, comp_tri = estimate_complexity(sizes_tri, times_tri).
    print(f"\n三角求解複雜度：k ≈ {k_tri:.2f}，約為 {comp_tri}")  # EN: Print formatted output to the console.

    # ========================================
    # 9. 複雜度總結
    # ========================================
    print_separator("9. 複雜度總結")  # EN: Call print_separator(...) to perform an operation.

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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("計算複雜度示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
