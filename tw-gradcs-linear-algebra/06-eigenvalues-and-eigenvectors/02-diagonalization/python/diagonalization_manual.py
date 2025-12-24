"""
對角化：手刻版本 (Diagonalization: Manual Implementation)

本程式示範：
1. 手動計算 2x2 矩陣的特徵值與特徵向量
2. 驗證對角化分解 A = P * D * P^(-1)
3. 冪次法 (Power Method)：觀察向量收斂到主特徵向量

This program demonstrates:
1. Manual computation of eigenvalues/eigenvectors for 2x2 matrices
2. Verification of diagonalization A = P * D * P^(-1)
3. Power Method: observing vector convergence to dominant eigenvector
"""

import math


def print_matrix(name: str, matrix: list[list[float]]) -> None:
    """印出矩陣 (Print matrix)"""
    print(f"{name} =")
    for row in matrix:
        print("  [", "  ".join(f"{x:8.4f}" for x in row), "]")
    print()


def print_vector(name: str, vec: list[float]) -> None:
    """印出向量 (Print vector)"""
    print(f"{name} = [{', '.join(f'{x:.4f}' for x in vec)}]")


def matrix_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    矩陣乘法 (Matrix multiplication)
    計算 A * B，其中 A 是 m×n，B 是 n×p，結果是 m×p
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])

    # 初始化結果矩陣 (Initialize result matrix)
    result = [[0.0] * p for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    return result


def matrix_vector_multiply(A: list[list[float]], v: list[float]) -> list[float]:
    """
    矩陣與向量相乘 (Matrix-vector multiplication)
    計算 A * v
    """
    n = len(A)
    result = [0.0] * n

    for i in range(n):
        for j in range(len(v)):
            result[i] += A[i][j] * v[j]

    return result


def compute_eigenvalues_2x2(A: list[list[float]]) -> tuple[float, float]:
    """
    計算 2x2 矩陣的特徵值 (Compute eigenvalues of 2x2 matrix)

    對於 A = [[a, b], [c, d]]，特徵多項式為：
    λ² - (a+d)λ + (ad-bc) = 0

    使用公式解：
    λ = [(a+d) ± sqrt((a+d)² - 4(ad-bc))] / 2
    """
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]

    # 跡 (trace) = a + d
    trace = a + d

    # 行列式 (determinant) = ad - bc
    det = a * d - b * c

    # 判別式 (discriminant)
    discriminant = trace * trace - 4 * det

    if discriminant < 0:
        raise ValueError("此矩陣有複數特徵值，本簡化版本不處理")

    sqrt_disc = math.sqrt(discriminant)

    # 兩個特徵值 (Two eigenvalues)
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2

    return lambda1, lambda2


def compute_eigenvector_2x2(A: list[list[float]], eigenvalue: float) -> list[float]:
    """
    計算 2x2 矩陣對應某特徵值的特徵向量
    (Compute eigenvector for a given eigenvalue of 2x2 matrix)

    解 (A - λI)x = 0
    """
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]

    # A - λI
    a_minus_lambda = a - eigenvalue
    d_minus_lambda = d - eigenvalue

    # 解 null space，取非零解
    # 如果 b ≠ 0，則 x = [b, λ-a] 是解（歸一化前）
    # 如果 b = 0 但 c ≠ 0，則 x = [λ-d, c] 是解

    if abs(b) > 1e-10:
        vec = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        vec = [eigenvalue - d, c]
    else:
        # 對角矩陣的情況 (Diagonal matrix case)
        if abs(a_minus_lambda) < 1e-10:
            vec = [1.0, 0.0]
        else:
            vec = [0.0, 1.0]

    # 正規化 (Normalize)
    norm = math.sqrt(vec[0]**2 + vec[1]**2)
    return [vec[0] / norm, vec[1] / norm]


def matrix_inverse_2x2(A: list[list[float]]) -> list[list[float]]:
    """
    計算 2x2 矩陣的反矩陣 (Compute inverse of 2x2 matrix)

    對於 A = [[a, b], [c, d]]
    A^(-1) = (1/det) * [[d, -b], [-c, a]]
    """
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]

    det = a * d - b * c

    if abs(det) < 1e-10:
        raise ValueError("矩陣不可逆 (Matrix is singular)")

    return [
        [d / det, -b / det],
        [-c / det, a / det]
    ]


def vector_norm(v: list[float]) -> float:
    """計算向量的長度 (Compute vector norm)"""
    return math.sqrt(sum(x**2 for x in v))


def normalize_vector(v: list[float]) -> list[float]:
    """正規化向量 (Normalize vector)"""
    norm = vector_norm(v)
    return [x / norm for x in v]


def power_method(A: list[list[float]], initial_vec: list[float], iterations: int = 20) -> None:
    """
    冪次法示範 (Power Method Demonstration)

    反覆計算 v_(k+1) = A * v_k 並正規化，
    觀察向量逐漸收斂到主特徵向量（對應最大特徵值的特徵向量）

    Repeatedly compute v_(k+1) = A * v_k and normalize,
    observe convergence to dominant eigenvector
    """
    print("=" * 60)
    print("冪次法示範 (Power Method Demonstration)")
    print("=" * 60)
    print(f"初始向量 (Initial vector): {initial_vec}")
    print()

    v = initial_vec.copy()

    for k in range(iterations):
        # 計算 A * v
        v_new = matrix_vector_multiply(A, v)

        # 計算 Rayleigh 商來估計特徵值 (Rayleigh quotient for eigenvalue estimation)
        # λ ≈ (v^T * A * v) / (v^T * v)
        numerator = sum(v[i] * v_new[i] for i in range(len(v)))
        denominator = sum(v[i] * v[i] for i in range(len(v)))
        rayleigh = numerator / denominator

        # 正規化 (Normalize)
        v = normalize_vector(v_new)

        # 每 5 次迭代印出一次 (Print every 5 iterations)
        if k < 5 or k % 5 == 4:
            print(f"迭代 {k+1:2d}: v = [{v[0]:8.5f}, {v[1]:8.5f}]  "
                  f"估計 λ = {rayleigh:.5f}")

    print()
    print(f"收斂結果 (Converged result): [{v[0]:.5f}, {v[1]:.5f}]")
    print("此即為主特徵向量 (This is the dominant eigenvector)")


def main():
    """主程式 (Main program)"""

    # ========================================
    # 範例矩陣 (Example matrix)
    # ========================================
    print("=" * 60)
    print("對角化示範 - 手刻版本")
    print("Diagonalization Demo - Manual Implementation")
    print("=" * 60)
    print()

    # 定義一個 2x2 對稱矩陣（保證可對角化且有實特徵值）
    # Define a 2x2 symmetric matrix (guaranteed diagonalizable with real eigenvalues)
    A = [
        [2.0, 1.0],
        [1.0, 2.0]
    ]

    print_matrix("A（原始矩陣 / Original matrix）", A)

    # ========================================
    # 步驟 1：計算特徵值 (Step 1: Compute eigenvalues)
    # ========================================
    print("-" * 40)
    print("步驟 1：計算特徵值 (Compute Eigenvalues)")
    print("-" * 40)

    lambda1, lambda2 = compute_eigenvalues_2x2(A)
    print(f"λ₁ = {lambda1:.4f}")
    print(f"λ₂ = {lambda2:.4f}")
    print()

    # ========================================
    # 步驟 2：計算特徵向量 (Step 2: Compute eigenvectors)
    # ========================================
    print("-" * 40)
    print("步驟 2：計算特徵向量 (Compute Eigenvectors)")
    print("-" * 40)

    v1 = compute_eigenvector_2x2(A, lambda1)
    v2 = compute_eigenvector_2x2(A, lambda2)

    print_vector(f"v₁ (對應 λ₁={lambda1:.2f})", v1)
    print_vector(f"v₂ (對應 λ₂={lambda2:.2f})", v2)
    print()

    # ========================================
    # 步驟 3：建構 P 和 D (Step 3: Construct P and D)
    # ========================================
    print("-" * 40)
    print("步驟 3：建構 P 和 D (Construct P and D)")
    print("-" * 40)

    # P = [v1 | v2]，特徵向量作為行 (eigenvectors as columns)
    P = [
        [v1[0], v2[0]],
        [v1[1], v2[1]]
    ]

    # D = diag(λ₁, λ₂)
    D = [
        [lambda1, 0.0],
        [0.0, lambda2]
    ]

    print_matrix("P（特徵向量矩陣 / Eigenvector matrix）", P)
    print_matrix("D（對角矩陣 / Diagonal matrix）", D)

    # ========================================
    # 步驟 4：驗證 A = P * D * P^(-1)
    # ========================================
    print("-" * 40)
    print("步驟 4：驗證 A = P·D·P⁻¹ (Verify Diagonalization)")
    print("-" * 40)

    P_inv = matrix_inverse_2x2(P)
    print_matrix("P⁻¹", P_inv)

    # 計算 P * D
    PD = matrix_multiply(P, D)

    # 計算 (P * D) * P^(-1)
    reconstructed_A = matrix_multiply(PD, P_inv)
    print_matrix("P·D·P⁻¹（重建的 A / Reconstructed A）", reconstructed_A)

    # 檢查誤差 (Check error)
    error = sum(abs(A[i][j] - reconstructed_A[i][j])
                for i in range(2) for j in range(2))
    print(f"重建誤差 (Reconstruction error): {error:.10f}")
    print()

    # ========================================
    # 步驟 5：冪次法示範 (Step 5: Power Method Demo)
    # ========================================
    initial_vector = [1.0, 0.0]  # 隨機初始向量 (Random initial vector)
    power_method(A, initial_vector, iterations=15)

    print()
    print("=" * 60)
    print("觀察：向量收斂到 [0.707, 0.707]，即 v₁ 的方向")
    print("Observation: Vector converges to [0.707, 0.707], direction of v₁")
    print(f"這是主特徵向量，對應最大特徵值 λ₁ = {lambda1}")
    print(f"This is the dominant eigenvector for largest eigenvalue λ₁ = {lambda1}")
    print("=" * 60)


if __name__ == "__main__":
    main()
