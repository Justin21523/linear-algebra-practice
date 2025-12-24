"""
對角化：NumPy 版本 (Diagonalization: NumPy Implementation)

本程式示範：
1. 使用 NumPy 計算特徵值與特徵向量
2. 驗證對角化分解 A = P * D * P^(-1)
3. 冪次法 (Power Method)：觀察向量收斂到主特徵向量
4. 利用對角化快速計算矩陣冪次 A^k

This program demonstrates:
1. Using NumPy to compute eigenvalues/eigenvectors
2. Verification of diagonalization A = P * D * P^(-1)
3. Power Method: observing vector convergence to dominant eigenvector
4. Fast matrix power computation using diagonalization
"""

import numpy as np

# 設定 NumPy 印出格式 (Set NumPy print format)
np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線與標題 (Print separator with title)"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def power_method_demo(A: np.ndarray, initial_vec: np.ndarray, iterations: int = 15) -> None:
    """
    冪次法示範 (Power Method Demonstration)

    反覆計算 v_(k+1) = A * v_k 並正規化，
    觀察向量逐漸收斂到主特徵向量

    Parameters:
    - A: 方陣 (square matrix)
    - initial_vec: 初始向量 (initial vector)
    - iterations: 迭代次數 (number of iterations)
    """
    print_separator("冪次法示範 (Power Method Demonstration)")
    print(f"初始向量 (Initial vector): {initial_vec}")
    print()

    v = initial_vec.copy().astype(float)
    v = v / np.linalg.norm(v)  # 正規化 (Normalize)

    for k in range(iterations):
        # 計算 A * v (Compute A * v)
        v_new = A @ v

        # Rayleigh 商估計特徵值 (Rayleigh quotient for eigenvalue estimation)
        rayleigh = (v @ v_new) / (v @ v)

        # 正規化 (Normalize)
        v = v_new / np.linalg.norm(v_new)

        if k < 5 or k % 5 == 4:
            print(f"迭代 {k+1:2d}: v = {v}  估計 λ = {rayleigh:.5f}")

    print()
    print(f"收斂結果 (Converged result): {v}")
    return v


def main():
    """主程式 (Main program)"""

    print_separator("對角化示範 - NumPy 版本\nDiagonalization Demo - NumPy Implementation")

    # ========================================
    # 範例矩陣 (Example matrix)
    # ========================================
    # 使用一個 3x3 對稱矩陣來展示 NumPy 的威力
    # Using a 3x3 symmetric matrix to showcase NumPy's power
    A = np.array([
        [4, 1, 1],
        [1, 3, 1],
        [1, 1, 2]
    ], dtype=float)

    print("\n原始矩陣 A (Original matrix A):")
    print(A)

    # ========================================
    # 步驟 1：使用 NumPy 計算特徵值與特徵向量
    # Step 1: Compute eigenvalues and eigenvectors using NumPy
    # ========================================
    print_separator("步驟 1：計算特徵值與特徵向量\nStep 1: Compute Eigenvalues and Eigenvectors")

    # np.linalg.eig() 回傳：
    # - eigenvalues: 特徵值陣列
    # - eigenvectors: 特徵向量矩陣（每一行是一個特徵向量）
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("特徵值 (Eigenvalues):")
    for i, lam in enumerate(eigenvalues):
        print(f"  λ_{i+1} = {lam:.4f}")

    print("\n特徵向量矩陣 P (Eigenvector matrix P):")
    print("（每一行是一個特徵向量 / Each column is an eigenvector）")
    print(eigenvectors)

    # ========================================
    # 步驟 2：驗證 A * v = λ * v
    # Step 2: Verify A * v = λ * v
    # ========================================
    print_separator("步驟 2：驗證特徵方程 A·v = λ·v\nStep 2: Verify Eigenequation")

    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]  # 第 i 個特徵向量 (i-th eigenvector)
        lam = eigenvalues[i]    # 對應的特徵值 (corresponding eigenvalue)

        # 計算 A * v 和 λ * v (Compute A*v and λ*v)
        Av = A @ v
        lambda_v = lam * v

        print(f"\n特徵對 {i+1} (Eigenpair {i+1}):")
        print(f"  λ = {lam:.4f}")
        print(f"  v = {v}")
        print(f"  A·v   = {Av}")
        print(f"  λ·v   = {lambda_v}")
        print(f"  誤差 (Error) = {np.linalg.norm(Av - lambda_v):.10f}")

    # ========================================
    # 步驟 3：建構對角矩陣 D 並驗證 A = P * D * P^(-1)
    # Step 3: Construct D and verify A = P * D * P^(-1)
    # ========================================
    print_separator("步驟 3：驗證對角化 A = P·D·P⁻¹\nStep 3: Verify Diagonalization")

    P = eigenvectors
    D = np.diag(eigenvalues)  # 建構對角矩陣 (Construct diagonal matrix)
    P_inv = np.linalg.inv(P)

    print("對角矩陣 D (Diagonal matrix D):")
    print(D)

    print("\nP⁻¹ (Inverse of P):")
    print(P_inv)

    # 重建 A (Reconstruct A)
    reconstructed_A = P @ D @ P_inv

    print("\n重建的 A = P·D·P⁻¹ (Reconstructed A):")
    print(reconstructed_A)

    print(f"\n重建誤差 (Reconstruction error): {np.linalg.norm(A - reconstructed_A):.10f}")

    # ========================================
    # 步驟 4：利用對角化計算矩陣冪次
    # Step 4: Use diagonalization for matrix power
    # ========================================
    print_separator("步驟 4：利用對角化計算 A^10\nStep 4: Matrix Power via Diagonalization")

    k = 10

    # 方法一：直接計算 (Direct computation)
    A_power_direct = np.linalg.matrix_power(A, k)

    # 方法二：對角化 A^k = P * D^k * P^(-1)
    # D^k 只需要對角線元素各自取 k 次方
    D_power = np.diag(eigenvalues ** k)
    A_power_diag = P @ D_power @ P_inv

    print(f"直接計算 A^{k} (Direct computation):")
    print(A_power_direct)

    print(f"\n對角化計算 A^{k} = P·D^{k}·P⁻¹:")
    print(A_power_diag)

    print(f"\n兩種方法的差異 (Difference): {np.linalg.norm(A_power_direct - A_power_diag):.10f}")

    # ========================================
    # 步驟 5：冪次法示範
    # Step 5: Power Method Demonstration
    # ========================================
    initial_vector = np.array([1.0, 0.0, 0.0])
    converged_v = power_method_demo(A, initial_vector, iterations=15)

    # 比較收斂結果與真正的主特徵向量
    # Compare converged result with true dominant eigenvector
    dominant_idx = np.argmax(np.abs(eigenvalues))
    dominant_eigenvector = eigenvectors[:, dominant_idx]

    # 調整符號以便比較（特徵向量方向可能相反）
    # Adjust sign for comparison (eigenvector direction may be flipped)
    if np.dot(converged_v, dominant_eigenvector) < 0:
        dominant_eigenvector = -dominant_eigenvector

    print_separator("冪次法結果分析 (Power Method Analysis)")
    print(f"冪次法收斂結果: {converged_v}")
    print(f"真正的主特徵向量: {dominant_eigenvector}")
    print(f"對應的特徵值: λ = {eigenvalues[dominant_idx]:.4f}")
    print(f"角度差異 (cosine similarity): {np.dot(converged_v, dominant_eigenvector):.6f}")

    # ========================================
    # 額外範例：2x2 簡單矩陣
    # Extra example: Simple 2x2 matrix
    # ========================================
    print_separator("額外範例：2x2 矩陣\nExtra Example: 2x2 Matrix")

    A_2x2 = np.array([
        [2, 1],
        [1, 2]
    ], dtype=float)

    print("矩陣 A:")
    print(A_2x2)

    eigenvalues_2x2, eigenvectors_2x2 = np.linalg.eig(A_2x2)

    print(f"\n特徵值: λ₁ = {eigenvalues_2x2[0]:.4f}, λ₂ = {eigenvalues_2x2[1]:.4f}")
    print("\n特徵向量矩陣 P:")
    print(eigenvectors_2x2)

    # 示範 2x2 的冪次法
    print("\n2x2 矩陣的冪次法收斂:")
    power_method_demo(A_2x2, np.array([1.0, 0.0]), iterations=10)


if __name__ == "__main__":
    main()
