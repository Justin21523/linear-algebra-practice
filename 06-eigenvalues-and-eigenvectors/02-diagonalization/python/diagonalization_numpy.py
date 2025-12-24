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
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.

# 設定 NumPy 印出格式 (Set NumPy print format)
np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線與標題 (Print separator with title)"""  # EN: Execute statement: """印出分隔線與標題 (Print separator with title)""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


def power_method_demo(A: np.ndarray, initial_vec: np.ndarray, iterations: int = 15) -> None:  # EN: Define power_method_demo and its behavior.
    """
    冪次法示範 (Power Method Demonstration)

    反覆計算 v_(k+1) = A * v_k 並正規化，
    觀察向量逐漸收斂到主特徵向量

    Parameters:
    - A: 方陣 (square matrix)
    - initial_vec: 初始向量 (initial vector)
    - iterations: 迭代次數 (number of iterations)
    """  # EN: Execute statement: """.
    print_separator("冪次法示範 (Power Method Demonstration)")  # EN: Call print_separator(...) to perform an operation.
    print(f"初始向量 (Initial vector): {initial_vec}")  # EN: Print formatted output to the console.
    print()  # EN: Print formatted output to the console.

    v = initial_vec.copy().astype(float)  # EN: Assign v from expression: initial_vec.copy().astype(float).
    v = v / np.linalg.norm(v)  # 正規化 (Normalize)  # EN: Assign v from expression: v / np.linalg.norm(v) # 正規化 (Normalize).

    for k in range(iterations):  # EN: Iterate with a for-loop: for k in range(iterations):.
        # 計算 A * v (Compute A * v)
        v_new = A @ v  # EN: Assign v_new from expression: A @ v.

        # Rayleigh 商估計特徵值 (Rayleigh quotient for eigenvalue estimation)
        rayleigh = (v @ v_new) / (v @ v)  # EN: Assign rayleigh from expression: (v @ v_new) / (v @ v).

        # 正規化 (Normalize)
        v = v_new / np.linalg.norm(v_new)  # EN: Assign v from expression: v_new / np.linalg.norm(v_new).

        if k < 5 or k % 5 == 4:  # EN: Branch on a condition: if k < 5 or k % 5 == 4:.
            print(f"迭代 {k+1:2d}: v = {v}  估計 λ = {rayleigh:.5f}")  # EN: Print formatted output to the console.

    print()  # EN: Print formatted output to the console.
    print(f"收斂結果 (Converged result): {v}")  # EN: Print formatted output to the console.
    return v  # EN: Return a value: return v.


def main():  # EN: Define main and its behavior.
    """主程式 (Main program)"""  # EN: Execute statement: """主程式 (Main program)""".

    print_separator("對角化示範 - NumPy 版本\nDiagonalization Demo - NumPy Implementation")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 範例矩陣 (Example matrix)
    # ========================================
    # 使用一個 3x3 對稱矩陣來展示 NumPy 的威力
    # Using a 3x3 symmetric matrix to showcase NumPy's power
    A = np.array([  # EN: Assign A from expression: np.array([.
        [4, 1, 1],  # EN: Execute statement: [4, 1, 1],.
        [1, 3, 1],  # EN: Execute statement: [1, 3, 1],.
        [1, 1, 2]  # EN: Execute statement: [1, 1, 2].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print("\n原始矩陣 A (Original matrix A):")  # EN: Print formatted output to the console.
    print(A)  # EN: Print formatted output to the console.

    # ========================================
    # 步驟 1：使用 NumPy 計算特徵值與特徵向量
    # Step 1: Compute eigenvalues and eigenvectors using NumPy
    # ========================================
    print_separator("步驟 1：計算特徵值與特徵向量\nStep 1: Compute Eigenvalues and Eigenvectors")  # EN: Call print_separator(...) to perform an operation.

    # np.linalg.eig() 回傳：
    # - eigenvalues: 特徵值陣列
    # - eigenvectors: 特徵向量矩陣（每一行是一個特徵向量）
    eigenvalues, eigenvectors = np.linalg.eig(A)  # EN: Execute statement: eigenvalues, eigenvectors = np.linalg.eig(A).

    print("特徵值 (Eigenvalues):")  # EN: Print formatted output to the console.
    for i, lam in enumerate(eigenvalues):  # EN: Iterate with a for-loop: for i, lam in enumerate(eigenvalues):.
        print(f"  λ_{i+1} = {lam:.4f}")  # EN: Print formatted output to the console.

    print("\n特徵向量矩陣 P (Eigenvector matrix P):")  # EN: Print formatted output to the console.
    print("（每一行是一個特徵向量 / Each column is an eigenvector）")  # EN: Print formatted output to the console.
    print(eigenvectors)  # EN: Print formatted output to the console.

    # ========================================
    # 步驟 2：驗證 A * v = λ * v
    # Step 2: Verify A * v = λ * v
    # ========================================
    print_separator("步驟 2：驗證特徵方程 A·v = λ·v\nStep 2: Verify Eigenequation")  # EN: Call print_separator(...) to perform an operation.

    for i in range(len(eigenvalues)):  # EN: Iterate with a for-loop: for i in range(len(eigenvalues)):.
        v = eigenvectors[:, i]  # 第 i 個特徵向量 (i-th eigenvector)  # EN: Assign v from expression: eigenvectors[:, i] # 第 i 個特徵向量 (i-th eigenvector).
        lam = eigenvalues[i]    # 對應的特徵值 (corresponding eigenvalue)  # EN: Assign lam from expression: eigenvalues[i] # 對應的特徵值 (corresponding eigenvalue).

        # 計算 A * v 和 λ * v (Compute A*v and λ*v)
        Av = A @ v  # EN: Assign Av from expression: A @ v.
        lambda_v = lam * v  # EN: Assign lambda_v from expression: lam * v.

        print(f"\n特徵對 {i+1} (Eigenpair {i+1}):")  # EN: Print formatted output to the console.
        print(f"  λ = {lam:.4f}")  # EN: Print formatted output to the console.
        print(f"  v = {v}")  # EN: Print formatted output to the console.
        print(f"  A·v   = {Av}")  # EN: Print formatted output to the console.
        print(f"  λ·v   = {lambda_v}")  # EN: Print formatted output to the console.
        print(f"  誤差 (Error) = {np.linalg.norm(Av - lambda_v):.10f}")  # EN: Print formatted output to the console.

    # ========================================
    # 步驟 3：建構對角矩陣 D 並驗證 A = P * D * P^(-1)
    # Step 3: Construct D and verify A = P * D * P^(-1)
    # ========================================
    print_separator("步驟 3：驗證對角化 A = P·D·P⁻¹\nStep 3: Verify Diagonalization")  # EN: Call print_separator(...) to perform an operation.

    P = eigenvectors  # EN: Assign P from expression: eigenvectors.
    D = np.diag(eigenvalues)  # 建構對角矩陣 (Construct diagonal matrix)  # EN: Assign D from expression: np.diag(eigenvalues) # 建構對角矩陣 (Construct diagonal matrix).
    P_inv = np.linalg.inv(P)  # EN: Assign P_inv from expression: np.linalg.inv(P).

    print("對角矩陣 D (Diagonal matrix D):")  # EN: Print formatted output to the console.
    print(D)  # EN: Print formatted output to the console.

    print("\nP⁻¹ (Inverse of P):")  # EN: Print formatted output to the console.
    print(P_inv)  # EN: Print formatted output to the console.

    # 重建 A (Reconstruct A)
    reconstructed_A = P @ D @ P_inv  # EN: Assign reconstructed_A from expression: P @ D @ P_inv.

    print("\n重建的 A = P·D·P⁻¹ (Reconstructed A):")  # EN: Print formatted output to the console.
    print(reconstructed_A)  # EN: Print formatted output to the console.

    print(f"\n重建誤差 (Reconstruction error): {np.linalg.norm(A - reconstructed_A):.10f}")  # EN: Print formatted output to the console.

    # ========================================
    # 步驟 4：利用對角化計算矩陣冪次
    # Step 4: Use diagonalization for matrix power
    # ========================================
    print_separator("步驟 4：利用對角化計算 A^10\nStep 4: Matrix Power via Diagonalization")  # EN: Call print_separator(...) to perform an operation.

    k = 10  # EN: Assign k from expression: 10.

    # 方法一：直接計算 (Direct computation)
    A_power_direct = np.linalg.matrix_power(A, k)  # EN: Assign A_power_direct from expression: np.linalg.matrix_power(A, k).

    # 方法二：對角化 A^k = P * D^k * P^(-1)
    # D^k 只需要對角線元素各自取 k 次方
    D_power = np.diag(eigenvalues ** k)  # EN: Assign D_power from expression: np.diag(eigenvalues ** k).
    A_power_diag = P @ D_power @ P_inv  # EN: Assign A_power_diag from expression: P @ D_power @ P_inv.

    print(f"直接計算 A^{k} (Direct computation):")  # EN: Print formatted output to the console.
    print(A_power_direct)  # EN: Print formatted output to the console.

    print(f"\n對角化計算 A^{k} = P·D^{k}·P⁻¹:")  # EN: Print formatted output to the console.
    print(A_power_diag)  # EN: Print formatted output to the console.

    print(f"\n兩種方法的差異 (Difference): {np.linalg.norm(A_power_direct - A_power_diag):.10f}")  # EN: Print formatted output to the console.

    # ========================================
    # 步驟 5：冪次法示範
    # Step 5: Power Method Demonstration
    # ========================================
    initial_vector = np.array([1.0, 0.0, 0.0])  # EN: Assign initial_vector from expression: np.array([1.0, 0.0, 0.0]).
    converged_v = power_method_demo(A, initial_vector, iterations=15)  # EN: Assign converged_v from expression: power_method_demo(A, initial_vector, iterations=15).

    # 比較收斂結果與真正的主特徵向量
    # Compare converged result with true dominant eigenvector
    dominant_idx = np.argmax(np.abs(eigenvalues))  # EN: Assign dominant_idx from expression: np.argmax(np.abs(eigenvalues)).
    dominant_eigenvector = eigenvectors[:, dominant_idx]  # EN: Assign dominant_eigenvector from expression: eigenvectors[:, dominant_idx].

    # 調整符號以便比較（特徵向量方向可能相反）
    # Adjust sign for comparison (eigenvector direction may be flipped)
    if np.dot(converged_v, dominant_eigenvector) < 0:  # EN: Branch on a condition: if np.dot(converged_v, dominant_eigenvector) < 0:.
        dominant_eigenvector = -dominant_eigenvector  # EN: Assign dominant_eigenvector from expression: -dominant_eigenvector.

    print_separator("冪次法結果分析 (Power Method Analysis)")  # EN: Call print_separator(...) to perform an operation.
    print(f"冪次法收斂結果: {converged_v}")  # EN: Print formatted output to the console.
    print(f"真正的主特徵向量: {dominant_eigenvector}")  # EN: Print formatted output to the console.
    print(f"對應的特徵值: λ = {eigenvalues[dominant_idx]:.4f}")  # EN: Print formatted output to the console.
    print(f"角度差異 (cosine similarity): {np.dot(converged_v, dominant_eigenvector):.6f}")  # EN: Print formatted output to the console.

    # ========================================
    # 額外範例：2x2 簡單矩陣
    # Extra example: Simple 2x2 matrix
    # ========================================
    print_separator("額外範例：2x2 矩陣\nExtra Example: 2x2 Matrix")  # EN: Call print_separator(...) to perform an operation.

    A_2x2 = np.array([  # EN: Assign A_2x2 from expression: np.array([.
        [2, 1],  # EN: Execute statement: [2, 1],.
        [1, 2]  # EN: Execute statement: [1, 2].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print("矩陣 A:")  # EN: Print formatted output to the console.
    print(A_2x2)  # EN: Print formatted output to the console.

    eigenvalues_2x2, eigenvectors_2x2 = np.linalg.eig(A_2x2)  # EN: Execute statement: eigenvalues_2x2, eigenvectors_2x2 = np.linalg.eig(A_2x2).

    print(f"\n特徵值: λ₁ = {eigenvalues_2x2[0]:.4f}, λ₂ = {eigenvalues_2x2[1]:.4f}")  # EN: Print formatted output to the console.
    print("\n特徵向量矩陣 P:")  # EN: Print formatted output to the console.
    print(eigenvectors_2x2)  # EN: Print formatted output to the console.

    # 示範 2x2 的冪次法
    print("\n2x2 矩陣的冪次法收斂:")  # EN: Print formatted output to the console.
    power_method_demo(A_2x2, np.array([1.0, 0.0]), iterations=10)  # EN: Call power_method_demo(...) to perform an operation.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
