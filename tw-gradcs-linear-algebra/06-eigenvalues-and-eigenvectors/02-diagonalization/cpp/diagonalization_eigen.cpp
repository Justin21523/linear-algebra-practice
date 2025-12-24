/**
 * 對角化：Eigen Library 版本 (Diagonalization: Eigen Library Implementation)
 *
 * 本程式示範：
 * 1. 使用 Eigen 計算特徵值與特徵向量
 * 2. 驗證對角化分解 A = P * D * P^(-1)
 * 3. 冪次法 (Power Method)：觀察向量收斂到主特徵向量
 * 4. 利用對角化快速計算矩陣冪次 A^k
 *
 * This program demonstrates:
 * 1. Using Eigen to compute eigenvalues/eigenvectors
 * 2. Verification of diagonalization A = P * D * P^(-1)
 * 3. Power Method: observing vector convergence to dominant eigenvector
 * 4. Fast matrix power computation using diagonalization
 *
 * 編譯方式 (Compilation):
 *   g++ -std=c++17 -I /path/to/eigen -o diagonalization_eigen diagonalization_eigen.cpp
 *
 * Eigen 安裝方式 (Installing Eigen):
 *   Ubuntu: sudo apt-get install libeigen3-dev
 *   MacOS:  brew install eigen
 */

#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/**
 * 印出分隔線 (Print separator)
 */
void printSeparator(const string& title) {
    cout << endl;
    cout << string(60, '=') << endl;
    cout << title << endl;
    cout << string(60, '=') << endl;
}

/**
 * 冪次法示範 (Power Method Demonstration)
 *
 * 反覆計算 v_(k+1) = A * v_k 並正規化，
 * 觀察向量逐漸收斂到主特徵向量
 */
template<int N>
Vector<double, N> powerMethod(const Matrix<double, N, N>& A,
                               Vector<double, N> initialVec,
                               int iterations) {
    printSeparator("冪次法示範 (Power Method Demonstration)");
    cout << "初始向量 (Initial vector): " << initialVec.transpose() << endl;
    cout << endl;

    Vector<double, N> v = initialVec.normalized();

    cout << fixed << setprecision(5);

    for (int k = 0; k < iterations; k++) {
        // 計算 A * v (Compute A * v)
        Vector<double, N> vNew = A * v;

        // Rayleigh 商估計特徵值 (Rayleigh quotient for eigenvalue estimation)
        double rayleigh = v.dot(vNew) / v.dot(v);

        // 正規化 (Normalize)
        v = vNew.normalized();

        // 印出結果 (Print result)
        if (k < 5 || k % 5 == 4) {
            cout << "迭代 " << setw(2) << (k + 1) << ": v = "
                 << v.transpose() << "  估計 λ = "
                 << setw(8) << rayleigh << endl;
        }
    }

    cout << endl;
    cout << "收斂結果 (Converged result): " << v.transpose() << endl;

    return v;
}

int main() {
    // 設定輸出格式 (Set output format)
    cout << fixed << setprecision(4);

    printSeparator("對角化示範 - Eigen Library 版本\nDiagonalization Demo - Eigen Implementation");

    // ========================================
    // 範例矩陣 (Example matrix)
    // ========================================
    // 使用一個 3x3 對稱矩陣來展示 Eigen 的威力
    // Using a 3x3 symmetric matrix to showcase Eigen's power
    Matrix3d A;
    A << 4, 1, 1,
         1, 3, 1,
         1, 1, 2;

    cout << "\n原始矩陣 A (Original matrix A):\n" << A << endl;

    // ========================================
    // 步驟 1：使用 Eigen 計算特徵值與特徵向量
    // Step 1: Compute eigenvalues and eigenvectors using Eigen
    // ========================================
    printSeparator("步驟 1：計算特徵值與特徵向量\nStep 1: Compute Eigenvalues and Eigenvectors");

    // 使用 EigenSolver（適用於一般方陣）
    // Using EigenSolver (for general square matrices)
    // 對於對稱矩陣，可使用 SelfAdjointEigenSolver 更高效
    // For symmetric matrices, SelfAdjointEigenSolver is more efficient
    SelfAdjointEigenSolver<Matrix3d> solver(A);

    Vector3d eigenvalues = solver.eigenvalues();
    Matrix3d eigenvectors = solver.eigenvectors();

    cout << "特徵值 (Eigenvalues):" << endl;
    for (int i = 0; i < 3; i++) {
        cout << "  λ_" << (i + 1) << " = " << eigenvalues(i) << endl;
    }

    cout << "\n特徵向量矩陣 P (Eigenvector matrix P):\n";
    cout << "（每一行是一個特徵向量 / Each column is an eigenvector）\n";
    cout << eigenvectors << endl;

    // ========================================
    // 步驟 2：驗證 A * v = λ * v
    // Step 2: Verify A * v = λ * v
    // ========================================
    printSeparator("步驟 2：驗證特徵方程 A·v = λ·v\nStep 2: Verify Eigenequation");

    for (int i = 0; i < 3; i++) {
        Vector3d v = eigenvectors.col(i);
        double lambda = eigenvalues(i);

        Vector3d Av = A * v;
        Vector3d lambda_v = lambda * v;

        cout << "\n特徵對 " << (i + 1) << " (Eigenpair " << (i + 1) << "):" << endl;
        cout << "  λ = " << lambda << endl;
        cout << "  v = " << v.transpose() << endl;
        cout << "  A·v   = " << Av.transpose() << endl;
        cout << "  λ·v   = " << lambda_v.transpose() << endl;
        cout << "  誤差 (Error) = " << scientific << setprecision(10)
             << (Av - lambda_v).norm() << fixed << setprecision(4) << endl;
    }

    // ========================================
    // 步驟 3：建構對角矩陣 D 並驗證 A = P * D * P^(-1)
    // Step 3: Construct D and verify A = P * D * P^(-1)
    // ========================================
    printSeparator("步驟 3：驗證對角化 A = P·D·P⁻¹\nStep 3: Verify Diagonalization");

    Matrix3d P = eigenvectors;
    Matrix3d D = eigenvalues.asDiagonal();  // 建構對角矩陣 (Construct diagonal matrix)
    Matrix3d P_inv = P.inverse();

    cout << "對角矩陣 D (Diagonal matrix D):\n" << D << endl;
    cout << "\nP⁻¹ (Inverse of P):\n" << P_inv << endl;

    // 重建 A (Reconstruct A)
    Matrix3d reconstructedA = P * D * P_inv;

    cout << "\n重建的 A = P·D·P⁻¹ (Reconstructed A):\n" << reconstructedA << endl;

    cout << "\n重建誤差 (Reconstruction error): " << scientific << setprecision(10)
         << (A - reconstructedA).norm() << fixed << setprecision(4) << endl;

    // ========================================
    // 步驟 4：利用對角化計算矩陣冪次
    // Step 4: Use diagonalization for matrix power
    // ========================================
    printSeparator("步驟 4：利用對角化計算 A^10\nStep 4: Matrix Power via Diagonalization");

    int k = 10;

    // 方法一：直接計算 (Direct computation using Eigen's pow)
    // Eigen 沒有直接的矩陣冪次，手動實現
    Matrix3d A_power_direct = Matrix3d::Identity();
    for (int i = 0; i < k; i++) {
        A_power_direct = A_power_direct * A;
    }

    // 方法二：對角化 A^k = P * D^k * P^(-1)
    // D^k 只需要對角線元素各自取 k 次方
    Vector3d eigenvalues_power_k = eigenvalues.array().pow(k);
    Matrix3d D_power = eigenvalues_power_k.asDiagonal();
    Matrix3d A_power_diag = P * D_power * P_inv;

    cout << "直接計算 A^" << k << " (Direct computation):\n" << A_power_direct << endl;
    cout << "\n對角化計算 A^" << k << " = P·D^" << k << "·P⁻¹:\n" << A_power_diag << endl;

    cout << "\n兩種方法的差異 (Difference): " << scientific << setprecision(10)
         << (A_power_direct - A_power_diag).norm() << fixed << setprecision(4) << endl;

    // ========================================
    // 步驟 5：冪次法示範
    // Step 5: Power Method Demonstration
    // ========================================
    Vector3d initialVector(1.0, 0.0, 0.0);
    Vector3d convergedV = powerMethod<3>(A, initialVector, 15);

    // 比較收斂結果與真正的主特徵向量
    // Compare converged result with true dominant eigenvector
    int dominantIdx;
    eigenvalues.cwiseAbs().maxCoeff(&dominantIdx);
    Vector3d dominantEigenvector = eigenvectors.col(dominantIdx);

    // 調整符號以便比較（特徵向量方向可能相反）
    // Adjust sign for comparison (eigenvector direction may be flipped)
    if (convergedV.dot(dominantEigenvector) < 0) {
        dominantEigenvector = -dominantEigenvector;
    }

    printSeparator("冪次法結果分析 (Power Method Analysis)");
    cout << "冪次法收斂結果: " << convergedV.transpose() << endl;
    cout << "真正的主特徵向量: " << dominantEigenvector.transpose() << endl;
    cout << "對應的特徵值: λ = " << eigenvalues(dominantIdx) << endl;
    cout << "餘弦相似度 (Cosine similarity): "
         << abs(convergedV.dot(dominantEigenvector)) << endl;

    // ========================================
    // 額外範例：2x2 簡單矩陣
    // Extra example: Simple 2x2 matrix
    // ========================================
    printSeparator("額外範例：2x2 矩陣\nExtra Example: 2x2 Matrix");

    Matrix2d A_2x2;
    A_2x2 << 2, 1,
             1, 2;

    cout << "矩陣 A:\n" << A_2x2 << endl;

    SelfAdjointEigenSolver<Matrix2d> solver2(A_2x2);
    Vector2d eigenvalues_2x2 = solver2.eigenvalues();
    Matrix2d eigenvectors_2x2 = solver2.eigenvectors();

    cout << "\n特徵值: λ₁ = " << eigenvalues_2x2(0)
         << ", λ₂ = " << eigenvalues_2x2(1) << endl;
    cout << "\n特徵向量矩陣 P:\n" << eigenvectors_2x2 << endl;

    // 示範 2x2 的冪次法
    cout << "\n2x2 矩陣的冪次法收斂:" << endl;
    powerMethod<2>(A_2x2, Vector2d(1.0, 0.0), 10);

    return 0;
}
