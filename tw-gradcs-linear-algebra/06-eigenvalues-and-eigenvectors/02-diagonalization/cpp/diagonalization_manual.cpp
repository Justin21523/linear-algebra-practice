/**
 * 對角化：手刻版本 (Diagonalization: Manual Implementation)
 *
 * 本程式示範：
 * 1. 手動計算 2x2 矩陣的特徵值與特徵向量
 * 2. 驗證對角化分解 A = P * D * P^(-1)
 * 3. 冪次法 (Power Method)：觀察向量收斂到主特徵向量
 *
 * This program demonstrates:
 * 1. Manual computation of eigenvalues/eigenvectors for 2x2 matrices
 * 2. Verification of diagonalization A = P * D * P^(-1)
 * 3. Power Method: observing vector convergence to dominant eigenvector
 *
 * 編譯方式 (Compilation):
 *   g++ -std=c++17 -o diagonalization_manual diagonalization_manual.cpp
 */

#include <iostream>
#include <cmath>
#include <iomanip>
#include <array>

using namespace std;

// 使用 2x2 固定大小的陣列型別 (Using fixed-size 2x2 array types)
using Matrix2x2 = array<array<double, 2>, 2>;
using Vector2 = array<double, 2>;

/**
 * 印出 2x2 矩陣 (Print 2x2 matrix)
 */
void printMatrix(const string& name, const Matrix2x2& M) {
    cout << name << " =" << endl;
    cout << fixed << setprecision(4);
    cout << "  [" << setw(8) << M[0][0] << "  " << setw(8) << M[0][1] << " ]" << endl;
    cout << "  [" << setw(8) << M[1][0] << "  " << setw(8) << M[1][1] << " ]" << endl;
    cout << endl;
}

/**
 * 印出向量 (Print vector)
 */
void printVector(const string& name, const Vector2& v) {
    cout << fixed << setprecision(4);
    cout << name << " = [" << v[0] << ", " << v[1] << "]" << endl;
}

/**
 * 計算向量長度 (Compute vector norm)
 */
double vectorNorm(const Vector2& v) {
    return sqrt(v[0] * v[0] + v[1] * v[1]);
}

/**
 * 正規化向量 (Normalize vector)
 */
Vector2 normalizeVector(const Vector2& v) {
    double norm = vectorNorm(v);
    return {v[0] / norm, v[1] / norm};
}

/**
 * 矩陣與向量相乘 (Matrix-vector multiplication)
 * 計算 M * v
 */
Vector2 matrixVectorMultiply(const Matrix2x2& M, const Vector2& v) {
    return {
        M[0][0] * v[0] + M[0][1] * v[1],
        M[1][0] * v[0] + M[1][1] * v[1]
    };
}

/**
 * 矩陣乘法 (Matrix multiplication)
 * 計算 A * B
 */
Matrix2x2 matrixMultiply(const Matrix2x2& A, const Matrix2x2& B) {
    Matrix2x2 result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < 2; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

/**
 * 計算 2x2 矩陣的特徵值 (Compute eigenvalues of 2x2 matrix)
 *
 * 對於 A = [[a, b], [c, d]]，特徵多項式為：
 * λ² - (a+d)λ + (ad-bc) = 0
 *
 * For A = [[a, b], [c, d]], characteristic polynomial is:
 * λ² - (a+d)λ + (ad-bc) = 0
 */
pair<double, double> computeEigenvalues(const Matrix2x2& A) {
    double a = A[0][0], b = A[0][1];
    double c = A[1][0], d = A[1][1];

    // 跡 (trace) = a + d
    double trace = a + d;

    // 行列式 (determinant) = ad - bc
    double det = a * d - b * c;

    // 判別式 (discriminant)
    double discriminant = trace * trace - 4 * det;

    if (discriminant < 0) {
        cerr << "錯誤：此矩陣有複數特徵值 (Error: Complex eigenvalues)" << endl;
        exit(1);
    }

    double sqrtDisc = sqrt(discriminant);

    // 兩個特徵值 (Two eigenvalues)
    double lambda1 = (trace + sqrtDisc) / 2;
    double lambda2 = (trace - sqrtDisc) / 2;

    return {lambda1, lambda2};
}

/**
 * 計算對應某特徵值的特徵向量 (Compute eigenvector for given eigenvalue)
 *
 * 解 (A - λI)x = 0
 */
Vector2 computeEigenvector(const Matrix2x2& A, double lambda) {
    double a = A[0][0], b = A[0][1];
    double c = A[1][0], d = A[1][1];

    Vector2 vec;

    // 解 null space
    // If b ≠ 0, then x = [b, λ-a] is a solution
    // If b = 0 but c ≠ 0, then x = [λ-d, c] is a solution

    const double EPS = 1e-10;

    if (abs(b) > EPS) {
        vec = {b, lambda - a};
    } else if (abs(c) > EPS) {
        vec = {lambda - d, c};
    } else {
        // 對角矩陣的情況 (Diagonal matrix case)
        if (abs(a - lambda) < EPS) {
            vec = {1.0, 0.0};
        } else {
            vec = {0.0, 1.0};
        }
    }

    // 正規化 (Normalize)
    return normalizeVector(vec);
}

/**
 * 計算 2x2 矩陣的反矩陣 (Compute inverse of 2x2 matrix)
 *
 * 對於 A = [[a, b], [c, d]]
 * A^(-1) = (1/det) * [[d, -b], [-c, a]]
 */
Matrix2x2 matrixInverse(const Matrix2x2& A) {
    double a = A[0][0], b = A[0][1];
    double c = A[1][0], d = A[1][1];

    double det = a * d - b * c;

    if (abs(det) < 1e-10) {
        cerr << "錯誤：矩陣不可逆 (Error: Singular matrix)" << endl;
        exit(1);
    }

    return {{
        {d / det, -b / det},
        {-c / det, a / det}
    }};
}

/**
 * 冪次法示範 (Power Method Demonstration)
 *
 * 反覆計算 v_(k+1) = A * v_k 並正規化，
 * 觀察向量逐漸收斂到主特徵向量
 *
 * Repeatedly compute v_(k+1) = A * v_k and normalize,
 * observe convergence to dominant eigenvector
 */
void powerMethod(const Matrix2x2& A, Vector2 initialVec, int iterations) {
    cout << string(60, '=') << endl;
    cout << "冪次法示範 (Power Method Demonstration)" << endl;
    cout << string(60, '=') << endl;
    printVector("初始向量 (Initial vector)", initialVec);
    cout << endl;

    Vector2 v = normalizeVector(initialVec);

    cout << fixed << setprecision(5);

    for (int k = 0; k < iterations; k++) {
        // 計算 A * v
        Vector2 vNew = matrixVectorMultiply(A, v);

        // Rayleigh 商估計特徵值 (Rayleigh quotient)
        double numerator = v[0] * vNew[0] + v[1] * vNew[1];
        double denominator = v[0] * v[0] + v[1] * v[1];
        double rayleigh = numerator / denominator;

        // 正規化 (Normalize)
        v = normalizeVector(vNew);

        // 印出結果 (Print result)
        if (k < 5 || k % 5 == 4) {
            cout << "迭代 " << setw(2) << (k + 1) << ": v = ["
                 << setw(8) << v[0] << ", " << setw(8) << v[1] << "]  "
                 << "估計 λ = " << setw(8) << rayleigh << endl;
        }
    }

    cout << endl;
    printVector("收斂結果 (Converged result)", v);
}

int main() {
    cout << string(60, '=') << endl;
    cout << "對角化示範 - 手刻版本" << endl;
    cout << "Diagonalization Demo - Manual Implementation" << endl;
    cout << string(60, '=') << endl;
    cout << endl;

    // ========================================
    // 範例矩陣 (Example matrix)
    // ========================================
    // 定義一個 2x2 對稱矩陣（保證可對角化且有實特徵值）
    // Define a 2x2 symmetric matrix (guaranteed diagonalizable with real eigenvalues)
    Matrix2x2 A = {{
        {2.0, 1.0},
        {1.0, 2.0}
    }};

    printMatrix("A（原始矩陣 / Original matrix）", A);

    // ========================================
    // 步驟 1：計算特徵值 (Step 1: Compute eigenvalues)
    // ========================================
    cout << string(40, '-') << endl;
    cout << "步驟 1：計算特徵值 (Compute Eigenvalues)" << endl;
    cout << string(40, '-') << endl;

    auto [lambda1, lambda2] = computeEigenvalues(A);
    cout << fixed << setprecision(4);
    cout << "λ₁ = " << lambda1 << endl;
    cout << "λ₂ = " << lambda2 << endl;
    cout << endl;

    // ========================================
    // 步驟 2：計算特徵向量 (Step 2: Compute eigenvectors)
    // ========================================
    cout << string(40, '-') << endl;
    cout << "步驟 2：計算特徵向量 (Compute Eigenvectors)" << endl;
    cout << string(40, '-') << endl;

    Vector2 v1 = computeEigenvector(A, lambda1);
    Vector2 v2 = computeEigenvector(A, lambda2);

    printVector("v₁ (對應 λ₁)", v1);
    printVector("v₂ (對應 λ₂)", v2);
    cout << endl;

    // ========================================
    // 步驟 3：建構 P 和 D (Step 3: Construct P and D)
    // ========================================
    cout << string(40, '-') << endl;
    cout << "步驟 3：建構 P 和 D (Construct P and D)" << endl;
    cout << string(40, '-') << endl;

    // P = [v1 | v2]，特徵向量作為行 (eigenvectors as columns)
    Matrix2x2 P = {{
        {v1[0], v2[0]},
        {v1[1], v2[1]}
    }};

    // D = diag(λ₁, λ₂)
    Matrix2x2 D = {{
        {lambda1, 0.0},
        {0.0, lambda2}
    }};

    printMatrix("P（特徵向量矩陣 / Eigenvector matrix）", P);
    printMatrix("D（對角矩陣 / Diagonal matrix）", D);

    // ========================================
    // 步驟 4：驗證 A = P * D * P^(-1)
    // Step 4: Verify A = P * D * P^(-1)
    // ========================================
    cout << string(40, '-') << endl;
    cout << "步驟 4：驗證 A = P·D·P⁻¹ (Verify Diagonalization)" << endl;
    cout << string(40, '-') << endl;

    Matrix2x2 P_inv = matrixInverse(P);
    printMatrix("P⁻¹", P_inv);

    // 計算 P * D * P^(-1) (Compute P * D * P^(-1))
    Matrix2x2 PD = matrixMultiply(P, D);
    Matrix2x2 reconstructedA = matrixMultiply(PD, P_inv);

    printMatrix("P·D·P⁻¹（重建的 A / Reconstructed A）", reconstructedA);

    // 計算誤差 (Compute error)
    double error = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            error += abs(A[i][j] - reconstructedA[i][j]);
        }
    }
    cout << "重建誤差 (Reconstruction error): " << scientific << setprecision(10)
         << error << endl;
    cout << endl;

    // ========================================
    // 步驟 5：冪次法示範 (Step 5: Power Method Demo)
    // ========================================
    Vector2 initialVector = {1.0, 0.0};
    powerMethod(A, initialVector, 15);

    cout << endl;
    cout << string(60, '=') << endl;
    cout << "觀察：向量收斂到 [0.707, 0.707]，即 v₁ 的方向" << endl;
    cout << "Observation: Vector converges to [0.707, 0.707], direction of v₁" << endl;
    cout << fixed << setprecision(2);
    cout << "這是主特徵向量，對應最大特徵值 λ₁ = " << lambda1 << endl;
    cout << "This is the dominant eigenvector for largest eigenvalue λ₁ = " << lambda1 << endl;
    cout << string(60, '=') << endl;

    return 0;
}
