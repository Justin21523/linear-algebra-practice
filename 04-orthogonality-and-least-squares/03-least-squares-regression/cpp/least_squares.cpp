/**
 * 最小平方回歸 (Least Squares Regression)
 *
 * 本程式示範：
 * 1. 正規方程求解最小平方問題
 * 2. 簡單線性迴歸
 * 3. 殘差分析
 *
 * 編譯：g++ -std=c++17 -O2 least_squares.cpp -o least_squares
 * 執行：./least_squares
 */

#include <iostream>  // EN: Include a header dependency: #include <iostream>.
#include <vector>  // EN: Include a header dependency: #include <vector>.
#include <cmath>  // EN: Include a header dependency: #include <cmath>.
#include <iomanip>  // EN: Include a header dependency: #include <iomanip>.
#include <string>  // EN: Include a header dependency: #include <string>.

using Vector = std::vector<double>;  // EN: Execute a statement: using Vector = std::vector<double>;.
using Matrix = std::vector<std::vector<double>>;  // EN: Execute a statement: using Matrix = std::vector<std::vector<double>>;.

// ========================================
// 輔助函數
// ========================================

void printSeparator(const std::string& title) {  // EN: Execute line: void printSeparator(const std::string& title) {.
    std::cout << "\n" << std::string(60, '=') << "\n";  // EN: Execute a statement: std::cout << "\n" << std::string(60, '=') << "\n";.
    std::cout << title << "\n";  // EN: Execute a statement: std::cout << title << "\n";.
    std::cout << std::string(60, '=') << "\n";  // EN: Execute a statement: std::cout << std::string(60, '=') << "\n";.
}  // EN: Structure delimiter for a block or scope.

void printVector(const std::string& name, const Vector& v) {  // EN: Execute line: void printVector(const std::string& name, const Vector& v) {.
    std::cout << name << " = [";  // EN: Execute a statement: std::cout << name << " = [";.
    for (size_t i = 0; i < v.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < v.size(); ++i) {.
        std::cout << std::fixed << std::setprecision(4) << v[i];  // EN: Execute a statement: std::cout << std::fixed << std::setprecision(4) << v[i];.
        if (i < v.size() - 1) std::cout << ", ";  // EN: Conditional control flow: if (i < v.size() - 1) std::cout << ", ";.
    }  // EN: Structure delimiter for a block or scope.
    std::cout << "]\n";  // EN: Execute a statement: std::cout << "]\n";.
}  // EN: Structure delimiter for a block or scope.

void printMatrix(const std::string& name, const Matrix& M) {  // EN: Execute line: void printMatrix(const std::string& name, const Matrix& M) {.
    std::cout << name << " =\n";  // EN: Execute a statement: std::cout << name << " =\n";.
    for (const auto& row : M) {  // EN: Loop control flow: for (const auto& row : M) {.
        std::cout << "  [";  // EN: Execute a statement: std::cout << " [";.
        for (size_t j = 0; j < row.size(); ++j) {  // EN: Loop control flow: for (size_t j = 0; j < row.size(); ++j) {.
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << row[j];  // EN: Execute a statement: std::cout << std::fixed << std::setprecision(4) << std::setw(8) << row[….
            if (j < row.size() - 1) std::cout << ", ";  // EN: Conditional control flow: if (j < row.size() - 1) std::cout << ", ";.
        }  // EN: Structure delimiter for a block or scope.
        std::cout << "]\n";  // EN: Execute a statement: std::cout << "]\n";.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 基本運算
// ========================================

double dotProduct(const Vector& x, const Vector& y) {  // EN: Execute line: double dotProduct(const Vector& x, const Vector& y) {.
    double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
    for (size_t i = 0; i < x.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < x.size(); ++i) {.
        result += x[i] * y[i];  // EN: Execute a statement: result += x[i] * y[i];.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

double vectorNorm(const Vector& x) {  // EN: Execute line: double vectorNorm(const Vector& x) {.
    return std::sqrt(dotProduct(x, x));  // EN: Return from the current function: return std::sqrt(dotProduct(x, x));.
}  // EN: Structure delimiter for a block or scope.

Vector vectorSubtract(const Vector& x, const Vector& y) {  // EN: Execute line: Vector vectorSubtract(const Vector& x, const Vector& y) {.
    Vector result(x.size());  // EN: Execute a statement: Vector result(x.size());.
    for (size_t i = 0; i < x.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < x.size(); ++i) {.
        result[i] = x[i] - y[i];  // EN: Execute a statement: result[i] = x[i] - y[i];.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

Matrix transpose(const Matrix& A) {  // EN: Execute line: Matrix transpose(const Matrix& A) {.
    size_t m = A.size(), n = A[0].size();  // EN: Execute a statement: size_t m = A.size(), n = A[0].size();.
    Matrix result(n, Vector(m));  // EN: Execute a statement: Matrix result(n, Vector(m));.
    for (size_t i = 0; i < m; ++i) {  // EN: Loop control flow: for (size_t i = 0; i < m; ++i) {.
        for (size_t j = 0; j < n; ++j) {  // EN: Loop control flow: for (size_t j = 0; j < n; ++j) {.
            result[j][i] = A[i][j];  // EN: Execute a statement: result[j][i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

Matrix matrixMultiply(const Matrix& A, const Matrix& B) {  // EN: Execute line: Matrix matrixMultiply(const Matrix& A, const Matrix& B) {.
    size_t m = A.size(), k = B.size(), n = B[0].size();  // EN: Execute a statement: size_t m = A.size(), k = B.size(), n = B[0].size();.
    Matrix result(m, Vector(n, 0.0));  // EN: Execute a statement: Matrix result(m, Vector(n, 0.0));.
    for (size_t i = 0; i < m; ++i) {  // EN: Loop control flow: for (size_t i = 0; i < m; ++i) {.
        for (size_t j = 0; j < n; ++j) {  // EN: Loop control flow: for (size_t j = 0; j < n; ++j) {.
            for (size_t p = 0; p < k; ++p) {  // EN: Loop control flow: for (size_t p = 0; p < k; ++p) {.
                result[i][j] += A[i][p] * B[p][j];  // EN: Execute a statement: result[i][j] += A[i][p] * B[p][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

Vector matrixVectorMultiply(const Matrix& A, const Vector& x) {  // EN: Execute line: Vector matrixVectorMultiply(const Matrix& A, const Vector& x) {.
    Vector result(A.size(), 0.0);  // EN: Execute a statement: Vector result(A.size(), 0.0);.
    for (size_t i = 0; i < A.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < A.size(); ++i) {.
        for (size_t j = 0; j < x.size(); ++j) {  // EN: Loop control flow: for (size_t j = 0; j < x.size(); ++j) {.
            result[i] += A[i][j] * x[j];  // EN: Execute a statement: result[i] += A[i][j] * x[j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

Vector solve2x2(const Matrix& A, const Vector& b) {  // EN: Execute line: Vector solve2x2(const Matrix& A, const Vector& b) {.
    double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Execute a statement: double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];.
    return {  // EN: Return from the current function: return {.
        (A[1][1] * b[0] - A[0][1] * b[1]) / det,  // EN: Execute line: (A[1][1] * b[0] - A[0][1] * b[1]) / det,.
        (-A[1][0] * b[0] + A[0][0] * b[1]) / det  // EN: Execute line: (-A[1][0] * b[0] + A[0][0] * b[1]) / det.
    };  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 最小平方求解
// ========================================

struct LeastSquaresResult {  // EN: Execute line: struct LeastSquaresResult {.
    Vector coefficients;  // EN: Execute a statement: Vector coefficients;.
    Vector fitted;  // EN: Execute a statement: Vector fitted;.
    Vector residual;  // EN: Execute a statement: Vector residual;.
    double residualNorm;  // EN: Execute a statement: double residualNorm;.
    double rSquared;  // EN: Execute a statement: double rSquared;.
};  // EN: Structure delimiter for a block or scope.

Matrix createDesignMatrixLinear(const Vector& t) {  // EN: Execute line: Matrix createDesignMatrixLinear(const Vector& t) {.
    Matrix A(t.size(), Vector(2));  // EN: Execute a statement: Matrix A(t.size(), Vector(2));.
    for (size_t i = 0; i < t.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < t.size(); ++i) {.
        A[i][0] = 1.0;  // EN: Execute a statement: A[i][0] = 1.0;.
        A[i][1] = t[i];  // EN: Execute a statement: A[i][1] = t[i];.
    }  // EN: Structure delimiter for a block or scope.
    return A;  // EN: Return from the current function: return A;.
}  // EN: Structure delimiter for a block or scope.

LeastSquaresResult leastSquaresSolve(const Matrix& A, const Vector& b) {  // EN: Execute line: LeastSquaresResult leastSquaresSolve(const Matrix& A, const Vector& b) {.
    // AᵀA
    Matrix AT = transpose(A);  // EN: Execute a statement: Matrix AT = transpose(A);.
    Matrix ATA = matrixMultiply(AT, A);  // EN: Execute a statement: Matrix ATA = matrixMultiply(AT, A);.

    // Aᵀb
    Vector ATb = matrixVectorMultiply(AT, b);  // EN: Execute a statement: Vector ATb = matrixVectorMultiply(AT, b);.

    // 解 AᵀA x̂ = Aᵀb（假設 2×2）
    Vector xHat = solve2x2(ATA, ATb);  // EN: Execute a statement: Vector xHat = solve2x2(ATA, ATb);.

    // 擬合值和殘差
    Vector yHat = matrixVectorMultiply(A, xHat);  // EN: Execute a statement: Vector yHat = matrixVectorMultiply(A, xHat);.
    Vector residual = vectorSubtract(b, yHat);  // EN: Execute a statement: Vector residual = vectorSubtract(b, yHat);.
    double residualNorm = vectorNorm(residual);  // EN: Execute a statement: double residualNorm = vectorNorm(residual);.

    // R²
    double bMean = 0.0;  // EN: Execute a statement: double bMean = 0.0;.
    for (double bi : b) bMean += bi;  // EN: Loop control flow: for (double bi : b) bMean += bi;.
    bMean /= b.size();  // EN: Execute a statement: bMean /= b.size();.

    double tss = 0.0, rss = 0.0;  // EN: Execute a statement: double tss = 0.0, rss = 0.0;.
    for (size_t i = 0; i < b.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < b.size(); ++i) {.
        tss += (b[i] - bMean) * (b[i] - bMean);  // EN: Execute a statement: tss += (b[i] - bMean) * (b[i] - bMean);.
        rss += residual[i] * residual[i];  // EN: Execute a statement: rss += residual[i] * residual[i];.
    }  // EN: Structure delimiter for a block or scope.
    double rSquared = (tss > 0) ? (1.0 - rss / tss) : 0.0;  // EN: Execute a statement: double rSquared = (tss > 0) ? (1.0 - rss / tss) : 0.0;.

    return {xHat, yHat, residual, residualNorm, rSquared};  // EN: Return from the current function: return {xHat, yHat, residual, residualNorm, rSquared};.
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 主程式
// ========================================

int main() {  // EN: Execute line: int main() {.
    std::cout << std::fixed << std::setprecision(4);  // EN: Execute a statement: std::cout << std::fixed << std::setprecision(4);.

    printSeparator("最小平方回歸示範 (C++)\nLeast Squares Regression Demo");  // EN: Execute a statement: printSeparator("最小平方回歸示範 (C++)\nLeast Squares Regression Demo");.

    // 1. 簡單線性迴歸
    printSeparator("1. 簡單線性迴歸：y = C + Dt");  // EN: Execute a statement: printSeparator("1. 簡單線性迴歸：y = C + Dt");.

    Vector t = {0.0, 1.0, 2.0};  // EN: Execute a statement: Vector t = {0.0, 1.0, 2.0};.
    Vector b = {1.0, 3.0, 4.0};  // EN: Execute a statement: Vector b = {1.0, 3.0, 4.0};.

    std::cout << "數據點：\n";  // EN: Execute a statement: std::cout << "數據點：\n";.
    for (size_t i = 0; i < t.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < t.size(); ++i) {.
        std::cout << "  t = " << t[i] << ", b = " << b[i] << "\n";  // EN: Execute a statement: std::cout << " t = " << t[i] << ", b = " << b[i] << "\n";.
    }  // EN: Structure delimiter for a block or scope.

    Matrix A = createDesignMatrixLinear(t);  // EN: Execute a statement: Matrix A = createDesignMatrixLinear(t);.
    printMatrix("\n設計矩陣 A [1, t]", A);  // EN: Execute a statement: printMatrix("\n設計矩陣 A [1, t]", A);.
    printVector("觀測值 b", b);  // EN: Execute a statement: printVector("觀測值 b", b);.

    auto result = leastSquaresSolve(A, b);  // EN: Execute a statement: auto result = leastSquaresSolve(A, b);.

    std::cout << "\n【解】\n";  // EN: Execute a statement: std::cout << "\n【解】\n";.
    std::cout << "C（截距）= " << result.coefficients[0] << "\n";  // EN: Execute a statement: std::cout << "C（截距）= " << result.coefficients[0] << "\n";.
    std::cout << "D（斜率）= " << result.coefficients[1] << "\n";  // EN: Execute a statement: std::cout << "D（斜率）= " << result.coefficients[1] << "\n";.
    std::cout << "\n最佳直線：y = " << result.coefficients[0]  // EN: Execute line: std::cout << "\n最佳直線：y = " << result.coefficients[0].
              << " + " << result.coefficients[1] << "t\n";  // EN: Execute a statement: << " + " << result.coefficients[1] << "t\n";.

    printVector("\n擬合值 ŷ", result.fitted);  // EN: Execute a statement: printVector("\n擬合值 ŷ", result.fitted);.
    printVector("殘差 e", result.residual);  // EN: Execute a statement: printVector("殘差 e", result.residual);.
    std::cout << "殘差範數 ‖e‖ = " << result.residualNorm << "\n";  // EN: Execute a statement: std::cout << "殘差範數 ‖e‖ = " << result.residualNorm << "\n";.
    std::cout << "R² = " << result.rSquared << "\n";  // EN: Execute a statement: std::cout << "R² = " << result.rSquared << "\n";.

    // 2. 更多數據
    printSeparator("2. 更多數據點");  // EN: Execute a statement: printSeparator("2. 更多數據點");.

    Vector t2 = {0.0, 1.0, 2.0, 3.0, 4.0};  // EN: Execute a statement: Vector t2 = {0.0, 1.0, 2.0, 3.0, 4.0};.
    Vector b2 = {1.0, 2.5, 3.5, 5.0, 6.5};  // EN: Execute a statement: Vector b2 = {1.0, 2.5, 3.5, 5.0, 6.5};.

    std::cout << "數據點：\n";  // EN: Execute a statement: std::cout << "數據點：\n";.
    for (size_t i = 0; i < t2.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < t2.size(); ++i) {.
        std::cout << "  (" << t2[i] << ", " << b2[i] << ")\n";  // EN: Execute a statement: std::cout << " (" << t2[i] << ", " << b2[i] << ")\n";.
    }  // EN: Structure delimiter for a block or scope.

    Matrix A2 = createDesignMatrixLinear(t2);  // EN: Execute a statement: Matrix A2 = createDesignMatrixLinear(t2);.
    auto result2 = leastSquaresSolve(A2, b2);  // EN: Execute a statement: auto result2 = leastSquaresSolve(A2, b2);.

    std::cout << "\n最佳直線：y = " << result2.coefficients[0]  // EN: Execute line: std::cout << "\n最佳直線：y = " << result2.coefficients[0].
              << " + " << result2.coefficients[1] << "t\n";  // EN: Execute a statement: << " + " << result2.coefficients[1] << "t\n";.
    std::cout << "R² = " << result2.rSquared << "\n";  // EN: Execute a statement: std::cout << "R² = " << result2.rSquared << "\n";.

    // 總結
    printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
    std::cout << R"(  // EN: Execute line: std::cout << R"(.
最小平方法核心公式：  // EN: Execute line: 最小平方法核心公式：.

1. 正規方程：AᵀA x̂ = Aᵀb  // EN: Execute line: 1. 正規方程：AᵀA x̂ = Aᵀb.

2. 解：x̂ = (AᵀA)⁻¹Aᵀb  // EN: Execute line: 2. 解：x̂ = (AᵀA)⁻¹Aᵀb.

3. 幾何意義：Ax̂ 是 b 在 C(A) 上的投影  // EN: Execute line: 3. 幾何意義：Ax̂ 是 b 在 C(A) 上的投影.

4. R² = 1 - RSS/TSS（越接近 1 越好）  // EN: Execute line: 4. R² = 1 - RSS/TSS（越接近 1 越好）.
)" << "\n";  // EN: Execute a statement: )" << "\n";.

    std::cout << std::string(60, '=') << "\n";  // EN: Execute a statement: std::cout << std::string(60, '=') << "\n";.
    std::cout << "示範完成！\n";  // EN: Execute a statement: std::cout << "示範完成！\n";.
    std::cout << std::string(60, '=') << "\n";  // EN: Execute a statement: std::cout << std::string(60, '=') << "\n";.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
