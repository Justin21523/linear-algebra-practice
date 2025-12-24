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

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

// ========================================
// 輔助函數
// ========================================

void printSeparator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void printVector(const std::string& name, const Vector& v) {
    std::cout << name << " = [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

void printMatrix(const std::string& name, const Matrix& M) {
    std::cout << name << " =\n";
    for (const auto& row : M) {
        std::cout << "  [";
        for (size_t j = 0; j < row.size(); ++j) {
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << row[j];
            if (j < row.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

// ========================================
// 基本運算
// ========================================

double dotProduct(const Vector& x, const Vector& y) {
    double result = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        result += x[i] * y[i];
    }
    return result;
}

double vectorNorm(const Vector& x) {
    return std::sqrt(dotProduct(x, x));
}

Vector vectorSubtract(const Vector& x, const Vector& y) {
    Vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] - y[i];
    }
    return result;
}

Matrix transpose(const Matrix& A) {
    size_t m = A.size(), n = A[0].size();
    Matrix result(n, Vector(m));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

Matrix matrixMultiply(const Matrix& A, const Matrix& B) {
    size_t m = A.size(), k = B.size(), n = B[0].size();
    Matrix result(m, Vector(n, 0.0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t p = 0; p < k; ++p) {
                result[i][j] += A[i][p] * B[p][j];
            }
        }
    }
    return result;
}

Vector matrixVectorMultiply(const Matrix& A, const Vector& x) {
    Vector result(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < x.size(); ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

Vector solve2x2(const Matrix& A, const Vector& b) {
    double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    return {
        (A[1][1] * b[0] - A[0][1] * b[1]) / det,
        (-A[1][0] * b[0] + A[0][0] * b[1]) / det
    };
}

// ========================================
// 最小平方求解
// ========================================

struct LeastSquaresResult {
    Vector coefficients;
    Vector fitted;
    Vector residual;
    double residualNorm;
    double rSquared;
};

Matrix createDesignMatrixLinear(const Vector& t) {
    Matrix A(t.size(), Vector(2));
    for (size_t i = 0; i < t.size(); ++i) {
        A[i][0] = 1.0;
        A[i][1] = t[i];
    }
    return A;
}

LeastSquaresResult leastSquaresSolve(const Matrix& A, const Vector& b) {
    // AᵀA
    Matrix AT = transpose(A);
    Matrix ATA = matrixMultiply(AT, A);

    // Aᵀb
    Vector ATb = matrixVectorMultiply(AT, b);

    // 解 AᵀA x̂ = Aᵀb（假設 2×2）
    Vector xHat = solve2x2(ATA, ATb);

    // 擬合值和殘差
    Vector yHat = matrixVectorMultiply(A, xHat);
    Vector residual = vectorSubtract(b, yHat);
    double residualNorm = vectorNorm(residual);

    // R²
    double bMean = 0.0;
    for (double bi : b) bMean += bi;
    bMean /= b.size();

    double tss = 0.0, rss = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        tss += (b[i] - bMean) * (b[i] - bMean);
        rss += residual[i] * residual[i];
    }
    double rSquared = (tss > 0) ? (1.0 - rss / tss) : 0.0;

    return {xHat, yHat, residual, residualNorm, rSquared};
}

// ========================================
// 主程式
// ========================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    printSeparator("最小平方回歸示範 (C++)\nLeast Squares Regression Demo");

    // 1. 簡單線性迴歸
    printSeparator("1. 簡單線性迴歸：y = C + Dt");

    Vector t = {0.0, 1.0, 2.0};
    Vector b = {1.0, 3.0, 4.0};

    std::cout << "數據點：\n";
    for (size_t i = 0; i < t.size(); ++i) {
        std::cout << "  t = " << t[i] << ", b = " << b[i] << "\n";
    }

    Matrix A = createDesignMatrixLinear(t);
    printMatrix("\n設計矩陣 A [1, t]", A);
    printVector("觀測值 b", b);

    auto result = leastSquaresSolve(A, b);

    std::cout << "\n【解】\n";
    std::cout << "C（截距）= " << result.coefficients[0] << "\n";
    std::cout << "D（斜率）= " << result.coefficients[1] << "\n";
    std::cout << "\n最佳直線：y = " << result.coefficients[0]
              << " + " << result.coefficients[1] << "t\n";

    printVector("\n擬合值 ŷ", result.fitted);
    printVector("殘差 e", result.residual);
    std::cout << "殘差範數 ‖e‖ = " << result.residualNorm << "\n";
    std::cout << "R² = " << result.rSquared << "\n";

    // 2. 更多數據
    printSeparator("2. 更多數據點");

    Vector t2 = {0.0, 1.0, 2.0, 3.0, 4.0};
    Vector b2 = {1.0, 2.5, 3.5, 5.0, 6.5};

    std::cout << "數據點：\n";
    for (size_t i = 0; i < t2.size(); ++i) {
        std::cout << "  (" << t2[i] << ", " << b2[i] << ")\n";
    }

    Matrix A2 = createDesignMatrixLinear(t2);
    auto result2 = leastSquaresSolve(A2, b2);

    std::cout << "\n最佳直線：y = " << result2.coefficients[0]
              << " + " << result2.coefficients[1] << "t\n";
    std::cout << "R² = " << result2.rSquared << "\n";

    // 總結
    printSeparator("總結");
    std::cout << R"(
最小平方法核心公式：

1. 正規方程：AᵀA x̂ = Aᵀb

2. 解：x̂ = (AᵀA)⁻¹Aᵀb

3. 幾何意義：Ax̂ 是 b 在 C(A) 上的投影

4. R² = 1 - RSS/TSS（越接近 1 越好）
)" << "\n";

    std::cout << std::string(60, '=') << "\n";
    std::cout << "示範完成！\n";
    std::cout << std::string(60, '=') << "\n";

    return 0;
}
