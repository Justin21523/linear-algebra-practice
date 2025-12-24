/**
 * 投影 (Projections)
 *
 * 本程式示範：
 * 1. 投影到直線
 * 2. 投影矩陣及其性質
 * 3. 投影到子空間
 * 4. 誤差向量的正交性驗證
 *
 * 編譯：g++ -std=c++17 -O2 projection.cpp -o projection
 * 執行：./projection
 */

#include <iostream>  // EN: Include a header dependency: #include <iostream>.
#include <vector>  // EN: Include a header dependency: #include <vector>.
#include <cmath>  // EN: Include a header dependency: #include <cmath>.
#include <iomanip>  // EN: Include a header dependency: #include <iomanip>.
#include <string>  // EN: Include a header dependency: #include <string>.

using Vector = std::vector<double>;  // EN: Execute a statement: using Vector = std::vector<double>;.
using Matrix = std::vector<std::vector<double>>;  // EN: Execute a statement: using Matrix = std::vector<std::vector<double>>;.

const double EPSILON = 1e-10;  // EN: Execute a statement: const double EPSILON = 1e-10;.

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

Vector scalarMultiply(double c, const Vector& x) {  // EN: Execute line: Vector scalarMultiply(double c, const Vector& x) {.
    Vector result(x.size());  // EN: Execute a statement: Vector result(x.size());.
    for (size_t i = 0; i < x.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < x.size(); ++i) {.
        result[i] = c * x[i];  // EN: Execute a statement: result[i] = c * x[i];.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

Vector vectorSubtract(const Vector& x, const Vector& y) {  // EN: Execute line: Vector vectorSubtract(const Vector& x, const Vector& y) {.
    Vector result(x.size());  // EN: Execute a statement: Vector result(x.size());.
    for (size_t i = 0; i < x.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < x.size(); ++i) {.
        result[i] = x[i] - y[i];  // EN: Execute a statement: result[i] = x[i] - y[i];.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

Matrix outerProduct(const Vector& x, const Vector& y) {  // EN: Execute line: Matrix outerProduct(const Vector& x, const Vector& y) {.
    Matrix result(x.size(), Vector(y.size()));  // EN: Execute a statement: Matrix result(x.size(), Vector(y.size()));.
    for (size_t i = 0; i < x.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < x.size(); ++i) {.
        for (size_t j = 0; j < y.size(); ++j) {  // EN: Loop control flow: for (size_t j = 0; j < y.size(); ++j) {.
            result[i][j] = x[i] * y[j];  // EN: Execute a statement: result[i][j] = x[i] * y[j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

Matrix matrixScalarMultiply(double c, const Matrix& A) {  // EN: Execute line: Matrix matrixScalarMultiply(double c, const Matrix& A) {.
    Matrix result = A;  // EN: Execute a statement: Matrix result = A;.
    for (auto& row : result) {  // EN: Loop control flow: for (auto& row : result) {.
        for (auto& val : row) {  // EN: Loop control flow: for (auto& val : row) {.
            val *= c;  // EN: Execute a statement: val *= c;.
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

// ========================================
// 投影函數
// ========================================

struct ProjectionResult {  // EN: Execute line: struct ProjectionResult {.
    double xHat;  // EN: Execute a statement: double xHat;.
    Vector projection;  // EN: Execute a statement: Vector projection;.
    Vector error;  // EN: Execute a statement: Vector error;.
    double errorNorm;  // EN: Execute a statement: double errorNorm;.
};  // EN: Structure delimiter for a block or scope.

/**
 * 投影到直線
 * p = (aᵀb / aᵀa) * a
 */
ProjectionResult projectOntoLine(const Vector& b, const Vector& a) {  // EN: Execute line: ProjectionResult projectOntoLine(const Vector& b, const Vector& a) {.
    double aTb = dotProduct(a, b);  // EN: Execute a statement: double aTb = dotProduct(a, b);.
    double aTa = dotProduct(a, a);  // EN: Execute a statement: double aTa = dotProduct(a, a);.

    double xHat = aTb / aTa;  // EN: Execute a statement: double xHat = aTb / aTa;.
    Vector p = scalarMultiply(xHat, a);  // EN: Execute a statement: Vector p = scalarMultiply(xHat, a);.
    Vector e = vectorSubtract(b, p);  // EN: Execute a statement: Vector e = vectorSubtract(b, p);.

    return {xHat, p, e, vectorNorm(e)};  // EN: Return from the current function: return {xHat, p, e, vectorNorm(e)};.
}  // EN: Structure delimiter for a block or scope.

/**
 * 投影到直線的投影矩陣
 * P = aaᵀ / (aᵀa)
 */
Matrix projectionMatrixLine(const Vector& a) {  // EN: Execute line: Matrix projectionMatrixLine(const Vector& a) {.
    double aTa = dotProduct(a, a);  // EN: Execute a statement: double aTa = dotProduct(a, a);.
    Matrix aaT = outerProduct(a, a);  // EN: Execute a statement: Matrix aaT = outerProduct(a, a);.
    return matrixScalarMultiply(1.0 / aTa, aaT);  // EN: Return from the current function: return matrixScalarMultiply(1.0 / aTa, aaT);.
}  // EN: Structure delimiter for a block or scope.

/**
 * 驗證投影矩陣的性質
 */
void verifyProjectionMatrix(const Matrix& P, const std::string& name = "P") {  // EN: Execute line: void verifyProjectionMatrix(const Matrix& P, const std::string& name = ….
    size_t n = P.size();  // EN: Execute a statement: size_t n = P.size();.

    std::cout << "\n驗證 " << name << " 的性質：\n";  // EN: Execute a statement: std::cout << "\n驗證 " << name << " 的性質：\n";.

    // 對稱性
    bool isSymmetric = true;  // EN: Execute a statement: bool isSymmetric = true;.
    for (size_t i = 0; i < n && isSymmetric; ++i) {  // EN: Loop control flow: for (size_t i = 0; i < n && isSymmetric; ++i) {.
        for (size_t j = 0; j < n && isSymmetric; ++j) {  // EN: Loop control flow: for (size_t j = 0; j < n && isSymmetric; ++j) {.
            if (std::abs(P[i][j] - P[j][i]) > EPSILON) {  // EN: Conditional control flow: if (std::abs(P[i][j] - P[j][i]) > EPSILON) {.
                isSymmetric = false;  // EN: Execute a statement: isSymmetric = false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    std::cout << "  對稱性 (" << name << "ᵀ = " << name << ")：" << (isSymmetric ? "true" : "false") << "\n";  // EN: Execute a statement: std::cout << " 對稱性 (" << name << "ᵀ = " << name << ")：" << (isSymmetric….

    // 冪等性
    Matrix P2 = matrixMultiply(P, P);  // EN: Execute a statement: Matrix P2 = matrixMultiply(P, P);.
    bool isIdempotent = true;  // EN: Execute a statement: bool isIdempotent = true;.
    for (size_t i = 0; i < n && isIdempotent; ++i) {  // EN: Loop control flow: for (size_t i = 0; i < n && isIdempotent; ++i) {.
        for (size_t j = 0; j < n && isIdempotent; ++j) {  // EN: Loop control flow: for (size_t j = 0; j < n && isIdempotent; ++j) {.
            if (std::abs(P[i][j] - P2[i][j]) > EPSILON) {  // EN: Conditional control flow: if (std::abs(P[i][j] - P2[i][j]) > EPSILON) {.
                isIdempotent = false;  // EN: Execute a statement: isIdempotent = false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    std::cout << "  冪等性 (" << name << "² = " << name << ")：" << (isIdempotent ? "true" : "false") << "\n";  // EN: Execute a statement: std::cout << " 冪等性 (" << name << "² = " << name << ")：" << (isIdempoten….
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 主程式
// ========================================

int main() {  // EN: Execute line: int main() {.
    std::cout << std::fixed << std::setprecision(4);  // EN: Execute a statement: std::cout << std::fixed << std::setprecision(4);.

    printSeparator("投影示範 (C++)\nProjection Demo");  // EN: Execute a statement: printSeparator("投影示範 (C++)\nProjection Demo");.

    // 1. 投影到直線
    printSeparator("1. 投影到直線");  // EN: Execute a statement: printSeparator("1. 投影到直線");.

    Vector a = {1.0, 1.0};  // EN: Execute a statement: Vector a = {1.0, 1.0};.
    Vector b = {2.0, 0.0};  // EN: Execute a statement: Vector b = {2.0, 0.0};.

    printVector("方向 a", a);  // EN: Execute a statement: printVector("方向 a", a);.
    printVector("向量 b", b);  // EN: Execute a statement: printVector("向量 b", b);.

    auto result = projectOntoLine(b, a);  // EN: Execute a statement: auto result = projectOntoLine(b, a);.

    std::cout << "\n投影係數 x̂ = (aᵀb)/(aᵀa) = " << result.xHat << "\n";  // EN: Execute a statement: std::cout << "\n投影係數 x̂ = (aᵀb)/(aᵀa) = " << result.xHat << "\n";.
    printVector("投影 p = x̂a", result.projection);  // EN: Execute a statement: printVector("投影 p = x̂a", result.projection);.
    printVector("誤差 e = b - p", result.error);  // EN: Execute a statement: printVector("誤差 e = b - p", result.error);.

    // 驗證正交性
    double eDotA = dotProduct(result.error, a);  // EN: Execute a statement: double eDotA = dotProduct(result.error, a);.
    std::cout << "\n驗證 e ⊥ a：e · a = " << eDotA << "\n";  // EN: Execute a statement: std::cout << "\n驗證 e ⊥ a：e · a = " << eDotA << "\n";.
    std::cout << "正交？ " << (std::abs(eDotA) < EPSILON ? "true" : "false") << "\n";  // EN: Execute a statement: std::cout << "正交？ " << (std::abs(eDotA) < EPSILON ? "true" : "false") <….

    // 2. 投影矩陣
    printSeparator("2. 投影矩陣（到直線）");  // EN: Execute a statement: printSeparator("2. 投影矩陣（到直線）");.

    Vector a2 = {1.0, 2.0};  // EN: Execute a statement: Vector a2 = {1.0, 2.0};.
    printVector("方向 a", a2);  // EN: Execute a statement: printVector("方向 a", a2);.

    Matrix P = projectionMatrixLine(a2);  // EN: Execute a statement: Matrix P = projectionMatrixLine(a2);.
    printMatrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P);  // EN: Execute a statement: printMatrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P);.

    verifyProjectionMatrix(P);  // EN: Execute a statement: verifyProjectionMatrix(P);.

    // 用投影矩陣計算投影
    Vector b2 = {3.0, 4.0};  // EN: Execute a statement: Vector b2 = {3.0, 4.0};.
    printVector("\n向量 b", b2);  // EN: Execute a statement: printVector("\n向量 b", b2);.

    Vector p = matrixVectorMultiply(P, b2);  // EN: Execute a statement: Vector p = matrixVectorMultiply(P, b2);.
    printVector("投影 p = Pb", p);  // EN: Execute a statement: printVector("投影 p = Pb", p);.

    // 3. 多個向量的投影
    printSeparator("3. 批次投影");  // EN: Execute a statement: printSeparator("3. 批次投影");.

    std::vector<Vector> vectors = {{1.0, 0.0}, {0.0, 1.0}, {2.0, 2.0}, {3.0, -1.0}};  // EN: Execute a statement: std::vector<Vector> vectors = {{1.0, 0.0}, {0.0, 1.0}, {2.0, 2.0}, {3.0….

    std::cout << "方向 a = [1, 2]\n";  // EN: Execute a statement: std::cout << "方向 a = [1, 2]\n";.
    std::cout << "\n各向量投影結果：\n";  // EN: Execute a statement: std::cout << "\n各向量投影結果：\n";.

    for (const auto& v : vectors) {  // EN: Loop control flow: for (const auto& v : vectors) {.
        auto proj = projectOntoLine(v, a2);  // EN: Execute a statement: auto proj = projectOntoLine(v, a2);.
        std::cout << "  [" << v[0] << ", " << v[1] << "] -> ["  // EN: Execute line: std::cout << " [" << v[0] << ", " << v[1] << "] -> [".
                  << proj.projection[0] << ", " << proj.projection[1] << "]\n";  // EN: Execute a statement: << proj.projection[0] << ", " << proj.projection[1] << "]\n";.
    }  // EN: Structure delimiter for a block or scope.

    // 總結
    printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
    std::cout << R"(  // EN: Execute line: std::cout << R"(.
投影公式：  // EN: Execute line: 投影公式：.

1. 投影到直線：  // EN: Execute line: 1. 投影到直線：.
   p = (aᵀb / aᵀa) a  // EN: Execute line: p = (aᵀb / aᵀa) a.
   P = aaᵀ / (aᵀa)  // EN: Execute line: P = aaᵀ / (aᵀa).

2. 投影到子空間：  // EN: Execute line: 2. 投影到子空間：.
   p = A(AᵀA)⁻¹Aᵀb  // EN: Execute line: p = A(AᵀA)⁻¹Aᵀb.
   P = A(AᵀA)⁻¹Aᵀ  // EN: Execute line: P = A(AᵀA)⁻¹Aᵀ.

3. 投影矩陣性質：  // EN: Execute line: 3. 投影矩陣性質：.
   Pᵀ = P（對稱）  // EN: Execute line: Pᵀ = P（對稱）.
   P² = P（冪等）  // EN: Execute line: P² = P（冪等）.
)" << "\n";  // EN: Execute a statement: )" << "\n";.

    std::cout << std::string(60, '=') << "\n";  // EN: Execute a statement: std::cout << std::string(60, '=') << "\n";.
    std::cout << "示範完成！\n";  // EN: Execute a statement: std::cout << "示範完成！\n";.
    std::cout << std::string(60, '=') << "\n";  // EN: Execute a statement: std::cout << std::string(60, '=') << "\n";.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
