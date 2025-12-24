/**
 * Gram-Schmidt 正交化 (Gram-Schmidt Process)
 *
 * 編譯：g++ -std=c++17 -O2 gram_schmidt.cpp -o gram_schmidt
 * 執行：./gram_schmidt
 */

#include <iostream>  // EN: Include a header dependency: #include <iostream>.
#include <vector>  // EN: Include a header dependency: #include <vector>.
#include <cmath>  // EN: Include a header dependency: #include <cmath>.
#include <iomanip>  // EN: Include a header dependency: #include <iomanip>.

using Vector = std::vector<double>;  // EN: Execute a statement: using Vector = std::vector<double>;.
using Matrix = std::vector<Vector>;  // EN: Execute a statement: using Matrix = std::vector<Vector>;.

void printSeparator(const std::string& title) {  // EN: Execute line: void printSeparator(const std::string& title) {.
    std::cout << "\n" << std::string(60, '=') << "\n" << title << "\n" << std::string(60, '=') << "\n";  // EN: Execute a statement: std::cout << "\n" << std::string(60, '=') << "\n" << title << "\n" << s….
}  // EN: Structure delimiter for a block or scope.

void printVector(const std::string& name, const Vector& v) {  // EN: Execute line: void printVector(const std::string& name, const Vector& v) {.
    std::cout << name << " = [";  // EN: Execute a statement: std::cout << name << " = [";.
    for (size_t i = 0; i < v.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < v.size(); ++i) {.
        std::cout << std::fixed << std::setprecision(4) << v[i];  // EN: Execute a statement: std::cout << std::fixed << std::setprecision(4) << v[i];.
        if (i < v.size() - 1) std::cout << ", ";  // EN: Conditional control flow: if (i < v.size() - 1) std::cout << ", ";.
    }  // EN: Structure delimiter for a block or scope.
    std::cout << "]\n";  // EN: Execute a statement: std::cout << "]\n";.
}  // EN: Structure delimiter for a block or scope.

double dotProduct(const Vector& x, const Vector& y) {  // EN: Execute line: double dotProduct(const Vector& x, const Vector& y) {.
    double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
    for (size_t i = 0; i < x.size(); ++i) result += x[i] * y[i];  // EN: Loop control flow: for (size_t i = 0; i < x.size(); ++i) result += x[i] * y[i];.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

double vectorNorm(const Vector& x) {  // EN: Execute line: double vectorNorm(const Vector& x) {.
    return std::sqrt(dotProduct(x, x));  // EN: Return from the current function: return std::sqrt(dotProduct(x, x));.
}  // EN: Structure delimiter for a block or scope.

Vector scalarMultiply(double c, const Vector& x) {  // EN: Execute line: Vector scalarMultiply(double c, const Vector& x) {.
    Vector result(x.size());  // EN: Execute a statement: Vector result(x.size());.
    for (size_t i = 0; i < x.size(); ++i) result[i] = c * x[i];  // EN: Loop control flow: for (size_t i = 0; i < x.size(); ++i) result[i] = c * x[i];.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

Vector vectorSubtract(const Vector& x, const Vector& y) {  // EN: Execute line: Vector vectorSubtract(const Vector& x, const Vector& y) {.
    Vector result(x.size());  // EN: Execute a statement: Vector result(x.size());.
    for (size_t i = 0; i < x.size(); ++i) result[i] = x[i] - y[i];  // EN: Loop control flow: for (size_t i = 0; i < x.size(); ++i) result[i] = x[i] - y[i];.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

Vector normalize(const Vector& x) {  // EN: Execute line: Vector normalize(const Vector& x) {.
    double norm = vectorNorm(x);  // EN: Execute a statement: double norm = vectorNorm(x);.
    return scalarMultiply(1.0 / norm, x);  // EN: Return from the current function: return scalarMultiply(1.0 / norm, x);.
}  // EN: Structure delimiter for a block or scope.

/**
 * Modified Gram-Schmidt
 */
Matrix modifiedGramSchmidt(const Matrix& A) {  // EN: Execute line: Matrix modifiedGramSchmidt(const Matrix& A) {.
    size_t n = A.size();  // EN: Execute a statement: size_t n = A.size();.
    Matrix Q = A;  // 複製  // EN: Execute line: Matrix Q = A; // 複製.

    for (size_t j = 0; j < n; ++j) {  // EN: Loop control flow: for (size_t j = 0; j < n; ++j) {.
        for (size_t i = 0; i < j; ++i) {  // EN: Loop control flow: for (size_t i = 0; i < j; ++i) {.
            double coeff = dotProduct(Q[i], Q[j]) / dotProduct(Q[i], Q[i]);  // EN: Execute a statement: double coeff = dotProduct(Q[i], Q[j]) / dotProduct(Q[i], Q[i]);.
            Q[j] = vectorSubtract(Q[j], scalarMultiply(coeff, Q[i]));  // EN: Execute a statement: Q[j] = vectorSubtract(Q[j], scalarMultiply(coeff, Q[i]));.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    return Q;  // EN: Return from the current function: return Q;.
}  // EN: Structure delimiter for a block or scope.

/**
 * Gram-Schmidt 正交化並標準化
 */
Matrix gramSchmidtNormalized(const Matrix& A) {  // EN: Execute line: Matrix gramSchmidtNormalized(const Matrix& A) {.
    Matrix Q = modifiedGramSchmidt(A);  // EN: Execute a statement: Matrix Q = modifiedGramSchmidt(A);.
    for (auto& q : Q) {  // EN: Loop control flow: for (auto& q : Q) {.
        q = normalize(q);  // EN: Execute a statement: q = normalize(q);.
    }  // EN: Structure delimiter for a block or scope.
    return Q;  // EN: Return from the current function: return Q;.
}  // EN: Structure delimiter for a block or scope.

bool verifyOrthogonality(const Matrix& Q, double tol = 1e-10) {  // EN: Execute line: bool verifyOrthogonality(const Matrix& Q, double tol = 1e-10) {.
    for (size_t i = 0; i < Q.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < Q.size(); ++i) {.
        for (size_t j = i + 1; j < Q.size(); ++j) {  // EN: Loop control flow: for (size_t j = i + 1; j < Q.size(); ++j) {.
            if (std::abs(dotProduct(Q[i], Q[j])) > tol) return false;  // EN: Conditional control flow: if (std::abs(dotProduct(Q[i], Q[j])) > tol) return false;.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return true;  // EN: Return from the current function: return true;.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    std::cout << std::fixed << std::setprecision(4);  // EN: Execute a statement: std::cout << std::fixed << std::setprecision(4);.

    printSeparator("Gram-Schmidt 正交化示範 (C++)");  // EN: Execute a statement: printSeparator("Gram-Schmidt 正交化示範 (C++)");.

    // 輸入向量組
    Matrix A = {  // EN: Execute line: Matrix A = {.
        {1.0, 1.0, 0.0},  // EN: Execute line: {1.0, 1.0, 0.0},.
        {1.0, 0.0, 1.0},  // EN: Execute line: {1.0, 0.0, 1.0},.
        {0.0, 1.0, 1.0}  // EN: Execute line: {0.0, 1.0, 1.0}.
    };  // EN: Structure delimiter for a block or scope.

    std::cout << "輸入向量組：\n";  // EN: Execute a statement: std::cout << "輸入向量組：\n";.
    for (size_t i = 0; i < A.size(); ++i)  // EN: Loop control flow: for (size_t i = 0; i < A.size(); ++i).
        printVector("a" + std::to_string(i+1), A[i]);  // EN: Execute a statement: printVector("a" + std::to_string(i+1), A[i]);.

    // MGS
    Matrix Q = modifiedGramSchmidt(A);  // EN: Execute a statement: Matrix Q = modifiedGramSchmidt(A);.

    std::cout << "\n正交化結果（MGS）：\n";  // EN: Execute a statement: std::cout << "\n正交化結果（MGS）：\n";.
    for (size_t i = 0; i < Q.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < Q.size(); ++i) {.
        printVector("q" + std::to_string(i+1), Q[i]);  // EN: Execute a statement: printVector("q" + std::to_string(i+1), Q[i]);.
        std::cout << "    ‖q" << i+1 << "‖ = " << vectorNorm(Q[i]) << "\n";  // EN: Execute a statement: std::cout << " ‖q" << i+1 << "‖ = " << vectorNorm(Q[i]) << "\n";.
    }  // EN: Structure delimiter for a block or scope.

    std::cout << "\n正交？ " << (verifyOrthogonality(Q) ? "true" : "false") << "\n";  // EN: Execute a statement: std::cout << "\n正交？ " << (verifyOrthogonality(Q) ? "true" : "false") <<….

    // 內積驗證
    std::cout << "\n內積驗證：\n";  // EN: Execute a statement: std::cout << "\n內積驗證：\n";.
    std::cout << "q₁ · q₂ = " << dotProduct(Q[0], Q[1]) << "\n";  // EN: Execute a statement: std::cout << "q₁ · q₂ = " << dotProduct(Q[0], Q[1]) << "\n";.
    std::cout << "q₁ · q₃ = " << dotProduct(Q[0], Q[2]) << "\n";  // EN: Execute a statement: std::cout << "q₁ · q₃ = " << dotProduct(Q[0], Q[2]) << "\n";.
    std::cout << "q₂ · q₃ = " << dotProduct(Q[1], Q[2]) << "\n";  // EN: Execute a statement: std::cout << "q₂ · q₃ = " << dotProduct(Q[1], Q[2]) << "\n";.

    // 標準化
    printSeparator("標準正交化");  // EN: Execute a statement: printSeparator("標準正交化");.

    Matrix E = gramSchmidtNormalized(A);  // EN: Execute a statement: Matrix E = gramSchmidtNormalized(A);.

    std::cout << "標準正交向量組：\n";  // EN: Execute a statement: std::cout << "標準正交向量組：\n";.
    for (size_t i = 0; i < E.size(); ++i) {  // EN: Loop control flow: for (size_t i = 0; i < E.size(); ++i) {.
        printVector("e" + std::to_string(i+1), E[i]);  // EN: Execute a statement: printVector("e" + std::to_string(i+1), E[i]);.
        std::cout << "    ‖e" << i+1 << "‖ = " << vectorNorm(E[i]) << "\n";  // EN: Execute a statement: std::cout << " ‖e" << i+1 << "‖ = " << vectorNorm(E[i]) << "\n";.
    }  // EN: Structure delimiter for a block or scope.

    // 總結
    printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
    std::cout << R"(  // EN: Execute line: std::cout << R"(.
Gram-Schmidt 核心公式：  // EN: Execute line: Gram-Schmidt 核心公式：.

proj_q(a) = (qᵀa / qᵀq) q  // EN: Execute line: proj_q(a) = (qᵀa / qᵀq) q.

q₁ = a₁  // EN: Execute line: q₁ = a₁.
qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ)  // EN: Execute line: qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ).

eᵢ = qᵢ / ‖qᵢ‖  // EN: Execute line: eᵢ = qᵢ / ‖qᵢ‖.
)" << "\n";  // EN: Execute a statement: )" << "\n";.

    std::cout << std::string(60, '=') << "\n示範完成！\n" << std::string(60, '=') << "\n";  // EN: Execute a statement: std::cout << std::string(60, '=') << "\n示範完成！\n" << std::string(60, '='….

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
