/**
 * Gram-Schmidt 正交化 (Gram-Schmidt Process)
 *
 * 編譯：g++ -std=c++17 -O2 gram_schmidt.cpp -o gram_schmidt
 * 執行：./gram_schmidt
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

void printSeparator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n" << title << "\n" << std::string(60, '=') << "\n";
}

void printVector(const std::string& name, const Vector& v) {
    std::cout << name << " = [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

double dotProduct(const Vector& x, const Vector& y) {
    double result = 0.0;
    for (size_t i = 0; i < x.size(); ++i) result += x[i] * y[i];
    return result;
}

double vectorNorm(const Vector& x) {
    return std::sqrt(dotProduct(x, x));
}

Vector scalarMultiply(double c, const Vector& x) {
    Vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) result[i] = c * x[i];
    return result;
}

Vector vectorSubtract(const Vector& x, const Vector& y) {
    Vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) result[i] = x[i] - y[i];
    return result;
}

Vector normalize(const Vector& x) {
    double norm = vectorNorm(x);
    return scalarMultiply(1.0 / norm, x);
}

/**
 * Modified Gram-Schmidt
 */
Matrix modifiedGramSchmidt(const Matrix& A) {
    size_t n = A.size();
    Matrix Q = A;  // 複製

    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < j; ++i) {
            double coeff = dotProduct(Q[i], Q[j]) / dotProduct(Q[i], Q[i]);
            Q[j] = vectorSubtract(Q[j], scalarMultiply(coeff, Q[i]));
        }
    }

    return Q;
}

/**
 * Gram-Schmidt 正交化並標準化
 */
Matrix gramSchmidtNormalized(const Matrix& A) {
    Matrix Q = modifiedGramSchmidt(A);
    for (auto& q : Q) {
        q = normalize(q);
    }
    return Q;
}

bool verifyOrthogonality(const Matrix& Q, double tol = 1e-10) {
    for (size_t i = 0; i < Q.size(); ++i) {
        for (size_t j = i + 1; j < Q.size(); ++j) {
            if (std::abs(dotProduct(Q[i], Q[j])) > tol) return false;
        }
    }
    return true;
}

int main() {
    std::cout << std::fixed << std::setprecision(4);

    printSeparator("Gram-Schmidt 正交化示範 (C++)");

    // 輸入向量組
    Matrix A = {
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    };

    std::cout << "輸入向量組：\n";
    for (size_t i = 0; i < A.size(); ++i)
        printVector("a" + std::to_string(i+1), A[i]);

    // MGS
    Matrix Q = modifiedGramSchmidt(A);

    std::cout << "\n正交化結果（MGS）：\n";
    for (size_t i = 0; i < Q.size(); ++i) {
        printVector("q" + std::to_string(i+1), Q[i]);
        std::cout << "    ‖q" << i+1 << "‖ = " << vectorNorm(Q[i]) << "\n";
    }

    std::cout << "\n正交？ " << (verifyOrthogonality(Q) ? "true" : "false") << "\n";

    // 內積驗證
    std::cout << "\n內積驗證：\n";
    std::cout << "q₁ · q₂ = " << dotProduct(Q[0], Q[1]) << "\n";
    std::cout << "q₁ · q₃ = " << dotProduct(Q[0], Q[2]) << "\n";
    std::cout << "q₂ · q₃ = " << dotProduct(Q[1], Q[2]) << "\n";

    // 標準化
    printSeparator("標準正交化");

    Matrix E = gramSchmidtNormalized(A);

    std::cout << "標準正交向量組：\n";
    for (size_t i = 0; i < E.size(); ++i) {
        printVector("e" + std::to_string(i+1), E[i]);
        std::cout << "    ‖e" << i+1 << "‖ = " << vectorNorm(E[i]) << "\n";
    }

    // 總結
    printSeparator("總結");
    std::cout << R"(
Gram-Schmidt 核心公式：

proj_q(a) = (qᵀa / qᵀq) q

q₁ = a₁
qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ)

eᵢ = qᵢ / ‖qᵢ‖
)" << "\n";

    std::cout << std::string(60, '=') << "\n示範完成！\n" << std::string(60, '=') << "\n";

    return 0;
}
