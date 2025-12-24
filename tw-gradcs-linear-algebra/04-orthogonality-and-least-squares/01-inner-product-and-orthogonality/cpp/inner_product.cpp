/**
 * 內積與正交性 (Inner Product and Orthogonality)
 *
 * 本程式示範：
 * 1. 向量內積計算
 * 2. 向量長度（範數）
 * 3. 向量夾角
 * 4. 正交性判斷
 * 5. 正交矩陣驗證
 * 6. Cauchy-Schwarz 不等式
 *
 * 編譯：g++ -std=c++17 -O2 inner_product.cpp -o inner_product
 * 執行：./inner_product
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

const double EPSILON = 1e-10;
const double PI = 3.14159265358979323846;

// ========================================
// 輔助函數
// ========================================

void printSeparator(const std::string& title) {
    std::cout << "\n";
    std::cout << std::string(60, '=') << "\n";
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
// 向量運算
// ========================================

/**
 * 計算兩向量的內積 (Dot Product)
 * x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ
 */
double dotProduct(const Vector& x, const Vector& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("向量維度必須相同");
    }

    double result = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        result += x[i] * y[i];
    }
    return result;
}

/**
 * 計算向量的長度（L2 範數）
 * ‖x‖ = √(x · x)
 */
double vectorNorm(const Vector& x) {
    return std::sqrt(dotProduct(x, x));
}

/**
 * 正規化向量為單位向量
 * û = x / ‖x‖
 */
Vector normalize(const Vector& x) {
    double norm = vectorNorm(x);
    if (norm < EPSILON) {
        throw std::invalid_argument("零向量無法正規化");
    }

    Vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] / norm;
    }
    return result;
}

/**
 * 計算兩向量的夾角（弧度）
 * cos θ = (x · y) / (‖x‖ ‖y‖)
 */
double vectorAngle(const Vector& x, const Vector& y) {
    double dot = dotProduct(x, y);
    double normX = vectorNorm(x);
    double normY = vectorNorm(y);

    if (normX < EPSILON || normY < EPSILON) {
        throw std::invalid_argument("零向量沒有定義夾角");
    }

    double cosTheta = dot / (normX * normY);
    // 處理浮點數誤差
    cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
    return std::acos(cosTheta);
}

/**
 * 判斷兩向量是否正交
 * x ⊥ y ⟺ x · y = 0
 */
bool isOrthogonal(const Vector& x, const Vector& y, double tol = EPSILON) {
    return std::abs(dotProduct(x, y)) < tol;
}

// ========================================
// 矩陣運算
// ========================================

/**
 * 矩陣轉置
 */
Matrix transpose(const Matrix& A) {
    if (A.empty()) return {};

    size_t m = A.size();
    size_t n = A[0].size();

    Matrix result(n, Vector(m));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

/**
 * 矩陣乘法
 */
Matrix matrixMultiply(const Matrix& A, const Matrix& B) {
    size_t m = A.size();
    size_t n = B[0].size();
    size_t k = B.size();

    if (A[0].size() != k) {
        throw std::invalid_argument("矩陣維度不匹配");
    }

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

/**
 * 矩陣乘向量
 */
Vector matrixVectorMultiply(const Matrix& A, const Vector& x) {
    size_t m = A.size();
    size_t n = A[0].size();

    if (x.size() != n) {
        throw std::invalid_argument("維度不匹配");
    }

    Vector result(m, 0.0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

/**
 * 判斷是否為單位矩陣
 */
bool isIdentity(const Matrix& A, double tol = EPSILON) {
    size_t n = A.size();
    if (A[0].size() != n) return false;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(A[i][j] - expected) > tol) {
                return false;
            }
        }
    }
    return true;
}

/**
 * 判斷矩陣是否為正交矩陣
 * QᵀQ = I
 */
bool isOrthogonalMatrix(const Matrix& Q, double tol = EPSILON) {
    Matrix QT = transpose(Q);
    Matrix product = matrixMultiply(QT, Q);
    return isIdentity(product, tol);
}

// ========================================
// 不等式驗證
// ========================================

struct InequalityResult {
    double leftSide;
    double rightSide;
    bool satisfied;
    bool equality;
};

/**
 * 驗證 Cauchy-Schwarz 不等式
 * |x · y| ≤ ‖x‖ ‖y‖
 */
InequalityResult verifyCauchySchwarz(const Vector& x, const Vector& y) {
    double dot = std::abs(dotProduct(x, y));
    double productOfNorms = vectorNorm(x) * vectorNorm(y);

    return {
        dot,
        productOfNorms,
        dot <= productOfNorms + EPSILON,
        std::abs(dot - productOfNorms) < EPSILON
    };
}

/**
 * 驗證三角不等式
 * ‖x + y‖ ≤ ‖x‖ + ‖y‖
 */
InequalityResult verifyTriangleInequality(const Vector& x, const Vector& y) {
    Vector sum(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        sum[i] = x[i] + y[i];
    }

    double normSum = vectorNorm(sum);
    double sumOfNorms = vectorNorm(x) + vectorNorm(y);

    return {
        normSum,
        sumOfNorms,
        normSum <= sumOfNorms + EPSILON,
        std::abs(normSum - sumOfNorms) < EPSILON
    };
}

// ========================================
// 主程式
// ========================================

int main() {
    std::cout << std::fixed << std::setprecision(4);

    printSeparator("內積與正交性示範 (C++)\nInner Product & Orthogonality Demo");

    // ========================================
    // 1. 內積計算
    // ========================================
    printSeparator("1. 內積計算 (Dot Product)");

    Vector x = {1.0, 2.0, 3.0};
    Vector y = {4.0, 5.0, 6.0};

    printVector("x", x);
    printVector("y", y);
    std::cout << "\nx · y = " << dotProduct(x, y) << "\n";
    std::cout << "計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32\n";

    // ========================================
    // 2. 向量長度
    // ========================================
    printSeparator("2. 向量長度 (Vector Norm)");

    Vector v = {3.0, 4.0};
    printVector("v", v);
    std::cout << "‖v‖ = " << vectorNorm(v) << "\n";
    std::cout << "計算：√(3² + 4²) = √25 = 5\n";

    // 正規化
    Vector vNormalized = normalize(v);
    std::cout << "\n單位向量：\n";
    printVector("v̂ = v/‖v‖", vNormalized);
    std::cout << "‖v̂‖ = " << vectorNorm(vNormalized) << "\n";

    // ========================================
    // 3. 向量夾角
    // ========================================
    printSeparator("3. 向量夾角 (Vector Angle)");

    Vector a = {1.0, 0.0};
    Vector b = {1.0, 1.0};

    printVector("a", a);
    printVector("b", b);

    double theta = vectorAngle(a, b);
    std::cout << "\n夾角 θ = " << theta << " rad = " << (theta * 180.0 / PI) << "°\n";
    std::cout << "cos θ = " << std::cos(theta) << "\n";
    std::cout << "預期：cos 45° = 1/√2 ≈ 0.7071\n";

    // ========================================
    // 4. 正交性判斷
    // ========================================
    printSeparator("4. 正交性判斷 (Orthogonality Check)");

    Vector u1 = {1.0, 2.0};
    Vector u2 = {-2.0, 1.0};

    printVector("u₁", u1);
    printVector("u₂", u2);
    std::cout << "\nu₁ · u₂ = " << dotProduct(u1, u2) << "\n";
    std::cout << "u₁ ⊥ u₂？ " << (isOrthogonal(u1, u2) ? "true" : "false") << "\n";

    // 非正交的例子
    Vector w1 = {1.0, 1.0};
    Vector w2 = {1.0, 2.0};

    std::cout << "\n另一組：\n";
    printVector("w₁", w1);
    printVector("w₂", w2);
    std::cout << "w₁ · w₂ = " << dotProduct(w1, w2) << "\n";
    std::cout << "w₁ ⊥ w₂？ " << (isOrthogonal(w1, w2) ? "true" : "false") << "\n";

    // ========================================
    // 5. 正交矩陣
    // ========================================
    printSeparator("5. 正交矩陣 (Orthogonal Matrix)");

    // 旋轉矩陣（45度）
    double angle = PI / 4;
    Matrix Q = {
        {std::cos(angle), -std::sin(angle)},
        {std::sin(angle), std::cos(angle)}
    };

    std::cout << "旋轉矩陣（θ = 45°）：\n";
    printMatrix("Q", Q);

    Matrix QT = transpose(Q);
    printMatrix("\nQᵀ", QT);

    Matrix QTQ = matrixMultiply(QT, Q);
    printMatrix("\nQᵀQ", QTQ);

    std::cout << "\nQ 是正交矩陣？ " << (isOrthogonalMatrix(Q) ? "true" : "false") << "\n";

    // 驗證保長度
    Vector xVec = {3.0, 4.0};
    Vector Qx = matrixVectorMultiply(Q, xVec);

    std::cout << "\n保長度驗證：\n";
    printVector("x", xVec);
    printVector("Qx", Qx);
    std::cout << "‖x‖ = " << vectorNorm(xVec) << "\n";
    std::cout << "‖Qx‖ = " << vectorNorm(Qx) << "\n";

    // ========================================
    // 6. Cauchy-Schwarz 不等式
    // ========================================
    printSeparator("6. Cauchy-Schwarz 不等式");

    Vector csX = {1.0, 2.0, 3.0};
    Vector csY = {4.0, 5.0, 6.0};

    printVector("x", csX);
    printVector("y", csY);

    auto csResult = verifyCauchySchwarz(csX, csY);
    std::cout << "\n|x · y| = " << csResult.leftSide << "\n";
    std::cout << "‖x‖ ‖y‖ = " << csResult.rightSide << "\n";
    std::cout << "|x · y| ≤ ‖x‖ ‖y‖？ " << (csResult.satisfied ? "true" : "false") << "\n";
    std::cout << "等號成立？ " << (csResult.equality ? "true（平行）" : "false") << "\n";

    // 平行向量
    std::cout << "\n平行向量的情況：\n";
    Vector p = {1.0, 2.0};
    Vector q = {2.0, 4.0};

    printVector("p", p);
    printVector("q = 2p", q);

    auto csParallel = verifyCauchySchwarz(p, q);
    std::cout << "|p · q| = " << csParallel.leftSide << "\n";
    std::cout << "‖p‖ ‖q‖ = " << csParallel.rightSide << "\n";
    std::cout << "等號成立？ " << (csParallel.equality ? "true" : "false") << "\n";

    // ========================================
    // 7. 三角不等式
    // ========================================
    printSeparator("7. 三角不等式");

    Vector triX = {3.0, 0.0};
    Vector triY = {0.0, 4.0};

    printVector("x", triX);
    printVector("y", triY);

    auto triResult = verifyTriangleInequality(triX, triY);
    std::cout << "\n‖x + y‖ = " << triResult.leftSide << "\n";
    std::cout << "‖x‖ + ‖y‖ = " << triResult.rightSide << "\n";
    std::cout << "‖x + y‖ ≤ ‖x‖ + ‖y‖？ " << (triResult.satisfied ? "true" : "false") << "\n";

    // 總結
    printSeparator("總結");
    std::cout << R"(
內積與正交性的核心公式：

1. 內積：x · y = Σ xᵢyᵢ

2. 長度：‖x‖ = √(x · x)

3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖)

4. 正交：x ⊥ y ⟺ x · y = 0

5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ

6. Cauchy-Schwarz：|x · y| ≤ ‖x‖ ‖y‖

7. 三角不等式：‖x + y‖ ≤ ‖x‖ + ‖y‖
)" << "\n";

    std::cout << std::string(60, '=') << "\n";
    std::cout << "示範完成！\n";
    std::cout << std::string(60, '=') << "\n";

    return 0;
}
