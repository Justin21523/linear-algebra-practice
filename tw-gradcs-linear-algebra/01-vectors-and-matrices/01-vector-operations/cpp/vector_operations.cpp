/**
 * 向量運算：C++ 實作 (Vector Operations: C++ Implementation)
 *
 * 本程式示範：
 * 1. 向量加法、減法 (Vector addition, subtraction)
 * 2. 純量乘法 (Scalar multiplication)
 * 3. 向量長度 (Vector norm)
 * 4. 向量正規化 (Normalization)
 * 5. 內積 (Dot product)
 * 6. 夾角計算 (Angle between vectors)
 * 7. 投影 (Projection)
 *
 * 編譯方式 (Compilation):
 *   g++ -std=c++17 -o vector_operations vector_operations.cpp
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>

using namespace std;

// 型別別名 (Type alias)
using Vector = vector<double>;

/**
 * 印出向量 (Print vector)
 */
void printVector(const string& name, const Vector& v) {
    cout << name << " = [";
    for (size_t i = 0; i < v.size(); i++) {
        cout << fixed << setprecision(4) << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}

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
 * 向量加法 (Vector addition)
 * u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]
 */
Vector vectorAdd(const Vector& u, const Vector& v) {
    if (u.size() != v.size()) {
        throw invalid_argument("向量維度必須相同 (Vectors must have same dimension)");
    }

    Vector result(u.size());
    for (size_t i = 0; i < u.size(); i++) {
        result[i] = u[i] + v[i];
    }
    return result;
}

/**
 * 向量減法 (Vector subtraction)
 * u - v = [u₁-v₁, u₂-v₂, ..., uₙ-vₙ]
 */
Vector vectorSubtract(const Vector& u, const Vector& v) {
    if (u.size() != v.size()) {
        throw invalid_argument("向量維度必須相同 (Vectors must have same dimension)");
    }

    Vector result(u.size());
    for (size_t i = 0; i < u.size(); i++) {
        result[i] = u[i] - v[i];
    }
    return result;
}

/**
 * 純量乘法 (Scalar multiplication)
 * c·v = [c·v₁, c·v₂, ..., c·vₙ]
 */
Vector scalarMultiply(double c, const Vector& v) {
    Vector result(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        result[i] = c * v[i];
    }
    return result;
}

/**
 * 內積 (Dot product)
 * u·v = u₁v₁ + u₂v₂ + ... + uₙvₙ
 */
double dotProduct(const Vector& u, const Vector& v) {
    if (u.size() != v.size()) {
        throw invalid_argument("向量維度必須相同 (Vectors must have same dimension)");
    }

    double result = 0.0;
    for (size_t i = 0; i < u.size(); i++) {
        result += u[i] * v[i];
    }
    return result;
}

/**
 * 向量長度/範數 (Vector norm)
 * ‖v‖ = √(v₁² + v₂² + ... + vₙ²)
 */
double vectorNorm(const Vector& v) {
    return sqrt(dotProduct(v, v));
}

/**
 * 向量正規化 (Normalization)
 * û = v / ‖v‖
 */
Vector normalize(const Vector& v) {
    double norm = vectorNorm(v);

    if (norm == 0) {
        throw invalid_argument("零向量無法正規化 (Cannot normalize zero vector)");
    }

    return scalarMultiply(1.0 / norm, v);
}

/**
 * 計算兩向量夾角（弧度）(Angle between vectors in radians)
 * θ = arccos((u·v) / (‖u‖‖v‖))
 */
double angleBetween(const Vector& u, const Vector& v) {
    double dot = dotProduct(u, v);
    double normU = vectorNorm(u);
    double normV = vectorNorm(v);

    if (normU == 0 || normV == 0) {
        throw invalid_argument("零向量沒有定義夾角 (Angle undefined for zero vector)");
    }

    // 處理數值誤差 (Handle numerical errors)
    double cosTheta = dot / (normU * normV);
    cosTheta = max(-1.0, min(1.0, cosTheta));

    return acos(cosTheta);
}

/**
 * 向量投影 (Vector projection)
 * proj_v(u) = ((u·v) / ‖v‖²) · v
 */
Vector project(const Vector& u, const Vector& v) {
    double dotUV = dotProduct(u, v);
    double normVSquared = dotProduct(v, v);

    if (normVSquared == 0) {
        throw invalid_argument("無法投影到零向量 (Cannot project onto zero vector)");
    }

    double scalar = dotUV / normVSquared;
    return scalarMultiply(scalar, v);
}

/**
 * 純量投影 (Scalar projection)
 * comp_v(u) = (u·v) / ‖v‖
 */
double scalarProjection(const Vector& u, const Vector& v) {
    double normV = vectorNorm(v);

    if (normV == 0) {
        throw invalid_argument("無法計算到零向量的投影");
    }

    return dotProduct(u, v) / normV;
}

/**
 * 檢查正交性 (Check orthogonality)
 */
bool isOrthogonal(const Vector& u, const Vector& v, double tolerance = 1e-10) {
    return abs(dotProduct(u, v)) < tolerance;
}

/**
 * 弧度轉角度 (Radians to degrees)
 */
double toDegrees(double radians) {
    return radians * 180.0 / M_PI;
}

int main() {
    printSeparator("向量運算示範 - C++ 版本\nVector Operations Demo - C++ Implementation");

    // ========================================
    // 定義範例向量 (Define example vectors)
    // ========================================
    Vector u = {3.0, 4.0};
    Vector v = {1.0, 2.0};

    cout << "\n範例向量 (Example vectors):" << endl;
    printVector("u", u);
    printVector("v", v);

    // ========================================
    // 1. 向量加法 (Vector Addition)
    // ========================================
    printSeparator("1. 向量加法 (Vector Addition)");

    Vector sum = vectorAdd(u, v);
    printVector("u + v", sum);

    // ========================================
    // 2. 向量減法 (Vector Subtraction)
    // ========================================
    printSeparator("2. 向量減法 (Vector Subtraction)");

    Vector diff = vectorSubtract(u, v);
    printVector("u - v", diff);

    // ========================================
    // 3. 純量乘法 (Scalar Multiplication)
    // ========================================
    printSeparator("3. 純量乘法 (Scalar Multiplication)");

    double c = 2.5;
    Vector scaled = scalarMultiply(c, u);
    cout << c << " × u = ";
    printVector("", scaled);

    Vector negated = scalarMultiply(-1, u);
    printVector("-u (反向量)", negated);

    // ========================================
    // 4. 向量長度 (Vector Norm)
    // ========================================
    printSeparator("4. 向量長度 (Vector Norm)");

    double normU = vectorNorm(u);
    cout << "‖u‖ = √(" << u[0] << "² + " << u[1] << "²) = √"
         << (u[0]*u[0] + u[1]*u[1]) << " = " << normU << endl;

    cout << "\n這就是經典的 3-4-5 直角三角形！" << endl;

    // ========================================
    // 5. 正規化 (Normalization)
    // ========================================
    printSeparator("5. 正規化 (Normalization)");

    Vector uHat = normalize(u);
    printVector("û (單位向量)", uHat);
    cout << "‖û‖ = " << fixed << setprecision(10) << vectorNorm(uHat)
         << " (應該是 1)" << endl;

    // ========================================
    // 6. 內積 (Dot Product)
    // ========================================
    printSeparator("6. 內積 (Dot Product)");

    double dot = dotProduct(u, v);
    cout << fixed << setprecision(4);
    cout << "u·v = " << u[0] << "×" << v[0] << " + " << u[1] << "×" << v[1]
         << " = " << dot << endl;

    // 驗證 ‖v‖² = v·v
    cout << "\n驗證 ‖v‖² = v·v:" << endl;
    cout << "‖v‖² = " << pow(vectorNorm(v), 2) << endl;
    cout << "v·v  = " << dotProduct(v, v) << endl;

    // ========================================
    // 7. 夾角計算 (Angle Calculation)
    // ========================================
    printSeparator("7. 夾角計算 (Angle Between Vectors)");

    double angleRad = angleBetween(u, v);
    double angleDeg = toDegrees(angleRad);

    cout << "θ = " << angleRad << " 弧度 (radians)" << endl;
    cout << "θ = " << angleDeg << "° 度 (degrees)" << endl;

    // ========================================
    // 8. 正交檢驗 (Orthogonality Check)
    // ========================================
    printSeparator("8. 正交檢驗 (Orthogonality Check)");

    Vector a = {1.0, 0.0};
    Vector b = {0.0, 1.0};

    printVector("a", a);
    printVector("b", b);
    cout << "a·b = " << dotProduct(a, b) << endl;
    cout << "a 和 b 是否正交？ " << (isOrthogonal(a, b) ? "是" : "否") << endl;

    cout << endl;
    printVector("u", u);
    printVector("v", v);
    cout << "u·v = " << dotProduct(u, v) << endl;
    cout << "u 和 v 是否正交？ " << (isOrthogonal(u, v) ? "是" : "否") << endl;

    // ========================================
    // 9. 向量投影 (Vector Projection)
    // ========================================
    printSeparator("9. 向量投影 (Vector Projection)");

    Vector proj = project(u, v);
    double comp = scalarProjection(u, v);

    cout << "將 u 投影到 v 上 (Project u onto v):" << endl;
    printVector("proj_v(u)", proj);
    cout << "comp_v(u) = " << comp << " (純量投影)" << endl;

    // 驗證
    Vector perp = vectorSubtract(u, proj);
    cout << "\n驗證：u = proj + perp，其中 perp ⊥ v" << endl;
    printVector("perp = u - proj", perp);
    cout << "perp · v = " << scientific << setprecision(10)
         << dotProduct(perp, v) << " (應該是 0)" << endl;

    // ========================================
    // 10. 3D 向量 (3D Vectors)
    // ========================================
    printSeparator("10. 3D 向量 (3D Vectors)");

    Vector p = {1.0, 2.0, 3.0};
    Vector q = {4.0, 5.0, 6.0};

    cout << fixed << setprecision(4);
    printVector("p", p);
    printVector("q", q);
    printVector("p + q", vectorAdd(p, q));
    cout << "p · q = " << dotProduct(p, q) << endl;
    cout << "‖p‖ = " << vectorNorm(p) << endl;
    cout << "夾角 = " << toDegrees(angleBetween(p, q)) << "°" << endl;

    // ========================================
    // 11. 線性組合 (Linear Combination)
    // ========================================
    printSeparator("11. 線性組合 (Linear Combination)");

    Vector e1 = {1.0, 0.0};
    Vector e2 = {0.0, 1.0};

    cout << "標準基底向量 (Standard basis vectors):" << endl;
    printVector("e₁", e1);
    printVector("e₂", e2);

    cout << "\n向量 [3, 4] = 3·e₁ + 4·e₂" << endl;
    Vector combination = vectorAdd(scalarMultiply(3, e1), scalarMultiply(4, e2));
    printVector("3·e₁ + 4·e₂", combination);

    cout << endl;
    cout << string(60, '=') << endl;
    cout << "所有向量運算示範完成！" << endl;
    cout << "All vector operations demonstrated!" << endl;
    cout << string(60, '=') << endl;

    return 0;
}
