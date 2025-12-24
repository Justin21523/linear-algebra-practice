/**
 * 克萊姆法則 (Cramer's Rule)
 *
 * 編譯：g++ -std=c++17 -O2 cramers_rule.cpp -o cramers_rule
 * 執行：./cramers_rule
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using Matrix = vector<vector<double>>;
using Vector = vector<double>;

void printSeparator(const string& title) {
    cout << endl;
    cout << string(60, '=') << endl;
    cout << title << endl;
    cout << string(60, '=') << endl;
}

void printMatrix(const string& name, const Matrix& M) {
    cout << name << " =" << endl;
    for (const auto& row : M) {
        cout << "  [";
        for (size_t i = 0; i < row.size(); i++) {
            cout << fixed << setprecision(4) << setw(8) << row[i];
            if (i < row.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
    }
}

void printVector(const string& name, const Vector& v) {
    cout << name << " = [";
    for (size_t i = 0; i < v.size(); i++) {
        cout << fixed << setprecision(4) << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}

// 2×2 行列式
double det2x2(const Matrix& A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

// 3×3 行列式
double det3x3(const Matrix& A) {
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

double determinant(const Matrix& A) {
    int n = A.size();
    if (n == 2) return det2x2(A);
    if (n == 3) return det3x3(A);
    throw runtime_error("僅支援 2×2 和 3×3 矩陣");
}

// 替換第 j 行
Matrix replaceColumn(const Matrix& A, const Vector& b, int j) {
    Matrix Aj = A;
    for (size_t i = 0; i < A.size(); i++) {
        Aj[i][j] = b[i];
    }
    return Aj;
}

// 克萊姆法則
Vector cramersRule(const Matrix& A, const Vector& b) {
    int n = A.size();
    double detA = determinant(A);

    if (abs(detA) < 1e-10) {
        throw runtime_error("矩陣奇異");
    }

    Vector x(n);
    for (int j = 0; j < n; j++) {
        Matrix Aj = replaceColumn(A, b, j);
        x[j] = determinant(Aj) / detA;
    }
    return x;
}

int main() {
    printSeparator("克萊姆法則示範 (C++)");

    // ========================================
    // 1. 2×2 系統
    // ========================================
    printSeparator("1. 2×2 系統");

    Matrix A2 = {{2, 3}, {4, 5}};
    Vector b2 = {8, 14};

    cout << "方程組：" << endl;
    cout << "  2x + 3y = 8" << endl;
    cout << "  4x + 5y = 14" << endl;

    printMatrix("\nA", A2);
    printVector("b", b2);

    double detA2 = determinant(A2);
    cout << "\ndet(A) = " << detA2 << endl;

    Vector x2 = cramersRule(A2, b2);

    for (int j = 0; j < 2; j++) {
        Matrix Aj = replaceColumn(A2, b2, j);
        double detAj = determinant(Aj);
        cout << "\nA" << j+1 << "（第 " << j+1 << " 行換成 b）：" << endl;
        printMatrix("", Aj);
        cout << "det(A" << j+1 << ") = " << detAj << endl;
        cout << "x" << j+1 << " = " << x2[j] << endl;
    }

    cout << "\n解：x = " << x2[0] << ", y = " << x2[1] << endl;

    // ========================================
    // 2. 3×3 系統
    // ========================================
    printSeparator("2. 3×3 系統");

    Matrix A3 = {
        {2, 1, -1},
        {-3, -1, 2},
        {-2, 1, 2}
    };
    Vector b3 = {8, -11, -3};

    cout << "方程組：" << endl;
    cout << "   2x +  y -  z =  8" << endl;
    cout << "  -3x -  y + 2z = -11" << endl;
    cout << "  -2x +  y + 2z = -3" << endl;

    printMatrix("\nA", A3);
    printVector("b", b3);

    Vector x3 = cramersRule(A3, b3);

    cout << "\n解：x = " << x3[0] << ", y = " << x3[1] << ", z = " << x3[2] << endl;

    // 驗證
    cout << "\n驗證：" << endl;
    cout << "  2(" << x3[0] << ") + (" << x3[1] << ") - (" << x3[2] << ") = "
         << 2*x3[0] + x3[1] - x3[2] << endl;
    cout << "  -3(" << x3[0] << ") - (" << x3[1] << ") + 2(" << x3[2] << ") = "
         << -3*x3[0] - x3[1] + 2*x3[2] << endl;
    cout << "  -2(" << x3[0] << ") + (" << x3[1] << ") + 2(" << x3[2] << ") = "
         << -2*x3[0] + x3[1] + 2*x3[2] << endl;

    // 總結
    printSeparator("總結");
    cout << R"(
克萊姆法則：
  xⱼ = det(Aⱼ) / det(A)
  Aⱼ = A 的第 j 行換成 b

適用條件：
  - det(A) ≠ 0
  - 方陣系統

時間複雜度：O(n! × n)
實際應用：使用 LU 分解更有效率
)" << endl;

    cout << string(60, '=') << endl;
    cout << "示範完成！" << endl;
    cout << string(60, '=') << endl;

    return 0;
}
