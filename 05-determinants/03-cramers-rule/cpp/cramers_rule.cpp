/**
 * 克萊姆法則 (Cramer's Rule)
 *
 * 編譯：g++ -std=c++17 -O2 cramers_rule.cpp -o cramers_rule
 * 執行：./cramers_rule
 */

#include <iostream>  // EN: Include a header dependency: #include <iostream>.
#include <vector>  // EN: Include a header dependency: #include <vector>.
#include <cmath>  // EN: Include a header dependency: #include <cmath>.
#include <iomanip>  // EN: Include a header dependency: #include <iomanip>.

using namespace std;  // EN: Execute a statement: using namespace std;.
using Matrix = vector<vector<double>>;  // EN: Execute a statement: using Matrix = vector<vector<double>>;.
using Vector = vector<double>;  // EN: Execute a statement: using Vector = vector<double>;.

void printSeparator(const string& title) {  // EN: Execute line: void printSeparator(const string& title) {.
    cout << endl;  // EN: Execute a statement: cout << endl;.
    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.
    cout << title << endl;  // EN: Execute a statement: cout << title << endl;.
    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.
}  // EN: Structure delimiter for a block or scope.

void printMatrix(const string& name, const Matrix& M) {  // EN: Execute line: void printMatrix(const string& name, const Matrix& M) {.
    cout << name << " =" << endl;  // EN: Execute a statement: cout << name << " =" << endl;.
    for (const auto& row : M) {  // EN: Loop control flow: for (const auto& row : M) {.
        cout << "  [";  // EN: Execute a statement: cout << " [";.
        for (size_t i = 0; i < row.size(); i++) {  // EN: Loop control flow: for (size_t i = 0; i < row.size(); i++) {.
            cout << fixed << setprecision(4) << setw(8) << row[i];  // EN: Execute a statement: cout << fixed << setprecision(4) << setw(8) << row[i];.
            if (i < row.size() - 1) cout << ", ";  // EN: Conditional control flow: if (i < row.size() - 1) cout << ", ";.
        }  // EN: Structure delimiter for a block or scope.
        cout << "]" << endl;  // EN: Execute a statement: cout << "]" << endl;.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

void printVector(const string& name, const Vector& v) {  // EN: Execute line: void printVector(const string& name, const Vector& v) {.
    cout << name << " = [";  // EN: Execute a statement: cout << name << " = [";.
    for (size_t i = 0; i < v.size(); i++) {  // EN: Loop control flow: for (size_t i = 0; i < v.size(); i++) {.
        cout << fixed << setprecision(4) << v[i];  // EN: Execute a statement: cout << fixed << setprecision(4) << v[i];.
        if (i < v.size() - 1) cout << ", ";  // EN: Conditional control flow: if (i < v.size() - 1) cout << ", ";.
    }  // EN: Structure delimiter for a block or scope.
    cout << "]" << endl;  // EN: Execute a statement: cout << "]" << endl;.
}  // EN: Structure delimiter for a block or scope.

// 2×2 行列式
double det2x2(const Matrix& A) {  // EN: Execute line: double det2x2(const Matrix& A) {.
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Return from the current function: return A[0][0] * A[1][1] - A[0][1] * A[1][0];.
}  // EN: Structure delimiter for a block or scope.

// 3×3 行列式
double det3x3(const Matrix& A) {  // EN: Execute line: double det3x3(const Matrix& A) {.
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])  // EN: Return from the current function: return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]).
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])  // EN: Execute line: - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]).
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);  // EN: Execute a statement: + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);.
}  // EN: Structure delimiter for a block or scope.

double determinant(const Matrix& A) {  // EN: Execute line: double determinant(const Matrix& A) {.
    int n = A.size();  // EN: Execute a statement: int n = A.size();.
    if (n == 2) return det2x2(A);  // EN: Conditional control flow: if (n == 2) return det2x2(A);.
    if (n == 3) return det3x3(A);  // EN: Conditional control flow: if (n == 3) return det3x3(A);.
    throw runtime_error("僅支援 2×2 和 3×3 矩陣");  // EN: Execute a statement: throw runtime_error("僅支援 2×2 和 3×3 矩陣");.
}  // EN: Structure delimiter for a block or scope.

// 替換第 j 行
Matrix replaceColumn(const Matrix& A, const Vector& b, int j) {  // EN: Execute line: Matrix replaceColumn(const Matrix& A, const Vector& b, int j) {.
    Matrix Aj = A;  // EN: Execute a statement: Matrix Aj = A;.
    for (size_t i = 0; i < A.size(); i++) {  // EN: Loop control flow: for (size_t i = 0; i < A.size(); i++) {.
        Aj[i][j] = b[i];  // EN: Execute a statement: Aj[i][j] = b[i];.
    }  // EN: Structure delimiter for a block or scope.
    return Aj;  // EN: Return from the current function: return Aj;.
}  // EN: Structure delimiter for a block or scope.

// 克萊姆法則
Vector cramersRule(const Matrix& A, const Vector& b) {  // EN: Execute line: Vector cramersRule(const Matrix& A, const Vector& b) {.
    int n = A.size();  // EN: Execute a statement: int n = A.size();.
    double detA = determinant(A);  // EN: Execute a statement: double detA = determinant(A);.

    if (abs(detA) < 1e-10) {  // EN: Conditional control flow: if (abs(detA) < 1e-10) {.
        throw runtime_error("矩陣奇異");  // EN: Execute a statement: throw runtime_error("矩陣奇異");.
    }  // EN: Structure delimiter for a block or scope.

    Vector x(n);  // EN: Execute a statement: Vector x(n);.
    for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
        Matrix Aj = replaceColumn(A, b, j);  // EN: Execute a statement: Matrix Aj = replaceColumn(A, b, j);.
        x[j] = determinant(Aj) / detA;  // EN: Execute a statement: x[j] = determinant(Aj) / detA;.
    }  // EN: Structure delimiter for a block or scope.
    return x;  // EN: Return from the current function: return x;.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    printSeparator("克萊姆法則示範 (C++)");  // EN: Execute a statement: printSeparator("克萊姆法則示範 (C++)");.

    // ========================================
    // 1. 2×2 系統
    // ========================================
    printSeparator("1. 2×2 系統");  // EN: Execute a statement: printSeparator("1. 2×2 系統");.

    Matrix A2 = {{2, 3}, {4, 5}};  // EN: Execute a statement: Matrix A2 = {{2, 3}, {4, 5}};.
    Vector b2 = {8, 14};  // EN: Execute a statement: Vector b2 = {8, 14};.

    cout << "方程組：" << endl;  // EN: Execute a statement: cout << "方程組：" << endl;.
    cout << "  2x + 3y = 8" << endl;  // EN: Execute a statement: cout << " 2x + 3y = 8" << endl;.
    cout << "  4x + 5y = 14" << endl;  // EN: Execute a statement: cout << " 4x + 5y = 14" << endl;.

    printMatrix("\nA", A2);  // EN: Execute a statement: printMatrix("\nA", A2);.
    printVector("b", b2);  // EN: Execute a statement: printVector("b", b2);.

    double detA2 = determinant(A2);  // EN: Execute a statement: double detA2 = determinant(A2);.
    cout << "\ndet(A) = " << detA2 << endl;  // EN: Execute a statement: cout << "\ndet(A) = " << detA2 << endl;.

    Vector x2 = cramersRule(A2, b2);  // EN: Execute a statement: Vector x2 = cramersRule(A2, b2);.

    for (int j = 0; j < 2; j++) {  // EN: Loop control flow: for (int j = 0; j < 2; j++) {.
        Matrix Aj = replaceColumn(A2, b2, j);  // EN: Execute a statement: Matrix Aj = replaceColumn(A2, b2, j);.
        double detAj = determinant(Aj);  // EN: Execute a statement: double detAj = determinant(Aj);.
        cout << "\nA" << j+1 << "（第 " << j+1 << " 行換成 b）：" << endl;  // EN: Execute a statement: cout << "\nA" << j+1 << "（第 " << j+1 << " 行換成 b）：" << endl;.
        printMatrix("", Aj);  // EN: Execute a statement: printMatrix("", Aj);.
        cout << "det(A" << j+1 << ") = " << detAj << endl;  // EN: Execute a statement: cout << "det(A" << j+1 << ") = " << detAj << endl;.
        cout << "x" << j+1 << " = " << x2[j] << endl;  // EN: Execute a statement: cout << "x" << j+1 << " = " << x2[j] << endl;.
    }  // EN: Structure delimiter for a block or scope.

    cout << "\n解：x = " << x2[0] << ", y = " << x2[1] << endl;  // EN: Execute a statement: cout << "\n解：x = " << x2[0] << ", y = " << x2[1] << endl;.

    // ========================================
    // 2. 3×3 系統
    // ========================================
    printSeparator("2. 3×3 系統");  // EN: Execute a statement: printSeparator("2. 3×3 系統");.

    Matrix A3 = {  // EN: Execute line: Matrix A3 = {.
        {2, 1, -1},  // EN: Execute line: {2, 1, -1},.
        {-3, -1, 2},  // EN: Execute line: {-3, -1, 2},.
        {-2, 1, 2}  // EN: Execute line: {-2, 1, 2}.
    };  // EN: Structure delimiter for a block or scope.
    Vector b3 = {8, -11, -3};  // EN: Execute a statement: Vector b3 = {8, -11, -3};.

    cout << "方程組：" << endl;  // EN: Execute a statement: cout << "方程組：" << endl;.
    cout << "   2x +  y -  z =  8" << endl;  // EN: Execute a statement: cout << " 2x + y - z = 8" << endl;.
    cout << "  -3x -  y + 2z = -11" << endl;  // EN: Execute a statement: cout << " -3x - y + 2z = -11" << endl;.
    cout << "  -2x +  y + 2z = -3" << endl;  // EN: Execute a statement: cout << " -2x + y + 2z = -3" << endl;.

    printMatrix("\nA", A3);  // EN: Execute a statement: printMatrix("\nA", A3);.
    printVector("b", b3);  // EN: Execute a statement: printVector("b", b3);.

    Vector x3 = cramersRule(A3, b3);  // EN: Execute a statement: Vector x3 = cramersRule(A3, b3);.

    cout << "\n解：x = " << x3[0] << ", y = " << x3[1] << ", z = " << x3[2] << endl;  // EN: Execute a statement: cout << "\n解：x = " << x3[0] << ", y = " << x3[1] << ", z = " << x3[2] <….

    // 驗證
    cout << "\n驗證：" << endl;  // EN: Execute a statement: cout << "\n驗證：" << endl;.
    cout << "  2(" << x3[0] << ") + (" << x3[1] << ") - (" << x3[2] << ") = "  // EN: Execute line: cout << " 2(" << x3[0] << ") + (" << x3[1] << ") - (" << x3[2] << ") = ".
         << 2*x3[0] + x3[1] - x3[2] << endl;  // EN: Execute a statement: << 2*x3[0] + x3[1] - x3[2] << endl;.
    cout << "  -3(" << x3[0] << ") - (" << x3[1] << ") + 2(" << x3[2] << ") = "  // EN: Execute line: cout << " -3(" << x3[0] << ") - (" << x3[1] << ") + 2(" << x3[2] << ") ….
         << -3*x3[0] - x3[1] + 2*x3[2] << endl;  // EN: Execute a statement: << -3*x3[0] - x3[1] + 2*x3[2] << endl;.
    cout << "  -2(" << x3[0] << ") + (" << x3[1] << ") + 2(" << x3[2] << ") = "  // EN: Execute line: cout << " -2(" << x3[0] << ") + (" << x3[1] << ") + 2(" << x3[2] << ") ….
         << -2*x3[0] + x3[1] + 2*x3[2] << endl;  // EN: Execute a statement: << -2*x3[0] + x3[1] + 2*x3[2] << endl;.

    // 總結
    printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
    cout << R"(  // EN: Execute line: cout << R"(.
克萊姆法則：  // EN: Execute line: 克萊姆法則：.
  xⱼ = det(Aⱼ) / det(A)  // EN: Execute line: xⱼ = det(Aⱼ) / det(A).
  Aⱼ = A 的第 j 行換成 b  // EN: Execute line: Aⱼ = A 的第 j 行換成 b.

適用條件：  // EN: Execute line: 適用條件：.
  - det(A) ≠ 0  // EN: Execute line: - det(A) ≠ 0.
  - 方陣系統  // EN: Execute line: - 方陣系統.

時間複雜度：O(n! × n)  // EN: Execute line: 時間複雜度：O(n! × n).
實際應用：使用 LU 分解更有效率  // EN: Execute line: 實際應用：使用 LU 分解更有效率.
)" << endl;  // EN: Execute a statement: )" << endl;.

    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.
    cout << "示範完成！" << endl;  // EN: Execute a statement: cout << "示範完成！" << endl;.
    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
