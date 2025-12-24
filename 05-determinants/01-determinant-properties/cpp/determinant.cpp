/**
 * 行列式的性質 (Determinant Properties)
 *
 * 編譯：g++ -std=c++17 -O2 determinant.cpp -o determinant
 * 執行：./determinant
 */

#include <iostream>  // EN: Include a header dependency: #include <iostream>.
#include <vector>  // EN: Include a header dependency: #include <vector>.
#include <cmath>  // EN: Include a header dependency: #include <cmath>.
#include <iomanip>  // EN: Include a header dependency: #include <iomanip>.

using namespace std;  // EN: Execute a statement: using namespace std;.

void printSeparator(const string& title) {  // EN: Execute line: void printSeparator(const string& title) {.
    cout << endl;  // EN: Execute a statement: cout << endl;.
    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.
    cout << title << endl;  // EN: Execute a statement: cout << title << endl;.
    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.
}  // EN: Structure delimiter for a block or scope.

void printMatrix(const string& name, const vector<vector<double>>& M) {  // EN: Execute line: void printMatrix(const string& name, const vector<vector<double>>& M) {.
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

// 2×2 行列式
double det2x2(const vector<vector<double>>& A) {  // EN: Execute line: double det2x2(const vector<vector<double>>& A) {.
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Return from the current function: return A[0][0] * A[1][1] - A[0][1] * A[1][0];.
}  // EN: Structure delimiter for a block or scope.

// 3×3 行列式
double det3x3(const vector<vector<double>>& A) {  // EN: Execute line: double det3x3(const vector<vector<double>>& A) {.
    double a = A[0][0], b = A[0][1], c = A[0][2];  // EN: Execute a statement: double a = A[0][0], b = A[0][1], c = A[0][2];.
    double d = A[1][0], e = A[1][1], f = A[1][2];  // EN: Execute a statement: double d = A[1][0], e = A[1][1], f = A[1][2];.
    double g = A[2][0], h = A[2][1], i = A[2][2];  // EN: Execute a statement: double g = A[2][0], h = A[2][1], i = A[2][2];.

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);  // EN: Return from the current function: return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);.
}  // EN: Structure delimiter for a block or scope.

// n×n 行列式（列運算化為上三角）
double detNxN(vector<vector<double>> M) {  // EN: Execute line: double detNxN(vector<vector<double>> M) {.
    int n = M.size();  // EN: Execute a statement: int n = M.size();.
    int sign = 1;  // EN: Execute a statement: int sign = 1;.

    for (int col = 0; col < n; col++) {  // EN: Loop control flow: for (int col = 0; col < n; col++) {.
        // 找主元
        int pivotRow = -1;  // EN: Execute a statement: int pivotRow = -1;.
        for (int row = col; row < n; row++) {  // EN: Loop control flow: for (int row = col; row < n; row++) {.
            if (abs(M[row][col]) > 1e-10) {  // EN: Conditional control flow: if (abs(M[row][col]) > 1e-10) {.
                pivotRow = row;  // EN: Execute a statement: pivotRow = row;.
                break;  // EN: Execute a statement: break;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.

        if (pivotRow == -1) return 0.0;  // EN: Conditional control flow: if (pivotRow == -1) return 0.0;.

        // 列交換
        if (pivotRow != col) {  // EN: Conditional control flow: if (pivotRow != col) {.
            swap(M[col], M[pivotRow]);  // EN: Execute a statement: swap(M[col], M[pivotRow]);.
            sign *= -1;  // EN: Execute a statement: sign *= -1;.
        }  // EN: Structure delimiter for a block or scope.

        // 消去
        for (int row = col + 1; row < n; row++) {  // EN: Loop control flow: for (int row = col + 1; row < n; row++) {.
            double factor = M[row][col] / M[col][col];  // EN: Execute a statement: double factor = M[row][col] / M[col][col];.
            for (int j = col; j < n; j++) {  // EN: Loop control flow: for (int j = col; j < n; j++) {.
                M[row][j] -= factor * M[col][j];  // EN: Execute a statement: M[row][j] -= factor * M[col][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    double det = sign;  // EN: Execute a statement: double det = sign;.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        det *= M[i][i];  // EN: Execute a statement: det *= M[i][i];.
    }  // EN: Structure delimiter for a block or scope.

    return det;  // EN: Return from the current function: return det;.
}  // EN: Structure delimiter for a block or scope.

// 矩陣乘法
vector<vector<double>> matrixMultiply(const vector<vector<double>>& A,  // EN: Execute line: vector<vector<double>> matrixMultiply(const vector<vector<double>>& A,.
                                       const vector<vector<double>>& B) {  // EN: Execute line: const vector<vector<double>>& B) {.
    int m = A.size(), k = B.size(), n = B[0].size();  // EN: Execute a statement: int m = A.size(), k = B.size(), n = B[0].size();.
    vector<vector<double>> result(m, vector<double>(n, 0.0));  // EN: Execute a statement: vector<vector<double>> result(m, vector<double>(n, 0.0));.

    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            for (int p = 0; p < k; p++) {  // EN: Loop control flow: for (int p = 0; p < k; p++) {.
                result[i][j] += A[i][p] * B[p][j];  // EN: Execute a statement: result[i][j] += A[i][p] * B[p][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

// 矩陣轉置
vector<vector<double>> transpose(const vector<vector<double>>& A) {  // EN: Execute line: vector<vector<double>> transpose(const vector<vector<double>>& A) {.
    int m = A.size(), n = A[0].size();  // EN: Execute a statement: int m = A.size(), n = A[0].size();.
    vector<vector<double>> result(n, vector<double>(m));  // EN: Execute a statement: vector<vector<double>> result(n, vector<double>(m));.
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[j][i] = A[i][j];  // EN: Execute a statement: result[j][i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

// 純量乘矩陣
vector<vector<double>> scalarMultiply(double c, const vector<vector<double>>& A) {  // EN: Execute line: vector<vector<double>> scalarMultiply(double c, const vector<vector<dou….
    vector<vector<double>> result = A;  // EN: Execute a statement: vector<vector<double>> result = A;.
    for (auto& row : result) {  // EN: Loop control flow: for (auto& row : result) {.
        for (auto& x : row) {  // EN: Loop control flow: for (auto& x : row) {.
            x *= c;  // EN: Execute a statement: x *= c;.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    printSeparator("行列式性質示範 (C++)");  // EN: Execute a statement: printSeparator("行列式性質示範 (C++)");.

    // ========================================
    // 1. 基本計算
    // ========================================
    printSeparator("1. 基本行列式計算");  // EN: Execute a statement: printSeparator("1. 基本行列式計算");.

    vector<vector<double>> A2 = {{3, 8}, {4, 6}};  // EN: Execute a statement: vector<vector<double>> A2 = {{3, 8}, {4, 6}};.
    printMatrix("A (2×2)", A2);  // EN: Execute a statement: printMatrix("A (2×2)", A2);.
    cout << "det(A) = " << det2x2(A2) << endl;  // EN: Execute a statement: cout << "det(A) = " << det2x2(A2) << endl;.

    vector<vector<double>> A3 = {  // EN: Execute line: vector<vector<double>> A3 = {.
        {1, 2, 3},  // EN: Execute line: {1, 2, 3},.
        {4, 5, 6},  // EN: Execute line: {4, 5, 6},.
        {7, 8, 10}  // EN: Execute line: {7, 8, 10}.
    };  // EN: Structure delimiter for a block or scope.
    printMatrix("\nA (3×3)", A3);  // EN: Execute a statement: printMatrix("\nA (3×3)", A3);.
    cout << "det(A) = " << det3x3(A3) << endl;  // EN: Execute a statement: cout << "det(A) = " << det3x3(A3) << endl;.

    // ========================================
    // 2. 性質 1：det(I) = 1
    // ========================================
    printSeparator("2. 性質 1：det(I) = 1");  // EN: Execute a statement: printSeparator("2. 性質 1：det(I) = 1");.

    vector<vector<double>> I3 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};  // EN: Execute a statement: vector<vector<double>> I3 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};.
    printMatrix("I₃", I3);  // EN: Execute a statement: printMatrix("I₃", I3);.
    cout << "det(I₃) = " << det3x3(I3) << endl;  // EN: Execute a statement: cout << "det(I₃) = " << det3x3(I3) << endl;.

    // ========================================
    // 3. 性質 2：列交換變號
    // ========================================
    printSeparator("3. 性質 2：列交換變號");  // EN: Execute a statement: printSeparator("3. 性質 2：列交換變號");.

    vector<vector<double>> A = {{1, 2}, {3, 4}};  // EN: Execute a statement: vector<vector<double>> A = {{1, 2}, {3, 4}};.
    printMatrix("A", A);  // EN: Execute a statement: printMatrix("A", A);.
    cout << "det(A) = " << det2x2(A) << endl;  // EN: Execute a statement: cout << "det(A) = " << det2x2(A) << endl;.

    vector<vector<double>> A_swap = {{3, 4}, {1, 2}};  // EN: Execute a statement: vector<vector<double>> A_swap = {{3, 4}, {1, 2}};.
    printMatrix("\nA（交換列）", A_swap);  // EN: Execute a statement: printMatrix("\nA（交換列）", A_swap);.
    cout << "det(交換後) = " << det2x2(A_swap) << endl;  // EN: Execute a statement: cout << "det(交換後) = " << det2x2(A_swap) << endl;.
    cout << "驗證：變號 ✓" << endl;  // EN: Execute a statement: cout << "驗證：變號 ✓" << endl;.

    // ========================================
    // 4. 乘積公式
    // ========================================
    printSeparator("4. 乘積公式：det(AB) = det(A)·det(B)");  // EN: Execute a statement: printSeparator("4. 乘積公式：det(AB) = det(A)·det(B)");.

    A = {{1, 2}, {3, 4}};  // EN: Execute a statement: A = {{1, 2}, {3, 4}};.
    vector<vector<double>> B = {{5, 6}, {7, 8}};  // EN: Execute a statement: vector<vector<double>> B = {{5, 6}, {7, 8}};.
    auto AB = matrixMultiply(A, B);  // EN: Execute a statement: auto AB = matrixMultiply(A, B);.

    printMatrix("A", A);  // EN: Execute a statement: printMatrix("A", A);.
    printMatrix("B", B);  // EN: Execute a statement: printMatrix("B", B);.
    printMatrix("AB", AB);  // EN: Execute a statement: printMatrix("AB", AB);.

    double detA = det2x2(A);  // EN: Execute a statement: double detA = det2x2(A);.
    double detB = det2x2(B);  // EN: Execute a statement: double detB = det2x2(B);.
    double detAB = det2x2(AB);  // EN: Execute a statement: double detAB = det2x2(AB);.

    cout << "\ndet(A) = " << detA << endl;  // EN: Execute a statement: cout << "\ndet(A) = " << detA << endl;.
    cout << "det(B) = " << detB << endl;  // EN: Execute a statement: cout << "det(B) = " << detB << endl;.
    cout << "det(A)·det(B) = " << detA * detB << endl;  // EN: Execute a statement: cout << "det(A)·det(B) = " << detA * detB << endl;.
    cout << "det(AB) = " << detAB << endl;  // EN: Execute a statement: cout << "det(AB) = " << detAB << endl;.

    // ========================================
    // 5. 轉置公式
    // ========================================
    printSeparator("5. 轉置公式：det(Aᵀ) = det(A)");  // EN: Execute a statement: printSeparator("5. 轉置公式：det(Aᵀ) = det(A)");.

    A3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};  // EN: Execute a statement: A3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};.
    auto AT = transpose(A3);  // EN: Execute a statement: auto AT = transpose(A3);.

    printMatrix("A", A3);  // EN: Execute a statement: printMatrix("A", A3);.
    printMatrix("Aᵀ", AT);  // EN: Execute a statement: printMatrix("Aᵀ", AT);.

    cout << "\ndet(A) = " << det3x3(A3) << endl;  // EN: Execute a statement: cout << "\ndet(A) = " << det3x3(A3) << endl;.
    cout << "det(Aᵀ) = " << det3x3(AT) << endl;  // EN: Execute a statement: cout << "det(Aᵀ) = " << det3x3(AT) << endl;.

    // ========================================
    // 6. 純量乘法
    // ========================================
    printSeparator("6. 純量乘法：det(cA) = cⁿ·det(A)");  // EN: Execute a statement: printSeparator("6. 純量乘法：det(cA) = cⁿ·det(A)");.

    A = {{1, 2}, {3, 4}};  // EN: Execute a statement: A = {{1, 2}, {3, 4}};.
    double c = 2;  // EN: Execute a statement: double c = 2;.
    auto cA = scalarMultiply(c, A);  // EN: Execute a statement: auto cA = scalarMultiply(c, A);.

    printMatrix("A (2×2)", A);  // EN: Execute a statement: printMatrix("A (2×2)", A);.
    cout << "c = " << c << endl;  // EN: Execute a statement: cout << "c = " << c << endl;.
    printMatrix("cA", cA);  // EN: Execute a statement: printMatrix("cA", cA);.

    detA = det2x2(A);  // EN: Execute a statement: detA = det2x2(A);.
    double detcA = det2x2(cA);  // EN: Execute a statement: double detcA = det2x2(cA);.
    int n = 2;  // EN: Execute a statement: int n = 2;.

    cout << "\ndet(A) = " << detA << endl;  // EN: Execute a statement: cout << "\ndet(A) = " << detA << endl;.
    cout << "cⁿ·det(A) = " << c << "² × " << detA << " = " << pow(c, n) * detA << endl;  // EN: Execute a statement: cout << "cⁿ·det(A) = " << c << "² × " << detA << " = " << pow(c, n) * d….
    cout << "det(cA) = " << detcA << endl;  // EN: Execute a statement: cout << "det(cA) = " << detcA << endl;.

    // ========================================
    // 7. 上三角矩陣
    // ========================================
    printSeparator("7. 上三角矩陣：det = 對角線乘積");  // EN: Execute a statement: printSeparator("7. 上三角矩陣：det = 對角線乘積");.

    vector<vector<double>> U = {{2, 3, 1}, {0, 4, 5}, {0, 0, 6}};  // EN: Execute a statement: vector<vector<double>> U = {{2, 3, 1}, {0, 4, 5}, {0, 0, 6}};.
    printMatrix("U（上三角）", U);  // EN: Execute a statement: printMatrix("U（上三角）", U);.
    cout << "對角線乘積：2 × 4 × 6 = " << 2 * 4 * 6 << endl;  // EN: Execute a statement: cout << "對角線乘積：2 × 4 × 6 = " << 2 * 4 * 6 << endl;.
    cout << "det(U) = " << det3x3(U) << endl;  // EN: Execute a statement: cout << "det(U) = " << det3x3(U) << endl;.

    // ========================================
    // 8. 奇異矩陣
    // ========================================
    printSeparator("8. 奇異矩陣：det(A) = 0");  // EN: Execute a statement: printSeparator("8. 奇異矩陣：det(A) = 0");.

    vector<vector<double>> A_singular = {{1, 2}, {2, 4}};  // EN: Execute a statement: vector<vector<double>> A_singular = {{1, 2}, {2, 4}};.
    printMatrix("A（列成比例）", A_singular);  // EN: Execute a statement: printMatrix("A（列成比例）", A_singular);.
    cout << "det(A) = " << det2x2(A_singular) << endl;  // EN: Execute a statement: cout << "det(A) = " << det2x2(A_singular) << endl;.
    cout << "此矩陣不可逆" << endl;  // EN: Execute a statement: cout << "此矩陣不可逆" << endl;.

    // 總結
    printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
    cout << R"(  // EN: Execute line: cout << R"(.
行列式三大性質：  // EN: Execute line: 行列式三大性質：.
1. det(I) = 1  // EN: Execute line: 1. det(I) = 1.
2. 列交換 → det 變號  // EN: Execute line: 2. 列交換 → det 變號.
3. 對單列線性  // EN: Execute line: 3. 對單列線性.

重要公式：  // EN: Execute line: 重要公式：.
- det(AB) = det(A)·det(B)  // EN: Execute line: - det(AB) = det(A)·det(B).
- det(Aᵀ) = det(A)  // EN: Execute line: - det(Aᵀ) = det(A).
- det(A⁻¹) = 1/det(A)  // EN: Execute line: - det(A⁻¹) = 1/det(A).
- det(cA) = cⁿ·det(A)  // EN: Execute line: - det(cA) = cⁿ·det(A).
)" << endl;  // EN: Execute a statement: )" << endl;.

    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.
    cout << "示範完成！" << endl;  // EN: Execute a statement: cout << "示範完成！" << endl;.
    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
