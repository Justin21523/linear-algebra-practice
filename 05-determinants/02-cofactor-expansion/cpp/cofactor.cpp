/**
 * 餘因子展開 (Cofactor Expansion)
 *
 * 編譯：g++ -std=c++17 -O2 cofactor.cpp -o cofactor
 * 執行：./cofactor
 */

#include <iostream>  // EN: Include a header dependency: #include <iostream>.
#include <vector>  // EN: Include a header dependency: #include <vector>.
#include <cmath>  // EN: Include a header dependency: #include <cmath>.
#include <iomanip>  // EN: Include a header dependency: #include <iomanip>.

using namespace std;  // EN: Execute a statement: using namespace std;.
using Matrix = vector<vector<double>>;  // EN: Execute a statement: using Matrix = vector<vector<double>>;.

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

// 取得子矩陣
Matrix getMinorMatrix(const Matrix& A, int row, int col) {  // EN: Execute line: Matrix getMinorMatrix(const Matrix& A, int row, int col) {.
    int n = A.size();  // EN: Execute a statement: int n = A.size();.
    Matrix sub(n - 1, vector<double>(n - 1));  // EN: Execute a statement: Matrix sub(n - 1, vector<double>(n - 1));.
    int si = 0;  // EN: Execute a statement: int si = 0;.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        if (i == row) continue;  // EN: Conditional control flow: if (i == row) continue;.
        int sj = 0;  // EN: Execute a statement: int sj = 0;.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            if (j == col) continue;  // EN: Conditional control flow: if (j == col) continue;.
            sub[si][sj] = A[i][j];  // EN: Execute a statement: sub[si][sj] = A[i][j];.
            sj++;  // EN: Execute a statement: sj++;.
        }  // EN: Structure delimiter for a block or scope.
        si++;  // EN: Execute a statement: si++;.
    }  // EN: Structure delimiter for a block or scope.
    return sub;  // EN: Return from the current function: return sub;.
}  // EN: Structure delimiter for a block or scope.

// 行列式（遞迴餘因子展開）
double determinant(const Matrix& A) {  // EN: Execute line: double determinant(const Matrix& A) {.
    int n = A.size();  // EN: Execute a statement: int n = A.size();.
    if (n == 1) return A[0][0];  // EN: Conditional control flow: if (n == 1) return A[0][0];.
    if (n == 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Conditional control flow: if (n == 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];.

    double det = 0.0;  // EN: Execute a statement: double det = 0.0;.
    for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
        Matrix sub = getMinorMatrix(A, 0, j);  // EN: Execute a statement: Matrix sub = getMinorMatrix(A, 0, j);.
        double sign = (j % 2 == 0) ? 1.0 : -1.0;  // EN: Execute a statement: double sign = (j % 2 == 0) ? 1.0 : -1.0;.
        det += sign * A[0][j] * determinant(sub);  // EN: Execute a statement: det += sign * A[0][j] * determinant(sub);.
    }  // EN: Structure delimiter for a block or scope.
    return det;  // EN: Return from the current function: return det;.
}  // EN: Structure delimiter for a block or scope.

// 子行列式
double minor(const Matrix& A, int i, int j) {  // EN: Execute line: double minor(const Matrix& A, int i, int j) {.
    return determinant(getMinorMatrix(A, i, j));  // EN: Return from the current function: return determinant(getMinorMatrix(A, i, j));.
}  // EN: Structure delimiter for a block or scope.

// 餘因子
double cofactor(const Matrix& A, int i, int j) {  // EN: Execute line: double cofactor(const Matrix& A, int i, int j) {.
    double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;  // EN: Execute a statement: double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;.
    return sign * minor(A, i, j);  // EN: Return from the current function: return sign * minor(A, i, j);.
}  // EN: Structure delimiter for a block or scope.

// 餘因子矩陣
Matrix cofactorMatrix(const Matrix& A) {  // EN: Execute line: Matrix cofactorMatrix(const Matrix& A) {.
    int n = A.size();  // EN: Execute a statement: int n = A.size();.
    Matrix C(n, vector<double>(n));  // EN: Execute a statement: Matrix C(n, vector<double>(n));.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            C[i][j] = cofactor(A, i, j);  // EN: Execute a statement: C[i][j] = cofactor(A, i, j);.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return C;  // EN: Return from the current function: return C;.
}  // EN: Structure delimiter for a block or scope.

// 轉置
Matrix transpose(const Matrix& A) {  // EN: Execute line: Matrix transpose(const Matrix& A) {.
    int n = A.size();  // EN: Execute a statement: int n = A.size();.
    Matrix T(n, vector<double>(n));  // EN: Execute a statement: Matrix T(n, vector<double>(n));.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            T[j][i] = A[i][j];  // EN: Execute a statement: T[j][i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return T;  // EN: Return from the current function: return T;.
}  // EN: Structure delimiter for a block or scope.

// 伴隨矩陣
Matrix adjugate(const Matrix& A) {  // EN: Execute line: Matrix adjugate(const Matrix& A) {.
    return transpose(cofactorMatrix(A));  // EN: Return from the current function: return transpose(cofactorMatrix(A));.
}  // EN: Structure delimiter for a block or scope.

// 逆矩陣
Matrix inverse(const Matrix& A) {  // EN: Execute line: Matrix inverse(const Matrix& A) {.
    double det = determinant(A);  // EN: Execute a statement: double det = determinant(A);.
    Matrix adj = adjugate(A);  // EN: Execute a statement: Matrix adj = adjugate(A);.
    int n = A.size();  // EN: Execute a statement: int n = A.size();.
    Matrix inv(n, vector<double>(n));  // EN: Execute a statement: Matrix inv(n, vector<double>(n));.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            inv[i][j] = adj[i][j] / det;  // EN: Execute a statement: inv[i][j] = adj[i][j] / det;.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return inv;  // EN: Return from the current function: return inv;.
}  // EN: Structure delimiter for a block or scope.

// 矩陣乘法
Matrix multiply(const Matrix& A, const Matrix& B) {  // EN: Execute line: Matrix multiply(const Matrix& A, const Matrix& B) {.
    int n = A.size();  // EN: Execute a statement: int n = A.size();.
    Matrix C(n, vector<double>(n, 0.0));  // EN: Execute a statement: Matrix C(n, vector<double>(n, 0.0));.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            for (int k = 0; k < n; k++) {  // EN: Loop control flow: for (int k = 0; k < n; k++) {.
                C[i][j] += A[i][k] * B[k][j];  // EN: Execute a statement: C[i][j] += A[i][k] * B[k][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return C;  // EN: Return from the current function: return C;.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    printSeparator("餘因子展開示範 (C++)");  // EN: Execute a statement: printSeparator("餘因子展開示範 (C++)");.

    // ========================================
    // 1. 子行列式與餘因子
    // ========================================
    printSeparator("1. 子行列式與餘因子");  // EN: Execute a statement: printSeparator("1. 子行列式與餘因子");.

    Matrix A = {  // EN: Execute line: Matrix A = {.
        {1, 2, 3},  // EN: Execute line: {1, 2, 3},.
        {4, 5, 6},  // EN: Execute line: {4, 5, 6},.
        {7, 8, 9}  // EN: Execute line: {7, 8, 9}.
    };  // EN: Structure delimiter for a block or scope.

    printMatrix("A", A);  // EN: Execute a statement: printMatrix("A", A);.

    cout << "\n所有餘因子 Cᵢⱼ：" << endl;  // EN: Execute a statement: cout << "\n所有餘因子 Cᵢⱼ：" << endl;.
    for (int i = 0; i < 3; i++) {  // EN: Loop control flow: for (int i = 0; i < 3; i++) {.
        for (int j = 0; j < 3; j++) {  // EN: Loop control flow: for (int j = 0; j < 3; j++) {.
            cout << "  C" << i+1 << j+1 << " = " << setw(8) << cofactor(A, i, j);  // EN: Execute a statement: cout << " C" << i+1 << j+1 << " = " << setw(8) << cofactor(A, i, j);.
        }  // EN: Structure delimiter for a block or scope.
        cout << endl;  // EN: Execute a statement: cout << endl;.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 2. 餘因子展開
    // ========================================
    printSeparator("2. 餘因子展開計算行列式");  // EN: Execute a statement: printSeparator("2. 餘因子展開計算行列式");.

    cout << "沿第一列展開：" << endl;  // EN: Execute a statement: cout << "沿第一列展開：" << endl;.
    cout << "det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃" << endl;  // EN: Execute a statement: cout << "det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃" << endl;.
    cout << "       = " << A[0][0] << "×" << cofactor(A, 0, 0)  // EN: Execute line: cout << " = " << A[0][0] << "×" << cofactor(A, 0, 0).
         << " + " << A[0][1] << "×" << cofactor(A, 0, 1)  // EN: Execute line: << " + " << A[0][1] << "×" << cofactor(A, 0, 1).
         << " + " << A[0][2] << "×" << cofactor(A, 0, 2) << endl;  // EN: Execute a statement: << " + " << A[0][2] << "×" << cofactor(A, 0, 2) << endl;.
    cout << "       = " << determinant(A) << endl;  // EN: Execute a statement: cout << " = " << determinant(A) << endl;.

    // ========================================
    // 3. 餘因子矩陣與伴隨矩陣
    // ========================================
    printSeparator("3. 餘因子矩陣與伴隨矩陣");  // EN: Execute a statement: printSeparator("3. 餘因子矩陣與伴隨矩陣");.

    Matrix B = {  // EN: Execute line: Matrix B = {.
        {2, 1, 3},  // EN: Execute line: {2, 1, 3},.
        {1, 0, 2},  // EN: Execute line: {1, 0, 2},.
        {4, 1, 5}  // EN: Execute line: {4, 1, 5}.
    };  // EN: Structure delimiter for a block or scope.

    printMatrix("A", B);  // EN: Execute a statement: printMatrix("A", B);.
    cout << "\ndet(A) = " << determinant(B) << endl;  // EN: Execute a statement: cout << "\ndet(A) = " << determinant(B) << endl;.

    Matrix C = cofactorMatrix(B);  // EN: Execute a statement: Matrix C = cofactorMatrix(B);.
    printMatrix("\n餘因子矩陣 C", C);  // EN: Execute a statement: printMatrix("\n餘因子矩陣 C", C);.

    Matrix adj = adjugate(B);  // EN: Execute a statement: Matrix adj = adjugate(B);.
    printMatrix("\n伴隨矩陣 adj(A) = Cᵀ", adj);  // EN: Execute a statement: printMatrix("\n伴隨矩陣 adj(A) = Cᵀ", adj);.

    // ========================================
    // 4. 用伴隨矩陣求逆矩陣
    // ========================================
    printSeparator("4. 用伴隨矩陣求逆矩陣");  // EN: Execute a statement: printSeparator("4. 用伴隨矩陣求逆矩陣");.

    cout << "A⁻¹ = adj(A) / det(A)" << endl;  // EN: Execute a statement: cout << "A⁻¹ = adj(A) / det(A)" << endl;.

    Matrix B_inv = inverse(B);  // EN: Execute a statement: Matrix B_inv = inverse(B);.
    printMatrix("\nA⁻¹", B_inv);  // EN: Execute a statement: printMatrix("\nA⁻¹", B_inv);.

    // 驗證
    Matrix I = multiply(B, B_inv);  // EN: Execute a statement: Matrix I = multiply(B, B_inv);.
    printMatrix("\n驗證 A × A⁻¹", I);  // EN: Execute a statement: printMatrix("\n驗證 A × A⁻¹", I);.

    // ========================================
    // 5. 2×2 特例
    // ========================================
    printSeparator("5. 2×2 伴隨矩陣公式");  // EN: Execute a statement: printSeparator("5. 2×2 伴隨矩陣公式");.

    Matrix A2 = {{3, 4}, {5, 6}};  // EN: Execute a statement: Matrix A2 = {{3, 4}, {5, 6}};.
    printMatrix("A", A2);  // EN: Execute a statement: printMatrix("A", A2);.

    cout << "\n對於 [[a,b],[c,d]]:" << endl;  // EN: Execute a statement: cout << "\n對於 [[a,b],[c,d]]:" << endl;.
    cout << "adj(A) = [[d,-b],[-c,a]] = [[" << A2[1][1] << "," << -A2[0][1]  // EN: Execute line: cout << "adj(A) = [[d,-b],[-c,a]] = [[" << A2[1][1] << "," << -A2[0][1].
         << "],[" << -A2[1][0] << "," << A2[0][0] << "]]" << endl;  // EN: Execute a statement: << "],[" << -A2[1][0] << "," << A2[0][0] << "]]" << endl;.

    Matrix adj2 = adjugate(A2);  // EN: Execute a statement: Matrix adj2 = adjugate(A2);.
    printMatrix("\n計算得到的 adj(A)", adj2);  // EN: Execute a statement: printMatrix("\n計算得到的 adj(A)", adj2);.

    // 總結
    printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
    cout << R"(  // EN: Execute line: cout << R"(.
餘因子展開公式：  // EN: Execute line: 餘因子展開公式：.
  Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ  // EN: Execute line: Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ.
  det(A) = Σⱼ aᵢⱼ Cᵢⱼ  // EN: Execute line: det(A) = Σⱼ aᵢⱼ Cᵢⱼ.

伴隨矩陣：  // EN: Execute line: 伴隨矩陣：.
  adj(A) = Cᵀ  // EN: Execute line: adj(A) = Cᵀ.

逆矩陣：  // EN: Execute line: 逆矩陣：.
  A⁻¹ = adj(A) / det(A)  // EN: Execute line: A⁻¹ = adj(A) / det(A).

時間複雜度：O(n!)  // EN: Execute line: 時間複雜度：O(n!).
)" << endl;  // EN: Execute a statement: )" << endl;.

    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.
    cout << "示範完成！" << endl;  // EN: Execute a statement: cout << "示範完成！" << endl;.
    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
