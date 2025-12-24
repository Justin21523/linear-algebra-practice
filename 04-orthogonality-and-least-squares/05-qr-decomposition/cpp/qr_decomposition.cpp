/**
 * QR 分解 (QR Decomposition)
 *
 * 編譯：g++ -std=c++17 -O2 qr_decomposition.cpp -o qr_decomposition
 * 執行：./qr_decomposition
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

void printVector(const string& name, const vector<double>& v) {  // EN: Execute line: void printVector(const string& name, const vector<double>& v) {.
    cout << name << " = [";  // EN: Execute a statement: cout << name << " = [";.
    for (size_t i = 0; i < v.size(); i++) {  // EN: Loop control flow: for (size_t i = 0; i < v.size(); i++) {.
        cout << fixed << setprecision(4) << v[i];  // EN: Execute a statement: cout << fixed << setprecision(4) << v[i];.
        if (i < v.size() - 1) cout << ", ";  // EN: Conditional control flow: if (i < v.size() - 1) cout << ", ";.
    }  // EN: Structure delimiter for a block or scope.
    cout << "]" << endl;  // EN: Execute a statement: cout << "]" << endl;.
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

// 基本向量運算
double dotProduct(const vector<double>& x, const vector<double>& y) {  // EN: Execute line: double dotProduct(const vector<double>& x, const vector<double>& y) {.
    double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
    for (size_t i = 0; i < x.size(); i++) {  // EN: Loop control flow: for (size_t i = 0; i < x.size(); i++) {.
        result += x[i] * y[i];  // EN: Execute a statement: result += x[i] * y[i];.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

double vectorNorm(const vector<double>& x) {  // EN: Execute line: double vectorNorm(const vector<double>& x) {.
    return sqrt(dotProduct(x, x));  // EN: Return from the current function: return sqrt(dotProduct(x, x));.
}  // EN: Structure delimiter for a block or scope.

vector<double> scalarMultiply(double c, const vector<double>& x) {  // EN: Execute line: vector<double> scalarMultiply(double c, const vector<double>& x) {.
    vector<double> result(x.size());  // EN: Execute a statement: vector<double> result(x.size());.
    for (size_t i = 0; i < x.size(); i++) {  // EN: Loop control flow: for (size_t i = 0; i < x.size(); i++) {.
        result[i] = c * x[i];  // EN: Execute a statement: result[i] = c * x[i];.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

vector<double> vectorSubtract(const vector<double>& x, const vector<double>& y) {  // EN: Execute line: vector<double> vectorSubtract(const vector<double>& x, const vector<dou….
    vector<double> result(x.size());  // EN: Execute a statement: vector<double> result(x.size());.
    for (size_t i = 0; i < x.size(); i++) {  // EN: Loop control flow: for (size_t i = 0; i < x.size(); i++) {.
        result[i] = x[i] - y[i];  // EN: Execute a statement: result[i] = x[i] - y[i];.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

// 取得矩陣的第 j 行（column）
vector<double> getColumn(const vector<vector<double>>& A, int j) {  // EN: Execute line: vector<double> getColumn(const vector<vector<double>>& A, int j) {.
    vector<double> col(A.size());  // EN: Execute a statement: vector<double> col(A.size());.
    for (size_t i = 0; i < A.size(); i++) {  // EN: Loop control flow: for (size_t i = 0; i < A.size(); i++) {.
        col[i] = A[i][j];  // EN: Execute a statement: col[i] = A[i][j];.
    }  // EN: Structure delimiter for a block or scope.
    return col;  // EN: Return from the current function: return col;.
}  // EN: Structure delimiter for a block or scope.

// Gram-Schmidt QR 分解
pair<vector<vector<double>>, vector<vector<double>>>  // EN: Execute line: pair<vector<vector<double>>, vector<vector<double>>>.
qrDecomposition(const vector<vector<double>>& A) {  // EN: Execute line: qrDecomposition(const vector<vector<double>>& A) {.
    int m = A.size();  // EN: Execute a statement: int m = A.size();.
    int n = A[0].size();  // EN: Execute a statement: int n = A[0].size();.

    // Q: m×n, R: n×n
    vector<vector<double>> Q(m, vector<double>(n, 0.0));  // EN: Execute a statement: vector<vector<double>> Q(m, vector<double>(n, 0.0));.
    vector<vector<double>> R(n, vector<double>(n, 0.0));  // EN: Execute a statement: vector<vector<double>> R(n, vector<double>(n, 0.0));.

    for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
        // 取得 A 的第 j 行
        vector<double> v = getColumn(A, j);  // EN: Execute a statement: vector<double> v = getColumn(A, j);.

        // 減去前面所有 q 向量的投影
        for (int i = 0; i < j; i++) {  // EN: Loop control flow: for (int i = 0; i < j; i++) {.
            vector<double> qi = getColumn(Q, i);  // EN: Execute a statement: vector<double> qi = getColumn(Q, i);.
            R[i][j] = dotProduct(qi, getColumn(A, j));  // EN: Execute a statement: R[i][j] = dotProduct(qi, getColumn(A, j));.
            vector<double> proj = scalarMultiply(R[i][j], qi);  // EN: Execute a statement: vector<double> proj = scalarMultiply(R[i][j], qi);.
            v = vectorSubtract(v, proj);  // EN: Execute a statement: v = vectorSubtract(v, proj);.
        }  // EN: Structure delimiter for a block or scope.

        // 標準化
        R[j][j] = vectorNorm(v);  // EN: Execute a statement: R[j][j] = vectorNorm(v);.

        if (R[j][j] > 1e-10) {  // EN: Conditional control flow: if (R[j][j] > 1e-10) {.
            for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
                Q[i][j] = v[i] / R[j][j];  // EN: Execute a statement: Q[i][j] = v[i] / R[j][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    return {Q, R};  // EN: Return from the current function: return {Q, R};.
}  // EN: Structure delimiter for a block or scope.

// 回代法解上三角方程組 Rx = b
vector<double> solveUpperTriangular(const vector<vector<double>>& R,  // EN: Execute line: vector<double> solveUpperTriangular(const vector<vector<double>>& R,.
                                     const vector<double>& b) {  // EN: Execute line: const vector<double>& b) {.
    int n = b.size();  // EN: Execute a statement: int n = b.size();.
    vector<double> x(n, 0.0);  // EN: Execute a statement: vector<double> x(n, 0.0);.

    for (int i = n - 1; i >= 0; i--) {  // EN: Loop control flow: for (int i = n - 1; i >= 0; i--) {.
        x[i] = b[i];  // EN: Execute a statement: x[i] = b[i];.
        for (int j = i + 1; j < n; j++) {  // EN: Loop control flow: for (int j = i + 1; j < n; j++) {.
            x[i] -= R[i][j] * x[j];  // EN: Execute a statement: x[i] -= R[i][j] * x[j];.
        }  // EN: Structure delimiter for a block or scope.
        x[i] /= R[i][i];  // EN: Execute a statement: x[i] /= R[i][i];.
    }  // EN: Structure delimiter for a block or scope.

    return x;  // EN: Return from the current function: return x;.
}  // EN: Structure delimiter for a block or scope.

// 用 QR 分解解最小平方問題
vector<double> qrLeastSquares(const vector<vector<double>>& A,  // EN: Execute line: vector<double> qrLeastSquares(const vector<vector<double>>& A,.
                               const vector<double>& b) {  // EN: Execute line: const vector<double>& b) {.
    auto [Q, R] = qrDecomposition(A);  // EN: Execute a statement: auto [Q, R] = qrDecomposition(A);.

    // 計算 Qᵀb
    int n = Q[0].size();  // EN: Execute a statement: int n = Q[0].size();.
    vector<double> Qt_b(n, 0.0);  // EN: Execute a statement: vector<double> Qt_b(n, 0.0);.
    for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
        vector<double> qj = getColumn(Q, j);  // EN: Execute a statement: vector<double> qj = getColumn(Q, j);.
        Qt_b[j] = dotProduct(qj, b);  // EN: Execute a statement: Qt_b[j] = dotProduct(qj, b);.
    }  // EN: Structure delimiter for a block or scope.

    // 解 Rx = Qᵀb
    return solveUpperTriangular(R, Qt_b);  // EN: Return from the current function: return solveUpperTriangular(R, Qt_b);.
}  // EN: Structure delimiter for a block or scope.

// 矩陣乘法
vector<vector<double>> matrixMultiply(const vector<vector<double>>& A,  // EN: Execute line: vector<vector<double>> matrixMultiply(const vector<vector<double>>& A,.
                                       const vector<vector<double>>& B) {  // EN: Execute line: const vector<vector<double>>& B) {.
    int m = A.size();  // EN: Execute a statement: int m = A.size();.
    int k = B.size();  // EN: Execute a statement: int k = B.size();.
    int n = B[0].size();  // EN: Execute a statement: int n = B[0].size();.

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
    int m = A.size();  // EN: Execute a statement: int m = A.size();.
    int n = A[0].size();  // EN: Execute a statement: int n = A[0].size();.
    vector<vector<double>> result(n, vector<double>(m));  // EN: Execute a statement: vector<vector<double>> result(n, vector<double>(m));.
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[j][i] = A[i][j];  // EN: Execute a statement: result[j][i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    printSeparator("QR 分解示範 (C++)");  // EN: Execute a statement: printSeparator("QR 分解示範 (C++)");.

    // ========================================
    // 1. 基本 QR 分解
    // ========================================
    printSeparator("1. 基本 QR 分解");  // EN: Execute a statement: printSeparator("1. 基本 QR 分解");.

    vector<vector<double>> A = {  // EN: Execute line: vector<vector<double>> A = {.
        {1.0, 1.0},  // EN: Execute line: {1.0, 1.0},.
        {1.0, 0.0},  // EN: Execute line: {1.0, 0.0},.
        {0.0, 1.0}  // EN: Execute line: {0.0, 1.0}.
    };  // EN: Structure delimiter for a block or scope.

    cout << "輸入矩陣 A：" << endl;  // EN: Execute a statement: cout << "輸入矩陣 A：" << endl;.
    printMatrix("A", A);  // EN: Execute a statement: printMatrix("A", A);.

    auto [Q, R] = qrDecomposition(A);  // EN: Execute a statement: auto [Q, R] = qrDecomposition(A);.

    cout << "\nQR 分解結果：" << endl;  // EN: Execute a statement: cout << "\nQR 分解結果：" << endl;.
    printMatrix("Q", Q);  // EN: Execute a statement: printMatrix("Q", Q);.
    printMatrix("\nR", R);  // EN: Execute a statement: printMatrix("\nR", R);.

    // 驗證 QᵀQ = I
    auto QT = transpose(Q);  // EN: Execute a statement: auto QT = transpose(Q);.
    auto QTQ = matrixMultiply(QT, Q);  // EN: Execute a statement: auto QTQ = matrixMultiply(QT, Q);.
    cout << "\n驗證 QᵀQ = I：" << endl;  // EN: Execute a statement: cout << "\n驗證 QᵀQ = I：" << endl;.
    printMatrix("QᵀQ", QTQ);  // EN: Execute a statement: printMatrix("QᵀQ", QTQ);.

    // 驗證 A = QR
    auto QR_result = matrixMultiply(Q, R);  // EN: Execute a statement: auto QR_result = matrixMultiply(Q, R);.
    cout << "\n驗證 A = QR：" << endl;  // EN: Execute a statement: cout << "\n驗證 A = QR：" << endl;.
    printMatrix("QR", QR_result);  // EN: Execute a statement: printMatrix("QR", QR_result);.

    // ========================================
    // 2. 用 QR 解最小平方
    // ========================================
    printSeparator("2. 用 QR 解最小平方");  // EN: Execute a statement: printSeparator("2. 用 QR 解最小平方");.

    // 數據
    vector<double> t = {0.0, 1.0, 2.0};  // EN: Execute a statement: vector<double> t = {0.0, 1.0, 2.0};.
    vector<double> b = {1.0, 3.0, 4.0};  // EN: Execute a statement: vector<double> b = {1.0, 3.0, 4.0};.

    cout << "數據點：" << endl;  // EN: Execute a statement: cout << "數據點：" << endl;.
    for (size_t i = 0; i < t.size(); i++) {  // EN: Loop control flow: for (size_t i = 0; i < t.size(); i++) {.
        cout << "  (" << t[i] << ", " << b[i] << ")" << endl;  // EN: Execute a statement: cout << " (" << t[i] << ", " << b[i] << ")" << endl;.
    }  // EN: Structure delimiter for a block or scope.

    // 設計矩陣
    vector<vector<double>> A_ls(t.size(), vector<double>(2));  // EN: Execute a statement: vector<vector<double>> A_ls(t.size(), vector<double>(2));.
    for (size_t i = 0; i < t.size(); i++) {  // EN: Loop control flow: for (size_t i = 0; i < t.size(); i++) {.
        A_ls[i][0] = 1.0;  // EN: Execute a statement: A_ls[i][0] = 1.0;.
        A_ls[i][1] = t[i];  // EN: Execute a statement: A_ls[i][1] = t[i];.
    }  // EN: Structure delimiter for a block or scope.

    cout << "\n設計矩陣 A：" << endl;  // EN: Execute a statement: cout << "\n設計矩陣 A：" << endl;.
    printMatrix("A", A_ls);  // EN: Execute a statement: printMatrix("A", A_ls);.
    printVector("觀測值 b", b);  // EN: Execute a statement: printVector("觀測值 b", b);.

    // QR 分解
    auto [Q_ls, R_ls] = qrDecomposition(A_ls);  // EN: Execute a statement: auto [Q_ls, R_ls] = qrDecomposition(A_ls);.
    printMatrix("\nQ", Q_ls);  // EN: Execute a statement: printMatrix("\nQ", Q_ls);.
    printMatrix("R", R_ls);  // EN: Execute a statement: printMatrix("R", R_ls);.

    // 解最小平方
    vector<double> x = qrLeastSquares(A_ls, b);  // EN: Execute a statement: vector<double> x = qrLeastSquares(A_ls, b);.
    printVector("\n解 x", x);  // EN: Execute a statement: printVector("\n解 x", x);.

    cout << "\n最佳直線：y = " << fixed << setprecision(4)  // EN: Execute line: cout << "\n最佳直線：y = " << fixed << setprecision(4).
         << x[0] << " + " << x[1] << "t" << endl;  // EN: Execute a statement: << x[0] << " + " << x[1] << "t" << endl;.

    // ========================================
    // 3. 3×3 矩陣的 QR 分解
    // ========================================
    printSeparator("3. 3×3 矩陣的 QR 分解");  // EN: Execute a statement: printSeparator("3. 3×3 矩陣的 QR 分解");.

    vector<vector<double>> A2 = {  // EN: Execute line: vector<vector<double>> A2 = {.
        {1.0, 1.0, 0.0},  // EN: Execute line: {1.0, 1.0, 0.0},.
        {1.0, 0.0, 1.0},  // EN: Execute line: {1.0, 0.0, 1.0},.
        {0.0, 1.0, 1.0}  // EN: Execute line: {0.0, 1.0, 1.0}.
    };  // EN: Structure delimiter for a block or scope.

    cout << "輸入矩陣 A：" << endl;  // EN: Execute a statement: cout << "輸入矩陣 A：" << endl;.
    printMatrix("A", A2);  // EN: Execute a statement: printMatrix("A", A2);.

    auto [Q2, R2] = qrDecomposition(A2);  // EN: Execute a statement: auto [Q2, R2] = qrDecomposition(A2);.

    cout << "\nQR 分解結果：" << endl;  // EN: Execute a statement: cout << "\nQR 分解結果：" << endl;.
    printMatrix("Q", Q2);  // EN: Execute a statement: printMatrix("Q", Q2);.
    printMatrix("\nR", R2);  // EN: Execute a statement: printMatrix("\nR", R2);.

    // 總結
    printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
    cout << R"(  // EN: Execute line: cout << R"(.
QR 分解核心：  // EN: Execute line: QR 分解核心：.

1. A = QR  // EN: Execute line: 1. A = QR.
   - Q: 標準正交矩陣 (QᵀQ = I)  // EN: Execute line: - Q: 標準正交矩陣 (QᵀQ = I).
   - R: 上三角矩陣  // EN: Execute line: - R: 上三角矩陣.

2. Gram-Schmidt 演算法：  // EN: Execute line: 2. Gram-Schmidt 演算法：.
   - 對 A 的行向量正交化得到 Q  // EN: Execute line: - 對 A 的行向量正交化得到 Q.
   - R 的元素是投影係數  // EN: Execute line: - R 的元素是投影係數.

3. 用 QR 解最小平方：  // EN: Execute line: 3. 用 QR 解最小平方：.
   min ‖Ax - b‖²  // EN: Execute line: min ‖Ax - b‖².
   → Rx = Qᵀb  // EN: Execute line: → Rx = Qᵀb.

4. 優勢：  // EN: Execute line: 4. 優勢：.
   - 比正規方程更穩定  // EN: Execute line: - 比正規方程更穩定.
   - 避免計算 AᵀA  // EN: Execute line: - 避免計算 AᵀA.
)" << endl;  // EN: Execute a statement: )" << endl;.

    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.
    cout << "示範完成！" << endl;  // EN: Execute a statement: cout << "示範完成！" << endl;.
    cout << string(60, '=') << endl;  // EN: Execute a statement: cout << string(60, '=') << endl;.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
