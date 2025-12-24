/**
 * QR 分解 (QR Decomposition)
 *
 * 編譯：g++ -std=c++17 -O2 qr_decomposition.cpp -o qr_decomposition
 * 執行：./qr_decomposition
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

void printSeparator(const string& title) {
    cout << endl;
    cout << string(60, '=') << endl;
    cout << title << endl;
    cout << string(60, '=') << endl;
}

void printVector(const string& name, const vector<double>& v) {
    cout << name << " = [";
    for (size_t i = 0; i < v.size(); i++) {
        cout << fixed << setprecision(4) << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}

void printMatrix(const string& name, const vector<vector<double>>& M) {
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

// 基本向量運算
double dotProduct(const vector<double>& x, const vector<double>& y) {
    double result = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        result += x[i] * y[i];
    }
    return result;
}

double vectorNorm(const vector<double>& x) {
    return sqrt(dotProduct(x, x));
}

vector<double> scalarMultiply(double c, const vector<double>& x) {
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        result[i] = c * x[i];
    }
    return result;
}

vector<double> vectorSubtract(const vector<double>& x, const vector<double>& y) {
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        result[i] = x[i] - y[i];
    }
    return result;
}

// 取得矩陣的第 j 行（column）
vector<double> getColumn(const vector<vector<double>>& A, int j) {
    vector<double> col(A.size());
    for (size_t i = 0; i < A.size(); i++) {
        col[i] = A[i][j];
    }
    return col;
}

// Gram-Schmidt QR 分解
pair<vector<vector<double>>, vector<vector<double>>>
qrDecomposition(const vector<vector<double>>& A) {
    int m = A.size();
    int n = A[0].size();

    // Q: m×n, R: n×n
    vector<vector<double>> Q(m, vector<double>(n, 0.0));
    vector<vector<double>> R(n, vector<double>(n, 0.0));

    for (int j = 0; j < n; j++) {
        // 取得 A 的第 j 行
        vector<double> v = getColumn(A, j);

        // 減去前面所有 q 向量的投影
        for (int i = 0; i < j; i++) {
            vector<double> qi = getColumn(Q, i);
            R[i][j] = dotProduct(qi, getColumn(A, j));
            vector<double> proj = scalarMultiply(R[i][j], qi);
            v = vectorSubtract(v, proj);
        }

        // 標準化
        R[j][j] = vectorNorm(v);

        if (R[j][j] > 1e-10) {
            for (int i = 0; i < m; i++) {
                Q[i][j] = v[i] / R[j][j];
            }
        }
    }

    return {Q, R};
}

// 回代法解上三角方程組 Rx = b
vector<double> solveUpperTriangular(const vector<vector<double>>& R,
                                     const vector<double>& b) {
    int n = b.size();
    vector<double> x(n, 0.0);

    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= R[i][j] * x[j];
        }
        x[i] /= R[i][i];
    }

    return x;
}

// 用 QR 分解解最小平方問題
vector<double> qrLeastSquares(const vector<vector<double>>& A,
                               const vector<double>& b) {
    auto [Q, R] = qrDecomposition(A);

    // 計算 Qᵀb
    int n = Q[0].size();
    vector<double> Qt_b(n, 0.0);
    for (int j = 0; j < n; j++) {
        vector<double> qj = getColumn(Q, j);
        Qt_b[j] = dotProduct(qj, b);
    }

    // 解 Rx = Qᵀb
    return solveUpperTriangular(R, Qt_b);
}

// 矩陣乘法
vector<vector<double>> matrixMultiply(const vector<vector<double>>& A,
                                       const vector<vector<double>>& B) {
    int m = A.size();
    int k = B.size();
    int n = B[0].size();

    vector<vector<double>> result(m, vector<double>(n, 0.0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                result[i][j] += A[i][p] * B[p][j];
            }
        }
    }
    return result;
}

// 矩陣轉置
vector<vector<double>> transpose(const vector<vector<double>>& A) {
    int m = A.size();
    int n = A[0].size();
    vector<vector<double>> result(n, vector<double>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

int main() {
    printSeparator("QR 分解示範 (C++)");

    // ========================================
    // 1. 基本 QR 分解
    // ========================================
    printSeparator("1. 基本 QR 分解");

    vector<vector<double>> A = {
        {1.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };

    cout << "輸入矩陣 A：" << endl;
    printMatrix("A", A);

    auto [Q, R] = qrDecomposition(A);

    cout << "\nQR 分解結果：" << endl;
    printMatrix("Q", Q);
    printMatrix("\nR", R);

    // 驗證 QᵀQ = I
    auto QT = transpose(Q);
    auto QTQ = matrixMultiply(QT, Q);
    cout << "\n驗證 QᵀQ = I：" << endl;
    printMatrix("QᵀQ", QTQ);

    // 驗證 A = QR
    auto QR_result = matrixMultiply(Q, R);
    cout << "\n驗證 A = QR：" << endl;
    printMatrix("QR", QR_result);

    // ========================================
    // 2. 用 QR 解最小平方
    // ========================================
    printSeparator("2. 用 QR 解最小平方");

    // 數據
    vector<double> t = {0.0, 1.0, 2.0};
    vector<double> b = {1.0, 3.0, 4.0};

    cout << "數據點：" << endl;
    for (size_t i = 0; i < t.size(); i++) {
        cout << "  (" << t[i] << ", " << b[i] << ")" << endl;
    }

    // 設計矩陣
    vector<vector<double>> A_ls(t.size(), vector<double>(2));
    for (size_t i = 0; i < t.size(); i++) {
        A_ls[i][0] = 1.0;
        A_ls[i][1] = t[i];
    }

    cout << "\n設計矩陣 A：" << endl;
    printMatrix("A", A_ls);
    printVector("觀測值 b", b);

    // QR 分解
    auto [Q_ls, R_ls] = qrDecomposition(A_ls);
    printMatrix("\nQ", Q_ls);
    printMatrix("R", R_ls);

    // 解最小平方
    vector<double> x = qrLeastSquares(A_ls, b);
    printVector("\n解 x", x);

    cout << "\n最佳直線：y = " << fixed << setprecision(4)
         << x[0] << " + " << x[1] << "t" << endl;

    // ========================================
    // 3. 3×3 矩陣的 QR 分解
    // ========================================
    printSeparator("3. 3×3 矩陣的 QR 分解");

    vector<vector<double>> A2 = {
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    };

    cout << "輸入矩陣 A：" << endl;
    printMatrix("A", A2);

    auto [Q2, R2] = qrDecomposition(A2);

    cout << "\nQR 分解結果：" << endl;
    printMatrix("Q", Q2);
    printMatrix("\nR", R2);

    // 總結
    printSeparator("總結");
    cout << R"(
QR 分解核心：

1. A = QR
   - Q: 標準正交矩陣 (QᵀQ = I)
   - R: 上三角矩陣

2. Gram-Schmidt 演算法：
   - 對 A 的行向量正交化得到 Q
   - R 的元素是投影係數

3. 用 QR 解最小平方：
   min ‖Ax - b‖²
   → Rx = Qᵀb

4. 優勢：
   - 比正規方程更穩定
   - 避免計算 AᵀA
)" << endl;

    cout << string(60, '=') << endl;
    cout << "示範完成！" << endl;
    cout << string(60, '=') << endl;

    return 0;
}
