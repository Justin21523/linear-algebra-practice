/**
 * 行列式的性質 (Determinant Properties)
 *
 * 編譯：g++ -std=c++17 -O2 determinant.cpp -o determinant
 * 執行：./determinant
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

// 2×2 行列式
double det2x2(const vector<vector<double>>& A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

// 3×3 行列式
double det3x3(const vector<vector<double>>& A) {
    double a = A[0][0], b = A[0][1], c = A[0][2];
    double d = A[1][0], e = A[1][1], f = A[1][2];
    double g = A[2][0], h = A[2][1], i = A[2][2];

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

// n×n 行列式（列運算化為上三角）
double detNxN(vector<vector<double>> M) {
    int n = M.size();
    int sign = 1;

    for (int col = 0; col < n; col++) {
        // 找主元
        int pivotRow = -1;
        for (int row = col; row < n; row++) {
            if (abs(M[row][col]) > 1e-10) {
                pivotRow = row;
                break;
            }
        }

        if (pivotRow == -1) return 0.0;

        // 列交換
        if (pivotRow != col) {
            swap(M[col], M[pivotRow]);
            sign *= -1;
        }

        // 消去
        for (int row = col + 1; row < n; row++) {
            double factor = M[row][col] / M[col][col];
            for (int j = col; j < n; j++) {
                M[row][j] -= factor * M[col][j];
            }
        }
    }

    double det = sign;
    for (int i = 0; i < n; i++) {
        det *= M[i][i];
    }

    return det;
}

// 矩陣乘法
vector<vector<double>> matrixMultiply(const vector<vector<double>>& A,
                                       const vector<vector<double>>& B) {
    int m = A.size(), k = B.size(), n = B[0].size();
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
    int m = A.size(), n = A[0].size();
    vector<vector<double>> result(n, vector<double>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

// 純量乘矩陣
vector<vector<double>> scalarMultiply(double c, const vector<vector<double>>& A) {
    vector<vector<double>> result = A;
    for (auto& row : result) {
        for (auto& x : row) {
            x *= c;
        }
    }
    return result;
}

int main() {
    printSeparator("行列式性質示範 (C++)");

    // ========================================
    // 1. 基本計算
    // ========================================
    printSeparator("1. 基本行列式計算");

    vector<vector<double>> A2 = {{3, 8}, {4, 6}};
    printMatrix("A (2×2)", A2);
    cout << "det(A) = " << det2x2(A2) << endl;

    vector<vector<double>> A3 = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 10}
    };
    printMatrix("\nA (3×3)", A3);
    cout << "det(A) = " << det3x3(A3) << endl;

    // ========================================
    // 2. 性質 1：det(I) = 1
    // ========================================
    printSeparator("2. 性質 1：det(I) = 1");

    vector<vector<double>> I3 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    printMatrix("I₃", I3);
    cout << "det(I₃) = " << det3x3(I3) << endl;

    // ========================================
    // 3. 性質 2：列交換變號
    // ========================================
    printSeparator("3. 性質 2：列交換變號");

    vector<vector<double>> A = {{1, 2}, {3, 4}};
    printMatrix("A", A);
    cout << "det(A) = " << det2x2(A) << endl;

    vector<vector<double>> A_swap = {{3, 4}, {1, 2}};
    printMatrix("\nA（交換列）", A_swap);
    cout << "det(交換後) = " << det2x2(A_swap) << endl;
    cout << "驗證：變號 ✓" << endl;

    // ========================================
    // 4. 乘積公式
    // ========================================
    printSeparator("4. 乘積公式：det(AB) = det(A)·det(B)");

    A = {{1, 2}, {3, 4}};
    vector<vector<double>> B = {{5, 6}, {7, 8}};
    auto AB = matrixMultiply(A, B);

    printMatrix("A", A);
    printMatrix("B", B);
    printMatrix("AB", AB);

    double detA = det2x2(A);
    double detB = det2x2(B);
    double detAB = det2x2(AB);

    cout << "\ndet(A) = " << detA << endl;
    cout << "det(B) = " << detB << endl;
    cout << "det(A)·det(B) = " << detA * detB << endl;
    cout << "det(AB) = " << detAB << endl;

    // ========================================
    // 5. 轉置公式
    // ========================================
    printSeparator("5. 轉置公式：det(Aᵀ) = det(A)");

    A3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
    auto AT = transpose(A3);

    printMatrix("A", A3);
    printMatrix("Aᵀ", AT);

    cout << "\ndet(A) = " << det3x3(A3) << endl;
    cout << "det(Aᵀ) = " << det3x3(AT) << endl;

    // ========================================
    // 6. 純量乘法
    // ========================================
    printSeparator("6. 純量乘法：det(cA) = cⁿ·det(A)");

    A = {{1, 2}, {3, 4}};
    double c = 2;
    auto cA = scalarMultiply(c, A);

    printMatrix("A (2×2)", A);
    cout << "c = " << c << endl;
    printMatrix("cA", cA);

    detA = det2x2(A);
    double detcA = det2x2(cA);
    int n = 2;

    cout << "\ndet(A) = " << detA << endl;
    cout << "cⁿ·det(A) = " << c << "² × " << detA << " = " << pow(c, n) * detA << endl;
    cout << "det(cA) = " << detcA << endl;

    // ========================================
    // 7. 上三角矩陣
    // ========================================
    printSeparator("7. 上三角矩陣：det = 對角線乘積");

    vector<vector<double>> U = {{2, 3, 1}, {0, 4, 5}, {0, 0, 6}};
    printMatrix("U（上三角）", U);
    cout << "對角線乘積：2 × 4 × 6 = " << 2 * 4 * 6 << endl;
    cout << "det(U) = " << det3x3(U) << endl;

    // ========================================
    // 8. 奇異矩陣
    // ========================================
    printSeparator("8. 奇異矩陣：det(A) = 0");

    vector<vector<double>> A_singular = {{1, 2}, {2, 4}};
    printMatrix("A（列成比例）", A_singular);
    cout << "det(A) = " << det2x2(A_singular) << endl;
    cout << "此矩陣不可逆" << endl;

    // 總結
    printSeparator("總結");
    cout << R"(
行列式三大性質：
1. det(I) = 1
2. 列交換 → det 變號
3. 對單列線性

重要公式：
- det(AB) = det(A)·det(B)
- det(Aᵀ) = det(A)
- det(A⁻¹) = 1/det(A)
- det(cA) = cⁿ·det(A)
)" << endl;

    cout << string(60, '=') << endl;
    cout << "示範完成！" << endl;
    cout << string(60, '=') << endl;

    return 0;
}
