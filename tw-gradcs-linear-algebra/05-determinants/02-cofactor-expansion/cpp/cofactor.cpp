/**
 * 餘因子展開 (Cofactor Expansion)
 *
 * 編譯：g++ -std=c++17 -O2 cofactor.cpp -o cofactor
 * 執行：./cofactor
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;
using Matrix = vector<vector<double>>;

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

// 取得子矩陣
Matrix getMinorMatrix(const Matrix& A, int row, int col) {
    int n = A.size();
    Matrix sub(n - 1, vector<double>(n - 1));
    int si = 0;
    for (int i = 0; i < n; i++) {
        if (i == row) continue;
        int sj = 0;
        for (int j = 0; j < n; j++) {
            if (j == col) continue;
            sub[si][sj] = A[i][j];
            sj++;
        }
        si++;
    }
    return sub;
}

// 行列式（遞迴餘因子展開）
double determinant(const Matrix& A) {
    int n = A.size();
    if (n == 1) return A[0][0];
    if (n == 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];

    double det = 0.0;
    for (int j = 0; j < n; j++) {
        Matrix sub = getMinorMatrix(A, 0, j);
        double sign = (j % 2 == 0) ? 1.0 : -1.0;
        det += sign * A[0][j] * determinant(sub);
    }
    return det;
}

// 子行列式
double minor(const Matrix& A, int i, int j) {
    return determinant(getMinorMatrix(A, i, j));
}

// 餘因子
double cofactor(const Matrix& A, int i, int j) {
    double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
    return sign * minor(A, i, j);
}

// 餘因子矩陣
Matrix cofactorMatrix(const Matrix& A) {
    int n = A.size();
    Matrix C(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = cofactor(A, i, j);
        }
    }
    return C;
}

// 轉置
Matrix transpose(const Matrix& A) {
    int n = A.size();
    Matrix T(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

// 伴隨矩陣
Matrix adjugate(const Matrix& A) {
    return transpose(cofactorMatrix(A));
}

// 逆矩陣
Matrix inverse(const Matrix& A) {
    double det = determinant(A);
    Matrix adj = adjugate(A);
    int n = A.size();
    Matrix inv(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inv[i][j] = adj[i][j] / det;
        }
    }
    return inv;
}

// 矩陣乘法
Matrix multiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

int main() {
    printSeparator("餘因子展開示範 (C++)");

    // ========================================
    // 1. 子行列式與餘因子
    // ========================================
    printSeparator("1. 子行列式與餘因子");

    Matrix A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    printMatrix("A", A);

    cout << "\n所有餘因子 Cᵢⱼ：" << endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cout << "  C" << i+1 << j+1 << " = " << setw(8) << cofactor(A, i, j);
        }
        cout << endl;
    }

    // ========================================
    // 2. 餘因子展開
    // ========================================
    printSeparator("2. 餘因子展開計算行列式");

    cout << "沿第一列展開：" << endl;
    cout << "det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃" << endl;
    cout << "       = " << A[0][0] << "×" << cofactor(A, 0, 0)
         << " + " << A[0][1] << "×" << cofactor(A, 0, 1)
         << " + " << A[0][2] << "×" << cofactor(A, 0, 2) << endl;
    cout << "       = " << determinant(A) << endl;

    // ========================================
    // 3. 餘因子矩陣與伴隨矩陣
    // ========================================
    printSeparator("3. 餘因子矩陣與伴隨矩陣");

    Matrix B = {
        {2, 1, 3},
        {1, 0, 2},
        {4, 1, 5}
    };

    printMatrix("A", B);
    cout << "\ndet(A) = " << determinant(B) << endl;

    Matrix C = cofactorMatrix(B);
    printMatrix("\n餘因子矩陣 C", C);

    Matrix adj = adjugate(B);
    printMatrix("\n伴隨矩陣 adj(A) = Cᵀ", adj);

    // ========================================
    // 4. 用伴隨矩陣求逆矩陣
    // ========================================
    printSeparator("4. 用伴隨矩陣求逆矩陣");

    cout << "A⁻¹ = adj(A) / det(A)" << endl;

    Matrix B_inv = inverse(B);
    printMatrix("\nA⁻¹", B_inv);

    // 驗證
    Matrix I = multiply(B, B_inv);
    printMatrix("\n驗證 A × A⁻¹", I);

    // ========================================
    // 5. 2×2 特例
    // ========================================
    printSeparator("5. 2×2 伴隨矩陣公式");

    Matrix A2 = {{3, 4}, {5, 6}};
    printMatrix("A", A2);

    cout << "\n對於 [[a,b],[c,d]]:" << endl;
    cout << "adj(A) = [[d,-b],[-c,a]] = [[" << A2[1][1] << "," << -A2[0][1]
         << "],[" << -A2[1][0] << "," << A2[0][0] << "]]" << endl;

    Matrix adj2 = adjugate(A2);
    printMatrix("\n計算得到的 adj(A)", adj2);

    // 總結
    printSeparator("總結");
    cout << R"(
餘因子展開公式：
  Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ
  det(A) = Σⱼ aᵢⱼ Cᵢⱼ

伴隨矩陣：
  adj(A) = Cᵀ

逆矩陣：
  A⁻¹ = adj(A) / det(A)

時間複雜度：O(n!)
)" << endl;

    cout << string(60, '=') << endl;
    cout << "示範完成！" << endl;
    cout << string(60, '=') << endl;

    return 0;
}
