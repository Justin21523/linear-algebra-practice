/**
 * QR 分解 (QR Decomposition)
 *
 * 編譯：gcc -std=c99 -O2 qr_decomposition.c -o qr_decomposition -lm
 * 執行：./qr_decomposition
 */

#include <stdio.h>  // EN: Include a header dependency: #include <stdio.h>.
#include <math.h>  // EN: Include a header dependency: #include <math.h>.
#include <string.h>  // EN: Include a header dependency: #include <string.h>.

#define MAX_DIM 10  // EN: Define a preprocessor macro: #define MAX_DIM 10.

void print_separator(const char* title) {  // EN: Execute line: void print_separator(const char* title) {.
    printf("\n");  // EN: Execute a statement: printf("\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n%s\n", title);  // EN: Execute a statement: printf("\n%s\n", title);.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.
}  // EN: Structure delimiter for a block or scope.

void print_vector(const char* name, const double* v, int n) {  // EN: Execute line: void print_vector(const char* name, const double* v, int n) {.
    printf("%s = [", name);  // EN: Execute a statement: printf("%s = [", name);.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        printf("%.4f", v[i]);  // EN: Execute a statement: printf("%.4f", v[i]);.
        if (i < n - 1) printf(", ");  // EN: Conditional control flow: if (i < n - 1) printf(", ");.
    }  // EN: Structure delimiter for a block or scope.
    printf("]\n");  // EN: Execute a statement: printf("]\n");.
}  // EN: Structure delimiter for a block or scope.

void print_matrix(const char* name, double M[MAX_DIM][MAX_DIM], int rows, int cols) {  // EN: Execute line: void print_matrix(const char* name, double M[MAX_DIM][MAX_DIM], int row….
    printf("%s =\n", name);  // EN: Execute a statement: printf("%s =\n", name);.
    for (int i = 0; i < rows; i++) {  // EN: Loop control flow: for (int i = 0; i < rows; i++) {.
        printf("  [");  // EN: Execute a statement: printf(" [");.
        for (int j = 0; j < cols; j++) {  // EN: Loop control flow: for (int j = 0; j < cols; j++) {.
            printf("%8.4f", M[i][j]);  // EN: Execute a statement: printf("%8.4f", M[i][j]);.
            if (j < cols - 1) printf(", ");  // EN: Conditional control flow: if (j < cols - 1) printf(", ");.
        }  // EN: Structure delimiter for a block or scope.
        printf("]\n");  // EN: Execute a statement: printf("]\n");.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

double dot_product(const double* x, const double* y, int n) {  // EN: Execute line: double dot_product(const double* x, const double* y, int n) {.
    double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
    for (int i = 0; i < n; i++) result += x[i] * y[i];  // EN: Loop control flow: for (int i = 0; i < n; i++) result += x[i] * y[i];.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

double vector_norm(const double* x, int n) {  // EN: Execute line: double vector_norm(const double* x, int n) {.
    return sqrt(dot_product(x, x, n));  // EN: Return from the current function: return sqrt(dot_product(x, x, n));.
}  // EN: Structure delimiter for a block or scope.

// 取得矩陣的第 j 行（column）
void get_column(double A[MAX_DIM][MAX_DIM], int j, double* col, int rows) {  // EN: Execute line: void get_column(double A[MAX_DIM][MAX_DIM], int j, double* col, int row….
    for (int i = 0; i < rows; i++) {  // EN: Loop control flow: for (int i = 0; i < rows; i++) {.
        col[i] = A[i][j];  // EN: Execute a statement: col[i] = A[i][j];.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// Gram-Schmidt QR 分解
// A: m×n, Q: m×n, R: n×n
void qr_decomposition(double A[MAX_DIM][MAX_DIM],  // EN: Execute line: void qr_decomposition(double A[MAX_DIM][MAX_DIM],.
                      double Q[MAX_DIM][MAX_DIM],  // EN: Execute line: double Q[MAX_DIM][MAX_DIM],.
                      double R[MAX_DIM][MAX_DIM],  // EN: Execute line: double R[MAX_DIM][MAX_DIM],.
                      int m, int n) {  // EN: Execute line: int m, int n) {.
    // 初始化
    for (int i = 0; i < m; i++)  // EN: Loop control flow: for (int i = 0; i < m; i++).
        for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            Q[i][j] = 0.0;  // EN: Execute a statement: Q[i][j] = 0.0;.

    for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
        for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            R[i][j] = 0.0;  // EN: Execute a statement: R[i][j] = 0.0;.

    double v[MAX_DIM], qi[MAX_DIM], aj[MAX_DIM];  // EN: Execute a statement: double v[MAX_DIM], qi[MAX_DIM], aj[MAX_DIM];.

    for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
        // 取得 A 的第 j 行
        get_column(A, j, v, m);  // EN: Execute a statement: get_column(A, j, v, m);.

        // 減去前面所有 q 向量的投影
        for (int i = 0; i < j; i++) {  // EN: Loop control flow: for (int i = 0; i < j; i++) {.
            get_column(Q, i, qi, m);  // EN: Execute a statement: get_column(Q, i, qi, m);.
            get_column(A, j, aj, m);  // EN: Execute a statement: get_column(A, j, aj, m);.
            R[i][j] = dot_product(qi, aj, m);  // EN: Execute a statement: R[i][j] = dot_product(qi, aj, m);.

            for (int k = 0; k < m; k++) {  // EN: Loop control flow: for (int k = 0; k < m; k++) {.
                v[k] -= R[i][j] * qi[k];  // EN: Execute a statement: v[k] -= R[i][j] * qi[k];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.

        // 標準化
        R[j][j] = vector_norm(v, m);  // EN: Execute a statement: R[j][j] = vector_norm(v, m);.

        if (R[j][j] > 1e-10) {  // EN: Conditional control flow: if (R[j][j] > 1e-10) {.
            for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
                Q[i][j] = v[i] / R[j][j];  // EN: Execute a statement: Q[i][j] = v[i] / R[j][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// 回代法解上三角方程組 Rx = b
void solve_upper_triangular(double R[MAX_DIM][MAX_DIM],  // EN: Execute line: void solve_upper_triangular(double R[MAX_DIM][MAX_DIM],.
                            const double* b,  // EN: Execute line: const double* b,.
                            double* x,  // EN: Execute line: double* x,.
                            int n) {  // EN: Execute line: int n) {.
    for (int i = 0; i < n; i++) x[i] = 0.0;  // EN: Loop control flow: for (int i = 0; i < n; i++) x[i] = 0.0;.

    for (int i = n - 1; i >= 0; i--) {  // EN: Loop control flow: for (int i = n - 1; i >= 0; i--) {.
        x[i] = b[i];  // EN: Execute a statement: x[i] = b[i];.
        for (int j = i + 1; j < n; j++) {  // EN: Loop control flow: for (int j = i + 1; j < n; j++) {.
            x[i] -= R[i][j] * x[j];  // EN: Execute a statement: x[i] -= R[i][j] * x[j];.
        }  // EN: Structure delimiter for a block or scope.
        x[i] /= R[i][i];  // EN: Execute a statement: x[i] /= R[i][i];.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// 用 QR 分解解最小平方問題
void qr_least_squares(double A[MAX_DIM][MAX_DIM],  // EN: Execute line: void qr_least_squares(double A[MAX_DIM][MAX_DIM],.
                      const double* b,  // EN: Execute line: const double* b,.
                      double* x,  // EN: Execute line: double* x,.
                      int m, int n) {  // EN: Execute line: int m, int n) {.
    double Q[MAX_DIM][MAX_DIM], R[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double Q[MAX_DIM][MAX_DIM], R[MAX_DIM][MAX_DIM];.
    qr_decomposition(A, Q, R, m, n);  // EN: Execute a statement: qr_decomposition(A, Q, R, m, n);.

    // 計算 Qᵀb
    double Qt_b[MAX_DIM];  // EN: Execute a statement: double Qt_b[MAX_DIM];.
    for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
        double qj[MAX_DIM];  // EN: Execute a statement: double qj[MAX_DIM];.
        get_column(Q, j, qj, m);  // EN: Execute a statement: get_column(Q, j, qj, m);.
        Qt_b[j] = dot_product(qj, b, m);  // EN: Execute a statement: Qt_b[j] = dot_product(qj, b, m);.
    }  // EN: Structure delimiter for a block or scope.

    // 解 Rx = Qᵀb
    solve_upper_triangular(R, Qt_b, x, n);  // EN: Execute a statement: solve_upper_triangular(R, Qt_b, x, n);.
}  // EN: Structure delimiter for a block or scope.

// 矩陣乘法
void matrix_multiply(double A[MAX_DIM][MAX_DIM],  // EN: Execute line: void matrix_multiply(double A[MAX_DIM][MAX_DIM],.
                     double B[MAX_DIM][MAX_DIM],  // EN: Execute line: double B[MAX_DIM][MAX_DIM],.
                     double result[MAX_DIM][MAX_DIM],  // EN: Execute line: double result[MAX_DIM][MAX_DIM],.
                     int m, int k, int n) {  // EN: Execute line: int m, int k, int n) {.
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[i][j] = 0.0;  // EN: Execute a statement: result[i][j] = 0.0;.
            for (int p = 0; p < k; p++) {  // EN: Loop control flow: for (int p = 0; p < k; p++) {.
                result[i][j] += A[i][p] * B[p][j];  // EN: Execute a statement: result[i][j] += A[i][p] * B[p][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// 矩陣轉置
void transpose(double A[MAX_DIM][MAX_DIM],  // EN: Execute line: void transpose(double A[MAX_DIM][MAX_DIM],.
               double result[MAX_DIM][MAX_DIM],  // EN: Execute line: double result[MAX_DIM][MAX_DIM],.
               int m, int n) {  // EN: Execute line: int m, int n) {.
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[j][i] = A[i][j];  // EN: Execute a statement: result[j][i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    print_separator("QR 分解示範 (C)");  // EN: Execute a statement: print_separator("QR 分解示範 (C)");.

    // ========================================
    // 1. 基本 QR 分解
    // ========================================
    print_separator("1. 基本 QR 分解");  // EN: Execute a statement: print_separator("1. 基本 QR 分解");.

    double A[MAX_DIM][MAX_DIM] = {  // EN: Execute line: double A[MAX_DIM][MAX_DIM] = {.
        {1.0, 1.0},  // EN: Execute line: {1.0, 1.0},.
        {1.0, 0.0},  // EN: Execute line: {1.0, 0.0},.
        {0.0, 1.0}  // EN: Execute line: {0.0, 1.0}.
    };  // EN: Structure delimiter for a block or scope.
    int m = 3, n = 2;  // EN: Execute a statement: int m = 3, n = 2;.

    printf("輸入矩陣 A：\n");  // EN: Execute a statement: printf("輸入矩陣 A：\n");.
    print_matrix("A", A, m, n);  // EN: Execute a statement: print_matrix("A", A, m, n);.

    double Q[MAX_DIM][MAX_DIM], R[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double Q[MAX_DIM][MAX_DIM], R[MAX_DIM][MAX_DIM];.
    qr_decomposition(A, Q, R, m, n);  // EN: Execute a statement: qr_decomposition(A, Q, R, m, n);.

    printf("\nQR 分解結果：\n");  // EN: Execute a statement: printf("\nQR 分解結果：\n");.
    print_matrix("Q", Q, m, n);  // EN: Execute a statement: print_matrix("Q", Q, m, n);.
    print_matrix("\nR", R, n, n);  // EN: Execute a statement: print_matrix("\nR", R, n, n);.

    // 驗證 QᵀQ = I
    double QT[MAX_DIM][MAX_DIM], QTQ[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double QT[MAX_DIM][MAX_DIM], QTQ[MAX_DIM][MAX_DIM];.
    transpose(Q, QT, m, n);  // EN: Execute a statement: transpose(Q, QT, m, n);.
    matrix_multiply(QT, Q, QTQ, n, m, n);  // EN: Execute a statement: matrix_multiply(QT, Q, QTQ, n, m, n);.
    printf("\n驗證 QᵀQ = I：\n");  // EN: Execute a statement: printf("\n驗證 QᵀQ = I：\n");.
    print_matrix("QᵀQ", QTQ, n, n);  // EN: Execute a statement: print_matrix("QᵀQ", QTQ, n, n);.

    // 驗證 A = QR
    double QR_result[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double QR_result[MAX_DIM][MAX_DIM];.
    matrix_multiply(Q, R, QR_result, m, n, n);  // EN: Execute a statement: matrix_multiply(Q, R, QR_result, m, n, n);.
    printf("\n驗證 A = QR：\n");  // EN: Execute a statement: printf("\n驗證 A = QR：\n");.
    print_matrix("QR", QR_result, m, n);  // EN: Execute a statement: print_matrix("QR", QR_result, m, n);.

    // ========================================
    // 2. 用 QR 解最小平方
    // ========================================
    print_separator("2. 用 QR 解最小平方");  // EN: Execute a statement: print_separator("2. 用 QR 解最小平方");.

    // 數據
    double t[] = {0.0, 1.0, 2.0};  // EN: Execute a statement: double t[] = {0.0, 1.0, 2.0};.
    double b[] = {1.0, 3.0, 4.0};  // EN: Execute a statement: double b[] = {1.0, 3.0, 4.0};.
    int data_size = 3;  // EN: Execute a statement: int data_size = 3;.

    printf("數據點：\n");  // EN: Execute a statement: printf("數據點：\n");.
    for (int i = 0; i < data_size; i++) {  // EN: Loop control flow: for (int i = 0; i < data_size; i++) {.
        printf("  (%.1f, %.1f)\n", t[i], b[i]);  // EN: Execute a statement: printf(" (%.1f, %.1f)\n", t[i], b[i]);.
    }  // EN: Structure delimiter for a block or scope.

    // 設計矩陣
    double A_ls[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double A_ls[MAX_DIM][MAX_DIM];.
    for (int i = 0; i < data_size; i++) {  // EN: Loop control flow: for (int i = 0; i < data_size; i++) {.
        A_ls[i][0] = 1.0;  // EN: Execute a statement: A_ls[i][0] = 1.0;.
        A_ls[i][1] = t[i];  // EN: Execute a statement: A_ls[i][1] = t[i];.
    }  // EN: Structure delimiter for a block or scope.

    printf("\n設計矩陣 A：\n");  // EN: Execute a statement: printf("\n設計矩陣 A：\n");.
    print_matrix("A", A_ls, data_size, 2);  // EN: Execute a statement: print_matrix("A", A_ls, data_size, 2);.
    print_vector("觀測值 b", b, data_size);  // EN: Execute a statement: print_vector("觀測值 b", b, data_size);.

    // QR 分解
    double Q_ls[MAX_DIM][MAX_DIM], R_ls[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double Q_ls[MAX_DIM][MAX_DIM], R_ls[MAX_DIM][MAX_DIM];.
    qr_decomposition(A_ls, Q_ls, R_ls, data_size, 2);  // EN: Execute a statement: qr_decomposition(A_ls, Q_ls, R_ls, data_size, 2);.
    print_matrix("\nQ", Q_ls, data_size, 2);  // EN: Execute a statement: print_matrix("\nQ", Q_ls, data_size, 2);.
    print_matrix("R", R_ls, 2, 2);  // EN: Execute a statement: print_matrix("R", R_ls, 2, 2);.

    // 解最小平方
    double x[MAX_DIM];  // EN: Execute a statement: double x[MAX_DIM];.
    qr_least_squares(A_ls, b, x, data_size, 2);  // EN: Execute a statement: qr_least_squares(A_ls, b, x, data_size, 2);.
    print_vector("\n解 x", x, 2);  // EN: Execute a statement: print_vector("\n解 x", x, 2);.

    printf("\n最佳直線：y = %.4f + %.4ft\n", x[0], x[1]);  // EN: Execute a statement: printf("\n最佳直線：y = %.4f + %.4ft\n", x[0], x[1]);.

    // ========================================
    // 3. 3×3 矩陣的 QR 分解
    // ========================================
    print_separator("3. 3×3 矩陣的 QR 分解");  // EN: Execute a statement: print_separator("3. 3×3 矩陣的 QR 分解");.

    double A2[MAX_DIM][MAX_DIM] = {  // EN: Execute line: double A2[MAX_DIM][MAX_DIM] = {.
        {1.0, 1.0, 0.0},  // EN: Execute line: {1.0, 1.0, 0.0},.
        {1.0, 0.0, 1.0},  // EN: Execute line: {1.0, 0.0, 1.0},.
        {0.0, 1.0, 1.0}  // EN: Execute line: {0.0, 1.0, 1.0}.
    };  // EN: Structure delimiter for a block or scope.

    printf("輸入矩陣 A：\n");  // EN: Execute a statement: printf("輸入矩陣 A：\n");.
    print_matrix("A", A2, 3, 3);  // EN: Execute a statement: print_matrix("A", A2, 3, 3);.

    double Q2[MAX_DIM][MAX_DIM], R2[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double Q2[MAX_DIM][MAX_DIM], R2[MAX_DIM][MAX_DIM];.
    qr_decomposition(A2, Q2, R2, 3, 3);  // EN: Execute a statement: qr_decomposition(A2, Q2, R2, 3, 3);.

    printf("\nQR 分解結果：\n");  // EN: Execute a statement: printf("\nQR 分解結果：\n");.
    print_matrix("Q", Q2, 3, 3);  // EN: Execute a statement: print_matrix("Q", Q2, 3, 3);.
    print_matrix("\nR", R2, 3, 3);  // EN: Execute a statement: print_matrix("\nR", R2, 3, 3);.

    // 總結
    print_separator("總結");  // EN: Execute a statement: print_separator("總結");.
    printf("\nQR 分解核心：\n\n");  // EN: Execute a statement: printf("\nQR 分解核心：\n\n");.
    printf("1. A = QR\n");  // EN: Execute a statement: printf("1. A = QR\n");.
    printf("   - Q: 標準正交矩陣 (QᵀQ = I)\n");  // EN: Execute a statement: printf(" - Q: 標準正交矩陣 (QᵀQ = I)\n");.
    printf("   - R: 上三角矩陣\n\n");  // EN: Execute a statement: printf(" - R: 上三角矩陣\n\n");.
    printf("2. Gram-Schmidt 演算法：\n");  // EN: Execute a statement: printf("2. Gram-Schmidt 演算法：\n");.
    printf("   - 對 A 的行向量正交化得到 Q\n");  // EN: Execute a statement: printf(" - 對 A 的行向量正交化得到 Q\n");.
    printf("   - R 的元素是投影係數\n\n");  // EN: Execute a statement: printf(" - R 的元素是投影係數\n\n");.
    printf("3. 用 QR 解最小平方：\n");  // EN: Execute a statement: printf("3. 用 QR 解最小平方：\n");.
    printf("   min ‖Ax - b‖²\n");  // EN: Execute a statement: printf(" min ‖Ax - b‖²\n");.
    printf("   → Rx = Qᵀb\n\n");  // EN: Execute a statement: printf(" → Rx = Qᵀb\n\n");.
    printf("4. 優勢：\n");  // EN: Execute a statement: printf("4. 優勢：\n");.
    printf("   - 比正規方程更穩定\n");  // EN: Execute a statement: printf(" - 比正規方程更穩定\n");.
    printf("   - 避免計算 AᵀA\n\n");  // EN: Execute a statement: printf(" - 避免計算 AᵀA\n\n");.

    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n示範完成！\n");  // EN: Execute a statement: printf("\n示範完成！\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
