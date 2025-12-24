/**
 * 最小平方回歸 (Least Squares Regression)
 *
 * 編譯：gcc -std=c99 -O2 least_squares.c -o least_squares -lm
 * 執行：./least_squares
 */

#include <stdio.h>  // EN: Include a header dependency: #include <stdio.h>.
#include <stdlib.h>  // EN: Include a header dependency: #include <stdlib.h>.
#include <math.h>  // EN: Include a header dependency: #include <math.h>.

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
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        result += x[i] * y[i];  // EN: Execute a statement: result += x[i] * y[i];.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

double vector_norm(const double* x, int n) {  // EN: Execute line: double vector_norm(const double* x, int n) {.
    return sqrt(dot_product(x, x, n));  // EN: Return from the current function: return sqrt(dot_product(x, x, n));.
}  // EN: Structure delimiter for a block or scope.

void matrix_transpose(double A[MAX_DIM][MAX_DIM], double result[MAX_DIM][MAX_DIM], int m, int n) {  // EN: Execute line: void matrix_transpose(double A[MAX_DIM][MAX_DIM], double result[MAX_DIM….
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[j][i] = A[i][j];  // EN: Execute a statement: result[j][i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

void matrix_multiply(double A[MAX_DIM][MAX_DIM], double B[MAX_DIM][MAX_DIM],  // EN: Execute line: void matrix_multiply(double A[MAX_DIM][MAX_DIM], double B[MAX_DIM][MAX_….
                     double result[MAX_DIM][MAX_DIM], int m, int k, int n) {  // EN: Execute line: double result[MAX_DIM][MAX_DIM], int m, int k, int n) {.
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[i][j] = 0.0;  // EN: Execute a statement: result[i][j] = 0.0;.
            for (int p = 0; p < k; p++) {  // EN: Loop control flow: for (int p = 0; p < k; p++) {.
                result[i][j] += A[i][p] * B[p][j];  // EN: Execute a statement: result[i][j] += A[i][p] * B[p][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

void matrix_vector_multiply(double A[MAX_DIM][MAX_DIM], const double* x, double* result, int m, int n) {  // EN: Execute line: void matrix_vector_multiply(double A[MAX_DIM][MAX_DIM], const double* x….
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        result[i] = 0.0;  // EN: Execute a statement: result[i] = 0.0;.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[i] += A[i][j] * x[j];  // EN: Execute a statement: result[i] += A[i][j] * x[j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

void solve_2x2(double A[MAX_DIM][MAX_DIM], const double* b, double* x) {  // EN: Execute line: void solve_2x2(double A[MAX_DIM][MAX_DIM], const double* b, double* x) {.
    double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Execute a statement: double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];.
    x[0] = (A[1][1] * b[0] - A[0][1] * b[1]) / det;  // EN: Execute a statement: x[0] = (A[1][1] * b[0] - A[0][1] * b[1]) / det;.
    x[1] = (-A[1][0] * b[0] + A[0][0] * b[1]) / det;  // EN: Execute a statement: x[1] = (-A[1][0] * b[0] + A[0][0] * b[1]) / det;.
}  // EN: Structure delimiter for a block or scope.

typedef struct {  // EN: Execute line: typedef struct {.
    double coefficients[MAX_DIM];  // EN: Execute a statement: double coefficients[MAX_DIM];.
    double fitted[MAX_DIM];  // EN: Execute a statement: double fitted[MAX_DIM];.
    double residual[MAX_DIM];  // EN: Execute a statement: double residual[MAX_DIM];.
    double residual_norm;  // EN: Execute a statement: double residual_norm;.
    double r_squared;  // EN: Execute a statement: double r_squared;.
} LeastSquaresResult;  // EN: Execute a statement: } LeastSquaresResult;.

void create_design_matrix_linear(const double* t, double A[MAX_DIM][MAX_DIM], int n) {  // EN: Execute line: void create_design_matrix_linear(const double* t, double A[MAX_DIM][MAX….
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        A[i][0] = 1.0;  // EN: Execute a statement: A[i][0] = 1.0;.
        A[i][1] = t[i];  // EN: Execute a statement: A[i][1] = t[i];.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

LeastSquaresResult least_squares_solve(double A[MAX_DIM][MAX_DIM], const double* b, int m, int n) {  // EN: Execute line: LeastSquaresResult least_squares_solve(double A[MAX_DIM][MAX_DIM], cons….
    LeastSquaresResult result;  // EN: Execute a statement: LeastSquaresResult result;.

    double AT[MAX_DIM][MAX_DIM], ATA[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double AT[MAX_DIM][MAX_DIM], ATA[MAX_DIM][MAX_DIM];.
    double ATb[MAX_DIM];  // EN: Execute a statement: double ATb[MAX_DIM];.

    matrix_transpose(A, AT, m, n);  // EN: Execute a statement: matrix_transpose(A, AT, m, n);.
    matrix_multiply(AT, A, ATA, n, m, n);  // EN: Execute a statement: matrix_multiply(AT, A, ATA, n, m, n);.

    // Aᵀb
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        ATb[i] = 0.0;  // EN: Execute a statement: ATb[i] = 0.0;.
        for (int j = 0; j < m; j++) {  // EN: Loop control flow: for (int j = 0; j < m; j++) {.
            ATb[i] += AT[i][j] * b[j];  // EN: Execute a statement: ATb[i] += AT[i][j] * b[j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    solve_2x2(ATA, ATb, result.coefficients);  // EN: Execute a statement: solve_2x2(ATA, ATb, result.coefficients);.

    // 擬合值
    matrix_vector_multiply(A, result.coefficients, result.fitted, m, n);  // EN: Execute a statement: matrix_vector_multiply(A, result.coefficients, result.fitted, m, n);.

    // 殘差
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        result.residual[i] = b[i] - result.fitted[i];  // EN: Execute a statement: result.residual[i] = b[i] - result.fitted[i];.
    }  // EN: Structure delimiter for a block or scope.
    result.residual_norm = vector_norm(result.residual, m);  // EN: Execute a statement: result.residual_norm = vector_norm(result.residual, m);.

    // R²
    double b_mean = 0.0;  // EN: Execute a statement: double b_mean = 0.0;.
    for (int i = 0; i < m; i++) b_mean += b[i];  // EN: Loop control flow: for (int i = 0; i < m; i++) b_mean += b[i];.
    b_mean /= m;  // EN: Execute a statement: b_mean /= m;.

    double tss = 0.0, rss = 0.0;  // EN: Execute a statement: double tss = 0.0, rss = 0.0;.
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        tss += (b[i] - b_mean) * (b[i] - b_mean);  // EN: Execute a statement: tss += (b[i] - b_mean) * (b[i] - b_mean);.
        rss += result.residual[i] * result.residual[i];  // EN: Execute a statement: rss += result.residual[i] * result.residual[i];.
    }  // EN: Structure delimiter for a block or scope.
    result.r_squared = (tss > 0) ? (1.0 - rss / tss) : 0.0;  // EN: Execute a statement: result.r_squared = (tss > 0) ? (1.0 - rss / tss) : 0.0;.

    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    print_separator("最小平方回歸示範 (C)\nLeast Squares Regression Demo");  // EN: Execute a statement: print_separator("最小平方回歸示範 (C)\nLeast Squares Regression Demo");.

    // 1. 簡單線性迴歸
    print_separator("1. 簡單線性迴歸：y = C + Dt");  // EN: Execute a statement: print_separator("1. 簡單線性迴歸：y = C + Dt");.

    double t[] = {0.0, 1.0, 2.0};  // EN: Execute a statement: double t[] = {0.0, 1.0, 2.0};.
    double b[] = {1.0, 3.0, 4.0};  // EN: Execute a statement: double b[] = {1.0, 3.0, 4.0};.
    int m = 3, n = 2;  // EN: Execute a statement: int m = 3, n = 2;.

    printf("數據點：\n");  // EN: Execute a statement: printf("數據點：\n");.
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        printf("  t = %.1f, b = %.1f\n", t[i], b[i]);  // EN: Execute a statement: printf(" t = %.1f, b = %.1f\n", t[i], b[i]);.
    }  // EN: Structure delimiter for a block or scope.

    double A[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double A[MAX_DIM][MAX_DIM];.
    create_design_matrix_linear(t, A, m);  // EN: Execute a statement: create_design_matrix_linear(t, A, m);.
    print_matrix("\n設計矩陣 A [1, t]", A, m, n);  // EN: Execute a statement: print_matrix("\n設計矩陣 A [1, t]", A, m, n);.
    print_vector("觀測值 b", b, m);  // EN: Execute a statement: print_vector("觀測值 b", b, m);.

    LeastSquaresResult result = least_squares_solve(A, b, m, n);  // EN: Execute a statement: LeastSquaresResult result = least_squares_solve(A, b, m, n);.

    printf("\n【解】\n");  // EN: Execute a statement: printf("\n【解】\n");.
    printf("C（截距）= %.4f\n", result.coefficients[0]);  // EN: Execute a statement: printf("C（截距）= %.4f\n", result.coefficients[0]);.
    printf("D（斜率）= %.4f\n", result.coefficients[1]);  // EN: Execute a statement: printf("D（斜率）= %.4f\n", result.coefficients[1]);.
    printf("\n最佳直線：y = %.4f + %.4ft\n", result.coefficients[0], result.coefficients[1]);  // EN: Execute a statement: printf("\n最佳直線：y = %.4f + %.4ft\n", result.coefficients[0], result.coef….

    print_vector("\n擬合值 ŷ", result.fitted, m);  // EN: Execute a statement: print_vector("\n擬合值 ŷ", result.fitted, m);.
    print_vector("殘差 e", result.residual, m);  // EN: Execute a statement: print_vector("殘差 e", result.residual, m);.
    printf("殘差範數 ‖e‖ = %.4f\n", result.residual_norm);  // EN: Execute a statement: printf("殘差範數 ‖e‖ = %.4f\n", result.residual_norm);.
    printf("R² = %.4f\n", result.r_squared);  // EN: Execute a statement: printf("R² = %.4f\n", result.r_squared);.

    // 總結
    print_separator("總結");  // EN: Execute a statement: print_separator("總結");.
    printf("\n最小平方法核心公式：\n\n");  // EN: Execute a statement: printf("\n最小平方法核心公式：\n\n");.
    printf("1. 正規方程：AᵀA x̂ = Aᵀb\n\n");  // EN: Execute a statement: printf("1. 正規方程：AᵀA x̂ = Aᵀb\n\n");.
    printf("2. 解：x̂ = (AᵀA)⁻¹Aᵀb\n\n");  // EN: Execute a statement: printf("2. 解：x̂ = (AᵀA)⁻¹Aᵀb\n\n");.
    printf("3. R² = 1 - RSS/TSS\n\n");  // EN: Execute a statement: printf("3. R² = 1 - RSS/TSS\n\n");.

    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n示範完成！\n");  // EN: Execute a statement: printf("\n示範完成！\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
