/**
 * 投影 (Projections)
 *
 * 本程式示範：
 * 1. 投影到直線
 * 2. 投影矩陣及其性質
 * 3. 誤差向量的正交性驗證
 *
 * 編譯：gcc -std=c99 -O2 projection.c -o projection -lm
 * 執行：./projection
 */

#include <stdio.h>  // EN: Include a header dependency: #include <stdio.h>.
#include <stdlib.h>  // EN: Include a header dependency: #include <stdlib.h>.
#include <math.h>  // EN: Include a header dependency: #include <math.h>.
#include <stdbool.h>  // EN: Include a header dependency: #include <stdbool.h>.

#define EPSILON 1e-10  // EN: Define a preprocessor macro: #define EPSILON 1e-10.
#define MAX_DIM 10  // EN: Define a preprocessor macro: #define MAX_DIM 10.

// ========================================
// 輔助函數
// ========================================

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

// ========================================
// 基本運算
// ========================================

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

void scalar_multiply(double c, const double* x, double* result, int n) {  // EN: Execute line: void scalar_multiply(double c, const double* x, double* result, int n) {.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        result[i] = c * x[i];  // EN: Execute a statement: result[i] = c * x[i];.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

void vector_subtract(const double* x, const double* y, double* result, int n) {  // EN: Execute line: void vector_subtract(const double* x, const double* y, double* result, ….
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        result[i] = x[i] - y[i];  // EN: Execute a statement: result[i] = x[i] - y[i];.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

void outer_product(const double* x, const double* y, double result[MAX_DIM][MAX_DIM], int m, int n) {  // EN: Execute line: void outer_product(const double* x, const double* y, double result[MAX_….
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[i][j] = x[i] * y[j];  // EN: Execute a statement: result[i][j] = x[i] * y[j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

void matrix_scalar_multiply(double c, double A[MAX_DIM][MAX_DIM], double result[MAX_DIM][MAX_DIM], int m, int n) {  // EN: Execute line: void matrix_scalar_multiply(double c, double A[MAX_DIM][MAX_DIM], doubl….
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[i][j] = c * A[i][j];  // EN: Execute a statement: result[i][j] = c * A[i][j];.
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

// ========================================
// 投影函數
// ========================================

typedef struct {  // EN: Execute line: typedef struct {.
    double x_hat;  // EN: Execute a statement: double x_hat;.
    double projection[MAX_DIM];  // EN: Execute a statement: double projection[MAX_DIM];.
    double error[MAX_DIM];  // EN: Execute a statement: double error[MAX_DIM];.
    double error_norm;  // EN: Execute a statement: double error_norm;.
} ProjectionResult;  // EN: Execute a statement: } ProjectionResult;.

/**
 * 投影到直線
 * p = (aᵀb / aᵀa) * a
 */
ProjectionResult project_onto_line(const double* b, const double* a, int n) {  // EN: Execute line: ProjectionResult project_onto_line(const double* b, const double* a, in….
    ProjectionResult result;  // EN: Execute a statement: ProjectionResult result;.

    double aTb = dot_product(a, b, n);  // EN: Execute a statement: double aTb = dot_product(a, b, n);.
    double aTa = dot_product(a, a, n);  // EN: Execute a statement: double aTa = dot_product(a, a, n);.

    result.x_hat = aTb / aTa;  // EN: Execute a statement: result.x_hat = aTb / aTa;.
    scalar_multiply(result.x_hat, a, result.projection, n);  // EN: Execute a statement: scalar_multiply(result.x_hat, a, result.projection, n);.
    vector_subtract(b, result.projection, result.error, n);  // EN: Execute a statement: vector_subtract(b, result.projection, result.error, n);.
    result.error_norm = vector_norm(result.error, n);  // EN: Execute a statement: result.error_norm = vector_norm(result.error, n);.

    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

/**
 * 投影到直線的投影矩陣
 * P = aaᵀ / (aᵀa)
 */
void projection_matrix_line(const double* a, double P[MAX_DIM][MAX_DIM], int n) {  // EN: Execute line: void projection_matrix_line(const double* a, double P[MAX_DIM][MAX_DIM]….
    double aTa = dot_product(a, a, n);  // EN: Execute a statement: double aTa = dot_product(a, a, n);.
    double aaT[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double aaT[MAX_DIM][MAX_DIM];.

    outer_product(a, a, aaT, n, n);  // EN: Execute a statement: outer_product(a, a, aaT, n, n);.
    matrix_scalar_multiply(1.0 / aTa, aaT, P, n, n);  // EN: Execute a statement: matrix_scalar_multiply(1.0 / aTa, aaT, P, n, n);.
}  // EN: Structure delimiter for a block or scope.

/**
 * 驗證投影矩陣的性質
 */
void verify_projection_matrix(double P[MAX_DIM][MAX_DIM], int n, const char* name) {  // EN: Execute line: void verify_projection_matrix(double P[MAX_DIM][MAX_DIM], int n, const ….
    printf("\n驗證 %s 的性質：\n", name);  // EN: Execute a statement: printf("\n驗證 %s 的性質：\n", name);.

    // 對稱性
    bool is_symmetric = true;  // EN: Execute a statement: bool is_symmetric = true;.
    for (int i = 0; i < n && is_symmetric; i++) {  // EN: Loop control flow: for (int i = 0; i < n && is_symmetric; i++) {.
        for (int j = 0; j < n && is_symmetric; j++) {  // EN: Loop control flow: for (int j = 0; j < n && is_symmetric; j++) {.
            if (fabs(P[i][j] - P[j][i]) > EPSILON) {  // EN: Conditional control flow: if (fabs(P[i][j] - P[j][i]) > EPSILON) {.
                is_symmetric = false;  // EN: Execute a statement: is_symmetric = false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    printf("  對稱性 (%sᵀ = %s)：%s\n", name, name, is_symmetric ? "true" : "false");  // EN: Execute a statement: printf(" 對稱性 (%sᵀ = %s)：%s\n", name, name, is_symmetric ? "true" : "fal….

    // 冪等性
    double P2[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double P2[MAX_DIM][MAX_DIM];.
    matrix_multiply(P, P, P2, n, n, n);  // EN: Execute a statement: matrix_multiply(P, P, P2, n, n, n);.

    bool is_idempotent = true;  // EN: Execute a statement: bool is_idempotent = true;.
    for (int i = 0; i < n && is_idempotent; i++) {  // EN: Loop control flow: for (int i = 0; i < n && is_idempotent; i++) {.
        for (int j = 0; j < n && is_idempotent; j++) {  // EN: Loop control flow: for (int j = 0; j < n && is_idempotent; j++) {.
            if (fabs(P[i][j] - P2[i][j]) > EPSILON) {  // EN: Conditional control flow: if (fabs(P[i][j] - P2[i][j]) > EPSILON) {.
                is_idempotent = false;  // EN: Execute a statement: is_idempotent = false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    printf("  冪等性 (%s² = %s)：%s\n", name, name, is_idempotent ? "true" : "false");  // EN: Execute a statement: printf(" 冪等性 (%s² = %s)：%s\n", name, name, is_idempotent ? "true" : "fa….
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 主程式
// ========================================

int main() {  // EN: Execute line: int main() {.
    print_separator("投影示範 (C)\nProjection Demo");  // EN: Execute a statement: print_separator("投影示範 (C)\nProjection Demo");.

    // 1. 投影到直線
    print_separator("1. 投影到直線");  // EN: Execute a statement: print_separator("1. 投影到直線");.

    double a[] = {1.0, 1.0};  // EN: Execute a statement: double a[] = {1.0, 1.0};.
    double b[] = {2.0, 0.0};  // EN: Execute a statement: double b[] = {2.0, 0.0};.

    print_vector("方向 a", a, 2);  // EN: Execute a statement: print_vector("方向 a", a, 2);.
    print_vector("向量 b", b, 2);  // EN: Execute a statement: print_vector("向量 b", b, 2);.

    ProjectionResult result = project_onto_line(b, a, 2);  // EN: Execute a statement: ProjectionResult result = project_onto_line(b, a, 2);.

    printf("\n投影係數 x̂ = (aᵀb)/(aᵀa) = %.4f\n", result.x_hat);  // EN: Execute a statement: printf("\n投影係數 x̂ = (aᵀb)/(aᵀa) = %.4f\n", result.x_hat);.
    print_vector("投影 p = x̂a", result.projection, 2);  // EN: Execute a statement: print_vector("投影 p = x̂a", result.projection, 2);.
    print_vector("誤差 e = b - p", result.error, 2);  // EN: Execute a statement: print_vector("誤差 e = b - p", result.error, 2);.

    // 驗證正交性
    double e_dot_a = dot_product(result.error, a, 2);  // EN: Execute a statement: double e_dot_a = dot_product(result.error, a, 2);.
    printf("\n驗證 e ⊥ a：e · a = %.6f\n", e_dot_a);  // EN: Execute a statement: printf("\n驗證 e ⊥ a：e · a = %.6f\n", e_dot_a);.
    printf("正交？ %s\n", fabs(e_dot_a) < EPSILON ? "true" : "false");  // EN: Execute a statement: printf("正交？ %s\n", fabs(e_dot_a) < EPSILON ? "true" : "false");.

    // 2. 投影矩陣
    print_separator("2. 投影矩陣（到直線）");  // EN: Execute a statement: print_separator("2. 投影矩陣（到直線）");.

    double a2[] = {1.0, 2.0};  // EN: Execute a statement: double a2[] = {1.0, 2.0};.
    print_vector("方向 a", a2, 2);  // EN: Execute a statement: print_vector("方向 a", a2, 2);.

    double P[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double P[MAX_DIM][MAX_DIM];.
    projection_matrix_line(a2, P, 2);  // EN: Execute a statement: projection_matrix_line(a2, P, 2);.
    print_matrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P, 2, 2);  // EN: Execute a statement: print_matrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P, 2, 2);.

    verify_projection_matrix(P, 2, "P");  // EN: Execute a statement: verify_projection_matrix(P, 2, "P");.

    // 用投影矩陣計算投影
    double b2[] = {3.0, 4.0};  // EN: Execute a statement: double b2[] = {3.0, 4.0};.
    print_vector("\n向量 b", b2, 2);  // EN: Execute a statement: print_vector("\n向量 b", b2, 2);.

    double p[MAX_DIM];  // EN: Execute a statement: double p[MAX_DIM];.
    matrix_vector_multiply(P, b2, p, 2, 2);  // EN: Execute a statement: matrix_vector_multiply(P, b2, p, 2, 2);.
    print_vector("投影 p = Pb", p, 2);  // EN: Execute a statement: print_vector("投影 p = Pb", p, 2);.

    // 3. 多個向量的投影
    print_separator("3. 批次投影");  // EN: Execute a statement: print_separator("3. 批次投影");.

    double vectors[][2] = {{1.0, 0.0}, {0.0, 1.0}, {2.0, 2.0}, {3.0, -1.0}};  // EN: Execute a statement: double vectors[][2] = {{1.0, 0.0}, {0.0, 1.0}, {2.0, 2.0}, {3.0, -1.0}};.
    int num_vectors = 4;  // EN: Execute a statement: int num_vectors = 4;.

    printf("方向 a = [1, 2]\n");  // EN: Execute a statement: printf("方向 a = [1, 2]\n");.
    printf("\n各向量投影結果：\n");  // EN: Execute a statement: printf("\n各向量投影結果：\n");.

    for (int i = 0; i < num_vectors; i++) {  // EN: Loop control flow: for (int i = 0; i < num_vectors; i++) {.
        ProjectionResult proj = project_onto_line(vectors[i], a2, 2);  // EN: Execute a statement: ProjectionResult proj = project_onto_line(vectors[i], a2, 2);.
        printf("  [%.1f, %.1f] -> [%.4f, %.4f]\n",  // EN: Execute line: printf(" [%.1f, %.1f] -> [%.4f, %.4f]\n",.
               vectors[i][0], vectors[i][1],  // EN: Execute line: vectors[i][0], vectors[i][1],.
               proj.projection[0], proj.projection[1]);  // EN: Execute a statement: proj.projection[0], proj.projection[1]);.
    }  // EN: Structure delimiter for a block or scope.

    // 總結
    print_separator("總結");  // EN: Execute a statement: print_separator("總結");.
    printf("\n");  // EN: Execute a statement: printf("\n");.
    printf("投影公式：\n\n");  // EN: Execute a statement: printf("投影公式：\n\n");.
    printf("1. 投影到直線：\n");  // EN: Execute a statement: printf("1. 投影到直線：\n");.
    printf("   p = (aᵀb / aᵀa) a\n");  // EN: Execute a statement: printf(" p = (aᵀb / aᵀa) a\n");.
    printf("   P = aaᵀ / (aᵀa)\n\n");  // EN: Execute a statement: printf(" P = aaᵀ / (aᵀa)\n\n");.
    printf("2. 投影到子空間：\n");  // EN: Execute a statement: printf("2. 投影到子空間：\n");.
    printf("   p = A(AᵀA)⁻¹Aᵀb\n");  // EN: Execute a statement: printf(" p = A(AᵀA)⁻¹Aᵀb\n");.
    printf("   P = A(AᵀA)⁻¹Aᵀ\n\n");  // EN: Execute a statement: printf(" P = A(AᵀA)⁻¹Aᵀ\n\n");.
    printf("3. 投影矩陣性質：\n");  // EN: Execute a statement: printf("3. 投影矩陣性質：\n");.
    printf("   Pᵀ = P（對稱）\n");  // EN: Execute a statement: printf(" Pᵀ = P（對稱）\n");.
    printf("   P² = P（冪等）\n\n");  // EN: Execute a statement: printf(" P² = P（冪等）\n\n");.

    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n示範完成！\n");  // EN: Execute a statement: printf("\n示範完成！\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
