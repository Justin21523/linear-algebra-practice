/**
 * 內積與正交性 (Inner Product and Orthogonality)
 *
 * 本程式示範：
 * 1. 向量內積計算
 * 2. 向量長度（範數）
 * 3. 向量夾角
 * 4. 正交性判斷
 * 5. 正交矩陣驗證
 *
 * 編譯：gcc -std=c99 -O2 inner_product.c -o inner_product -lm
 * 執行：./inner_product
 */

#include <stdio.h>  // EN: Include a header dependency: #include <stdio.h>.
#include <stdlib.h>  // EN: Include a header dependency: #include <stdlib.h>.
#include <math.h>  // EN: Include a header dependency: #include <math.h>.
#include <stdbool.h>  // EN: Include a header dependency: #include <stdbool.h>.

#define EPSILON 1e-10  // EN: Define a preprocessor macro: #define EPSILON 1e-10.
#define PI 3.14159265358979323846  // EN: Define a preprocessor macro: #define PI 3.14159265358979323846.
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

void print_matrix(const char* name, const double M[MAX_DIM][MAX_DIM], int rows, int cols) {  // EN: Execute line: void print_matrix(const char* name, const double M[MAX_DIM][MAX_DIM], i….
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
// 向量運算
// ========================================

/**
 * 計算兩向量的內積 (Dot Product)
 * x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ
 */
double dot_product(const double* x, const double* y, int n) {  // EN: Execute line: double dot_product(const double* x, const double* y, int n) {.
    double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        result += x[i] * y[i];  // EN: Execute a statement: result += x[i] * y[i];.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

/**
 * 計算向量的長度（L2 範數）
 * ‖x‖ = √(x · x)
 */
double vector_norm(const double* x, int n) {  // EN: Execute line: double vector_norm(const double* x, int n) {.
    return sqrt(dot_product(x, x, n));  // EN: Return from the current function: return sqrt(dot_product(x, x, n));.
}  // EN: Structure delimiter for a block or scope.

/**
 * 正規化向量為單位向量
 * result = x / ‖x‖
 */
void normalize(const double* x, double* result, int n) {  // EN: Execute line: void normalize(const double* x, double* result, int n) {.
    double norm = vector_norm(x, n);  // EN: Execute a statement: double norm = vector_norm(x, n);.
    if (norm < EPSILON) {  // EN: Conditional control flow: if (norm < EPSILON) {.
        printf("錯誤：零向量無法正規化\n");  // EN: Execute a statement: printf("錯誤：零向量無法正規化\n");.
        return;  // EN: Return from the current function: return;.
    }  // EN: Structure delimiter for a block or scope.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        result[i] = x[i] / norm;  // EN: Execute a statement: result[i] = x[i] / norm;.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

/**
 * 計算兩向量的夾角（弧度）
 * cos θ = (x · y) / (‖x‖ ‖y‖)
 */
double vector_angle(const double* x, const double* y, int n) {  // EN: Execute line: double vector_angle(const double* x, const double* y, int n) {.
    double dot = dot_product(x, y, n);  // EN: Execute a statement: double dot = dot_product(x, y, n);.
    double norm_x = vector_norm(x, n);  // EN: Execute a statement: double norm_x = vector_norm(x, n);.
    double norm_y = vector_norm(y, n);  // EN: Execute a statement: double norm_y = vector_norm(y, n);.

    if (norm_x < EPSILON || norm_y < EPSILON) {  // EN: Conditional control flow: if (norm_x < EPSILON || norm_y < EPSILON) {.
        printf("錯誤：零向量沒有定義夾角\n");  // EN: Execute a statement: printf("錯誤：零向量沒有定義夾角\n");.
        return 0.0;  // EN: Return from the current function: return 0.0;.
    }  // EN: Structure delimiter for a block or scope.

    double cos_theta = dot / (norm_x * norm_y);  // EN: Execute a statement: double cos_theta = dot / (norm_x * norm_y);.
    // 處理浮點數誤差
    if (cos_theta > 1.0) cos_theta = 1.0;  // EN: Conditional control flow: if (cos_theta > 1.0) cos_theta = 1.0;.
    if (cos_theta < -1.0) cos_theta = -1.0;  // EN: Conditional control flow: if (cos_theta < -1.0) cos_theta = -1.0;.

    return acos(cos_theta);  // EN: Return from the current function: return acos(cos_theta);.
}  // EN: Structure delimiter for a block or scope.

/**
 * 判斷兩向量是否正交
 * x ⊥ y ⟺ x · y = 0
 */
bool is_orthogonal(const double* x, const double* y, int n) {  // EN: Execute line: bool is_orthogonal(const double* x, const double* y, int n) {.
    return fabs(dot_product(x, y, n)) < EPSILON;  // EN: Return from the current function: return fabs(dot_product(x, y, n)) < EPSILON;.
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 矩陣運算
// ========================================

/**
 * 矩陣轉置
 */
void matrix_transpose(const double A[MAX_DIM][MAX_DIM], double result[MAX_DIM][MAX_DIM],  // EN: Execute line: void matrix_transpose(const double A[MAX_DIM][MAX_DIM], double result[M….
                      int rows, int cols) {  // EN: Execute line: int rows, int cols) {.
    for (int i = 0; i < rows; i++) {  // EN: Loop control flow: for (int i = 0; i < rows; i++) {.
        for (int j = 0; j < cols; j++) {  // EN: Loop control flow: for (int j = 0; j < cols; j++) {.
            result[j][i] = A[i][j];  // EN: Execute a statement: result[j][i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

/**
 * 矩陣乘法
 */
void matrix_multiply(const double A[MAX_DIM][MAX_DIM], const double B[MAX_DIM][MAX_DIM],  // EN: Execute line: void matrix_multiply(const double A[MAX_DIM][MAX_DIM], const double B[M….
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

/**
 * 矩陣乘向量
 */
void matrix_vector_multiply(const double A[MAX_DIM][MAX_DIM], const double* x,  // EN: Execute line: void matrix_vector_multiply(const double A[MAX_DIM][MAX_DIM], const dou….
                            double* result, int m, int n) {  // EN: Execute line: double* result, int m, int n) {.
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        result[i] = 0.0;  // EN: Execute a statement: result[i] = 0.0;.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[i] += A[i][j] * x[j];  // EN: Execute a statement: result[i] += A[i][j] * x[j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

/**
 * 判斷是否為單位矩陣
 */
bool is_identity(const double A[MAX_DIM][MAX_DIM], int n) {  // EN: Execute line: bool is_identity(const double A[MAX_DIM][MAX_DIM], int n) {.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            double expected = (i == j) ? 1.0 : 0.0;  // EN: Execute a statement: double expected = (i == j) ? 1.0 : 0.0;.
            if (fabs(A[i][j] - expected) > EPSILON) {  // EN: Conditional control flow: if (fabs(A[i][j] - expected) > EPSILON) {.
                return false;  // EN: Return from the current function: return false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return true;  // EN: Return from the current function: return true;.
}  // EN: Structure delimiter for a block or scope.

/**
 * 判斷矩陣是否為正交矩陣
 * QᵀQ = I
 */
bool is_orthogonal_matrix(const double Q[MAX_DIM][MAX_DIM], int n) {  // EN: Execute line: bool is_orthogonal_matrix(const double Q[MAX_DIM][MAX_DIM], int n) {.
    double QT[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double QT[MAX_DIM][MAX_DIM];.
    double product[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double product[MAX_DIM][MAX_DIM];.

    matrix_transpose(Q, QT, n, n);  // EN: Execute a statement: matrix_transpose(Q, QT, n, n);.
    matrix_multiply(QT, Q, product, n, n, n);  // EN: Execute a statement: matrix_multiply(QT, Q, product, n, n, n);.

    return is_identity(product, n);  // EN: Return from the current function: return is_identity(product, n);.
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 主程式
// ========================================

int main() {  // EN: Execute line: int main() {.
    print_separator("內積與正交性示範 (C)\nInner Product & Orthogonality Demo");  // EN: Execute a statement: print_separator("內積與正交性示範 (C)\nInner Product & Orthogonality Demo");.

    // 1. 內積計算
    print_separator("1. 內積計算 (Dot Product)");  // EN: Execute a statement: print_separator("1. 內積計算 (Dot Product)");.

    double x[] = {1.0, 2.0, 3.0};  // EN: Execute a statement: double x[] = {1.0, 2.0, 3.0};.
    double y[] = {4.0, 5.0, 6.0};  // EN: Execute a statement: double y[] = {4.0, 5.0, 6.0};.

    print_vector("x", x, 3);  // EN: Execute a statement: print_vector("x", x, 3);.
    print_vector("y", y, 3);  // EN: Execute a statement: print_vector("y", y, 3);.
    printf("\nx · y = %.4f\n", dot_product(x, y, 3));  // EN: Execute a statement: printf("\nx · y = %.4f\n", dot_product(x, y, 3));.
    printf("計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32\n");  // EN: Execute a statement: printf("計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32\n");.

    // 2. 向量長度
    print_separator("2. 向量長度 (Vector Norm)");  // EN: Execute a statement: print_separator("2. 向量長度 (Vector Norm)");.

    double v[] = {3.0, 4.0};  // EN: Execute a statement: double v[] = {3.0, 4.0};.
    print_vector("v", v, 2);  // EN: Execute a statement: print_vector("v", v, 2);.
    printf("‖v‖ = %.4f\n", vector_norm(v, 2));  // EN: Execute a statement: printf("‖v‖ = %.4f\n", vector_norm(v, 2));.
    printf("計算：√(3² + 4²) = √25 = 5\n");  // EN: Execute a statement: printf("計算：√(3² + 4²) = √25 = 5\n");.

    // 正規化
    double v_normalized[2];  // EN: Execute a statement: double v_normalized[2];.
    normalize(v, v_normalized, 2);  // EN: Execute a statement: normalize(v, v_normalized, 2);.
    printf("\n單位向量：\n");  // EN: Execute a statement: printf("\n單位向量：\n");.
    print_vector("v̂ = v/‖v‖", v_normalized, 2);  // EN: Execute a statement: print_vector("v̂ = v/‖v‖", v_normalized, 2);.
    printf("‖v̂‖ = %.4f\n", vector_norm(v_normalized, 2));  // EN: Execute a statement: printf("‖v̂‖ = %.4f\n", vector_norm(v_normalized, 2));.

    // 3. 向量夾角
    print_separator("3. 向量夾角 (Vector Angle)");  // EN: Execute a statement: print_separator("3. 向量夾角 (Vector Angle)");.

    double a[] = {1.0, 0.0};  // EN: Execute a statement: double a[] = {1.0, 0.0};.
    double b[] = {1.0, 1.0};  // EN: Execute a statement: double b[] = {1.0, 1.0};.

    print_vector("a", a, 2);  // EN: Execute a statement: print_vector("a", a, 2);.
    print_vector("b", b, 2);  // EN: Execute a statement: print_vector("b", b, 2);.

    double theta = vector_angle(a, b, 2);  // EN: Execute a statement: double theta = vector_angle(a, b, 2);.
    printf("\n夾角 θ = %.4f rad = %.2f°\n", theta, theta * 180.0 / PI);  // EN: Execute a statement: printf("\n夾角 θ = %.4f rad = %.2f°\n", theta, theta * 180.0 / PI);.
    printf("cos θ = %.4f\n", cos(theta));  // EN: Execute a statement: printf("cos θ = %.4f\n", cos(theta));.
    printf("預期：cos 45° = 1/√2 ≈ 0.7071\n");  // EN: Execute a statement: printf("預期：cos 45° = 1/√2 ≈ 0.7071\n");.

    // 4. 正交性判斷
    print_separator("4. 正交性判斷 (Orthogonality Check)");  // EN: Execute a statement: print_separator("4. 正交性判斷 (Orthogonality Check)");.

    double u1[] = {1.0, 2.0};  // EN: Execute a statement: double u1[] = {1.0, 2.0};.
    double u2[] = {-2.0, 1.0};  // EN: Execute a statement: double u2[] = {-2.0, 1.0};.

    print_vector("u₁", u1, 2);  // EN: Execute a statement: print_vector("u₁", u1, 2);.
    print_vector("u₂", u2, 2);  // EN: Execute a statement: print_vector("u₂", u2, 2);.
    printf("\nu₁ · u₂ = %.4f\n", dot_product(u1, u2, 2));  // EN: Execute a statement: printf("\nu₁ · u₂ = %.4f\n", dot_product(u1, u2, 2));.
    printf("u₁ ⊥ u₂？ %s\n", is_orthogonal(u1, u2, 2) ? "true" : "false");  // EN: Execute a statement: printf("u₁ ⊥ u₂？ %s\n", is_orthogonal(u1, u2, 2) ? "true" : "false");.

    // 非正交
    double w1[] = {1.0, 1.0};  // EN: Execute a statement: double w1[] = {1.0, 1.0};.
    double w2[] = {1.0, 2.0};  // EN: Execute a statement: double w2[] = {1.0, 2.0};.

    printf("\n另一組：\n");  // EN: Execute a statement: printf("\n另一組：\n");.
    print_vector("w₁", w1, 2);  // EN: Execute a statement: print_vector("w₁", w1, 2);.
    print_vector("w₂", w2, 2);  // EN: Execute a statement: print_vector("w₂", w2, 2);.
    printf("w₁ · w₂ = %.4f\n", dot_product(w1, w2, 2));  // EN: Execute a statement: printf("w₁ · w₂ = %.4f\n", dot_product(w1, w2, 2));.
    printf("w₁ ⊥ w₂？ %s\n", is_orthogonal(w1, w2, 2) ? "true" : "false");  // EN: Execute a statement: printf("w₁ ⊥ w₂？ %s\n", is_orthogonal(w1, w2, 2) ? "true" : "false");.

    // 5. 正交矩陣
    print_separator("5. 正交矩陣 (Orthogonal Matrix)");  // EN: Execute a statement: print_separator("5. 正交矩陣 (Orthogonal Matrix)");.

    double angle = PI / 4;  // EN: Execute a statement: double angle = PI / 4;.
    double Q[MAX_DIM][MAX_DIM] = {  // EN: Execute line: double Q[MAX_DIM][MAX_DIM] = {.
        {cos(angle), -sin(angle)},  // EN: Execute line: {cos(angle), -sin(angle)},.
        {sin(angle), cos(angle)}  // EN: Execute line: {sin(angle), cos(angle)}.
    };  // EN: Structure delimiter for a block or scope.

    printf("旋轉矩陣（θ = 45°）：\n");  // EN: Execute a statement: printf("旋轉矩陣（θ = 45°）：\n");.
    print_matrix("Q", Q, 2, 2);  // EN: Execute a statement: print_matrix("Q", Q, 2, 2);.

    double QT[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double QT[MAX_DIM][MAX_DIM];.
    matrix_transpose(Q, QT, 2, 2);  // EN: Execute a statement: matrix_transpose(Q, QT, 2, 2);.
    print_matrix("\nQᵀ", QT, 2, 2);  // EN: Execute a statement: print_matrix("\nQᵀ", QT, 2, 2);.

    double QTQ[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double QTQ[MAX_DIM][MAX_DIM];.
    matrix_multiply(QT, Q, QTQ, 2, 2, 2);  // EN: Execute a statement: matrix_multiply(QT, Q, QTQ, 2, 2, 2);.
    print_matrix("\nQᵀQ", QTQ, 2, 2);  // EN: Execute a statement: print_matrix("\nQᵀQ", QTQ, 2, 2);.

    printf("\nQ 是正交矩陣？ %s\n", is_orthogonal_matrix(Q, 2) ? "true" : "false");  // EN: Execute a statement: printf("\nQ 是正交矩陣？ %s\n", is_orthogonal_matrix(Q, 2) ? "true" : "false"….

    // 驗證保長度
    double x_vec[] = {3.0, 4.0};  // EN: Execute a statement: double x_vec[] = {3.0, 4.0};.
    double Qx[2];  // EN: Execute a statement: double Qx[2];.
    matrix_vector_multiply(Q, x_vec, Qx, 2, 2);  // EN: Execute a statement: matrix_vector_multiply(Q, x_vec, Qx, 2, 2);.

    printf("\n保長度驗證：\n");  // EN: Execute a statement: printf("\n保長度驗證：\n");.
    print_vector("x", x_vec, 2);  // EN: Execute a statement: print_vector("x", x_vec, 2);.
    print_vector("Qx", Qx, 2);  // EN: Execute a statement: print_vector("Qx", Qx, 2);.
    printf("‖x‖ = %.4f\n", vector_norm(x_vec, 2));  // EN: Execute a statement: printf("‖x‖ = %.4f\n", vector_norm(x_vec, 2));.
    printf("‖Qx‖ = %.4f\n", vector_norm(Qx, 2));  // EN: Execute a statement: printf("‖Qx‖ = %.4f\n", vector_norm(Qx, 2));.

    // 6. Cauchy-Schwarz 不等式
    print_separator("6. Cauchy-Schwarz 不等式");  // EN: Execute a statement: print_separator("6. Cauchy-Schwarz 不等式");.

    double cs_x[] = {1.0, 2.0, 3.0};  // EN: Execute a statement: double cs_x[] = {1.0, 2.0, 3.0};.
    double cs_y[] = {4.0, 5.0, 6.0};  // EN: Execute a statement: double cs_y[] = {4.0, 5.0, 6.0};.

    print_vector("x", cs_x, 3);  // EN: Execute a statement: print_vector("x", cs_x, 3);.
    print_vector("y", cs_y, 3);  // EN: Execute a statement: print_vector("y", cs_y, 3);.

    double left_side = fabs(dot_product(cs_x, cs_y, 3));  // EN: Execute a statement: double left_side = fabs(dot_product(cs_x, cs_y, 3));.
    double right_side = vector_norm(cs_x, 3) * vector_norm(cs_y, 3);  // EN: Execute a statement: double right_side = vector_norm(cs_x, 3) * vector_norm(cs_y, 3);.

    printf("\n|x · y| = %.4f\n", left_side);  // EN: Execute a statement: printf("\n|x · y| = %.4f\n", left_side);.
    printf("‖x‖ ‖y‖ = %.4f\n", right_side);  // EN: Execute a statement: printf("‖x‖ ‖y‖ = %.4f\n", right_side);.
    printf("|x · y| ≤ ‖x‖ ‖y‖？ %s\n", (left_side <= right_side + EPSILON) ? "true" : "false");  // EN: Execute a statement: printf("|x · y| ≤ ‖x‖ ‖y‖？ %s\n", (left_side <= right_side + EPSILON) ?….

    // 總結
    print_separator("總結");  // EN: Execute a statement: print_separator("總結");.
    printf("\n");  // EN: Execute a statement: printf("\n");.
    printf("內積與正交性的核心公式：\n\n");  // EN: Execute a statement: printf("內積與正交性的核心公式：\n\n");.
    printf("1. 內積：x · y = Σ xᵢyᵢ\n\n");  // EN: Execute a statement: printf("1. 內積：x · y = Σ xᵢyᵢ\n\n");.
    printf("2. 長度：‖x‖ = √(x · x)\n\n");  // EN: Execute a statement: printf("2. 長度：‖x‖ = √(x · x)\n\n");.
    printf("3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖)\n\n");  // EN: Execute a statement: printf("3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖)\n\n");.
    printf("4. 正交：x ⊥ y ⟺ x · y = 0\n\n");  // EN: Execute a statement: printf("4. 正交：x ⊥ y ⟺ x · y = 0\n\n");.
    printf("5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ\n\n");  // EN: Execute a statement: printf("5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ\n\n");.

    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n示範完成！\n");  // EN: Execute a statement: printf("\n示範完成！\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
