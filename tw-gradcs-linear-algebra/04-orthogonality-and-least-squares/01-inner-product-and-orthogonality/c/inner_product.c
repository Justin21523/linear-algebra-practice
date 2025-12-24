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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define EPSILON 1e-10
#define PI 3.14159265358979323846
#define MAX_DIM 10

// ========================================
// 輔助函數
// ========================================

void print_separator(const char* title) {
    printf("\n");
    for (int i = 0; i < 60; i++) printf("=");
    printf("\n%s\n", title);
    for (int i = 0; i < 60; i++) printf("=");
    printf("\n");
}

void print_vector(const char* name, const double* v, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("%.4f", v[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

void print_matrix(const char* name, const double M[MAX_DIM][MAX_DIM], int rows, int cols) {
    printf("%s =\n", name);
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            printf("%8.4f", M[i][j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

// ========================================
// 向量運算
// ========================================

/**
 * 計算兩向量的內積 (Dot Product)
 * x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ
 */
double dot_product(const double* x, const double* y, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
}

/**
 * 計算向量的長度（L2 範數）
 * ‖x‖ = √(x · x)
 */
double vector_norm(const double* x, int n) {
    return sqrt(dot_product(x, x, n));
}

/**
 * 正規化向量為單位向量
 * result = x / ‖x‖
 */
void normalize(const double* x, double* result, int n) {
    double norm = vector_norm(x, n);
    if (norm < EPSILON) {
        printf("錯誤：零向量無法正規化\n");
        return;
    }
    for (int i = 0; i < n; i++) {
        result[i] = x[i] / norm;
    }
}

/**
 * 計算兩向量的夾角（弧度）
 * cos θ = (x · y) / (‖x‖ ‖y‖)
 */
double vector_angle(const double* x, const double* y, int n) {
    double dot = dot_product(x, y, n);
    double norm_x = vector_norm(x, n);
    double norm_y = vector_norm(y, n);

    if (norm_x < EPSILON || norm_y < EPSILON) {
        printf("錯誤：零向量沒有定義夾角\n");
        return 0.0;
    }

    double cos_theta = dot / (norm_x * norm_y);
    // 處理浮點數誤差
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;

    return acos(cos_theta);
}

/**
 * 判斷兩向量是否正交
 * x ⊥ y ⟺ x · y = 0
 */
bool is_orthogonal(const double* x, const double* y, int n) {
    return fabs(dot_product(x, y, n)) < EPSILON;
}

// ========================================
// 矩陣運算
// ========================================

/**
 * 矩陣轉置
 */
void matrix_transpose(const double A[MAX_DIM][MAX_DIM], double result[MAX_DIM][MAX_DIM],
                      int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = A[i][j];
        }
    }
}

/**
 * 矩陣乘法
 */
void matrix_multiply(const double A[MAX_DIM][MAX_DIM], const double B[MAX_DIM][MAX_DIM],
                     double result[MAX_DIM][MAX_DIM], int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = 0.0;
            for (int p = 0; p < k; p++) {
                result[i][j] += A[i][p] * B[p][j];
            }
        }
    }
}

/**
 * 矩陣乘向量
 */
void matrix_vector_multiply(const double A[MAX_DIM][MAX_DIM], const double* x,
                            double* result, int m, int n) {
    for (int i = 0; i < m; i++) {
        result[i] = 0.0;
        for (int j = 0; j < n; j++) {
            result[i] += A[i][j] * x[j];
        }
    }
}

/**
 * 判斷是否為單位矩陣
 */
bool is_identity(const double A[MAX_DIM][MAX_DIM], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(A[i][j] - expected) > EPSILON) {
                return false;
            }
        }
    }
    return true;
}

/**
 * 判斷矩陣是否為正交矩陣
 * QᵀQ = I
 */
bool is_orthogonal_matrix(const double Q[MAX_DIM][MAX_DIM], int n) {
    double QT[MAX_DIM][MAX_DIM];
    double product[MAX_DIM][MAX_DIM];

    matrix_transpose(Q, QT, n, n);
    matrix_multiply(QT, Q, product, n, n, n);

    return is_identity(product, n);
}

// ========================================
// 主程式
// ========================================

int main() {
    print_separator("內積與正交性示範 (C)\nInner Product & Orthogonality Demo");

    // 1. 內積計算
    print_separator("1. 內積計算 (Dot Product)");

    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};

    print_vector("x", x, 3);
    print_vector("y", y, 3);
    printf("\nx · y = %.4f\n", dot_product(x, y, 3));
    printf("計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32\n");

    // 2. 向量長度
    print_separator("2. 向量長度 (Vector Norm)");

    double v[] = {3.0, 4.0};
    print_vector("v", v, 2);
    printf("‖v‖ = %.4f\n", vector_norm(v, 2));
    printf("計算：√(3² + 4²) = √25 = 5\n");

    // 正規化
    double v_normalized[2];
    normalize(v, v_normalized, 2);
    printf("\n單位向量：\n");
    print_vector("v̂ = v/‖v‖", v_normalized, 2);
    printf("‖v̂‖ = %.4f\n", vector_norm(v_normalized, 2));

    // 3. 向量夾角
    print_separator("3. 向量夾角 (Vector Angle)");

    double a[] = {1.0, 0.0};
    double b[] = {1.0, 1.0};

    print_vector("a", a, 2);
    print_vector("b", b, 2);

    double theta = vector_angle(a, b, 2);
    printf("\n夾角 θ = %.4f rad = %.2f°\n", theta, theta * 180.0 / PI);
    printf("cos θ = %.4f\n", cos(theta));
    printf("預期：cos 45° = 1/√2 ≈ 0.7071\n");

    // 4. 正交性判斷
    print_separator("4. 正交性判斷 (Orthogonality Check)");

    double u1[] = {1.0, 2.0};
    double u2[] = {-2.0, 1.0};

    print_vector("u₁", u1, 2);
    print_vector("u₂", u2, 2);
    printf("\nu₁ · u₂ = %.4f\n", dot_product(u1, u2, 2));
    printf("u₁ ⊥ u₂？ %s\n", is_orthogonal(u1, u2, 2) ? "true" : "false");

    // 非正交
    double w1[] = {1.0, 1.0};
    double w2[] = {1.0, 2.0};

    printf("\n另一組：\n");
    print_vector("w₁", w1, 2);
    print_vector("w₂", w2, 2);
    printf("w₁ · w₂ = %.4f\n", dot_product(w1, w2, 2));
    printf("w₁ ⊥ w₂？ %s\n", is_orthogonal(w1, w2, 2) ? "true" : "false");

    // 5. 正交矩陣
    print_separator("5. 正交矩陣 (Orthogonal Matrix)");

    double angle = PI / 4;
    double Q[MAX_DIM][MAX_DIM] = {
        {cos(angle), -sin(angle)},
        {sin(angle), cos(angle)}
    };

    printf("旋轉矩陣（θ = 45°）：\n");
    print_matrix("Q", Q, 2, 2);

    double QT[MAX_DIM][MAX_DIM];
    matrix_transpose(Q, QT, 2, 2);
    print_matrix("\nQᵀ", QT, 2, 2);

    double QTQ[MAX_DIM][MAX_DIM];
    matrix_multiply(QT, Q, QTQ, 2, 2, 2);
    print_matrix("\nQᵀQ", QTQ, 2, 2);

    printf("\nQ 是正交矩陣？ %s\n", is_orthogonal_matrix(Q, 2) ? "true" : "false");

    // 驗證保長度
    double x_vec[] = {3.0, 4.0};
    double Qx[2];
    matrix_vector_multiply(Q, x_vec, Qx, 2, 2);

    printf("\n保長度驗證：\n");
    print_vector("x", x_vec, 2);
    print_vector("Qx", Qx, 2);
    printf("‖x‖ = %.4f\n", vector_norm(x_vec, 2));
    printf("‖Qx‖ = %.4f\n", vector_norm(Qx, 2));

    // 6. Cauchy-Schwarz 不等式
    print_separator("6. Cauchy-Schwarz 不等式");

    double cs_x[] = {1.0, 2.0, 3.0};
    double cs_y[] = {4.0, 5.0, 6.0};

    print_vector("x", cs_x, 3);
    print_vector("y", cs_y, 3);

    double left_side = fabs(dot_product(cs_x, cs_y, 3));
    double right_side = vector_norm(cs_x, 3) * vector_norm(cs_y, 3);

    printf("\n|x · y| = %.4f\n", left_side);
    printf("‖x‖ ‖y‖ = %.4f\n", right_side);
    printf("|x · y| ≤ ‖x‖ ‖y‖？ %s\n", (left_side <= right_side + EPSILON) ? "true" : "false");

    // 總結
    print_separator("總結");
    printf("\n");
    printf("內積與正交性的核心公式：\n\n");
    printf("1. 內積：x · y = Σ xᵢyᵢ\n\n");
    printf("2. 長度：‖x‖ = √(x · x)\n\n");
    printf("3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖)\n\n");
    printf("4. 正交：x ⊥ y ⟺ x · y = 0\n\n");
    printf("5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ\n\n");

    for (int i = 0; i < 60; i++) printf("=");
    printf("\n示範完成！\n");
    for (int i = 0; i < 60; i++) printf("=");
    printf("\n");

    return 0;
}
