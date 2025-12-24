/**
 * 行列式的性質 (Determinant Properties)
 *
 * 編譯：gcc -std=c99 -O2 determinant.c -o determinant -lm
 * 執行：./determinant
 */

#include <stdio.h>
#include <math.h>

#define MAX_DIM 10

void print_separator(const char* title) {
    printf("\n");
    for (int i = 0; i < 60; i++) printf("=");
    printf("\n%s\n", title);
    for (int i = 0; i < 60; i++) printf("=");
    printf("\n");
}

void print_matrix(const char* name, double M[MAX_DIM][MAX_DIM], int rows, int cols) {
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

// 2×2 行列式
double det_2x2(double A[MAX_DIM][MAX_DIM]) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

// 3×3 行列式
double det_3x3(double A[MAX_DIM][MAX_DIM]) {
    double a = A[0][0], b = A[0][1], c = A[0][2];
    double d = A[1][0], e = A[1][1], f = A[1][2];
    double g = A[2][0], h = A[2][1], i = A[2][2];

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

// n×n 行列式（列運算化為上三角）
double det_nxn(double A[MAX_DIM][MAX_DIM], int n) {
    double M[MAX_DIM][MAX_DIM];

    // 複製矩陣
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i][j] = A[i][j];
        }
    }

    int sign = 1;

    for (int col = 0; col < n; col++) {
        // 找主元
        int pivot_row = -1;
        for (int row = col; row < n; row++) {
            if (fabs(M[row][col]) > 1e-10) {
                pivot_row = row;
                break;
            }
        }

        if (pivot_row == -1) return 0.0;

        // 列交換
        if (pivot_row != col) {
            for (int j = 0; j < n; j++) {
                double temp = M[col][j];
                M[col][j] = M[pivot_row][j];
                M[pivot_row][j] = temp;
            }
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
void matrix_multiply(double A[MAX_DIM][MAX_DIM], double B[MAX_DIM][MAX_DIM],
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

// 矩陣轉置
void transpose(double A[MAX_DIM][MAX_DIM], double result[MAX_DIM][MAX_DIM], int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[j][i] = A[i][j];
        }
    }
}

// 純量乘矩陣
void scalar_multiply(double c, double A[MAX_DIM][MAX_DIM],
                     double result[MAX_DIM][MAX_DIM], int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = c * A[i][j];
        }
    }
}

int main() {
    print_separator("行列式性質示範 (C)");

    // ========================================
    // 1. 基本計算
    // ========================================
    print_separator("1. 基本行列式計算");

    double A2[MAX_DIM][MAX_DIM] = {{3, 8}, {4, 6}};
    print_matrix("A (2×2)", A2, 2, 2);
    printf("det(A) = %.4f\n", det_2x2(A2));

    double A3[MAX_DIM][MAX_DIM] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 10}
    };
    print_matrix("\nA (3×3)", A3, 3, 3);
    printf("det(A) = %.4f\n", det_3x3(A3));

    // ========================================
    // 2. 性質 1：det(I) = 1
    // ========================================
    print_separator("2. 性質 1：det(I) = 1");

    double I3[MAX_DIM][MAX_DIM] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    print_matrix("I₃", I3, 3, 3);
    printf("det(I₃) = %.4f\n", det_3x3(I3));

    // ========================================
    // 3. 性質 2：列交換變號
    // ========================================
    print_separator("3. 性質 2：列交換變號");

    double A[MAX_DIM][MAX_DIM] = {{1, 2}, {3, 4}};
    print_matrix("A", A, 2, 2);
    printf("det(A) = %.4f\n", det_2x2(A));

    double A_swap[MAX_DIM][MAX_DIM] = {{3, 4}, {1, 2}};
    print_matrix("\nA（交換列）", A_swap, 2, 2);
    printf("det(交換後) = %.4f\n", det_2x2(A_swap));
    printf("驗證：變號 ✓\n");

    // ========================================
    // 4. 乘積公式
    // ========================================
    print_separator("4. 乘積公式：det(AB) = det(A)·det(B)");

    double A4[MAX_DIM][MAX_DIM] = {{1, 2}, {3, 4}};
    double B[MAX_DIM][MAX_DIM] = {{5, 6}, {7, 8}};
    double AB[MAX_DIM][MAX_DIM];
    matrix_multiply(A4, B, AB, 2, 2, 2);

    print_matrix("A", A4, 2, 2);
    print_matrix("B", B, 2, 2);
    print_matrix("AB", AB, 2, 2);

    double detA = det_2x2(A4);
    double detB = det_2x2(B);
    double detAB = det_2x2(AB);

    printf("\ndet(A) = %.4f\n", detA);
    printf("det(B) = %.4f\n", detB);
    printf("det(A)·det(B) = %.4f\n", detA * detB);
    printf("det(AB) = %.4f\n", detAB);

    // ========================================
    // 5. 轉置公式
    // ========================================
    print_separator("5. 轉置公式：det(Aᵀ) = det(A)");

    double A5[MAX_DIM][MAX_DIM] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
    double AT[MAX_DIM][MAX_DIM];
    transpose(A5, AT, 3, 3);

    print_matrix("A", A5, 3, 3);
    print_matrix("Aᵀ", AT, 3, 3);

    printf("\ndet(A) = %.4f\n", det_3x3(A5));
    printf("det(Aᵀ) = %.4f\n", det_3x3(AT));

    // ========================================
    // 6. 純量乘法
    // ========================================
    print_separator("6. 純量乘法：det(cA) = cⁿ·det(A)");

    double A6[MAX_DIM][MAX_DIM] = {{1, 2}, {3, 4}};
    double c = 2;
    double cA[MAX_DIM][MAX_DIM];
    scalar_multiply(c, A6, cA, 2, 2);

    print_matrix("A (2×2)", A6, 2, 2);
    printf("c = %.0f\n", c);
    print_matrix("cA", cA, 2, 2);

    detA = det_2x2(A6);
    double detcA = det_2x2(cA);
    int n = 2;

    printf("\ndet(A) = %.4f\n", detA);
    printf("cⁿ·det(A) = %.0f² × %.4f = %.4f\n", c, detA, pow(c, n) * detA);
    printf("det(cA) = %.4f\n", detcA);

    // ========================================
    // 7. 上三角矩陣
    // ========================================
    print_separator("7. 上三角矩陣：det = 對角線乘積");

    double U[MAX_DIM][MAX_DIM] = {{2, 3, 1}, {0, 4, 5}, {0, 0, 6}};
    print_matrix("U（上三角）", U, 3, 3);
    printf("對角線乘積：2 × 4 × 6 = %d\n", 2 * 4 * 6);
    printf("det(U) = %.4f\n", det_3x3(U));

    // ========================================
    // 8. 奇異矩陣
    // ========================================
    print_separator("8. 奇異矩陣：det(A) = 0");

    double A_singular[MAX_DIM][MAX_DIM] = {{1, 2}, {2, 4}};
    print_matrix("A（列成比例）", A_singular, 2, 2);
    printf("det(A) = %.4f\n", det_2x2(A_singular));
    printf("此矩陣不可逆\n");

    // 總結
    print_separator("總結");
    printf("\n行列式三大性質：\n");
    printf("1. det(I) = 1\n");
    printf("2. 列交換 → det 變號\n");
    printf("3. 對單列線性\n\n");
    printf("重要公式：\n");
    printf("- det(AB) = det(A)·det(B)\n");
    printf("- det(Aᵀ) = det(A)\n");
    printf("- det(A⁻¹) = 1/det(A)\n");
    printf("- det(cA) = cⁿ·det(A)\n\n");

    for (int i = 0; i < 60; i++) printf("=");
    printf("\n示範完成！\n");
    for (int i = 0; i < 60; i++) printf("=");
    printf("\n");

    return 0;
}
