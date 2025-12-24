/**
 * 克萊姆法則 (Cramer's Rule)
 *
 * 編譯：gcc -std=c99 -O2 cramers_rule.c -o cramers_rule -lm
 * 執行：./cramers_rule
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

void print_matrix(const char* name, double M[MAX_DIM][MAX_DIM], int n) {
    printf("%s =\n", name);
    for (int i = 0; i < n; i++) {
        printf("  [");
        for (int j = 0; j < n; j++) {
            printf("%8.4f", M[i][j]);
            if (j < n - 1) printf(", ");
        }
        printf("]\n");
    }
}

void print_vector(const char* name, double* v, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("%.4f", v[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

// 2×2 行列式
double det_2x2(double A[MAX_DIM][MAX_DIM]) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

// 3×3 行列式
double det_3x3(double A[MAX_DIM][MAX_DIM]) {
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

double determinant(double A[MAX_DIM][MAX_DIM], int n) {
    if (n == 2) return det_2x2(A);
    if (n == 3) return det_3x3(A);
    return 0.0;
}

// 替換第 j 行
void replace_column(double A[MAX_DIM][MAX_DIM], double* b, int j, int n,
                    double Aj[MAX_DIM][MAX_DIM]) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            Aj[i][k] = (k == j) ? b[i] : A[i][k];
        }
    }
}

// 克萊姆法則
int cramers_rule(double A[MAX_DIM][MAX_DIM], double* b, int n, double* x) {
    double det_A = determinant(A, n);

    if (fabs(det_A) < 1e-10) {
        return 0;  // 矩陣奇異
    }

    for (int j = 0; j < n; j++) {
        double Aj[MAX_DIM][MAX_DIM];
        replace_column(A, b, j, n, Aj);
        x[j] = determinant(Aj, n) / det_A;
    }
    return 1;
}

int main() {
    print_separator("克萊姆法則示範 (C)");

    // ========================================
    // 1. 2×2 系統
    // ========================================
    print_separator("1. 2×2 系統");

    double A2[MAX_DIM][MAX_DIM] = {{2, 3}, {4, 5}};
    double b2[] = {8, 14};

    printf("方程組：\n");
    printf("  2x + 3y = 8\n");
    printf("  4x + 5y = 14\n");

    print_matrix("\nA", A2, 2);
    print_vector("b", b2, 2);

    double det_A2 = determinant(A2, 2);
    printf("\ndet(A) = %.4f\n", det_A2);

    double x2[MAX_DIM];
    cramers_rule(A2, b2, 2, x2);

    for (int j = 0; j < 2; j++) {
        double Aj[MAX_DIM][MAX_DIM];
        replace_column(A2, b2, j, 2, Aj);
        double det_Aj = determinant(Aj, 2);
        printf("\nA%d（第 %d 行換成 b）：\n", j+1, j+1);
        print_matrix("", Aj, 2);
        printf("det(A%d) = %.4f\n", j+1, det_Aj);
        printf("x%d = %.4f\n", j+1, x2[j]);
    }

    printf("\n解：x = %.4f, y = %.4f\n", x2[0], x2[1]);

    // ========================================
    // 2. 3×3 系統
    // ========================================
    print_separator("2. 3×3 系統");

    double A3[MAX_DIM][MAX_DIM] = {
        {2, 1, -1},
        {-3, -1, 2},
        {-2, 1, 2}
    };
    double b3[] = {8, -11, -3};

    printf("方程組：\n");
    printf("   2x +  y -  z =  8\n");
    printf("  -3x -  y + 2z = -11\n");
    printf("  -2x +  y + 2z = -3\n");

    print_matrix("\nA", A3, 3);
    print_vector("b", b3, 3);

    double x3[MAX_DIM];
    cramers_rule(A3, b3, 3, x3);

    printf("\n解：x = %.4f, y = %.4f, z = %.4f\n", x3[0], x3[1], x3[2]);

    // 驗證
    printf("\n驗證：\n");
    printf("  2(%.0f) + (%.0f) - (%.0f) = %.4f\n",
        x3[0], x3[1], x3[2], 2*x3[0] + x3[1] - x3[2]);
    printf("  -3(%.0f) - (%.0f) + 2(%.0f) = %.4f\n",
        x3[0], x3[1], x3[2], -3*x3[0] - x3[1] + 2*x3[2]);
    printf("  -2(%.0f) + (%.0f) + 2(%.0f) = %.4f\n",
        x3[0], x3[1], x3[2], -2*x3[0] + x3[1] + 2*x3[2]);

    // 總結
    print_separator("總結");
    printf("\n克萊姆法則：\n");
    printf("  xⱼ = det(Aⱼ) / det(A)\n");
    printf("  Aⱼ = A 的第 j 行換成 b\n\n");
    printf("適用條件：\n");
    printf("  - det(A) ≠ 0\n");
    printf("  - 方陣系統\n\n");
    printf("時間複雜度：O(n! × n)\n\n");

    for (int i = 0; i < 60; i++) printf("=");
    printf("\n示範完成！\n");
    for (int i = 0; i < 60; i++) printf("=");
    printf("\n");

    return 0;
}
