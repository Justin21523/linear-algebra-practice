/**
 * 克萊姆法則 (Cramer's Rule)
 *
 * 編譯：gcc -std=c99 -O2 cramers_rule.c -o cramers_rule -lm
 * 執行：./cramers_rule
 */

#include <stdio.h>  // EN: Include a header dependency: #include <stdio.h>.
#include <math.h>  // EN: Include a header dependency: #include <math.h>.

#define MAX_DIM 10  // EN: Define a preprocessor macro: #define MAX_DIM 10.

void print_separator(const char* title) {  // EN: Execute line: void print_separator(const char* title) {.
    printf("\n");  // EN: Execute a statement: printf("\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n%s\n", title);  // EN: Execute a statement: printf("\n%s\n", title);.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.
}  // EN: Structure delimiter for a block or scope.

void print_matrix(const char* name, double M[MAX_DIM][MAX_DIM], int n) {  // EN: Execute line: void print_matrix(const char* name, double M[MAX_DIM][MAX_DIM], int n) {.
    printf("%s =\n", name);  // EN: Execute a statement: printf("%s =\n", name);.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        printf("  [");  // EN: Execute a statement: printf(" [");.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            printf("%8.4f", M[i][j]);  // EN: Execute a statement: printf("%8.4f", M[i][j]);.
            if (j < n - 1) printf(", ");  // EN: Conditional control flow: if (j < n - 1) printf(", ");.
        }  // EN: Structure delimiter for a block or scope.
        printf("]\n");  // EN: Execute a statement: printf("]\n");.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

void print_vector(const char* name, double* v, int n) {  // EN: Execute line: void print_vector(const char* name, double* v, int n) {.
    printf("%s = [", name);  // EN: Execute a statement: printf("%s = [", name);.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        printf("%.4f", v[i]);  // EN: Execute a statement: printf("%.4f", v[i]);.
        if (i < n - 1) printf(", ");  // EN: Conditional control flow: if (i < n - 1) printf(", ");.
    }  // EN: Structure delimiter for a block or scope.
    printf("]\n");  // EN: Execute a statement: printf("]\n");.
}  // EN: Structure delimiter for a block or scope.

// 2×2 行列式
double det_2x2(double A[MAX_DIM][MAX_DIM]) {  // EN: Execute line: double det_2x2(double A[MAX_DIM][MAX_DIM]) {.
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Return from the current function: return A[0][0] * A[1][1] - A[0][1] * A[1][0];.
}  // EN: Structure delimiter for a block or scope.

// 3×3 行列式
double det_3x3(double A[MAX_DIM][MAX_DIM]) {  // EN: Execute line: double det_3x3(double A[MAX_DIM][MAX_DIM]) {.
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])  // EN: Return from the current function: return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]).
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])  // EN: Execute line: - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]).
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);  // EN: Execute a statement: + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);.
}  // EN: Structure delimiter for a block or scope.

double determinant(double A[MAX_DIM][MAX_DIM], int n) {  // EN: Execute line: double determinant(double A[MAX_DIM][MAX_DIM], int n) {.
    if (n == 2) return det_2x2(A);  // EN: Conditional control flow: if (n == 2) return det_2x2(A);.
    if (n == 3) return det_3x3(A);  // EN: Conditional control flow: if (n == 3) return det_3x3(A);.
    return 0.0;  // EN: Return from the current function: return 0.0;.
}  // EN: Structure delimiter for a block or scope.

// 替換第 j 行
void replace_column(double A[MAX_DIM][MAX_DIM], double* b, int j, int n,  // EN: Execute line: void replace_column(double A[MAX_DIM][MAX_DIM], double* b, int j, int n,.
                    double Aj[MAX_DIM][MAX_DIM]) {  // EN: Execute line: double Aj[MAX_DIM][MAX_DIM]) {.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        for (int k = 0; k < n; k++) {  // EN: Loop control flow: for (int k = 0; k < n; k++) {.
            Aj[i][k] = (k == j) ? b[i] : A[i][k];  // EN: Execute a statement: Aj[i][k] = (k == j) ? b[i] : A[i][k];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// 克萊姆法則
int cramers_rule(double A[MAX_DIM][MAX_DIM], double* b, int n, double* x) {  // EN: Execute line: int cramers_rule(double A[MAX_DIM][MAX_DIM], double* b, int n, double* ….
    double det_A = determinant(A, n);  // EN: Execute a statement: double det_A = determinant(A, n);.

    if (fabs(det_A) < 1e-10) {  // EN: Conditional control flow: if (fabs(det_A) < 1e-10) {.
        return 0;  // 矩陣奇異  // EN: Return from the current function: return 0; // 矩陣奇異.
    }  // EN: Structure delimiter for a block or scope.

    for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
        double Aj[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double Aj[MAX_DIM][MAX_DIM];.
        replace_column(A, b, j, n, Aj);  // EN: Execute a statement: replace_column(A, b, j, n, Aj);.
        x[j] = determinant(Aj, n) / det_A;  // EN: Execute a statement: x[j] = determinant(Aj, n) / det_A;.
    }  // EN: Structure delimiter for a block or scope.
    return 1;  // EN: Return from the current function: return 1;.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    print_separator("克萊姆法則示範 (C)");  // EN: Execute a statement: print_separator("克萊姆法則示範 (C)");.

    // ========================================
    // 1. 2×2 系統
    // ========================================
    print_separator("1. 2×2 系統");  // EN: Execute a statement: print_separator("1. 2×2 系統");.

    double A2[MAX_DIM][MAX_DIM] = {{2, 3}, {4, 5}};  // EN: Execute a statement: double A2[MAX_DIM][MAX_DIM] = {{2, 3}, {4, 5}};.
    double b2[] = {8, 14};  // EN: Execute a statement: double b2[] = {8, 14};.

    printf("方程組：\n");  // EN: Execute a statement: printf("方程組：\n");.
    printf("  2x + 3y = 8\n");  // EN: Execute a statement: printf(" 2x + 3y = 8\n");.
    printf("  4x + 5y = 14\n");  // EN: Execute a statement: printf(" 4x + 5y = 14\n");.

    print_matrix("\nA", A2, 2);  // EN: Execute a statement: print_matrix("\nA", A2, 2);.
    print_vector("b", b2, 2);  // EN: Execute a statement: print_vector("b", b2, 2);.

    double det_A2 = determinant(A2, 2);  // EN: Execute a statement: double det_A2 = determinant(A2, 2);.
    printf("\ndet(A) = %.4f\n", det_A2);  // EN: Execute a statement: printf("\ndet(A) = %.4f\n", det_A2);.

    double x2[MAX_DIM];  // EN: Execute a statement: double x2[MAX_DIM];.
    cramers_rule(A2, b2, 2, x2);  // EN: Execute a statement: cramers_rule(A2, b2, 2, x2);.

    for (int j = 0; j < 2; j++) {  // EN: Loop control flow: for (int j = 0; j < 2; j++) {.
        double Aj[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double Aj[MAX_DIM][MAX_DIM];.
        replace_column(A2, b2, j, 2, Aj);  // EN: Execute a statement: replace_column(A2, b2, j, 2, Aj);.
        double det_Aj = determinant(Aj, 2);  // EN: Execute a statement: double det_Aj = determinant(Aj, 2);.
        printf("\nA%d（第 %d 行換成 b）：\n", j+1, j+1);  // EN: Execute a statement: printf("\nA%d（第 %d 行換成 b）：\n", j+1, j+1);.
        print_matrix("", Aj, 2);  // EN: Execute a statement: print_matrix("", Aj, 2);.
        printf("det(A%d) = %.4f\n", j+1, det_Aj);  // EN: Execute a statement: printf("det(A%d) = %.4f\n", j+1, det_Aj);.
        printf("x%d = %.4f\n", j+1, x2[j]);  // EN: Execute a statement: printf("x%d = %.4f\n", j+1, x2[j]);.
    }  // EN: Structure delimiter for a block or scope.

    printf("\n解：x = %.4f, y = %.4f\n", x2[0], x2[1]);  // EN: Execute a statement: printf("\n解：x = %.4f, y = %.4f\n", x2[0], x2[1]);.

    // ========================================
    // 2. 3×3 系統
    // ========================================
    print_separator("2. 3×3 系統");  // EN: Execute a statement: print_separator("2. 3×3 系統");.

    double A3[MAX_DIM][MAX_DIM] = {  // EN: Execute line: double A3[MAX_DIM][MAX_DIM] = {.
        {2, 1, -1},  // EN: Execute line: {2, 1, -1},.
        {-3, -1, 2},  // EN: Execute line: {-3, -1, 2},.
        {-2, 1, 2}  // EN: Execute line: {-2, 1, 2}.
    };  // EN: Structure delimiter for a block or scope.
    double b3[] = {8, -11, -3};  // EN: Execute a statement: double b3[] = {8, -11, -3};.

    printf("方程組：\n");  // EN: Execute a statement: printf("方程組：\n");.
    printf("   2x +  y -  z =  8\n");  // EN: Execute a statement: printf(" 2x + y - z = 8\n");.
    printf("  -3x -  y + 2z = -11\n");  // EN: Execute a statement: printf(" -3x - y + 2z = -11\n");.
    printf("  -2x +  y + 2z = -3\n");  // EN: Execute a statement: printf(" -2x + y + 2z = -3\n");.

    print_matrix("\nA", A3, 3);  // EN: Execute a statement: print_matrix("\nA", A3, 3);.
    print_vector("b", b3, 3);  // EN: Execute a statement: print_vector("b", b3, 3);.

    double x3[MAX_DIM];  // EN: Execute a statement: double x3[MAX_DIM];.
    cramers_rule(A3, b3, 3, x3);  // EN: Execute a statement: cramers_rule(A3, b3, 3, x3);.

    printf("\n解：x = %.4f, y = %.4f, z = %.4f\n", x3[0], x3[1], x3[2]);  // EN: Execute a statement: printf("\n解：x = %.4f, y = %.4f, z = %.4f\n", x3[0], x3[1], x3[2]);.

    // 驗證
    printf("\n驗證：\n");  // EN: Execute a statement: printf("\n驗證：\n");.
    printf("  2(%.0f) + (%.0f) - (%.0f) = %.4f\n",  // EN: Execute line: printf(" 2(%.0f) + (%.0f) - (%.0f) = %.4f\n",.
        x3[0], x3[1], x3[2], 2*x3[0] + x3[1] - x3[2]);  // EN: Execute a statement: x3[0], x3[1], x3[2], 2*x3[0] + x3[1] - x3[2]);.
    printf("  -3(%.0f) - (%.0f) + 2(%.0f) = %.4f\n",  // EN: Execute line: printf(" -3(%.0f) - (%.0f) + 2(%.0f) = %.4f\n",.
        x3[0], x3[1], x3[2], -3*x3[0] - x3[1] + 2*x3[2]);  // EN: Execute a statement: x3[0], x3[1], x3[2], -3*x3[0] - x3[1] + 2*x3[2]);.
    printf("  -2(%.0f) + (%.0f) + 2(%.0f) = %.4f\n",  // EN: Execute line: printf(" -2(%.0f) + (%.0f) + 2(%.0f) = %.4f\n",.
        x3[0], x3[1], x3[2], -2*x3[0] + x3[1] + 2*x3[2]);  // EN: Execute a statement: x3[0], x3[1], x3[2], -2*x3[0] + x3[1] + 2*x3[2]);.

    // 總結
    print_separator("總結");  // EN: Execute a statement: print_separator("總結");.
    printf("\n克萊姆法則：\n");  // EN: Execute a statement: printf("\n克萊姆法則：\n");.
    printf("  xⱼ = det(Aⱼ) / det(A)\n");  // EN: Execute a statement: printf(" xⱼ = det(Aⱼ) / det(A)\n");.
    printf("  Aⱼ = A 的第 j 行換成 b\n\n");  // EN: Execute a statement: printf(" Aⱼ = A 的第 j 行換成 b\n\n");.
    printf("適用條件：\n");  // EN: Execute a statement: printf("適用條件：\n");.
    printf("  - det(A) ≠ 0\n");  // EN: Execute a statement: printf(" - det(A) ≠ 0\n");.
    printf("  - 方陣系統\n\n");  // EN: Execute a statement: printf(" - 方陣系統\n\n");.
    printf("時間複雜度：O(n! × n)\n\n");  // EN: Execute a statement: printf("時間複雜度：O(n! × n)\n\n");.

    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n示範完成！\n");  // EN: Execute a statement: printf("\n示範完成！\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
