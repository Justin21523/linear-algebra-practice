/**
 * 行列式的性質 (Determinant Properties)
 *
 * 編譯：gcc -std=c99 -O2 determinant.c -o determinant -lm
 * 執行：./determinant
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

// 2×2 行列式
double det_2x2(double A[MAX_DIM][MAX_DIM]) {  // EN: Execute line: double det_2x2(double A[MAX_DIM][MAX_DIM]) {.
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Return from the current function: return A[0][0] * A[1][1] - A[0][1] * A[1][0];.
}  // EN: Structure delimiter for a block or scope.

// 3×3 行列式
double det_3x3(double A[MAX_DIM][MAX_DIM]) {  // EN: Execute line: double det_3x3(double A[MAX_DIM][MAX_DIM]) {.
    double a = A[0][0], b = A[0][1], c = A[0][2];  // EN: Execute a statement: double a = A[0][0], b = A[0][1], c = A[0][2];.
    double d = A[1][0], e = A[1][1], f = A[1][2];  // EN: Execute a statement: double d = A[1][0], e = A[1][1], f = A[1][2];.
    double g = A[2][0], h = A[2][1], i = A[2][2];  // EN: Execute a statement: double g = A[2][0], h = A[2][1], i = A[2][2];.

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);  // EN: Return from the current function: return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);.
}  // EN: Structure delimiter for a block or scope.

// n×n 行列式（列運算化為上三角）
double det_nxn(double A[MAX_DIM][MAX_DIM], int n) {  // EN: Execute line: double det_nxn(double A[MAX_DIM][MAX_DIM], int n) {.
    double M[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double M[MAX_DIM][MAX_DIM];.

    // 複製矩陣
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            M[i][j] = A[i][j];  // EN: Execute a statement: M[i][j] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    int sign = 1;  // EN: Execute a statement: int sign = 1;.

    for (int col = 0; col < n; col++) {  // EN: Loop control flow: for (int col = 0; col < n; col++) {.
        // 找主元
        int pivot_row = -1;  // EN: Execute a statement: int pivot_row = -1;.
        for (int row = col; row < n; row++) {  // EN: Loop control flow: for (int row = col; row < n; row++) {.
            if (fabs(M[row][col]) > 1e-10) {  // EN: Conditional control flow: if (fabs(M[row][col]) > 1e-10) {.
                pivot_row = row;  // EN: Execute a statement: pivot_row = row;.
                break;  // EN: Execute a statement: break;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.

        if (pivot_row == -1) return 0.0;  // EN: Conditional control flow: if (pivot_row == -1) return 0.0;.

        // 列交換
        if (pivot_row != col) {  // EN: Conditional control flow: if (pivot_row != col) {.
            for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
                double temp = M[col][j];  // EN: Execute a statement: double temp = M[col][j];.
                M[col][j] = M[pivot_row][j];  // EN: Execute a statement: M[col][j] = M[pivot_row][j];.
                M[pivot_row][j] = temp;  // EN: Execute a statement: M[pivot_row][j] = temp;.
            }  // EN: Structure delimiter for a block or scope.
            sign *= -1;  // EN: Execute a statement: sign *= -1;.
        }  // EN: Structure delimiter for a block or scope.

        // 消去
        for (int row = col + 1; row < n; row++) {  // EN: Loop control flow: for (int row = col + 1; row < n; row++) {.
            double factor = M[row][col] / M[col][col];  // EN: Execute a statement: double factor = M[row][col] / M[col][col];.
            for (int j = col; j < n; j++) {  // EN: Loop control flow: for (int j = col; j < n; j++) {.
                M[row][j] -= factor * M[col][j];  // EN: Execute a statement: M[row][j] -= factor * M[col][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    double det = sign;  // EN: Execute a statement: double det = sign;.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        det *= M[i][i];  // EN: Execute a statement: det *= M[i][i];.
    }  // EN: Structure delimiter for a block or scope.

    return det;  // EN: Return from the current function: return det;.
}  // EN: Structure delimiter for a block or scope.

// 矩陣乘法
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

// 矩陣轉置
void transpose(double A[MAX_DIM][MAX_DIM], double result[MAX_DIM][MAX_DIM], int m, int n) {  // EN: Execute line: void transpose(double A[MAX_DIM][MAX_DIM], double result[MAX_DIM][MAX_D….
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[j][i] = A[i][j];  // EN: Execute a statement: result[j][i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// 純量乘矩陣
void scalar_multiply(double c, double A[MAX_DIM][MAX_DIM],  // EN: Execute line: void scalar_multiply(double c, double A[MAX_DIM][MAX_DIM],.
                     double result[MAX_DIM][MAX_DIM], int m, int n) {  // EN: Execute line: double result[MAX_DIM][MAX_DIM], int m, int n) {.
    for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            result[i][j] = c * A[i][j];  // EN: Execute a statement: result[i][j] = c * A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    print_separator("行列式性質示範 (C)");  // EN: Execute a statement: print_separator("行列式性質示範 (C)");.

    // ========================================
    // 1. 基本計算
    // ========================================
    print_separator("1. 基本行列式計算");  // EN: Execute a statement: print_separator("1. 基本行列式計算");.

    double A2[MAX_DIM][MAX_DIM] = {{3, 8}, {4, 6}};  // EN: Execute a statement: double A2[MAX_DIM][MAX_DIM] = {{3, 8}, {4, 6}};.
    print_matrix("A (2×2)", A2, 2, 2);  // EN: Execute a statement: print_matrix("A (2×2)", A2, 2, 2);.
    printf("det(A) = %.4f\n", det_2x2(A2));  // EN: Execute a statement: printf("det(A) = %.4f\n", det_2x2(A2));.

    double A3[MAX_DIM][MAX_DIM] = {  // EN: Execute line: double A3[MAX_DIM][MAX_DIM] = {.
        {1, 2, 3},  // EN: Execute line: {1, 2, 3},.
        {4, 5, 6},  // EN: Execute line: {4, 5, 6},.
        {7, 8, 10}  // EN: Execute line: {7, 8, 10}.
    };  // EN: Structure delimiter for a block or scope.
    print_matrix("\nA (3×3)", A3, 3, 3);  // EN: Execute a statement: print_matrix("\nA (3×3)", A3, 3, 3);.
    printf("det(A) = %.4f\n", det_3x3(A3));  // EN: Execute a statement: printf("det(A) = %.4f\n", det_3x3(A3));.

    // ========================================
    // 2. 性質 1：det(I) = 1
    // ========================================
    print_separator("2. 性質 1：det(I) = 1");  // EN: Execute a statement: print_separator("2. 性質 1：det(I) = 1");.

    double I3[MAX_DIM][MAX_DIM] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};  // EN: Execute a statement: double I3[MAX_DIM][MAX_DIM] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};.
    print_matrix("I₃", I3, 3, 3);  // EN: Execute a statement: print_matrix("I₃", I3, 3, 3);.
    printf("det(I₃) = %.4f\n", det_3x3(I3));  // EN: Execute a statement: printf("det(I₃) = %.4f\n", det_3x3(I3));.

    // ========================================
    // 3. 性質 2：列交換變號
    // ========================================
    print_separator("3. 性質 2：列交換變號");  // EN: Execute a statement: print_separator("3. 性質 2：列交換變號");.

    double A[MAX_DIM][MAX_DIM] = {{1, 2}, {3, 4}};  // EN: Execute a statement: double A[MAX_DIM][MAX_DIM] = {{1, 2}, {3, 4}};.
    print_matrix("A", A, 2, 2);  // EN: Execute a statement: print_matrix("A", A, 2, 2);.
    printf("det(A) = %.4f\n", det_2x2(A));  // EN: Execute a statement: printf("det(A) = %.4f\n", det_2x2(A));.

    double A_swap[MAX_DIM][MAX_DIM] = {{3, 4}, {1, 2}};  // EN: Execute a statement: double A_swap[MAX_DIM][MAX_DIM] = {{3, 4}, {1, 2}};.
    print_matrix("\nA（交換列）", A_swap, 2, 2);  // EN: Execute a statement: print_matrix("\nA（交換列）", A_swap, 2, 2);.
    printf("det(交換後) = %.4f\n", det_2x2(A_swap));  // EN: Execute a statement: printf("det(交換後) = %.4f\n", det_2x2(A_swap));.
    printf("驗證：變號 ✓\n");  // EN: Execute a statement: printf("驗證：變號 ✓\n");.

    // ========================================
    // 4. 乘積公式
    // ========================================
    print_separator("4. 乘積公式：det(AB) = det(A)·det(B)");  // EN: Execute a statement: print_separator("4. 乘積公式：det(AB) = det(A)·det(B)");.

    double A4[MAX_DIM][MAX_DIM] = {{1, 2}, {3, 4}};  // EN: Execute a statement: double A4[MAX_DIM][MAX_DIM] = {{1, 2}, {3, 4}};.
    double B[MAX_DIM][MAX_DIM] = {{5, 6}, {7, 8}};  // EN: Execute a statement: double B[MAX_DIM][MAX_DIM] = {{5, 6}, {7, 8}};.
    double AB[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double AB[MAX_DIM][MAX_DIM];.
    matrix_multiply(A4, B, AB, 2, 2, 2);  // EN: Execute a statement: matrix_multiply(A4, B, AB, 2, 2, 2);.

    print_matrix("A", A4, 2, 2);  // EN: Execute a statement: print_matrix("A", A4, 2, 2);.
    print_matrix("B", B, 2, 2);  // EN: Execute a statement: print_matrix("B", B, 2, 2);.
    print_matrix("AB", AB, 2, 2);  // EN: Execute a statement: print_matrix("AB", AB, 2, 2);.

    double detA = det_2x2(A4);  // EN: Execute a statement: double detA = det_2x2(A4);.
    double detB = det_2x2(B);  // EN: Execute a statement: double detB = det_2x2(B);.
    double detAB = det_2x2(AB);  // EN: Execute a statement: double detAB = det_2x2(AB);.

    printf("\ndet(A) = %.4f\n", detA);  // EN: Execute a statement: printf("\ndet(A) = %.4f\n", detA);.
    printf("det(B) = %.4f\n", detB);  // EN: Execute a statement: printf("det(B) = %.4f\n", detB);.
    printf("det(A)·det(B) = %.4f\n", detA * detB);  // EN: Execute a statement: printf("det(A)·det(B) = %.4f\n", detA * detB);.
    printf("det(AB) = %.4f\n", detAB);  // EN: Execute a statement: printf("det(AB) = %.4f\n", detAB);.

    // ========================================
    // 5. 轉置公式
    // ========================================
    print_separator("5. 轉置公式：det(Aᵀ) = det(A)");  // EN: Execute a statement: print_separator("5. 轉置公式：det(Aᵀ) = det(A)");.

    double A5[MAX_DIM][MAX_DIM] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};  // EN: Execute a statement: double A5[MAX_DIM][MAX_DIM] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};.
    double AT[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double AT[MAX_DIM][MAX_DIM];.
    transpose(A5, AT, 3, 3);  // EN: Execute a statement: transpose(A5, AT, 3, 3);.

    print_matrix("A", A5, 3, 3);  // EN: Execute a statement: print_matrix("A", A5, 3, 3);.
    print_matrix("Aᵀ", AT, 3, 3);  // EN: Execute a statement: print_matrix("Aᵀ", AT, 3, 3);.

    printf("\ndet(A) = %.4f\n", det_3x3(A5));  // EN: Execute a statement: printf("\ndet(A) = %.4f\n", det_3x3(A5));.
    printf("det(Aᵀ) = %.4f\n", det_3x3(AT));  // EN: Execute a statement: printf("det(Aᵀ) = %.4f\n", det_3x3(AT));.

    // ========================================
    // 6. 純量乘法
    // ========================================
    print_separator("6. 純量乘法：det(cA) = cⁿ·det(A)");  // EN: Execute a statement: print_separator("6. 純量乘法：det(cA) = cⁿ·det(A)");.

    double A6[MAX_DIM][MAX_DIM] = {{1, 2}, {3, 4}};  // EN: Execute a statement: double A6[MAX_DIM][MAX_DIM] = {{1, 2}, {3, 4}};.
    double c = 2;  // EN: Execute a statement: double c = 2;.
    double cA[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double cA[MAX_DIM][MAX_DIM];.
    scalar_multiply(c, A6, cA, 2, 2);  // EN: Execute a statement: scalar_multiply(c, A6, cA, 2, 2);.

    print_matrix("A (2×2)", A6, 2, 2);  // EN: Execute a statement: print_matrix("A (2×2)", A6, 2, 2);.
    printf("c = %.0f\n", c);  // EN: Execute a statement: printf("c = %.0f\n", c);.
    print_matrix("cA", cA, 2, 2);  // EN: Execute a statement: print_matrix("cA", cA, 2, 2);.

    detA = det_2x2(A6);  // EN: Execute a statement: detA = det_2x2(A6);.
    double detcA = det_2x2(cA);  // EN: Execute a statement: double detcA = det_2x2(cA);.
    int n = 2;  // EN: Execute a statement: int n = 2;.

    printf("\ndet(A) = %.4f\n", detA);  // EN: Execute a statement: printf("\ndet(A) = %.4f\n", detA);.
    printf("cⁿ·det(A) = %.0f² × %.4f = %.4f\n", c, detA, pow(c, n) * detA);  // EN: Execute a statement: printf("cⁿ·det(A) = %.0f² × %.4f = %.4f\n", c, detA, pow(c, n) * detA);.
    printf("det(cA) = %.4f\n", detcA);  // EN: Execute a statement: printf("det(cA) = %.4f\n", detcA);.

    // ========================================
    // 7. 上三角矩陣
    // ========================================
    print_separator("7. 上三角矩陣：det = 對角線乘積");  // EN: Execute a statement: print_separator("7. 上三角矩陣：det = 對角線乘積");.

    double U[MAX_DIM][MAX_DIM] = {{2, 3, 1}, {0, 4, 5}, {0, 0, 6}};  // EN: Execute a statement: double U[MAX_DIM][MAX_DIM] = {{2, 3, 1}, {0, 4, 5}, {0, 0, 6}};.
    print_matrix("U（上三角）", U, 3, 3);  // EN: Execute a statement: print_matrix("U（上三角）", U, 3, 3);.
    printf("對角線乘積：2 × 4 × 6 = %d\n", 2 * 4 * 6);  // EN: Execute a statement: printf("對角線乘積：2 × 4 × 6 = %d\n", 2 * 4 * 6);.
    printf("det(U) = %.4f\n", det_3x3(U));  // EN: Execute a statement: printf("det(U) = %.4f\n", det_3x3(U));.

    // ========================================
    // 8. 奇異矩陣
    // ========================================
    print_separator("8. 奇異矩陣：det(A) = 0");  // EN: Execute a statement: print_separator("8. 奇異矩陣：det(A) = 0");.

    double A_singular[MAX_DIM][MAX_DIM] = {{1, 2}, {2, 4}};  // EN: Execute a statement: double A_singular[MAX_DIM][MAX_DIM] = {{1, 2}, {2, 4}};.
    print_matrix("A（列成比例）", A_singular, 2, 2);  // EN: Execute a statement: print_matrix("A（列成比例）", A_singular, 2, 2);.
    printf("det(A) = %.4f\n", det_2x2(A_singular));  // EN: Execute a statement: printf("det(A) = %.4f\n", det_2x2(A_singular));.
    printf("此矩陣不可逆\n");  // EN: Execute a statement: printf("此矩陣不可逆\n");.

    // 總結
    print_separator("總結");  // EN: Execute a statement: print_separator("總結");.
    printf("\n行列式三大性質：\n");  // EN: Execute a statement: printf("\n行列式三大性質：\n");.
    printf("1. det(I) = 1\n");  // EN: Execute a statement: printf("1. det(I) = 1\n");.
    printf("2. 列交換 → det 變號\n");  // EN: Execute a statement: printf("2. 列交換 → det 變號\n");.
    printf("3. 對單列線性\n\n");  // EN: Execute a statement: printf("3. 對單列線性\n\n");.
    printf("重要公式：\n");  // EN: Execute a statement: printf("重要公式：\n");.
    printf("- det(AB) = det(A)·det(B)\n");  // EN: Execute a statement: printf("- det(AB) = det(A)·det(B)\n");.
    printf("- det(Aᵀ) = det(A)\n");  // EN: Execute a statement: printf("- det(Aᵀ) = det(A)\n");.
    printf("- det(A⁻¹) = 1/det(A)\n");  // EN: Execute a statement: printf("- det(A⁻¹) = 1/det(A)\n");.
    printf("- det(cA) = cⁿ·det(A)\n\n");  // EN: Execute a statement: printf("- det(cA) = cⁿ·det(A)\n\n");.

    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n示範完成！\n");  // EN: Execute a statement: printf("\n示範完成！\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
