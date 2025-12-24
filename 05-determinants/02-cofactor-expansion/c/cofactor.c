/**
 * 餘因子展開 (Cofactor Expansion)
 *
 * 編譯：gcc -std=c99 -O2 cofactor.c -o cofactor -lm
 * 執行：./cofactor
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

// 取得子矩陣
void get_minor_matrix(double A[MAX_DIM][MAX_DIM], int n, int row, int col,  // EN: Execute line: void get_minor_matrix(double A[MAX_DIM][MAX_DIM], int n, int row, int c….
                      double sub[MAX_DIM][MAX_DIM]) {  // EN: Execute line: double sub[MAX_DIM][MAX_DIM]) {.
    int si = 0;  // EN: Execute a statement: int si = 0;.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        if (i == row) continue;  // EN: Conditional control flow: if (i == row) continue;.
        int sj = 0;  // EN: Execute a statement: int sj = 0;.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            if (j == col) continue;  // EN: Conditional control flow: if (j == col) continue;.
            sub[si][sj] = A[i][j];  // EN: Execute a statement: sub[si][sj] = A[i][j];.
            sj++;  // EN: Execute a statement: sj++;.
        }  // EN: Structure delimiter for a block or scope.
        si++;  // EN: Execute a statement: si++;.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// 前向宣告
double determinant(double A[MAX_DIM][MAX_DIM], int n);  // EN: Execute a statement: double determinant(double A[MAX_DIM][MAX_DIM], int n);.

// 子行列式
double minor_det(double A[MAX_DIM][MAX_DIM], int n, int i, int j) {  // EN: Execute line: double minor_det(double A[MAX_DIM][MAX_DIM], int n, int i, int j) {.
    double sub[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double sub[MAX_DIM][MAX_DIM];.
    get_minor_matrix(A, n, i, j, sub);  // EN: Execute a statement: get_minor_matrix(A, n, i, j, sub);.
    return determinant(sub, n - 1);  // EN: Return from the current function: return determinant(sub, n - 1);.
}  // EN: Structure delimiter for a block or scope.

// 餘因子
double cofactor(double A[MAX_DIM][MAX_DIM], int n, int i, int j) {  // EN: Execute line: double cofactor(double A[MAX_DIM][MAX_DIM], int n, int i, int j) {.
    double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;  // EN: Execute a statement: double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;.
    return sign * minor_det(A, n, i, j);  // EN: Return from the current function: return sign * minor_det(A, n, i, j);.
}  // EN: Structure delimiter for a block or scope.

// 行列式（遞迴餘因子展開）
double determinant(double A[MAX_DIM][MAX_DIM], int n) {  // EN: Execute line: double determinant(double A[MAX_DIM][MAX_DIM], int n) {.
    if (n == 1) return A[0][0];  // EN: Conditional control flow: if (n == 1) return A[0][0];.
    if (n == 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Conditional control flow: if (n == 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];.

    double det = 0.0;  // EN: Execute a statement: double det = 0.0;.
    for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
        det += A[0][j] * cofactor(A, n, 0, j);  // EN: Execute a statement: det += A[0][j] * cofactor(A, n, 0, j);.
    }  // EN: Structure delimiter for a block or scope.
    return det;  // EN: Return from the current function: return det;.
}  // EN: Structure delimiter for a block or scope.

// 餘因子矩陣
void cofactor_matrix(double A[MAX_DIM][MAX_DIM], int n, double C[MAX_DIM][MAX_DIM]) {  // EN: Execute line: void cofactor_matrix(double A[MAX_DIM][MAX_DIM], int n, double C[MAX_DI….
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            C[i][j] = cofactor(A, n, i, j);  // EN: Execute a statement: C[i][j] = cofactor(A, n, i, j);.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// 轉置
void transpose(double A[MAX_DIM][MAX_DIM], int n, double T[MAX_DIM][MAX_DIM]) {  // EN: Execute line: void transpose(double A[MAX_DIM][MAX_DIM], int n, double T[MAX_DIM][MAX….
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            T[j][i] = A[i][j];  // EN: Execute a statement: T[j][i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// 伴隨矩陣
void adjugate(double A[MAX_DIM][MAX_DIM], int n, double adj[MAX_DIM][MAX_DIM]) {  // EN: Execute line: void adjugate(double A[MAX_DIM][MAX_DIM], int n, double adj[MAX_DIM][MA….
    double C[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double C[MAX_DIM][MAX_DIM];.
    cofactor_matrix(A, n, C);  // EN: Execute a statement: cofactor_matrix(A, n, C);.
    transpose(C, n, adj);  // EN: Execute a statement: transpose(C, n, adj);.
}  // EN: Structure delimiter for a block or scope.

// 逆矩陣
void inverse(double A[MAX_DIM][MAX_DIM], int n, double inv[MAX_DIM][MAX_DIM]) {  // EN: Execute line: void inverse(double A[MAX_DIM][MAX_DIM], int n, double inv[MAX_DIM][MAX….
    double det = determinant(A, n);  // EN: Execute a statement: double det = determinant(A, n);.
    double adj[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double adj[MAX_DIM][MAX_DIM];.
    adjugate(A, n, adj);  // EN: Execute a statement: adjugate(A, n, adj);.

    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            inv[i][j] = adj[i][j] / det;  // EN: Execute a statement: inv[i][j] = adj[i][j] / det;.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// 矩陣乘法
void multiply(double A[MAX_DIM][MAX_DIM], double B[MAX_DIM][MAX_DIM],  // EN: Execute line: void multiply(double A[MAX_DIM][MAX_DIM], double B[MAX_DIM][MAX_DIM],.
              int n, double C[MAX_DIM][MAX_DIM]) {  // EN: Execute line: int n, double C[MAX_DIM][MAX_DIM]) {.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            C[i][j] = 0.0;  // EN: Execute a statement: C[i][j] = 0.0;.
            for (int k = 0; k < n; k++) {  // EN: Loop control flow: for (int k = 0; k < n; k++) {.
                C[i][j] += A[i][k] * B[k][j];  // EN: Execute a statement: C[i][j] += A[i][k] * B[k][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    print_separator("餘因子展開示範 (C)");  // EN: Execute a statement: print_separator("餘因子展開示範 (C)");.

    // ========================================
    // 1. 子行列式與餘因子
    // ========================================
    print_separator("1. 子行列式與餘因子");  // EN: Execute a statement: print_separator("1. 子行列式與餘因子");.

    double A[MAX_DIM][MAX_DIM] = {  // EN: Execute line: double A[MAX_DIM][MAX_DIM] = {.
        {1, 2, 3},  // EN: Execute line: {1, 2, 3},.
        {4, 5, 6},  // EN: Execute line: {4, 5, 6},.
        {7, 8, 9}  // EN: Execute line: {7, 8, 9}.
    };  // EN: Structure delimiter for a block or scope.

    print_matrix("A", A, 3);  // EN: Execute a statement: print_matrix("A", A, 3);.

    printf("\n所有餘因子 Cᵢⱼ：\n");  // EN: Execute a statement: printf("\n所有餘因子 Cᵢⱼ：\n");.
    for (int i = 0; i < 3; i++) {  // EN: Loop control flow: for (int i = 0; i < 3; i++) {.
        for (int j = 0; j < 3; j++) {  // EN: Loop control flow: for (int j = 0; j < 3; j++) {.
            printf("  C%d%d = %8.4f", i+1, j+1, cofactor(A, 3, i, j));  // EN: Execute a statement: printf(" C%d%d = %8.4f", i+1, j+1, cofactor(A, 3, i, j));.
        }  // EN: Structure delimiter for a block or scope.
        printf("\n");  // EN: Execute a statement: printf("\n");.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 2. 餘因子展開
    // ========================================
    print_separator("2. 餘因子展開計算行列式");  // EN: Execute a statement: print_separator("2. 餘因子展開計算行列式");.

    printf("沿第一列展開：\n");  // EN: Execute a statement: printf("沿第一列展開：\n");.
    printf("det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃\n");  // EN: Execute a statement: printf("det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃\n");.
    printf("       = %.0f×%.0f + %.0f×%.0f + %.0f×%.0f\n",  // EN: Execute line: printf(" = %.0f×%.0f + %.0f×%.0f + %.0f×%.0f\n",.
        A[0][0], cofactor(A, 3, 0, 0),  // EN: Execute line: A[0][0], cofactor(A, 3, 0, 0),.
        A[0][1], cofactor(A, 3, 0, 1),  // EN: Execute line: A[0][1], cofactor(A, 3, 0, 1),.
        A[0][2], cofactor(A, 3, 0, 2));  // EN: Execute a statement: A[0][2], cofactor(A, 3, 0, 2));.
    printf("       = %.4f\n", determinant(A, 3));  // EN: Execute a statement: printf(" = %.4f\n", determinant(A, 3));.

    // ========================================
    // 3. 餘因子矩陣與伴隨矩陣
    // ========================================
    print_separator("3. 餘因子矩陣與伴隨矩陣");  // EN: Execute a statement: print_separator("3. 餘因子矩陣與伴隨矩陣");.

    double B[MAX_DIM][MAX_DIM] = {  // EN: Execute line: double B[MAX_DIM][MAX_DIM] = {.
        {2, 1, 3},  // EN: Execute line: {2, 1, 3},.
        {1, 0, 2},  // EN: Execute line: {1, 0, 2},.
        {4, 1, 5}  // EN: Execute line: {4, 1, 5}.
    };  // EN: Structure delimiter for a block or scope.

    print_matrix("A", B, 3);  // EN: Execute a statement: print_matrix("A", B, 3);.
    printf("\ndet(A) = %.4f\n", determinant(B, 3));  // EN: Execute a statement: printf("\ndet(A) = %.4f\n", determinant(B, 3));.

    double C[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double C[MAX_DIM][MAX_DIM];.
    cofactor_matrix(B, 3, C);  // EN: Execute a statement: cofactor_matrix(B, 3, C);.
    print_matrix("\n餘因子矩陣 C", C, 3);  // EN: Execute a statement: print_matrix("\n餘因子矩陣 C", C, 3);.

    double adj[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double adj[MAX_DIM][MAX_DIM];.
    adjugate(B, 3, adj);  // EN: Execute a statement: adjugate(B, 3, adj);.
    print_matrix("\n伴隨矩陣 adj(A) = Cᵀ", adj, 3);  // EN: Execute a statement: print_matrix("\n伴隨矩陣 adj(A) = Cᵀ", adj, 3);.

    // ========================================
    // 4. 用伴隨矩陣求逆矩陣
    // ========================================
    print_separator("4. 用伴隨矩陣求逆矩陣");  // EN: Execute a statement: print_separator("4. 用伴隨矩陣求逆矩陣");.

    printf("A⁻¹ = adj(A) / det(A)\n");  // EN: Execute a statement: printf("A⁻¹ = adj(A) / det(A)\n");.

    double B_inv[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double B_inv[MAX_DIM][MAX_DIM];.
    inverse(B, 3, B_inv);  // EN: Execute a statement: inverse(B, 3, B_inv);.
    print_matrix("\nA⁻¹", B_inv, 3);  // EN: Execute a statement: print_matrix("\nA⁻¹", B_inv, 3);.

    // 驗證
    double I[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double I[MAX_DIM][MAX_DIM];.
    multiply(B, B_inv, 3, I);  // EN: Execute a statement: multiply(B, B_inv, 3, I);.
    print_matrix("\n驗證 A × A⁻¹", I, 3);  // EN: Execute a statement: print_matrix("\n驗證 A × A⁻¹", I, 3);.

    // ========================================
    // 5. 2×2 特例
    // ========================================
    print_separator("5. 2×2 伴隨矩陣公式");  // EN: Execute a statement: print_separator("5. 2×2 伴隨矩陣公式");.

    double A2[MAX_DIM][MAX_DIM] = {{3, 4}, {5, 6}};  // EN: Execute a statement: double A2[MAX_DIM][MAX_DIM] = {{3, 4}, {5, 6}};.
    print_matrix("A", A2, 2);  // EN: Execute a statement: print_matrix("A", A2, 2);.

    printf("\n對於 [[a,b],[c,d]]:\n");  // EN: Execute a statement: printf("\n對於 [[a,b],[c,d]]:\n");.
    printf("adj(A) = [[d,-b],[-c,a]] = [[%.0f,%.0f],[%.0f,%.0f]]\n",  // EN: Execute line: printf("adj(A) = [[d,-b],[-c,a]] = [[%.0f,%.0f],[%.0f,%.0f]]\n",.
        A2[1][1], -A2[0][1], -A2[1][0], A2[0][0]);  // EN: Execute a statement: A2[1][1], -A2[0][1], -A2[1][0], A2[0][0]);.

    double adj2[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double adj2[MAX_DIM][MAX_DIM];.
    adjugate(A2, 2, adj2);  // EN: Execute a statement: adjugate(A2, 2, adj2);.
    print_matrix("\n計算得到的 adj(A)", adj2, 2);  // EN: Execute a statement: print_matrix("\n計算得到的 adj(A)", adj2, 2);.

    // 總結
    print_separator("總結");  // EN: Execute a statement: print_separator("總結");.
    printf("\n餘因子展開公式：\n");  // EN: Execute a statement: printf("\n餘因子展開公式：\n");.
    printf("  Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ\n");  // EN: Execute a statement: printf(" Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ\n");.
    printf("  det(A) = Σⱼ aᵢⱼ Cᵢⱼ\n\n");  // EN: Execute a statement: printf(" det(A) = Σⱼ aᵢⱼ Cᵢⱼ\n\n");.
    printf("伴隨矩陣：\n");  // EN: Execute a statement: printf("伴隨矩陣：\n");.
    printf("  adj(A) = Cᵀ\n\n");  // EN: Execute a statement: printf(" adj(A) = Cᵀ\n\n");.
    printf("逆矩陣：\n");  // EN: Execute a statement: printf("逆矩陣：\n");.
    printf("  A⁻¹ = adj(A) / det(A)\n\n");  // EN: Execute a statement: printf(" A⁻¹ = adj(A) / det(A)\n\n");.
    printf("時間複雜度：O(n!)\n\n");  // EN: Execute a statement: printf("時間複雜度：O(n!)\n\n");.

    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n示範完成！\n");  // EN: Execute a statement: printf("\n示範完成！\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
