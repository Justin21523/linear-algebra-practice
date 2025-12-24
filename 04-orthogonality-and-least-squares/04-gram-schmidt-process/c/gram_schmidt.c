/**
 * Gram-Schmidt 正交化 (Gram-Schmidt Process)
 *
 * 編譯：gcc -std=c99 -O2 gram_schmidt.c -o gram_schmidt -lm
 * 執行：./gram_schmidt
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

void print_vector(const char* name, const double* v, int n) {  // EN: Execute line: void print_vector(const char* name, const double* v, int n) {.
    printf("%s = [", name);  // EN: Execute a statement: printf("%s = [", name);.
    for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
        printf("%.4f", v[i]);  // EN: Execute a statement: printf("%.4f", v[i]);.
        if (i < n - 1) printf(", ");  // EN: Conditional control flow: if (i < n - 1) printf(", ");.
    }  // EN: Structure delimiter for a block or scope.
    printf("]\n");  // EN: Execute a statement: printf("]\n");.
}  // EN: Structure delimiter for a block or scope.

double dot_product(const double* x, const double* y, int n) {  // EN: Execute line: double dot_product(const double* x, const double* y, int n) {.
    double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
    for (int i = 0; i < n; i++) result += x[i] * y[i];  // EN: Loop control flow: for (int i = 0; i < n; i++) result += x[i] * y[i];.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

double vector_norm(const double* x, int n) {  // EN: Execute line: double vector_norm(const double* x, int n) {.
    return sqrt(dot_product(x, x, n));  // EN: Return from the current function: return sqrt(dot_product(x, x, n));.
}  // EN: Structure delimiter for a block or scope.

void modified_gram_schmidt(double A[MAX_DIM][MAX_DIM], double Q[MAX_DIM][MAX_DIM], int num_vecs, int dim) {  // EN: Execute line: void modified_gram_schmidt(double A[MAX_DIM][MAX_DIM], double Q[MAX_DIM….
    // 複製
    for (int i = 0; i < num_vecs; i++)  // EN: Loop control flow: for (int i = 0; i < num_vecs; i++).
        for (int j = 0; j < dim; j++)  // EN: Loop control flow: for (int j = 0; j < dim; j++).
            Q[i][j] = A[i][j];  // EN: Execute a statement: Q[i][j] = A[i][j];.

    for (int j = 0; j < num_vecs; j++) {  // EN: Loop control flow: for (int j = 0; j < num_vecs; j++) {.
        for (int i = 0; i < j; i++) {  // EN: Loop control flow: for (int i = 0; i < j; i++) {.
            double coeff = dot_product(Q[i], Q[j], dim) / dot_product(Q[i], Q[i], dim);  // EN: Execute a statement: double coeff = dot_product(Q[i], Q[j], dim) / dot_product(Q[i], Q[i], d….
            for (int k = 0; k < dim; k++)  // EN: Loop control flow: for (int k = 0; k < dim; k++).
                Q[j][k] -= coeff * Q[i][k];  // EN: Execute a statement: Q[j][k] -= coeff * Q[i][k];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

void normalize_vectors(double Q[MAX_DIM][MAX_DIM], int num_vecs, int dim) {  // EN: Execute line: void normalize_vectors(double Q[MAX_DIM][MAX_DIM], int num_vecs, int di….
    for (int i = 0; i < num_vecs; i++) {  // EN: Loop control flow: for (int i = 0; i < num_vecs; i++) {.
        double norm = vector_norm(Q[i], dim);  // EN: Execute a statement: double norm = vector_norm(Q[i], dim);.
        for (int j = 0; j < dim; j++)  // EN: Loop control flow: for (int j = 0; j < dim; j++).
            Q[i][j] /= norm;  // EN: Execute a statement: Q[i][j] /= norm;.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

int verify_orthogonality(double Q[MAX_DIM][MAX_DIM], int num_vecs, int dim) {  // EN: Execute line: int verify_orthogonality(double Q[MAX_DIM][MAX_DIM], int num_vecs, int ….
    for (int i = 0; i < num_vecs; i++) {  // EN: Loop control flow: for (int i = 0; i < num_vecs; i++) {.
        for (int j = i + 1; j < num_vecs; j++) {  // EN: Loop control flow: for (int j = i + 1; j < num_vecs; j++) {.
            if (fabs(dot_product(Q[i], Q[j], dim)) > 1e-10)  // EN: Conditional control flow: if (fabs(dot_product(Q[i], Q[j], dim)) > 1e-10).
                return 0;  // EN: Return from the current function: return 0;.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return 1;  // EN: Return from the current function: return 1;.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    print_separator("Gram-Schmidt 正交化示範 (C)");  // EN: Execute a statement: print_separator("Gram-Schmidt 正交化示範 (C)");.

    double A[MAX_DIM][MAX_DIM] = {  // EN: Execute line: double A[MAX_DIM][MAX_DIM] = {.
        {1.0, 1.0, 0.0},  // EN: Execute line: {1.0, 1.0, 0.0},.
        {1.0, 0.0, 1.0},  // EN: Execute line: {1.0, 0.0, 1.0},.
        {0.0, 1.0, 1.0}  // EN: Execute line: {0.0, 1.0, 1.0}.
    };  // EN: Structure delimiter for a block or scope.
    int num_vecs = 3, dim = 3;  // EN: Execute a statement: int num_vecs = 3, dim = 3;.

    printf("輸入向量組：\n");  // EN: Execute a statement: printf("輸入向量組：\n");.
    for (int i = 0; i < num_vecs; i++) {  // EN: Loop control flow: for (int i = 0; i < num_vecs; i++) {.
        char name[10];  // EN: Execute a statement: char name[10];.
        sprintf(name, "a%d", i+1);  // EN: Execute a statement: sprintf(name, "a%d", i+1);.
        print_vector(name, A[i], dim);  // EN: Execute a statement: print_vector(name, A[i], dim);.
    }  // EN: Structure delimiter for a block or scope.

    double Q[MAX_DIM][MAX_DIM];  // EN: Execute a statement: double Q[MAX_DIM][MAX_DIM];.
    modified_gram_schmidt(A, Q, num_vecs, dim);  // EN: Execute a statement: modified_gram_schmidt(A, Q, num_vecs, dim);.

    printf("\n正交化結果（MGS）：\n");  // EN: Execute a statement: printf("\n正交化結果（MGS）：\n");.
    for (int i = 0; i < num_vecs; i++) {  // EN: Loop control flow: for (int i = 0; i < num_vecs; i++) {.
        char name[10];  // EN: Execute a statement: char name[10];.
        sprintf(name, "q%d", i+1);  // EN: Execute a statement: sprintf(name, "q%d", i+1);.
        print_vector(name, Q[i], dim);  // EN: Execute a statement: print_vector(name, Q[i], dim);.
        printf("    ‖q%d‖ = %.4f\n", i+1, vector_norm(Q[i], dim));  // EN: Execute a statement: printf(" ‖q%d‖ = %.4f\n", i+1, vector_norm(Q[i], dim));.
    }  // EN: Structure delimiter for a block or scope.

    printf("\n正交？ %s\n", verify_orthogonality(Q, num_vecs, dim) ? "true" : "false");  // EN: Execute a statement: printf("\n正交？ %s\n", verify_orthogonality(Q, num_vecs, dim) ? "true" : ….

    printf("\n內積驗證：\n");  // EN: Execute a statement: printf("\n內積驗證：\n");.
    printf("q₁ · q₂ = %.6f\n", dot_product(Q[0], Q[1], dim));  // EN: Execute a statement: printf("q₁ · q₂ = %.6f\n", dot_product(Q[0], Q[1], dim));.
    printf("q₁ · q₃ = %.6f\n", dot_product(Q[0], Q[2], dim));  // EN: Execute a statement: printf("q₁ · q₃ = %.6f\n", dot_product(Q[0], Q[2], dim));.
    printf("q₂ · q₃ = %.6f\n", dot_product(Q[1], Q[2], dim));  // EN: Execute a statement: printf("q₂ · q₃ = %.6f\n", dot_product(Q[1], Q[2], dim));.

    print_separator("標準正交化");  // EN: Execute a statement: print_separator("標準正交化");.

    normalize_vectors(Q, num_vecs, dim);  // EN: Execute a statement: normalize_vectors(Q, num_vecs, dim);.

    printf("標準正交向量組：\n");  // EN: Execute a statement: printf("標準正交向量組：\n");.
    for (int i = 0; i < num_vecs; i++) {  // EN: Loop control flow: for (int i = 0; i < num_vecs; i++) {.
        char name[10];  // EN: Execute a statement: char name[10];.
        sprintf(name, "e%d", i+1);  // EN: Execute a statement: sprintf(name, "e%d", i+1);.
        print_vector(name, Q[i], dim);  // EN: Execute a statement: print_vector(name, Q[i], dim);.
        printf("    ‖e%d‖ = %.4f\n", i+1, vector_norm(Q[i], dim));  // EN: Execute a statement: printf(" ‖e%d‖ = %.4f\n", i+1, vector_norm(Q[i], dim));.
    }  // EN: Structure delimiter for a block or scope.

    print_separator("總結");  // EN: Execute a statement: print_separator("總結");.
    printf("\nGram-Schmidt 核心公式：\n\n");  // EN: Execute a statement: printf("\nGram-Schmidt 核心公式：\n\n");.
    printf("proj_q(a) = (qᵀa / qᵀq) q\n\n");  // EN: Execute a statement: printf("proj_q(a) = (qᵀa / qᵀq) q\n\n");.
    printf("q₁ = a₁\n");  // EN: Execute a statement: printf("q₁ = a₁\n");.
    printf("qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ)\n\n");  // EN: Execute a statement: printf("qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ)\n\n");.
    printf("eᵢ = qᵢ / ‖qᵢ‖\n\n");  // EN: Execute a statement: printf("eᵢ = qᵢ / ‖qᵢ‖\n\n");.

    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n示範完成！\n");  // EN: Execute a statement: printf("\n示範完成！\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
