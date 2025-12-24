/**
 * Gram-Schmidt 正交化 (Gram-Schmidt Process)
 *
 * 編譯：gcc -std=c99 -O2 gram_schmidt.c -o gram_schmidt -lm
 * 執行：./gram_schmidt
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

void print_vector(const char* name, const double* v, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("%.4f", v[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

double dot_product(const double* x, const double* y, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) result += x[i] * y[i];
    return result;
}

double vector_norm(const double* x, int n) {
    return sqrt(dot_product(x, x, n));
}

void modified_gram_schmidt(double A[MAX_DIM][MAX_DIM], double Q[MAX_DIM][MAX_DIM], int num_vecs, int dim) {
    // 複製
    for (int i = 0; i < num_vecs; i++)
        for (int j = 0; j < dim; j++)
            Q[i][j] = A[i][j];

    for (int j = 0; j < num_vecs; j++) {
        for (int i = 0; i < j; i++) {
            double coeff = dot_product(Q[i], Q[j], dim) / dot_product(Q[i], Q[i], dim);
            for (int k = 0; k < dim; k++)
                Q[j][k] -= coeff * Q[i][k];
        }
    }
}

void normalize_vectors(double Q[MAX_DIM][MAX_DIM], int num_vecs, int dim) {
    for (int i = 0; i < num_vecs; i++) {
        double norm = vector_norm(Q[i], dim);
        for (int j = 0; j < dim; j++)
            Q[i][j] /= norm;
    }
}

int verify_orthogonality(double Q[MAX_DIM][MAX_DIM], int num_vecs, int dim) {
    for (int i = 0; i < num_vecs; i++) {
        for (int j = i + 1; j < num_vecs; j++) {
            if (fabs(dot_product(Q[i], Q[j], dim)) > 1e-10)
                return 0;
        }
    }
    return 1;
}

int main() {
    print_separator("Gram-Schmidt 正交化示範 (C)");

    double A[MAX_DIM][MAX_DIM] = {
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    };
    int num_vecs = 3, dim = 3;

    printf("輸入向量組：\n");
    for (int i = 0; i < num_vecs; i++) {
        char name[10];
        sprintf(name, "a%d", i+1);
        print_vector(name, A[i], dim);
    }

    double Q[MAX_DIM][MAX_DIM];
    modified_gram_schmidt(A, Q, num_vecs, dim);

    printf("\n正交化結果（MGS）：\n");
    for (int i = 0; i < num_vecs; i++) {
        char name[10];
        sprintf(name, "q%d", i+1);
        print_vector(name, Q[i], dim);
        printf("    ‖q%d‖ = %.4f\n", i+1, vector_norm(Q[i], dim));
    }

    printf("\n正交？ %s\n", verify_orthogonality(Q, num_vecs, dim) ? "true" : "false");

    printf("\n內積驗證：\n");
    printf("q₁ · q₂ = %.6f\n", dot_product(Q[0], Q[1], dim));
    printf("q₁ · q₃ = %.6f\n", dot_product(Q[0], Q[2], dim));
    printf("q₂ · q₃ = %.6f\n", dot_product(Q[1], Q[2], dim));

    print_separator("標準正交化");

    normalize_vectors(Q, num_vecs, dim);

    printf("標準正交向量組：\n");
    for (int i = 0; i < num_vecs; i++) {
        char name[10];
        sprintf(name, "e%d", i+1);
        print_vector(name, Q[i], dim);
        printf("    ‖e%d‖ = %.4f\n", i+1, vector_norm(Q[i], dim));
    }

    print_separator("總結");
    printf("\nGram-Schmidt 核心公式：\n\n");
    printf("proj_q(a) = (qᵀa / qᵀq) q\n\n");
    printf("q₁ = a₁\n");
    printf("qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ)\n\n");
    printf("eᵢ = qᵢ / ‖qᵢ‖\n\n");

    for (int i = 0; i < 60; i++) printf("=");
    printf("\n示範完成！\n");
    for (int i = 0; i < 60; i++) printf("=");
    printf("\n");

    return 0;
}
