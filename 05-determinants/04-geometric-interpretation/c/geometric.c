/**
 * 行列式的幾何解釋 (Geometric Interpretation)
 *
 * 編譯：gcc -std=c99 -O2 geometric.c -o geometric -lm
 * 執行：./geometric
 */

#include <stdio.h>  // EN: Include a header dependency: #include <stdio.h>.
#include <math.h>  // EN: Include a header dependency: #include <math.h>.

void print_separator(const char* title) {  // EN: Execute line: void print_separator(const char* title) {.
    printf("\n");  // EN: Execute a statement: printf("\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n%s\n", title);  // EN: Execute a statement: printf("\n%s\n", title);.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.
}  // EN: Structure delimiter for a block or scope.

void print_vector_2d(const char* name, double* v) {  // EN: Execute line: void print_vector_2d(const char* name, double* v) {.
    printf("%s = [%.4f, %.4f]\n", name, v[0], v[1]);  // EN: Execute a statement: printf("%s = [%.4f, %.4f]\n", name, v[0], v[1]);.
}  // EN: Structure delimiter for a block or scope.

void print_vector_3d(const char* name, double* v) {  // EN: Execute line: void print_vector_3d(const char* name, double* v) {.
    printf("%s = [%.4f, %.4f, %.4f]\n", name, v[0], v[1], v[2]);  // EN: Execute a statement: printf("%s = [%.4f, %.4f, %.4f]\n", name, v[0], v[1], v[2]);.
}  // EN: Structure delimiter for a block or scope.

void print_matrix_2x2(const char* name, double A[2][2]) {  // EN: Execute line: void print_matrix_2x2(const char* name, double A[2][2]) {.
    printf("%s =\n", name);  // EN: Execute a statement: printf("%s =\n", name);.
    printf("  [%8.4f, %8.4f]\n", A[0][0], A[0][1]);  // EN: Execute a statement: printf(" [%8.4f, %8.4f]\n", A[0][0], A[0][1]);.
    printf("  [%8.4f, %8.4f]\n", A[1][0], A[1][1]);  // EN: Execute a statement: printf(" [%8.4f, %8.4f]\n", A[1][0], A[1][1]);.
}  // EN: Structure delimiter for a block or scope.

// 2D 叉積（純量）
double cross_2d(double* a, double* b) {  // EN: Execute line: double cross_2d(double* a, double* b) {.
    return a[0] * b[1] - a[1] * b[0];  // EN: Return from the current function: return a[0] * b[1] - a[1] * b[0];.
}  // EN: Structure delimiter for a block or scope.

// 3D 叉積
void cross_3d(double* a, double* b, double* result) {  // EN: Execute line: void cross_3d(double* a, double* b, double* result) {.
    result[0] = a[1] * b[2] - a[2] * b[1];  // EN: Execute a statement: result[0] = a[1] * b[2] - a[2] * b[1];.
    result[1] = a[2] * b[0] - a[0] * b[2];  // EN: Execute a statement: result[1] = a[2] * b[0] - a[0] * b[2];.
    result[2] = a[0] * b[1] - a[1] * b[0];  // EN: Execute a statement: result[2] = a[0] * b[1] - a[1] * b[0];.
}  // EN: Structure delimiter for a block or scope.

// 3D 內積
double dot_3d(double* a, double* b) {  // EN: Execute line: double dot_3d(double* a, double* b) {.
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];  // EN: Return from the current function: return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];.
}  // EN: Structure delimiter for a block or scope.

// 2×2 行列式
double det_2x2(double A[2][2]) {  // EN: Execute line: double det_2x2(double A[2][2]) {.
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Return from the current function: return A[0][0] * A[1][1] - A[0][1] * A[1][0];.
}  // EN: Structure delimiter for a block or scope.

// 平行四邊形面積
double parallelogram_area(double* a, double* b) {  // EN: Execute line: double parallelogram_area(double* a, double* b) {.
    return fabs(cross_2d(a, b));  // EN: Return from the current function: return fabs(cross_2d(a, b));.
}  // EN: Structure delimiter for a block or scope.

// 平行六面體體積
double parallelepiped_volume(double* a, double* b, double* c) {  // EN: Execute line: double parallelepiped_volume(double* a, double* b, double* c) {.
    double bxc[3];  // EN: Execute a statement: double bxc[3];.
    cross_3d(b, c, bxc);  // EN: Execute a statement: cross_3d(b, c, bxc);.
    return fabs(dot_3d(a, bxc));  // EN: Return from the current function: return fabs(dot_3d(a, bxc));.
}  // EN: Structure delimiter for a block or scope.

// 三角形面積
double triangle_area(double x1, double y1,  // EN: Execute line: double triangle_area(double x1, double y1,.
                     double x2, double y2,  // EN: Execute line: double x2, double y2,.
                     double x3, double y3) {  // EN: Execute line: double x3, double y3) {.
    return fabs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;  // EN: Return from the current function: return fabs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;.
}  // EN: Structure delimiter for a block or scope.

int main() {  // EN: Execute line: int main() {.
    print_separator("行列式幾何解釋示範 (C)");  // EN: Execute a statement: print_separator("行列式幾何解釋示範 (C)");.

    // ========================================
    // 1. 平行四邊形面積
    // ========================================
    print_separator("1. 平行四邊形面積");  // EN: Execute a statement: print_separator("1. 平行四邊形面積");.

    double a[] = {3, 0};  // EN: Execute a statement: double a[] = {3, 0};.
    double b[] = {1, 2};  // EN: Execute a statement: double b[] = {1, 2};.

    print_vector_2d("a", a);  // EN: Execute a statement: print_vector_2d("a", a);.
    print_vector_2d("b", b);  // EN: Execute a statement: print_vector_2d("b", b);.

    double area = parallelogram_area(a, b);  // EN: Execute a statement: double area = parallelogram_area(a, b);.
    double signed_area = cross_2d(a, b);  // EN: Execute a statement: double signed_area = cross_2d(a, b);.

    printf("\n平行四邊形：\n");  // EN: Execute a statement: printf("\n平行四邊形：\n");.
    printf("  有號面積 = a × b = %.4f\n", signed_area);  // EN: Execute a statement: printf(" 有號面積 = a × b = %.4f\n", signed_area);.
    printf("  面積 = |a × b| = %.4f\n", area);  // EN: Execute a statement: printf(" 面積 = |a × b| = %.4f\n", area);.

    // ========================================
    // 2. 定向判斷
    // ========================================
    print_separator("2. 定向判斷");  // EN: Execute a statement: print_separator("2. 定向判斷");.

    double a2[] = {1, 0};  // EN: Execute a statement: double a2[] = {1, 0};.
    double b2[] = {0, 1};  // EN: Execute a statement: double b2[] = {0, 1};.
    double signed_val = cross_2d(a2, b2);  // EN: Execute a statement: double signed_val = cross_2d(a2, b2);.

    print_vector_2d("a", a2);  // EN: Execute a statement: print_vector_2d("a", a2);.
    print_vector_2d("b", b2);  // EN: Execute a statement: print_vector_2d("b", b2);.
    printf("有號面積 = %.4f\n", signed_val);  // EN: Execute a statement: printf("有號面積 = %.4f\n", signed_val);.
    printf("定向：%s\n", signed_val > 0 ? "逆時針（正向）" : "順時針（負向）");  // EN: Execute a statement: printf("定向：%s\n", signed_val > 0 ? "逆時針（正向）" : "順時針（負向）");.

    printf("\n交換 a, b 順序：\n");  // EN: Execute a statement: printf("\n交換 a, b 順序：\n");.
    signed_val = cross_2d(b2, a2);  // EN: Execute a statement: signed_val = cross_2d(b2, a2);.
    printf("有號面積 = %.4f\n", signed_val);  // EN: Execute a statement: printf("有號面積 = %.4f\n", signed_val);.
    printf("定向：%s\n", signed_val > 0 ? "逆時針（正向）" : "順時針（負向）");  // EN: Execute a statement: printf("定向：%s\n", signed_val > 0 ? "逆時針（正向）" : "順時針（負向）");.

    // ========================================
    // 3. 平行六面體體積
    // ========================================
    print_separator("3. 平行六面體體積");  // EN: Execute a statement: print_separator("3. 平行六面體體積");.

    double v1[] = {1, 0, 0};  // EN: Execute a statement: double v1[] = {1, 0, 0};.
    double v2[] = {0, 2, 0};  // EN: Execute a statement: double v2[] = {0, 2, 0};.
    double v3[] = {0, 0, 3};  // EN: Execute a statement: double v3[] = {0, 0, 3};.

    print_vector_3d("a", v1);  // EN: Execute a statement: print_vector_3d("a", v1);.
    print_vector_3d("b", v2);  // EN: Execute a statement: print_vector_3d("b", v2);.
    print_vector_3d("c", v3);  // EN: Execute a statement: print_vector_3d("c", v3);.

    double vol = parallelepiped_volume(v1, v2, v3);  // EN: Execute a statement: double vol = parallelepiped_volume(v1, v2, v3);.
    printf("\n體積 = |a · (b × c)| = %.4f\n", vol);  // EN: Execute a statement: printf("\n體積 = |a · (b × c)| = %.4f\n", vol);.

    // ========================================
    // 4. 三角形面積
    // ========================================
    print_separator("4. 三角形面積");  // EN: Execute a statement: print_separator("4. 三角形面積");.

    double x1 = 0, y1 = 0;  // EN: Execute a statement: double x1 = 0, y1 = 0;.
    double x2 = 4, y2 = 0;  // EN: Execute a statement: double x2 = 4, y2 = 0;.
    double x3 = 0, y3 = 3;  // EN: Execute a statement: double x3 = 0, y3 = 3;.

    printf("三角形頂點：\n");  // EN: Execute a statement: printf("三角形頂點：\n");.
    printf("  P1 = (%.0f, %.0f)\n", x1, y1);  // EN: Execute a statement: printf(" P1 = (%.0f, %.0f)\n", x1, y1);.
    printf("  P2 = (%.0f, %.0f)\n", x2, y2);  // EN: Execute a statement: printf(" P2 = (%.0f, %.0f)\n", x2, y2);.
    printf("  P3 = (%.0f, %.0f)\n", x3, y3);  // EN: Execute a statement: printf(" P3 = (%.0f, %.0f)\n", x3, y3);.

    double tri_area = triangle_area(x1, y1, x2, y2, x3, y3);  // EN: Execute a statement: double tri_area = triangle_area(x1, y1, x2, y2, x3, y3);.
    printf("\n面積 = %.4f\n", tri_area);  // EN: Execute a statement: printf("\n面積 = %.4f\n", tri_area);.

    // ========================================
    // 5. 線性變換的體積縮放
    // ========================================
    print_separator("5. 線性變換的體積縮放");  // EN: Execute a statement: print_separator("5. 線性變換的體積縮放");.

    double A[2][2] = {{2, 0}, {0, 3}};  // EN: Execute a statement: double A[2][2] = {{2, 0}, {0, 3}};.
    print_matrix_2x2("縮放矩陣 A", A);  // EN: Execute a statement: print_matrix_2x2("縮放矩陣 A", A);.
    printf("det(A) = %.4f\n", det_2x2(A));  // EN: Execute a statement: printf("det(A) = %.4f\n", det_2x2(A));.
    printf("\n單位正方形 → 2×3 長方形\n");  // EN: Execute a statement: printf("\n單位正方形 → 2×3 長方形\n");.
    printf("面積從 1 變成 %.4f\n", fabs(det_2x2(A)));  // EN: Execute a statement: printf("面積從 1 變成 %.4f\n", fabs(det_2x2(A)));.

    double theta = M_PI / 4;  // EN: Execute a statement: double theta = M_PI / 4;.
    double R[2][2] = {  // EN: Execute line: double R[2][2] = {.
        {cos(theta), -sin(theta)},  // EN: Execute line: {cos(theta), -sin(theta)},.
        {sin(theta), cos(theta)}  // EN: Execute line: {sin(theta), cos(theta)}.
    };  // EN: Structure delimiter for a block or scope.
    printf("\n旋轉矩陣：det(R) = %.4f（面積不變）\n", det_2x2(R));  // EN: Execute a statement: printf("\n旋轉矩陣：det(R) = %.4f（面積不變）\n", det_2x2(R));.

    double H[2][2] = {{1, 0}, {0, -1}};  // EN: Execute a statement: double H[2][2] = {{1, 0}, {0, -1}};.
    printf("反射矩陣：det(H) = %.4f（面積不變，定向反轉）\n", det_2x2(H));  // EN: Execute a statement: printf("反射矩陣：det(H) = %.4f（面積不變，定向反轉）\n", det_2x2(H));.

    double S[2][2] = {{1, 2}, {0, 1}};  // EN: Execute a statement: double S[2][2] = {{1, 2}, {0, 1}};.
    printf("剪切矩陣：det(S) = %.4f（面積不變）\n", det_2x2(S));  // EN: Execute a statement: printf("剪切矩陣：det(S) = %.4f（面積不變）\n", det_2x2(S));.

    // 總結
    print_separator("總結");  // EN: Execute a statement: print_separator("總結");.
    printf("\n行列式的幾何意義：\n\n");  // EN: Execute a statement: printf("\n行列式的幾何意義：\n\n");.
    printf("1. |det| = 體積/面積的縮放因子\n");  // EN: Execute a statement: printf("1. |det| = 體積/面積的縮放因子\n");.
    printf("2. sign(det) = 定向保持/反轉\n");  // EN: Execute a statement: printf("2. sign(det) = 定向保持/反轉\n");.
    printf("3. det = 0 → 降維\n\n");  // EN: Execute a statement: printf("3. det = 0 → 降維\n\n");.
    printf("特殊矩陣：\n");  // EN: Execute a statement: printf("特殊矩陣：\n");.
    printf("   - 旋轉：det = 1\n");  // EN: Execute a statement: printf(" - 旋轉：det = 1\n");.
    printf("   - 反射：det = -1\n");  // EN: Execute a statement: printf(" - 反射：det = -1\n");.
    printf("   - 剪切：det = 1\n\n");  // EN: Execute a statement: printf(" - 剪切：det = 1\n\n");.

    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n示範完成！\n");  // EN: Execute a statement: printf("\n示範完成！\n");.
    for (int i = 0; i < 60; i++) printf("=");  // EN: Loop control flow: for (int i = 0; i < 60; i++) printf("=");.
    printf("\n");  // EN: Execute a statement: printf("\n");.

    return 0;  // EN: Return from the current function: return 0;.
}  // EN: Structure delimiter for a block or scope.
