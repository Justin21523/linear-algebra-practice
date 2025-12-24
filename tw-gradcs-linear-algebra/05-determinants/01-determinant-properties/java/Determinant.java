/**
 * 行列式的性質 (Determinant Properties)
 *
 * 編譯：javac Determinant.java
 * 執行：java Determinant
 */

public class Determinant {

    static void printSeparator(String title) {
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println(title);
        System.out.println("=".repeat(60));
    }

    static void printMatrix(String name, double[][] M) {
        System.out.println(name + " =");
        for (double[] row : M) {
            StringBuilder sb = new StringBuilder("  [");
            for (int i = 0; i < row.length; i++) {
                sb.append(String.format("%8.4f", row[i]));
                if (i < row.length - 1) sb.append(", ");
            }
            sb.append("]");
            System.out.println(sb.toString());
        }
    }

    // 2×2 行列式
    static double det2x2(double[][] A) {
        return A[0][0] * A[1][1] - A[0][1] * A[1][0];
    }

    // 3×3 行列式
    static double det3x3(double[][] A) {
        double a = A[0][0], b = A[0][1], c = A[0][2];
        double d = A[1][0], e = A[1][1], f = A[1][2];
        double g = A[2][0], h = A[2][1], i = A[2][2];

        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    }

    // n×n 行列式（列運算化為上三角）
    static double detNxN(double[][] A) {
        int n = A.length;
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++) {
            M[i] = A[i].clone();
        }

        int sign = 1;

        for (int col = 0; col < n; col++) {
            // 找主元
            int pivotRow = -1;
            for (int row = col; row < n; row++) {
                if (Math.abs(M[row][col]) > 1e-10) {
                    pivotRow = row;
                    break;
                }
            }

            if (pivotRow == -1) return 0.0;

            // 列交換
            if (pivotRow != col) {
                double[] temp = M[col];
                M[col] = M[pivotRow];
                M[pivotRow] = temp;
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
    static double[][] matrixMultiply(double[][] A, double[][] B) {
        int m = A.length, k = B.length, n = B[0].length;
        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int p = 0; p < k; p++) {
                    result[i][j] += A[i][p] * B[p][j];
                }
            }
        }
        return result;
    }

    // 矩陣轉置
    static double[][] transpose(double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] result = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[j][i] = A[i][j];
            }
        }
        return result;
    }

    // 純量乘矩陣
    static double[][] scalarMultiply(double c, double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = c * A[i][j];
            }
        }
        return result;
    }

    public static void main(String[] args) {
        printSeparator("行列式性質示範 (Java)");

        // ========================================
        // 1. 基本計算
        // ========================================
        printSeparator("1. 基本行列式計算");

        double[][] A2 = {{3, 8}, {4, 6}};
        printMatrix("A (2×2)", A2);
        System.out.printf("det(A) = %.4f%n", det2x2(A2));

        double[][] A3 = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 10}
        };
        printMatrix("\nA (3×3)", A3);
        System.out.printf("det(A) = %.4f%n", det3x3(A3));

        // ========================================
        // 2. 性質 1：det(I) = 1
        // ========================================
        printSeparator("2. 性質 1：det(I) = 1");

        double[][] I3 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        printMatrix("I₃", I3);
        System.out.printf("det(I₃) = %.4f%n", det3x3(I3));

        // ========================================
        // 3. 性質 2：列交換變號
        // ========================================
        printSeparator("3. 性質 2：列交換變號");

        double[][] A = {{1, 2}, {3, 4}};
        printMatrix("A", A);
        System.out.printf("det(A) = %.4f%n", det2x2(A));

        double[][] A_swap = {{3, 4}, {1, 2}};
        printMatrix("\nA（交換列）", A_swap);
        System.out.printf("det(交換後) = %.4f%n", det2x2(A_swap));
        System.out.println("驗證：變號 ✓");

        // ========================================
        // 4. 乘積公式
        // ========================================
        printSeparator("4. 乘積公式：det(AB) = det(A)·det(B)");

        A = new double[][]{{1, 2}, {3, 4}};
        double[][] B = {{5, 6}, {7, 8}};
        double[][] AB = matrixMultiply(A, B);

        printMatrix("A", A);
        printMatrix("B", B);
        printMatrix("AB", AB);

        double detA = det2x2(A);
        double detB = det2x2(B);
        double detAB = det2x2(AB);

        System.out.printf("%ndet(A) = %.4f%n", detA);
        System.out.printf("det(B) = %.4f%n", detB);
        System.out.printf("det(A)·det(B) = %.4f%n", detA * detB);
        System.out.printf("det(AB) = %.4f%n", detAB);

        // ========================================
        // 5. 轉置公式
        // ========================================
        printSeparator("5. 轉置公式：det(Aᵀ) = det(A)");

        A3 = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
        double[][] AT = transpose(A3);

        printMatrix("A", A3);
        printMatrix("Aᵀ", AT);

        System.out.printf("%ndet(A) = %.4f%n", det3x3(A3));
        System.out.printf("det(Aᵀ) = %.4f%n", det3x3(AT));

        // ========================================
        // 6. 純量乘法
        // ========================================
        printSeparator("6. 純量乘法：det(cA) = cⁿ·det(A)");

        A = new double[][]{{1, 2}, {3, 4}};
        double c = 2;
        double[][] cA = scalarMultiply(c, A);

        printMatrix("A (2×2)", A);
        System.out.println("c = " + c);
        printMatrix("cA", cA);

        detA = det2x2(A);
        double detcA = det2x2(cA);
        int n = 2;

        System.out.printf("%ndet(A) = %.4f%n", detA);
        System.out.printf("cⁿ·det(A) = %.0f² × %.4f = %.4f%n", c, detA, Math.pow(c, n) * detA);
        System.out.printf("det(cA) = %.4f%n", detcA);

        // ========================================
        // 7. 上三角矩陣
        // ========================================
        printSeparator("7. 上三角矩陣：det = 對角線乘積");

        double[][] U = {{2, 3, 1}, {0, 4, 5}, {0, 0, 6}};
        printMatrix("U（上三角）", U);
        System.out.println("對角線乘積：2 × 4 × 6 = " + (2 * 4 * 6));
        System.out.printf("det(U) = %.4f%n", det3x3(U));

        // ========================================
        // 8. 奇異矩陣
        // ========================================
        printSeparator("8. 奇異矩陣：det(A) = 0");

        double[][] A_singular = {{1, 2}, {2, 4}};
        printMatrix("A（列成比例）", A_singular);
        System.out.printf("det(A) = %.4f%n", det2x2(A_singular));
        System.out.println("此矩陣不可逆");

        // 總結
        printSeparator("總結");
        System.out.println("""

行列式三大性質：
1. det(I) = 1
2. 列交換 → det 變號
3. 對單列線性

重要公式：
- det(AB) = det(A)·det(B)
- det(Aᵀ) = det(A)
- det(A⁻¹) = 1/det(A)
- det(cA) = cⁿ·det(A)
            """);

        System.out.println("=".repeat(60));
        System.out.println("示範完成！");
        System.out.println("=".repeat(60));
    }
}
