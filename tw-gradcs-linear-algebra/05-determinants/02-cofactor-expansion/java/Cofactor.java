/**
 * 餘因子展開 (Cofactor Expansion)
 *
 * 編譯：javac Cofactor.java
 * 執行：java Cofactor
 */

public class Cofactor {

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

    // 取得子矩陣
    static double[][] getMinorMatrix(double[][] A, int row, int col) {
        int n = A.length;
        double[][] sub = new double[n - 1][n - 1];
        int si = 0;
        for (int i = 0; i < n; i++) {
            if (i == row) continue;
            int sj = 0;
            for (int j = 0; j < n; j++) {
                if (j == col) continue;
                sub[si][sj] = A[i][j];
                sj++;
            }
            si++;
        }
        return sub;
    }

    // 行列式（遞迴餘因子展開）
    static double determinant(double[][] A) {
        int n = A.length;
        if (n == 1) return A[0][0];
        if (n == 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];

        double det = 0.0;
        for (int j = 0; j < n; j++) {
            double[][] sub = getMinorMatrix(A, 0, j);
            double sign = (j % 2 == 0) ? 1.0 : -1.0;
            det += sign * A[0][j] * determinant(sub);
        }
        return det;
    }

    // 子行列式
    static double minor(double[][] A, int i, int j) {
        return determinant(getMinorMatrix(A, i, j));
    }

    // 餘因子
    static double cofactor(double[][] A, int i, int j) {
        double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        return sign * minor(A, i, j);
    }

    // 餘因子矩陣
    static double[][] cofactorMatrix(double[][] A) {
        int n = A.length;
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = cofactor(A, i, j);
            }
        }
        return C;
    }

    // 轉置
    static double[][] transpose(double[][] A) {
        int n = A.length;
        double[][] T = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                T[j][i] = A[i][j];
            }
        }
        return T;
    }

    // 伴隨矩陣
    static double[][] adjugate(double[][] A) {
        return transpose(cofactorMatrix(A));
    }

    // 逆矩陣
    static double[][] inverse(double[][] A) {
        double det = determinant(A);
        double[][] adj = adjugate(A);
        int n = A.length;
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inv[i][j] = adj[i][j] / det;
            }
        }
        return inv;
    }

    // 矩陣乘法
    static double[][] multiply(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    public static void main(String[] args) {
        printSeparator("餘因子展開示範 (Java)");

        // ========================================
        // 1. 子行列式與餘因子
        // ========================================
        printSeparator("1. 子行列式與餘因子");

        double[][] A = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        printMatrix("A", A);

        System.out.println("\n所有餘因子 Cᵢⱼ：");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.printf("  C%d%d = %8.4f", i+1, j+1, cofactor(A, i, j));
            }
            System.out.println();
        }

        // ========================================
        // 2. 餘因子展開
        // ========================================
        printSeparator("2. 餘因子展開計算行列式");

        System.out.println("沿第一列展開：");
        System.out.println("det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃");
        System.out.printf("       = %.0f×%.0f + %.0f×%.0f + %.0f×%.0f%n",
            A[0][0], cofactor(A, 0, 0),
            A[0][1], cofactor(A, 0, 1),
            A[0][2], cofactor(A, 0, 2));
        System.out.printf("       = %.4f%n", determinant(A));

        // ========================================
        // 3. 餘因子矩陣與伴隨矩陣
        // ========================================
        printSeparator("3. 餘因子矩陣與伴隨矩陣");

        double[][] B = {
            {2, 1, 3},
            {1, 0, 2},
            {4, 1, 5}
        };

        printMatrix("A", B);
        System.out.printf("%ndet(A) = %.4f%n", determinant(B));

        double[][] C = cofactorMatrix(B);
        printMatrix("\n餘因子矩陣 C", C);

        double[][] adj = adjugate(B);
        printMatrix("\n伴隨矩陣 adj(A) = Cᵀ", adj);

        // ========================================
        // 4. 用伴隨矩陣求逆矩陣
        // ========================================
        printSeparator("4. 用伴隨矩陣求逆矩陣");

        System.out.println("A⁻¹ = adj(A) / det(A)");

        double[][] B_inv = inverse(B);
        printMatrix("\nA⁻¹", B_inv);

        // 驗證
        double[][] I = multiply(B, B_inv);
        printMatrix("\n驗證 A × A⁻¹", I);

        // ========================================
        // 5. 2×2 特例
        // ========================================
        printSeparator("5. 2×2 伴隨矩陣公式");

        double[][] A2 = {{3, 4}, {5, 6}};
        printMatrix("A", A2);

        System.out.println("\n對於 [[a,b],[c,d]]:");
        System.out.printf("adj(A) = [[d,-b],[-c,a]] = [[%.0f,%.0f],[%.0f,%.0f]]%n",
            A2[1][1], -A2[0][1], -A2[1][0], A2[0][0]);

        double[][] adj2 = adjugate(A2);
        printMatrix("\n計算得到的 adj(A)", adj2);

        // 總結
        printSeparator("總結");
        System.out.println("""

餘因子展開公式：
  Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ
  det(A) = Σⱼ aᵢⱼ Cᵢⱼ

伴隨矩陣：
  adj(A) = Cᵀ

逆矩陣：
  A⁻¹ = adj(A) / det(A)

時間複雜度：O(n!)
            """);

        System.out.println("=".repeat(60));
        System.out.println("示範完成！");
        System.out.println("=".repeat(60));
    }
}
