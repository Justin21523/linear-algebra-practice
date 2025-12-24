/**
 * QR 分解 (QR Decomposition)
 *
 * 編譯：javac QRDecomposition.java
 * 執行：java QRDecomposition
 */

public class QRDecomposition {

    static void printSeparator(String title) {
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println(title);
        System.out.println("=".repeat(60));
    }

    static void printVector(String name, double[] v) {
        StringBuilder sb = new StringBuilder();
        sb.append(name).append(" = [");
        for (int i = 0; i < v.length; i++) {
            sb.append(String.format("%.4f", v[i]));
            if (i < v.length - 1) sb.append(", ");
        }
        sb.append("]");
        System.out.println(sb.toString());
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

    // 基本向量運算
    static double dotProduct(double[] x, double[] y) {
        double result = 0.0;
        for (int i = 0; i < x.length; i++) {
            result += x[i] * y[i];
        }
        return result;
    }

    static double vectorNorm(double[] x) {
        return Math.sqrt(dotProduct(x, x));
    }

    static double[] scalarMultiply(double c, double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = c * x[i];
        }
        return result;
    }

    static double[] vectorSubtract(double[] x, double[] y) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = x[i] - y[i];
        }
        return result;
    }

    // 取得矩陣的第 j 行（column）
    static double[] getColumn(double[][] A, int j) {
        double[] col = new double[A.length];
        for (int i = 0; i < A.length; i++) {
            col[i] = A[i][j];
        }
        return col;
    }

    // Gram-Schmidt QR 分解
    static double[][][] qrDecomposition(double[][] A) {
        int m = A.length;
        int n = A[0].length;

        // Q: m×n, R: n×n
        double[][] Q = new double[m][n];
        double[][] R = new double[n][n];

        for (int j = 0; j < n; j++) {
            // 取得 A 的第 j 行
            double[] v = getColumn(A, j);

            // 減去前面所有 q 向量的投影
            for (int i = 0; i < j; i++) {
                double[] qi = getColumn(Q, i);
                R[i][j] = dotProduct(qi, getColumn(A, j));
                double[] proj = scalarMultiply(R[i][j], qi);
                v = vectorSubtract(v, proj);
            }

            // 標準化
            R[j][j] = vectorNorm(v);

            if (R[j][j] > 1e-10) {
                for (int i = 0; i < m; i++) {
                    Q[i][j] = v[i] / R[j][j];
                }
            }
        }

        return new double[][][] {Q, R};
    }

    // 回代法解上三角方程組 Rx = b
    static double[] solveUpperTriangular(double[][] R, double[] b) {
        int n = b.length;
        double[] x = new double[n];

        for (int i = n - 1; i >= 0; i--) {
            x[i] = b[i];
            for (int j = i + 1; j < n; j++) {
                x[i] -= R[i][j] * x[j];
            }
            x[i] /= R[i][i];
        }

        return x;
    }

    // 用 QR 分解解最小平方問題
    static double[] qrLeastSquares(double[][] A, double[] b) {
        double[][][] qr = qrDecomposition(A);
        double[][] Q = qr[0];
        double[][] R = qr[1];

        // 計算 Qᵀb
        int n = Q[0].length;
        double[] Qt_b = new double[n];
        for (int j = 0; j < n; j++) {
            double[] qj = getColumn(Q, j);
            Qt_b[j] = dotProduct(qj, b);
        }

        // 解 Rx = Qᵀb
        return solveUpperTriangular(R, Qt_b);
    }

    // 矩陣乘法
    static double[][] matrixMultiply(double[][] A, double[][] B) {
        int m = A.length;
        int k = B.length;
        int n = B[0].length;

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
        int m = A.length;
        int n = A[0].length;
        double[][] result = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[j][i] = A[i][j];
            }
        }
        return result;
    }

    public static void main(String[] args) {
        printSeparator("QR 分解示範 (Java)");

        // ========================================
        // 1. 基本 QR 分解
        // ========================================
        printSeparator("1. 基本 QR 分解");

        double[][] A = {
            {1.0, 1.0},
            {1.0, 0.0},
            {0.0, 1.0}
        };

        System.out.println("輸入矩陣 A：");
        printMatrix("A", A);

        double[][][] qr = qrDecomposition(A);
        double[][] Q = qr[0];
        double[][] R = qr[1];

        System.out.println("\nQR 分解結果：");
        printMatrix("Q", Q);
        printMatrix("\nR", R);

        // 驗證 QᵀQ = I
        double[][] QT = transpose(Q);
        double[][] QTQ = matrixMultiply(QT, Q);
        System.out.println("\n驗證 QᵀQ = I：");
        printMatrix("QᵀQ", QTQ);

        // 驗證 A = QR
        double[][] QR_result = matrixMultiply(Q, R);
        System.out.println("\n驗證 A = QR：");
        printMatrix("QR", QR_result);

        // ========================================
        // 2. 用 QR 解最小平方
        // ========================================
        printSeparator("2. 用 QR 解最小平方");

        // 數據
        double[] t = {0.0, 1.0, 2.0};
        double[] b = {1.0, 3.0, 4.0};

        System.out.println("數據點：");
        for (int i = 0; i < t.length; i++) {
            System.out.printf("  (%.1f, %.1f)%n", t[i], b[i]);
        }

        // 設計矩陣
        double[][] A_ls = new double[t.length][2];
        for (int i = 0; i < t.length; i++) {
            A_ls[i][0] = 1.0;
            A_ls[i][1] = t[i];
        }

        System.out.println("\n設計矩陣 A：");
        printMatrix("A", A_ls);
        printVector("觀測值 b", b);

        // QR 分解
        double[][][] qr_ls = qrDecomposition(A_ls);
        double[][] Q_ls = qr_ls[0];
        double[][] R_ls = qr_ls[1];
        printMatrix("\nQ", Q_ls);
        printMatrix("R", R_ls);

        // 解最小平方
        double[] x = qrLeastSquares(A_ls, b);
        printVector("\n解 x", x);

        System.out.printf("%n最佳直線：y = %.4f + %.4ft%n", x[0], x[1]);

        // ========================================
        // 3. 3×3 矩陣的 QR 分解
        // ========================================
        printSeparator("3. 3×3 矩陣的 QR 分解");

        double[][] A2 = {
            {1.0, 1.0, 0.0},
            {1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0}
        };

        System.out.println("輸入矩陣 A：");
        printMatrix("A", A2);

        double[][][] qr2 = qrDecomposition(A2);
        double[][] Q2 = qr2[0];
        double[][] R2 = qr2[1];

        System.out.println("\nQR 分解結果：");
        printMatrix("Q", Q2);
        printMatrix("\nR", R2);

        // 總結
        printSeparator("總結");
        System.out.println("""

QR 分解核心：

1. A = QR
   - Q: 標準正交矩陣 (QᵀQ = I)
   - R: 上三角矩陣

2. Gram-Schmidt 演算法：
   - 對 A 的行向量正交化得到 Q
   - R 的元素是投影係數

3. 用 QR 解最小平方：
   min ‖Ax - b‖²
   → Rx = Qᵀb

4. 優勢：
   - 比正規方程更穩定
   - 避免計算 AᵀA
            """);

        System.out.println("=".repeat(60));
        System.out.println("示範完成！");
        System.out.println("=".repeat(60));
    }
}
