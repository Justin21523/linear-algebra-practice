/**
 * QR 分解 (QR Decomposition)
 *
 * 編譯：javac QRDecomposition.java
 * 執行：java QRDecomposition
 */

public class QRDecomposition {  // EN: Execute line: public class QRDecomposition {.

    static void printSeparator(String title) {  // EN: Execute line: static void printSeparator(String title) {.
        System.out.println();  // EN: Execute a statement: System.out.println();.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
        System.out.println(title);  // EN: Execute a statement: System.out.println(title);.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
    }  // EN: Structure delimiter for a block or scope.

    static void printVector(String name, double[] v) {  // EN: Execute line: static void printVector(String name, double[] v) {.
        StringBuilder sb = new StringBuilder();  // EN: Execute a statement: StringBuilder sb = new StringBuilder();.
        sb.append(name).append(" = [");  // EN: Execute a statement: sb.append(name).append(" = [");.
        for (int i = 0; i < v.length; i++) {  // EN: Loop control flow: for (int i = 0; i < v.length; i++) {.
            sb.append(String.format("%.4f", v[i]));  // EN: Execute a statement: sb.append(String.format("%.4f", v[i]));.
            if (i < v.length - 1) sb.append(", ");  // EN: Conditional control flow: if (i < v.length - 1) sb.append(", ");.
        }  // EN: Structure delimiter for a block or scope.
        sb.append("]");  // EN: Execute a statement: sb.append("]");.
        System.out.println(sb.toString());  // EN: Execute a statement: System.out.println(sb.toString());.
    }  // EN: Structure delimiter for a block or scope.

    static void printMatrix(String name, double[][] M) {  // EN: Execute line: static void printMatrix(String name, double[][] M) {.
        System.out.println(name + " =");  // EN: Execute a statement: System.out.println(name + " =");.
        for (double[] row : M) {  // EN: Loop control flow: for (double[] row : M) {.
            StringBuilder sb = new StringBuilder("  [");  // EN: Execute a statement: StringBuilder sb = new StringBuilder(" [");.
            for (int i = 0; i < row.length; i++) {  // EN: Loop control flow: for (int i = 0; i < row.length; i++) {.
                sb.append(String.format("%8.4f", row[i]));  // EN: Execute a statement: sb.append(String.format("%8.4f", row[i]));.
                if (i < row.length - 1) sb.append(", ");  // EN: Conditional control flow: if (i < row.length - 1) sb.append(", ");.
            }  // EN: Structure delimiter for a block or scope.
            sb.append("]");  // EN: Execute a statement: sb.append("]");.
            System.out.println(sb.toString());  // EN: Execute a statement: System.out.println(sb.toString());.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    // 基本向量運算
    static double dotProduct(double[] x, double[] y) {  // EN: Execute line: static double dotProduct(double[] x, double[] y) {.
        double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
        for (int i = 0; i < x.length; i++) {  // EN: Loop control flow: for (int i = 0; i < x.length; i++) {.
            result += x[i] * y[i];  // EN: Execute a statement: result += x[i] * y[i];.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double vectorNorm(double[] x) {  // EN: Execute line: static double vectorNorm(double[] x) {.
        return Math.sqrt(dotProduct(x, x));  // EN: Return from the current function: return Math.sqrt(dotProduct(x, x));.
    }  // EN: Structure delimiter for a block or scope.

    static double[] scalarMultiply(double c, double[] x) {  // EN: Execute line: static double[] scalarMultiply(double c, double[] x) {.
        double[] result = new double[x.length];  // EN: Execute a statement: double[] result = new double[x.length];.
        for (int i = 0; i < x.length; i++) {  // EN: Loop control flow: for (int i = 0; i < x.length; i++) {.
            result[i] = c * x[i];  // EN: Execute a statement: result[i] = c * x[i];.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[] vectorSubtract(double[] x, double[] y) {  // EN: Execute line: static double[] vectorSubtract(double[] x, double[] y) {.
        double[] result = new double[x.length];  // EN: Execute a statement: double[] result = new double[x.length];.
        for (int i = 0; i < x.length; i++) {  // EN: Loop control flow: for (int i = 0; i < x.length; i++) {.
            result[i] = x[i] - y[i];  // EN: Execute a statement: result[i] = x[i] - y[i];.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    // 取得矩陣的第 j 行（column）
    static double[] getColumn(double[][] A, int j) {  // EN: Execute line: static double[] getColumn(double[][] A, int j) {.
        double[] col = new double[A.length];  // EN: Execute a statement: double[] col = new double[A.length];.
        for (int i = 0; i < A.length; i++) {  // EN: Loop control flow: for (int i = 0; i < A.length; i++) {.
            col[i] = A[i][j];  // EN: Execute a statement: col[i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
        return col;  // EN: Return from the current function: return col;.
    }  // EN: Structure delimiter for a block or scope.

    // Gram-Schmidt QR 分解
    static double[][][] qrDecomposition(double[][] A) {  // EN: Execute line: static double[][][] qrDecomposition(double[][] A) {.
        int m = A.length;  // EN: Execute a statement: int m = A.length;.
        int n = A[0].length;  // EN: Execute a statement: int n = A[0].length;.

        // Q: m×n, R: n×n
        double[][] Q = new double[m][n];  // EN: Execute a statement: double[][] Q = new double[m][n];.
        double[][] R = new double[n][n];  // EN: Execute a statement: double[][] R = new double[n][n];.

        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            // 取得 A 的第 j 行
            double[] v = getColumn(A, j);  // EN: Execute a statement: double[] v = getColumn(A, j);.

            // 減去前面所有 q 向量的投影
            for (int i = 0; i < j; i++) {  // EN: Loop control flow: for (int i = 0; i < j; i++) {.
                double[] qi = getColumn(Q, i);  // EN: Execute a statement: double[] qi = getColumn(Q, i);.
                R[i][j] = dotProduct(qi, getColumn(A, j));  // EN: Execute a statement: R[i][j] = dotProduct(qi, getColumn(A, j));.
                double[] proj = scalarMultiply(R[i][j], qi);  // EN: Execute a statement: double[] proj = scalarMultiply(R[i][j], qi);.
                v = vectorSubtract(v, proj);  // EN: Execute a statement: v = vectorSubtract(v, proj);.
            }  // EN: Structure delimiter for a block or scope.

            // 標準化
            R[j][j] = vectorNorm(v);  // EN: Execute a statement: R[j][j] = vectorNorm(v);.

            if (R[j][j] > 1e-10) {  // EN: Conditional control flow: if (R[j][j] > 1e-10) {.
                for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
                    Q[i][j] = v[i] / R[j][j];  // EN: Execute a statement: Q[i][j] = v[i] / R[j][j];.
                }  // EN: Structure delimiter for a block or scope.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.

        return new double[][][] {Q, R};  // EN: Return from the current function: return new double[][][] {Q, R};.
    }  // EN: Structure delimiter for a block or scope.

    // 回代法解上三角方程組 Rx = b
    static double[] solveUpperTriangular(double[][] R, double[] b) {  // EN: Execute line: static double[] solveUpperTriangular(double[][] R, double[] b) {.
        int n = b.length;  // EN: Execute a statement: int n = b.length;.
        double[] x = new double[n];  // EN: Execute a statement: double[] x = new double[n];.

        for (int i = n - 1; i >= 0; i--) {  // EN: Loop control flow: for (int i = n - 1; i >= 0; i--) {.
            x[i] = b[i];  // EN: Execute a statement: x[i] = b[i];.
            for (int j = i + 1; j < n; j++) {  // EN: Loop control flow: for (int j = i + 1; j < n; j++) {.
                x[i] -= R[i][j] * x[j];  // EN: Execute a statement: x[i] -= R[i][j] * x[j];.
            }  // EN: Structure delimiter for a block or scope.
            x[i] /= R[i][i];  // EN: Execute a statement: x[i] /= R[i][i];.
        }  // EN: Structure delimiter for a block or scope.

        return x;  // EN: Return from the current function: return x;.
    }  // EN: Structure delimiter for a block or scope.

    // 用 QR 分解解最小平方問題
    static double[] qrLeastSquares(double[][] A, double[] b) {  // EN: Execute line: static double[] qrLeastSquares(double[][] A, double[] b) {.
        double[][][] qr = qrDecomposition(A);  // EN: Execute a statement: double[][][] qr = qrDecomposition(A);.
        double[][] Q = qr[0];  // EN: Execute a statement: double[][] Q = qr[0];.
        double[][] R = qr[1];  // EN: Execute a statement: double[][] R = qr[1];.

        // 計算 Qᵀb
        int n = Q[0].length;  // EN: Execute a statement: int n = Q[0].length;.
        double[] Qt_b = new double[n];  // EN: Execute a statement: double[] Qt_b = new double[n];.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            double[] qj = getColumn(Q, j);  // EN: Execute a statement: double[] qj = getColumn(Q, j);.
            Qt_b[j] = dotProduct(qj, b);  // EN: Execute a statement: Qt_b[j] = dotProduct(qj, b);.
        }  // EN: Structure delimiter for a block or scope.

        // 解 Rx = Qᵀb
        return solveUpperTriangular(R, Qt_b);  // EN: Return from the current function: return solveUpperTriangular(R, Qt_b);.
    }  // EN: Structure delimiter for a block or scope.

    // 矩陣乘法
    static double[][] matrixMultiply(double[][] A, double[][] B) {  // EN: Execute line: static double[][] matrixMultiply(double[][] A, double[][] B) {.
        int m = A.length;  // EN: Execute a statement: int m = A.length;.
        int k = B.length;  // EN: Execute a statement: int k = B.length;.
        int n = B[0].length;  // EN: Execute a statement: int n = B[0].length;.

        double[][] result = new double[m][n];  // EN: Execute a statement: double[][] result = new double[m][n];.
        for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
            for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
                for (int p = 0; p < k; p++) {  // EN: Loop control flow: for (int p = 0; p < k; p++) {.
                    result[i][j] += A[i][p] * B[p][j];  // EN: Execute a statement: result[i][j] += A[i][p] * B[p][j];.
                }  // EN: Structure delimiter for a block or scope.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    // 矩陣轉置
    static double[][] transpose(double[][] A) {  // EN: Execute line: static double[][] transpose(double[][] A) {.
        int m = A.length;  // EN: Execute a statement: int m = A.length;.
        int n = A[0].length;  // EN: Execute a statement: int n = A[0].length;.
        double[][] result = new double[n][m];  // EN: Execute a statement: double[][] result = new double[n][m];.
        for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
            for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
                result[j][i] = A[i][j];  // EN: Execute a statement: result[j][i] = A[i][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    public static void main(String[] args) {  // EN: Execute line: public static void main(String[] args) {.
        printSeparator("QR 分解示範 (Java)");  // EN: Execute a statement: printSeparator("QR 分解示範 (Java)");.

        // ========================================
        // 1. 基本 QR 分解
        // ========================================
        printSeparator("1. 基本 QR 分解");  // EN: Execute a statement: printSeparator("1. 基本 QR 分解");.

        double[][] A = {  // EN: Execute line: double[][] A = {.
            {1.0, 1.0},  // EN: Execute line: {1.0, 1.0},.
            {1.0, 0.0},  // EN: Execute line: {1.0, 0.0},.
            {0.0, 1.0}  // EN: Execute line: {0.0, 1.0}.
        };  // EN: Structure delimiter for a block or scope.

        System.out.println("輸入矩陣 A：");  // EN: Execute a statement: System.out.println("輸入矩陣 A：");.
        printMatrix("A", A);  // EN: Execute a statement: printMatrix("A", A);.

        double[][][] qr = qrDecomposition(A);  // EN: Execute a statement: double[][][] qr = qrDecomposition(A);.
        double[][] Q = qr[0];  // EN: Execute a statement: double[][] Q = qr[0];.
        double[][] R = qr[1];  // EN: Execute a statement: double[][] R = qr[1];.

        System.out.println("\nQR 分解結果：");  // EN: Execute a statement: System.out.println("\nQR 分解結果：");.
        printMatrix("Q", Q);  // EN: Execute a statement: printMatrix("Q", Q);.
        printMatrix("\nR", R);  // EN: Execute a statement: printMatrix("\nR", R);.

        // 驗證 QᵀQ = I
        double[][] QT = transpose(Q);  // EN: Execute a statement: double[][] QT = transpose(Q);.
        double[][] QTQ = matrixMultiply(QT, Q);  // EN: Execute a statement: double[][] QTQ = matrixMultiply(QT, Q);.
        System.out.println("\n驗證 QᵀQ = I：");  // EN: Execute a statement: System.out.println("\n驗證 QᵀQ = I：");.
        printMatrix("QᵀQ", QTQ);  // EN: Execute a statement: printMatrix("QᵀQ", QTQ);.

        // 驗證 A = QR
        double[][] QR_result = matrixMultiply(Q, R);  // EN: Execute a statement: double[][] QR_result = matrixMultiply(Q, R);.
        System.out.println("\n驗證 A = QR：");  // EN: Execute a statement: System.out.println("\n驗證 A = QR：");.
        printMatrix("QR", QR_result);  // EN: Execute a statement: printMatrix("QR", QR_result);.

        // ========================================
        // 2. 用 QR 解最小平方
        // ========================================
        printSeparator("2. 用 QR 解最小平方");  // EN: Execute a statement: printSeparator("2. 用 QR 解最小平方");.

        // 數據
        double[] t = {0.0, 1.0, 2.0};  // EN: Execute a statement: double[] t = {0.0, 1.0, 2.0};.
        double[] b = {1.0, 3.0, 4.0};  // EN: Execute a statement: double[] b = {1.0, 3.0, 4.0};.

        System.out.println("數據點：");  // EN: Execute a statement: System.out.println("數據點：");.
        for (int i = 0; i < t.length; i++) {  // EN: Loop control flow: for (int i = 0; i < t.length; i++) {.
            System.out.printf("  (%.1f, %.1f)%n", t[i], b[i]);  // EN: Execute a statement: System.out.printf(" (%.1f, %.1f)%n", t[i], b[i]);.
        }  // EN: Structure delimiter for a block or scope.

        // 設計矩陣
        double[][] A_ls = new double[t.length][2];  // EN: Execute a statement: double[][] A_ls = new double[t.length][2];.
        for (int i = 0; i < t.length; i++) {  // EN: Loop control flow: for (int i = 0; i < t.length; i++) {.
            A_ls[i][0] = 1.0;  // EN: Execute a statement: A_ls[i][0] = 1.0;.
            A_ls[i][1] = t[i];  // EN: Execute a statement: A_ls[i][1] = t[i];.
        }  // EN: Structure delimiter for a block or scope.

        System.out.println("\n設計矩陣 A：");  // EN: Execute a statement: System.out.println("\n設計矩陣 A：");.
        printMatrix("A", A_ls);  // EN: Execute a statement: printMatrix("A", A_ls);.
        printVector("觀測值 b", b);  // EN: Execute a statement: printVector("觀測值 b", b);.

        // QR 分解
        double[][][] qr_ls = qrDecomposition(A_ls);  // EN: Execute a statement: double[][][] qr_ls = qrDecomposition(A_ls);.
        double[][] Q_ls = qr_ls[0];  // EN: Execute a statement: double[][] Q_ls = qr_ls[0];.
        double[][] R_ls = qr_ls[1];  // EN: Execute a statement: double[][] R_ls = qr_ls[1];.
        printMatrix("\nQ", Q_ls);  // EN: Execute a statement: printMatrix("\nQ", Q_ls);.
        printMatrix("R", R_ls);  // EN: Execute a statement: printMatrix("R", R_ls);.

        // 解最小平方
        double[] x = qrLeastSquares(A_ls, b);  // EN: Execute a statement: double[] x = qrLeastSquares(A_ls, b);.
        printVector("\n解 x", x);  // EN: Execute a statement: printVector("\n解 x", x);.

        System.out.printf("%n最佳直線：y = %.4f + %.4ft%n", x[0], x[1]);  // EN: Execute a statement: System.out.printf("%n最佳直線：y = %.4f + %.4ft%n", x[0], x[1]);.

        // ========================================
        // 3. 3×3 矩陣的 QR 分解
        // ========================================
        printSeparator("3. 3×3 矩陣的 QR 分解");  // EN: Execute a statement: printSeparator("3. 3×3 矩陣的 QR 分解");.

        double[][] A2 = {  // EN: Execute line: double[][] A2 = {.
            {1.0, 1.0, 0.0},  // EN: Execute line: {1.0, 1.0, 0.0},.
            {1.0, 0.0, 1.0},  // EN: Execute line: {1.0, 0.0, 1.0},.
            {0.0, 1.0, 1.0}  // EN: Execute line: {0.0, 1.0, 1.0}.
        };  // EN: Structure delimiter for a block or scope.

        System.out.println("輸入矩陣 A：");  // EN: Execute a statement: System.out.println("輸入矩陣 A：");.
        printMatrix("A", A2);  // EN: Execute a statement: printMatrix("A", A2);.

        double[][][] qr2 = qrDecomposition(A2);  // EN: Execute a statement: double[][][] qr2 = qrDecomposition(A2);.
        double[][] Q2 = qr2[0];  // EN: Execute a statement: double[][] Q2 = qr2[0];.
        double[][] R2 = qr2[1];  // EN: Execute a statement: double[][] R2 = qr2[1];.

        System.out.println("\nQR 分解結果：");  // EN: Execute a statement: System.out.println("\nQR 分解結果：");.
        printMatrix("Q", Q2);  // EN: Execute a statement: printMatrix("Q", Q2);.
        printMatrix("\nR", R2);  // EN: Execute a statement: printMatrix("\nR", R2);.

        // 總結
        printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
        System.out.println("""  // EN: Execute line: System.out.println(""".

QR 分解核心：  // EN: Execute line: QR 分解核心：.

1. A = QR  // EN: Execute line: 1. A = QR.
   - Q: 標準正交矩陣 (QᵀQ = I)  // EN: Execute line: - Q: 標準正交矩陣 (QᵀQ = I).
   - R: 上三角矩陣  // EN: Execute line: - R: 上三角矩陣.

2. Gram-Schmidt 演算法：  // EN: Execute line: 2. Gram-Schmidt 演算法：.
   - 對 A 的行向量正交化得到 Q  // EN: Execute line: - 對 A 的行向量正交化得到 Q.
   - R 的元素是投影係數  // EN: Execute line: - R 的元素是投影係數.

3. 用 QR 解最小平方：  // EN: Execute line: 3. 用 QR 解最小平方：.
   min ‖Ax - b‖²  // EN: Execute line: min ‖Ax - b‖².
   → Rx = Qᵀb  // EN: Execute line: → Rx = Qᵀb.

4. 優勢：  // EN: Execute line: 4. 優勢：.
   - 比正規方程更穩定  // EN: Execute line: - 比正規方程更穩定.
   - 避免計算 AᵀA  // EN: Execute line: - 避免計算 AᵀA.
            """);  // EN: Execute a statement: """);.

        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
        System.out.println("示範完成！");  // EN: Execute a statement: System.out.println("示範完成！");.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
