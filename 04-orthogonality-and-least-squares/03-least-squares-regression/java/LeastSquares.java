/**
 * 最小平方回歸 (Least Squares Regression)
 *
 * 本程式示範：
 * 1. 正規方程求解最小平方問題
 * 2. 簡單線性迴歸
 * 3. 殘差分析
 *
 * 編譯：javac LeastSquares.java
 * 執行：java LeastSquares
 */

public class LeastSquares {  // EN: Execute line: public class LeastSquares {.

    // ========================================
    // 輔助方法
    // ========================================

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
            for (int j = 0; j < row.length; j++) {  // EN: Loop control flow: for (int j = 0; j < row.length; j++) {.
                sb.append(String.format("%8.4f", row[j]));  // EN: Execute a statement: sb.append(String.format("%8.4f", row[j]));.
                if (j < row.length - 1) sb.append(", ");  // EN: Conditional control flow: if (j < row.length - 1) sb.append(", ");.
            }  // EN: Structure delimiter for a block or scope.
            sb.append("]");  // EN: Execute a statement: sb.append("]");.
            System.out.println(sb.toString());  // EN: Execute a statement: System.out.println(sb.toString());.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 基本運算
    // ========================================

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

    static double[] vectorSubtract(double[] x, double[] y) {  // EN: Execute line: static double[] vectorSubtract(double[] x, double[] y) {.
        double[] result = new double[x.length];  // EN: Execute a statement: double[] result = new double[x.length];.
        for (int i = 0; i < x.length; i++) {  // EN: Loop control flow: for (int i = 0; i < x.length; i++) {.
            result[i] = x[i] - y[i];  // EN: Execute a statement: result[i] = x[i] - y[i];.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[][] transpose(double[][] A) {  // EN: Execute line: static double[][] transpose(double[][] A) {.
        int m = A.length, n = A[0].length;  // EN: Execute a statement: int m = A.length, n = A[0].length;.
        double[][] result = new double[n][m];  // EN: Execute a statement: double[][] result = new double[n][m];.
        for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
            for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
                result[j][i] = A[i][j];  // EN: Execute a statement: result[j][i] = A[i][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[][] matrixMultiply(double[][] A, double[][] B) {  // EN: Execute line: static double[][] matrixMultiply(double[][] A, double[][] B) {.
        int m = A.length, k = B.length, n = B[0].length;  // EN: Execute a statement: int m = A.length, k = B.length, n = B[0].length;.
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

    static double[] matrixVectorMultiply(double[][] A, double[] x) {  // EN: Execute line: static double[] matrixVectorMultiply(double[][] A, double[] x) {.
        double[] result = new double[A.length];  // EN: Execute a statement: double[] result = new double[A.length];.
        for (int i = 0; i < A.length; i++) {  // EN: Loop control flow: for (int i = 0; i < A.length; i++) {.
            for (int j = 0; j < x.length; j++) {  // EN: Loop control flow: for (int j = 0; j < x.length; j++) {.
                result[i] += A[i][j] * x[j];  // EN: Execute a statement: result[i] += A[i][j] * x[j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[] solve2x2(double[][] A, double[] b) {  // EN: Execute line: static double[] solve2x2(double[][] A, double[] b) {.
        double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Execute a statement: double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];.
        return new double[] {  // EN: Return from the current function: return new double[] {.
            (A[1][1] * b[0] - A[0][1] * b[1]) / det,  // EN: Execute line: (A[1][1] * b[0] - A[0][1] * b[1]) / det,.
            (-A[1][0] * b[0] + A[0][0] * b[1]) / det  // EN: Execute line: (-A[1][0] * b[0] + A[0][0] * b[1]) / det.
        };  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 最小平方求解
    // ========================================

    static class LeastSquaresResult {  // EN: Execute line: static class LeastSquaresResult {.
        double[] coefficients;  // EN: Execute a statement: double[] coefficients;.
        double[] fitted;  // EN: Execute a statement: double[] fitted;.
        double[] residual;  // EN: Execute a statement: double[] residual;.
        double residualNorm;  // EN: Execute a statement: double residualNorm;.
        double rSquared;  // EN: Execute a statement: double rSquared;.
    }  // EN: Structure delimiter for a block or scope.

    static double[][] createDesignMatrixLinear(double[] t) {  // EN: Execute line: static double[][] createDesignMatrixLinear(double[] t) {.
        double[][] A = new double[t.length][2];  // EN: Execute a statement: double[][] A = new double[t.length][2];.
        for (int i = 0; i < t.length; i++) {  // EN: Loop control flow: for (int i = 0; i < t.length; i++) {.
            A[i][0] = 1.0;  // EN: Execute a statement: A[i][0] = 1.0;.
            A[i][1] = t[i];  // EN: Execute a statement: A[i][1] = t[i];.
        }  // EN: Structure delimiter for a block or scope.
        return A;  // EN: Return from the current function: return A;.
    }  // EN: Structure delimiter for a block or scope.

    static LeastSquaresResult leastSquaresSolve(double[][] A, double[] b) {  // EN: Execute line: static LeastSquaresResult leastSquaresSolve(double[][] A, double[] b) {.
        LeastSquaresResult result = new LeastSquaresResult();  // EN: Execute a statement: LeastSquaresResult result = new LeastSquaresResult();.

        // AᵀA
        double[][] AT = transpose(A);  // EN: Execute a statement: double[][] AT = transpose(A);.
        double[][] ATA = matrixMultiply(AT, A);  // EN: Execute a statement: double[][] ATA = matrixMultiply(AT, A);.

        // Aᵀb
        double[] ATb = matrixVectorMultiply(AT, b);  // EN: Execute a statement: double[] ATb = matrixVectorMultiply(AT, b);.

        // 解
        result.coefficients = solve2x2(ATA, ATb);  // EN: Execute a statement: result.coefficients = solve2x2(ATA, ATb);.

        // 擬合值和殘差
        result.fitted = matrixVectorMultiply(A, result.coefficients);  // EN: Execute a statement: result.fitted = matrixVectorMultiply(A, result.coefficients);.
        result.residual = vectorSubtract(b, result.fitted);  // EN: Execute a statement: result.residual = vectorSubtract(b, result.fitted);.
        result.residualNorm = vectorNorm(result.residual);  // EN: Execute a statement: result.residualNorm = vectorNorm(result.residual);.

        // R²
        double bMean = 0.0;  // EN: Execute a statement: double bMean = 0.0;.
        for (double bi : b) bMean += bi;  // EN: Loop control flow: for (double bi : b) bMean += bi;.
        bMean /= b.length;  // EN: Execute a statement: bMean /= b.length;.

        double tss = 0.0, rss = 0.0;  // EN: Execute a statement: double tss = 0.0, rss = 0.0;.
        for (int i = 0; i < b.length; i++) {  // EN: Loop control flow: for (int i = 0; i < b.length; i++) {.
            tss += (b[i] - bMean) * (b[i] - bMean);  // EN: Execute a statement: tss += (b[i] - bMean) * (b[i] - bMean);.
            rss += result.residual[i] * result.residual[i];  // EN: Execute a statement: rss += result.residual[i] * result.residual[i];.
        }  // EN: Structure delimiter for a block or scope.
        result.rSquared = (tss > 0) ? (1.0 - rss / tss) : 0.0;  // EN: Execute a statement: result.rSquared = (tss > 0) ? (1.0 - rss / tss) : 0.0;.

        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 主程式
    // ========================================

    public static void main(String[] args) {  // EN: Execute line: public static void main(String[] args) {.
        printSeparator("最小平方回歸示範 (Java)\nLeast Squares Regression Demo");  // EN: Execute a statement: printSeparator("最小平方回歸示範 (Java)\nLeast Squares Regression Demo");.

        // 1. 簡單線性迴歸
        printSeparator("1. 簡單線性迴歸：y = C + Dt");  // EN: Execute a statement: printSeparator("1. 簡單線性迴歸：y = C + Dt");.

        double[] t = {0.0, 1.0, 2.0};  // EN: Execute a statement: double[] t = {0.0, 1.0, 2.0};.
        double[] b = {1.0, 3.0, 4.0};  // EN: Execute a statement: double[] b = {1.0, 3.0, 4.0};.

        System.out.println("數據點：");  // EN: Execute a statement: System.out.println("數據點：");.
        for (int i = 0; i < t.length; i++) {  // EN: Loop control flow: for (int i = 0; i < t.length; i++) {.
            System.out.printf("  t = %.1f, b = %.1f%n", t[i], b[i]);  // EN: Execute a statement: System.out.printf(" t = %.1f, b = %.1f%n", t[i], b[i]);.
        }  // EN: Structure delimiter for a block or scope.

        double[][] A = createDesignMatrixLinear(t);  // EN: Execute a statement: double[][] A = createDesignMatrixLinear(t);.
        printMatrix("\n設計矩陣 A [1, t]", A);  // EN: Execute a statement: printMatrix("\n設計矩陣 A [1, t]", A);.
        printVector("觀測值 b", b);  // EN: Execute a statement: printVector("觀測值 b", b);.

        LeastSquaresResult result = leastSquaresSolve(A, b);  // EN: Execute a statement: LeastSquaresResult result = leastSquaresSolve(A, b);.

        System.out.println("\n【解】");  // EN: Execute a statement: System.out.println("\n【解】");.
        System.out.printf("C（截距）= %.4f%n", result.coefficients[0]);  // EN: Execute a statement: System.out.printf("C（截距）= %.4f%n", result.coefficients[0]);.
        System.out.printf("D（斜率）= %.4f%n", result.coefficients[1]);  // EN: Execute a statement: System.out.printf("D（斜率）= %.4f%n", result.coefficients[1]);.
        System.out.printf("%n最佳直線：y = %.4f + %.4ft%n",  // EN: Execute line: System.out.printf("%n最佳直線：y = %.4f + %.4ft%n",.
            result.coefficients[0], result.coefficients[1]);  // EN: Execute a statement: result.coefficients[0], result.coefficients[1]);.

        printVector("\n擬合值 ŷ", result.fitted);  // EN: Execute a statement: printVector("\n擬合值 ŷ", result.fitted);.
        printVector("殘差 e", result.residual);  // EN: Execute a statement: printVector("殘差 e", result.residual);.
        System.out.printf("殘差範數 ‖e‖ = %.4f%n", result.residualNorm);  // EN: Execute a statement: System.out.printf("殘差範數 ‖e‖ = %.4f%n", result.residualNorm);.
        System.out.printf("R² = %.4f%n", result.rSquared);  // EN: Execute a statement: System.out.printf("R² = %.4f%n", result.rSquared);.

        // 2. 更多數據
        printSeparator("2. 更多數據點");  // EN: Execute a statement: printSeparator("2. 更多數據點");.

        double[] t2 = {0.0, 1.0, 2.0, 3.0, 4.0};  // EN: Execute a statement: double[] t2 = {0.0, 1.0, 2.0, 3.0, 4.0};.
        double[] b2 = {1.0, 2.5, 3.5, 5.0, 6.5};  // EN: Execute a statement: double[] b2 = {1.0, 2.5, 3.5, 5.0, 6.5};.

        System.out.println("數據點：");  // EN: Execute a statement: System.out.println("數據點：");.
        for (int i = 0; i < t2.length; i++) {  // EN: Loop control flow: for (int i = 0; i < t2.length; i++) {.
            System.out.printf("  (%.1f, %.1f)%n", t2[i], b2[i]);  // EN: Execute a statement: System.out.printf(" (%.1f, %.1f)%n", t2[i], b2[i]);.
        }  // EN: Structure delimiter for a block or scope.

        double[][] A2 = createDesignMatrixLinear(t2);  // EN: Execute a statement: double[][] A2 = createDesignMatrixLinear(t2);.
        LeastSquaresResult result2 = leastSquaresSolve(A2, b2);  // EN: Execute a statement: LeastSquaresResult result2 = leastSquaresSolve(A2, b2);.

        System.out.printf("%n最佳直線：y = %.4f + %.4ft%n",  // EN: Execute line: System.out.printf("%n最佳直線：y = %.4f + %.4ft%n",.
            result2.coefficients[0], result2.coefficients[1]);  // EN: Execute a statement: result2.coefficients[0], result2.coefficients[1]);.
        System.out.printf("R² = %.4f%n", result2.rSquared);  // EN: Execute a statement: System.out.printf("R² = %.4f%n", result2.rSquared);.

        // 總結
        printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
        System.out.println("""  // EN: Execute line: System.out.println(""".

最小平方法核心公式：  // EN: Execute line: 最小平方法核心公式：.

1. 正規方程：AᵀA x̂ = Aᵀb  // EN: Execute line: 1. 正規方程：AᵀA x̂ = Aᵀb.

2. 解：x̂ = (AᵀA)⁻¹Aᵀb  // EN: Execute line: 2. 解：x̂ = (AᵀA)⁻¹Aᵀb.

3. 幾何意義：Ax̂ 是 b 在 C(A) 上的投影  // EN: Execute line: 3. 幾何意義：Ax̂ 是 b 在 C(A) 上的投影.

4. R² = 1 - RSS/TSS（越接近 1 越好）  // EN: Execute line: 4. R² = 1 - RSS/TSS（越接近 1 越好）.
""");  // EN: Execute a statement: """);.

        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
        System.out.println("示範完成！");  // EN: Execute a statement: System.out.println("示範完成！");.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
