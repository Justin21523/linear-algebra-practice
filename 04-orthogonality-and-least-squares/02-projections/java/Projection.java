/**
 * 投影 (Projections)
 *
 * 本程式示範：
 * 1. 投影到直線
 * 2. 投影矩陣及其性質
 * 3. 誤差向量的正交性驗證
 *
 * 編譯：javac Projection.java
 * 執行：java Projection
 */

public class Projection {  // EN: Execute line: public class Projection {.

    private static final double EPSILON = 1e-10;  // EN: Execute a statement: private static final double EPSILON = 1e-10;.

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

    static double[][] outerProduct(double[] x, double[] y) {  // EN: Execute line: static double[][] outerProduct(double[] x, double[] y) {.
        double[][] result = new double[x.length][y.length];  // EN: Execute a statement: double[][] result = new double[x.length][y.length];.
        for (int i = 0; i < x.length; i++) {  // EN: Loop control flow: for (int i = 0; i < x.length; i++) {.
            for (int j = 0; j < y.length; j++) {  // EN: Loop control flow: for (int j = 0; j < y.length; j++) {.
                result[i][j] = x[i] * y[j];  // EN: Execute a statement: result[i][j] = x[i] * y[j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[][] matrixScalarMultiply(double c, double[][] A) {  // EN: Execute line: static double[][] matrixScalarMultiply(double c, double[][] A) {.
        double[][] result = new double[A.length][A[0].length];  // EN: Execute a statement: double[][] result = new double[A.length][A[0].length];.
        for (int i = 0; i < A.length; i++) {  // EN: Loop control flow: for (int i = 0; i < A.length; i++) {.
            for (int j = 0; j < A[0].length; j++) {  // EN: Loop control flow: for (int j = 0; j < A[0].length; j++) {.
                result[i][j] = c * A[i][j];  // EN: Execute a statement: result[i][j] = c * A[i][j];.
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

    // ========================================
    // 投影函數
    // ========================================

    /**
     * 投影結果類別
     */
    static class ProjectionResult {  // EN: Execute line: static class ProjectionResult {.
        double xHat;  // EN: Execute a statement: double xHat;.
        double[] projection;  // EN: Execute a statement: double[] projection;.
        double[] error;  // EN: Execute a statement: double[] error;.
        double errorNorm;  // EN: Execute a statement: double errorNorm;.

        ProjectionResult(double xHat, double[] projection, double[] error, double errorNorm) {  // EN: Execute line: ProjectionResult(double xHat, double[] projection, double[] error, doub….
            this.xHat = xHat;  // EN: Execute a statement: this.xHat = xHat;.
            this.projection = projection;  // EN: Execute a statement: this.projection = projection;.
            this.error = error;  // EN: Execute a statement: this.error = error;.
            this.errorNorm = errorNorm;  // EN: Execute a statement: this.errorNorm = errorNorm;.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    /**
     * 投影到直線
     * p = (aᵀb / aᵀa) * a
     */
    static ProjectionResult projectOntoLine(double[] b, double[] a) {  // EN: Execute line: static ProjectionResult projectOntoLine(double[] b, double[] a) {.
        double aTb = dotProduct(a, b);  // EN: Execute a statement: double aTb = dotProduct(a, b);.
        double aTa = dotProduct(a, a);  // EN: Execute a statement: double aTa = dotProduct(a, a);.

        double xHat = aTb / aTa;  // EN: Execute a statement: double xHat = aTb / aTa;.
        double[] p = scalarMultiply(xHat, a);  // EN: Execute a statement: double[] p = scalarMultiply(xHat, a);.
        double[] e = vectorSubtract(b, p);  // EN: Execute a statement: double[] e = vectorSubtract(b, p);.

        return new ProjectionResult(xHat, p, e, vectorNorm(e));  // EN: Return from the current function: return new ProjectionResult(xHat, p, e, vectorNorm(e));.
    }  // EN: Structure delimiter for a block or scope.

    /**
     * 投影到直線的投影矩陣
     * P = aaᵀ / (aᵀa)
     */
    static double[][] projectionMatrixLine(double[] a) {  // EN: Execute line: static double[][] projectionMatrixLine(double[] a) {.
        double aTa = dotProduct(a, a);  // EN: Execute a statement: double aTa = dotProduct(a, a);.
        double[][] aaT = outerProduct(a, a);  // EN: Execute a statement: double[][] aaT = outerProduct(a, a);.
        return matrixScalarMultiply(1.0 / aTa, aaT);  // EN: Return from the current function: return matrixScalarMultiply(1.0 / aTa, aaT);.
    }  // EN: Structure delimiter for a block or scope.

    /**
     * 驗證投影矩陣的性質
     */
    static void verifyProjectionMatrix(double[][] P, String name) {  // EN: Execute line: static void verifyProjectionMatrix(double[][] P, String name) {.
        int n = P.length;  // EN: Execute a statement: int n = P.length;.

        System.out.println("\n驗證 " + name + " 的性質：");  // EN: Execute a statement: System.out.println("\n驗證 " + name + " 的性質：");.

        // 對稱性
        boolean isSymmetric = true;  // EN: Execute a statement: boolean isSymmetric = true;.
        for (int i = 0; i < n && isSymmetric; i++) {  // EN: Loop control flow: for (int i = 0; i < n && isSymmetric; i++) {.
            for (int j = 0; j < n && isSymmetric; j++) {  // EN: Loop control flow: for (int j = 0; j < n && isSymmetric; j++) {.
                if (Math.abs(P[i][j] - P[j][i]) > EPSILON) {  // EN: Conditional control flow: if (Math.abs(P[i][j] - P[j][i]) > EPSILON) {.
                    isSymmetric = false;  // EN: Execute a statement: isSymmetric = false;.
                }  // EN: Structure delimiter for a block or scope.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        System.out.println("  對稱性 (" + name + "ᵀ = " + name + ")：" + isSymmetric);  // EN: Execute a statement: System.out.println(" 對稱性 (" + name + "ᵀ = " + name + ")：" + isSymmetric….

        // 冪等性
        double[][] P2 = matrixMultiply(P, P);  // EN: Execute a statement: double[][] P2 = matrixMultiply(P, P);.
        boolean isIdempotent = true;  // EN: Execute a statement: boolean isIdempotent = true;.
        for (int i = 0; i < n && isIdempotent; i++) {  // EN: Loop control flow: for (int i = 0; i < n && isIdempotent; i++) {.
            for (int j = 0; j < n && isIdempotent; j++) {  // EN: Loop control flow: for (int j = 0; j < n && isIdempotent; j++) {.
                if (Math.abs(P[i][j] - P2[i][j]) > EPSILON) {  // EN: Conditional control flow: if (Math.abs(P[i][j] - P2[i][j]) > EPSILON) {.
                    isIdempotent = false;  // EN: Execute a statement: isIdempotent = false;.
                }  // EN: Structure delimiter for a block or scope.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        System.out.println("  冪等性 (" + name + "² = " + name + ")：" + isIdempotent);  // EN: Execute a statement: System.out.println(" 冪等性 (" + name + "² = " + name + ")：" + isIdempoten….
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 主程式
    // ========================================

    public static void main(String[] args) {  // EN: Execute line: public static void main(String[] args) {.
        printSeparator("投影示範 (Java)\nProjection Demo");  // EN: Execute a statement: printSeparator("投影示範 (Java)\nProjection Demo");.

        // 1. 投影到直線
        printSeparator("1. 投影到直線");  // EN: Execute a statement: printSeparator("1. 投影到直線");.

        double[] a = {1.0, 1.0};  // EN: Execute a statement: double[] a = {1.0, 1.0};.
        double[] b = {2.0, 0.0};  // EN: Execute a statement: double[] b = {2.0, 0.0};.

        printVector("方向 a", a);  // EN: Execute a statement: printVector("方向 a", a);.
        printVector("向量 b", b);  // EN: Execute a statement: printVector("向量 b", b);.

        ProjectionResult result = projectOntoLine(b, a);  // EN: Execute a statement: ProjectionResult result = projectOntoLine(b, a);.

        System.out.printf("\n投影係數 x̂ = (aᵀb)/(aᵀa) = %.4f%n", result.xHat);  // EN: Execute a statement: System.out.printf("\n投影係數 x̂ = (aᵀb)/(aᵀa) = %.4f%n", result.xHat);.
        printVector("投影 p = x̂a", result.projection);  // EN: Execute a statement: printVector("投影 p = x̂a", result.projection);.
        printVector("誤差 e = b - p", result.error);  // EN: Execute a statement: printVector("誤差 e = b - p", result.error);.

        // 驗證正交性
        double eDotA = dotProduct(result.error, a);  // EN: Execute a statement: double eDotA = dotProduct(result.error, a);.
        System.out.printf("\n驗證 e ⊥ a：e · a = %.6f%n", eDotA);  // EN: Execute a statement: System.out.printf("\n驗證 e ⊥ a：e · a = %.6f%n", eDotA);.
        System.out.println("正交？ " + (Math.abs(eDotA) < EPSILON));  // EN: Execute a statement: System.out.println("正交？ " + (Math.abs(eDotA) < EPSILON));.

        // 2. 投影矩陣
        printSeparator("2. 投影矩陣（到直線）");  // EN: Execute a statement: printSeparator("2. 投影矩陣（到直線）");.

        double[] a2 = {1.0, 2.0};  // EN: Execute a statement: double[] a2 = {1.0, 2.0};.
        printVector("方向 a", a2);  // EN: Execute a statement: printVector("方向 a", a2);.

        double[][] P = projectionMatrixLine(a2);  // EN: Execute a statement: double[][] P = projectionMatrixLine(a2);.
        printMatrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P);  // EN: Execute a statement: printMatrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P);.

        verifyProjectionMatrix(P, "P");  // EN: Execute a statement: verifyProjectionMatrix(P, "P");.

        // 用投影矩陣計算投影
        double[] b2 = {3.0, 4.0};  // EN: Execute a statement: double[] b2 = {3.0, 4.0};.
        printVector("\n向量 b", b2);  // EN: Execute a statement: printVector("\n向量 b", b2);.

        double[] p = matrixVectorMultiply(P, b2);  // EN: Execute a statement: double[] p = matrixVectorMultiply(P, b2);.
        printVector("投影 p = Pb", p);  // EN: Execute a statement: printVector("投影 p = Pb", p);.

        // 3. 多個向量的投影
        printSeparator("3. 批次投影");  // EN: Execute a statement: printSeparator("3. 批次投影");.

        double[][] vectors = {{1.0, 0.0}, {0.0, 1.0}, {2.0, 2.0}, {3.0, -1.0}};  // EN: Execute a statement: double[][] vectors = {{1.0, 0.0}, {0.0, 1.0}, {2.0, 2.0}, {3.0, -1.0}};.

        System.out.println("方向 a = [1, 2]");  // EN: Execute a statement: System.out.println("方向 a = [1, 2]");.
        System.out.println("\n各向量投影結果：");  // EN: Execute a statement: System.out.println("\n各向量投影結果：");.

        for (double[] v : vectors) {  // EN: Loop control flow: for (double[] v : vectors) {.
            ProjectionResult proj = projectOntoLine(v, a2);  // EN: Execute a statement: ProjectionResult proj = projectOntoLine(v, a2);.
            System.out.printf("  [%.1f, %.1f] -> [%.4f, %.4f]%n",  // EN: Execute line: System.out.printf(" [%.1f, %.1f] -> [%.4f, %.4f]%n",.
                v[0], v[1], proj.projection[0], proj.projection[1]);  // EN: Execute a statement: v[0], v[1], proj.projection[0], proj.projection[1]);.
        }  // EN: Structure delimiter for a block or scope.

        // 總結
        printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
        System.out.println("""  // EN: Execute line: System.out.println(""".

投影公式：  // EN: Execute line: 投影公式：.

1. 投影到直線：  // EN: Execute line: 1. 投影到直線：.
   p = (aᵀb / aᵀa) a  // EN: Execute line: p = (aᵀb / aᵀa) a.
   P = aaᵀ / (aᵀa)  // EN: Execute line: P = aaᵀ / (aᵀa).

2. 投影到子空間：  // EN: Execute line: 2. 投影到子空間：.
   p = A(AᵀA)⁻¹Aᵀb  // EN: Execute line: p = A(AᵀA)⁻¹Aᵀb.
   P = A(AᵀA)⁻¹Aᵀ  // EN: Execute line: P = A(AᵀA)⁻¹Aᵀ.

3. 投影矩陣性質：  // EN: Execute line: 3. 投影矩陣性質：.
   Pᵀ = P（對稱）  // EN: Execute line: Pᵀ = P（對稱）.
   P² = P（冪等）  // EN: Execute line: P² = P（冪等）.
""");  // EN: Execute a statement: """);.

        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
        System.out.println("示範完成！");  // EN: Execute a statement: System.out.println("示範完成！");.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
