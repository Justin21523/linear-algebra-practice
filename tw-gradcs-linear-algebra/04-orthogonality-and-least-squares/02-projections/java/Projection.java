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

public class Projection {

    private static final double EPSILON = 1e-10;

    // ========================================
    // 輔助方法
    // ========================================

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
            for (int j = 0; j < row.length; j++) {
                sb.append(String.format("%8.4f", row[j]));
                if (j < row.length - 1) sb.append(", ");
            }
            sb.append("]");
            System.out.println(sb.toString());
        }
    }

    // ========================================
    // 基本運算
    // ========================================

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

    static double[][] outerProduct(double[] x, double[] y) {
        double[][] result = new double[x.length][y.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < y.length; j++) {
                result[i][j] = x[i] * y[j];
            }
        }
        return result;
    }

    static double[][] matrixScalarMultiply(double c, double[][] A) {
        double[][] result = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                result[i][j] = c * A[i][j];
            }
        }
        return result;
    }

    static double[] matrixVectorMultiply(double[][] A, double[] x) {
        double[] result = new double[A.length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < x.length; j++) {
                result[i] += A[i][j] * x[j];
            }
        }
        return result;
    }

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

    // ========================================
    // 投影函數
    // ========================================

    /**
     * 投影結果類別
     */
    static class ProjectionResult {
        double xHat;
        double[] projection;
        double[] error;
        double errorNorm;

        ProjectionResult(double xHat, double[] projection, double[] error, double errorNorm) {
            this.xHat = xHat;
            this.projection = projection;
            this.error = error;
            this.errorNorm = errorNorm;
        }
    }

    /**
     * 投影到直線
     * p = (aᵀb / aᵀa) * a
     */
    static ProjectionResult projectOntoLine(double[] b, double[] a) {
        double aTb = dotProduct(a, b);
        double aTa = dotProduct(a, a);

        double xHat = aTb / aTa;
        double[] p = scalarMultiply(xHat, a);
        double[] e = vectorSubtract(b, p);

        return new ProjectionResult(xHat, p, e, vectorNorm(e));
    }

    /**
     * 投影到直線的投影矩陣
     * P = aaᵀ / (aᵀa)
     */
    static double[][] projectionMatrixLine(double[] a) {
        double aTa = dotProduct(a, a);
        double[][] aaT = outerProduct(a, a);
        return matrixScalarMultiply(1.0 / aTa, aaT);
    }

    /**
     * 驗證投影矩陣的性質
     */
    static void verifyProjectionMatrix(double[][] P, String name) {
        int n = P.length;

        System.out.println("\n驗證 " + name + " 的性質：");

        // 對稱性
        boolean isSymmetric = true;
        for (int i = 0; i < n && isSymmetric; i++) {
            for (int j = 0; j < n && isSymmetric; j++) {
                if (Math.abs(P[i][j] - P[j][i]) > EPSILON) {
                    isSymmetric = false;
                }
            }
        }
        System.out.println("  對稱性 (" + name + "ᵀ = " + name + ")：" + isSymmetric);

        // 冪等性
        double[][] P2 = matrixMultiply(P, P);
        boolean isIdempotent = true;
        for (int i = 0; i < n && isIdempotent; i++) {
            for (int j = 0; j < n && isIdempotent; j++) {
                if (Math.abs(P[i][j] - P2[i][j]) > EPSILON) {
                    isIdempotent = false;
                }
            }
        }
        System.out.println("  冪等性 (" + name + "² = " + name + ")：" + isIdempotent);
    }

    // ========================================
    // 主程式
    // ========================================

    public static void main(String[] args) {
        printSeparator("投影示範 (Java)\nProjection Demo");

        // 1. 投影到直線
        printSeparator("1. 投影到直線");

        double[] a = {1.0, 1.0};
        double[] b = {2.0, 0.0};

        printVector("方向 a", a);
        printVector("向量 b", b);

        ProjectionResult result = projectOntoLine(b, a);

        System.out.printf("\n投影係數 x̂ = (aᵀb)/(aᵀa) = %.4f%n", result.xHat);
        printVector("投影 p = x̂a", result.projection);
        printVector("誤差 e = b - p", result.error);

        // 驗證正交性
        double eDotA = dotProduct(result.error, a);
        System.out.printf("\n驗證 e ⊥ a：e · a = %.6f%n", eDotA);
        System.out.println("正交？ " + (Math.abs(eDotA) < EPSILON));

        // 2. 投影矩陣
        printSeparator("2. 投影矩陣（到直線）");

        double[] a2 = {1.0, 2.0};
        printVector("方向 a", a2);

        double[][] P = projectionMatrixLine(a2);
        printMatrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P);

        verifyProjectionMatrix(P, "P");

        // 用投影矩陣計算投影
        double[] b2 = {3.0, 4.0};
        printVector("\n向量 b", b2);

        double[] p = matrixVectorMultiply(P, b2);
        printVector("投影 p = Pb", p);

        // 3. 多個向量的投影
        printSeparator("3. 批次投影");

        double[][] vectors = {{1.0, 0.0}, {0.0, 1.0}, {2.0, 2.0}, {3.0, -1.0}};

        System.out.println("方向 a = [1, 2]");
        System.out.println("\n各向量投影結果：");

        for (double[] v : vectors) {
            ProjectionResult proj = projectOntoLine(v, a2);
            System.out.printf("  [%.1f, %.1f] -> [%.4f, %.4f]%n",
                v[0], v[1], proj.projection[0], proj.projection[1]);
        }

        // 總結
        printSeparator("總結");
        System.out.println("""

投影公式：

1. 投影到直線：
   p = (aᵀb / aᵀa) a
   P = aaᵀ / (aᵀa)

2. 投影到子空間：
   p = A(AᵀA)⁻¹Aᵀb
   P = A(AᵀA)⁻¹Aᵀ

3. 投影矩陣性質：
   Pᵀ = P（對稱）
   P² = P（冪等）
""");

        System.out.println("=".repeat(60));
        System.out.println("示範完成！");
        System.out.println("=".repeat(60));
    }
}
