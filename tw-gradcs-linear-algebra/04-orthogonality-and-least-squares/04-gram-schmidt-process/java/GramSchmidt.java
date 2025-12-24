/**
 * Gram-Schmidt 正交化 (Gram-Schmidt Process)
 *
 * 編譯：javac GramSchmidt.java
 * 執行：java GramSchmidt
 */

public class GramSchmidt {

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

    static double dotProduct(double[] x, double[] y) {
        double result = 0.0;
        for (int i = 0; i < x.length; i++) result += x[i] * y[i];
        return result;
    }

    static double vectorNorm(double[] x) {
        return Math.sqrt(dotProduct(x, x));
    }

    static double[] scalarMultiply(double c, double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) result[i] = c * x[i];
        return result;
    }

    static double[] vectorSubtract(double[] x, double[] y) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) result[i] = x[i] - y[i];
        return result;
    }

    static double[] normalize(double[] x) {
        return scalarMultiply(1.0 / vectorNorm(x), x);
    }

    /**
     * Modified Gram-Schmidt
     */
    static double[][] modifiedGramSchmidt(double[][] A) {
        int n = A.length;
        double[][] Q = new double[n][];

        for (int i = 0; i < n; i++) {
            Q[i] = A[i].clone();
        }

        for (int j = 0; j < n; j++) {
            for (int i = 0; i < j; i++) {
                double coeff = dotProduct(Q[i], Q[j]) / dotProduct(Q[i], Q[i]);
                Q[j] = vectorSubtract(Q[j], scalarMultiply(coeff, Q[i]));
            }
        }

        return Q;
    }

    static double[][] gramSchmidtNormalized(double[][] A) {
        double[][] Q = modifiedGramSchmidt(A);
        for (int i = 0; i < Q.length; i++) {
            Q[i] = normalize(Q[i]);
        }
        return Q;
    }

    static boolean verifyOrthogonality(double[][] Q) {
        for (int i = 0; i < Q.length; i++) {
            for (int j = i + 1; j < Q.length; j++) {
                if (Math.abs(dotProduct(Q[i], Q[j])) > 1e-10) return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
        printSeparator("Gram-Schmidt 正交化示範 (Java)");

        double[][] A = {
            {1.0, 1.0, 0.0},
            {1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0}
        };

        System.out.println("輸入向量組：");
        for (int i = 0; i < A.length; i++)
            printVector("a" + (i+1), A[i]);

        double[][] Q = modifiedGramSchmidt(A);

        System.out.println("\n正交化結果（MGS）：");
        for (int i = 0; i < Q.length; i++) {
            printVector("q" + (i+1), Q[i]);
            System.out.printf("    ‖q%d‖ = %.4f%n", i+1, vectorNorm(Q[i]));
        }

        System.out.println("\n正交？ " + verifyOrthogonality(Q));

        System.out.println("\n內積驗證：");
        System.out.printf("q₁ · q₂ = %.6f%n", dotProduct(Q[0], Q[1]));
        System.out.printf("q₁ · q₃ = %.6f%n", dotProduct(Q[0], Q[2]));
        System.out.printf("q₂ · q₃ = %.6f%n", dotProduct(Q[1], Q[2]));

        printSeparator("標準正交化");

        double[][] E = gramSchmidtNormalized(A);

        System.out.println("標準正交向量組：");
        for (int i = 0; i < E.length; i++) {
            printVector("e" + (i+1), E[i]);
            System.out.printf("    ‖e%d‖ = %.4f%n", i+1, vectorNorm(E[i]));
        }

        printSeparator("總結");
        System.out.println("""

Gram-Schmidt 核心公式：

proj_q(a) = (qᵀa / qᵀq) q

q₁ = a₁
qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ)

eᵢ = qᵢ / ‖qᵢ‖
""");

        System.out.println("=".repeat(60));
        System.out.println("示範完成！");
        System.out.println("=".repeat(60));
    }
}
