/**
 * Gram-Schmidt 正交化 (Gram-Schmidt Process)
 *
 * 編譯：javac GramSchmidt.java
 * 執行：java GramSchmidt
 */

public class GramSchmidt {  // EN: Execute line: public class GramSchmidt {.

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

    static double dotProduct(double[] x, double[] y) {  // EN: Execute line: static double dotProduct(double[] x, double[] y) {.
        double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
        for (int i = 0; i < x.length; i++) result += x[i] * y[i];  // EN: Loop control flow: for (int i = 0; i < x.length; i++) result += x[i] * y[i];.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double vectorNorm(double[] x) {  // EN: Execute line: static double vectorNorm(double[] x) {.
        return Math.sqrt(dotProduct(x, x));  // EN: Return from the current function: return Math.sqrt(dotProduct(x, x));.
    }  // EN: Structure delimiter for a block or scope.

    static double[] scalarMultiply(double c, double[] x) {  // EN: Execute line: static double[] scalarMultiply(double c, double[] x) {.
        double[] result = new double[x.length];  // EN: Execute a statement: double[] result = new double[x.length];.
        for (int i = 0; i < x.length; i++) result[i] = c * x[i];  // EN: Loop control flow: for (int i = 0; i < x.length; i++) result[i] = c * x[i];.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[] vectorSubtract(double[] x, double[] y) {  // EN: Execute line: static double[] vectorSubtract(double[] x, double[] y) {.
        double[] result = new double[x.length];  // EN: Execute a statement: double[] result = new double[x.length];.
        for (int i = 0; i < x.length; i++) result[i] = x[i] - y[i];  // EN: Loop control flow: for (int i = 0; i < x.length; i++) result[i] = x[i] - y[i];.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[] normalize(double[] x) {  // EN: Execute line: static double[] normalize(double[] x) {.
        return scalarMultiply(1.0 / vectorNorm(x), x);  // EN: Return from the current function: return scalarMultiply(1.0 / vectorNorm(x), x);.
    }  // EN: Structure delimiter for a block or scope.

    /**
     * Modified Gram-Schmidt
     */
    static double[][] modifiedGramSchmidt(double[][] A) {  // EN: Execute line: static double[][] modifiedGramSchmidt(double[][] A) {.
        int n = A.length;  // EN: Execute a statement: int n = A.length;.
        double[][] Q = new double[n][];  // EN: Execute a statement: double[][] Q = new double[n][];.

        for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
            Q[i] = A[i].clone();  // EN: Execute a statement: Q[i] = A[i].clone();.
        }  // EN: Structure delimiter for a block or scope.

        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            for (int i = 0; i < j; i++) {  // EN: Loop control flow: for (int i = 0; i < j; i++) {.
                double coeff = dotProduct(Q[i], Q[j]) / dotProduct(Q[i], Q[i]);  // EN: Execute a statement: double coeff = dotProduct(Q[i], Q[j]) / dotProduct(Q[i], Q[i]);.
                Q[j] = vectorSubtract(Q[j], scalarMultiply(coeff, Q[i]));  // EN: Execute a statement: Q[j] = vectorSubtract(Q[j], scalarMultiply(coeff, Q[i]));.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.

        return Q;  // EN: Return from the current function: return Q;.
    }  // EN: Structure delimiter for a block or scope.

    static double[][] gramSchmidtNormalized(double[][] A) {  // EN: Execute line: static double[][] gramSchmidtNormalized(double[][] A) {.
        double[][] Q = modifiedGramSchmidt(A);  // EN: Execute a statement: double[][] Q = modifiedGramSchmidt(A);.
        for (int i = 0; i < Q.length; i++) {  // EN: Loop control flow: for (int i = 0; i < Q.length; i++) {.
            Q[i] = normalize(Q[i]);  // EN: Execute a statement: Q[i] = normalize(Q[i]);.
        }  // EN: Structure delimiter for a block or scope.
        return Q;  // EN: Return from the current function: return Q;.
    }  // EN: Structure delimiter for a block or scope.

    static boolean verifyOrthogonality(double[][] Q) {  // EN: Execute line: static boolean verifyOrthogonality(double[][] Q) {.
        for (int i = 0; i < Q.length; i++) {  // EN: Loop control flow: for (int i = 0; i < Q.length; i++) {.
            for (int j = i + 1; j < Q.length; j++) {  // EN: Loop control flow: for (int j = i + 1; j < Q.length; j++) {.
                if (Math.abs(dotProduct(Q[i], Q[j])) > 1e-10) return false;  // EN: Conditional control flow: if (Math.abs(dotProduct(Q[i], Q[j])) > 1e-10) return false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return true;  // EN: Return from the current function: return true;.
    }  // EN: Structure delimiter for a block or scope.

    public static void main(String[] args) {  // EN: Execute line: public static void main(String[] args) {.
        printSeparator("Gram-Schmidt 正交化示範 (Java)");  // EN: Execute a statement: printSeparator("Gram-Schmidt 正交化示範 (Java)");.

        double[][] A = {  // EN: Execute line: double[][] A = {.
            {1.0, 1.0, 0.0},  // EN: Execute line: {1.0, 1.0, 0.0},.
            {1.0, 0.0, 1.0},  // EN: Execute line: {1.0, 0.0, 1.0},.
            {0.0, 1.0, 1.0}  // EN: Execute line: {0.0, 1.0, 1.0}.
        };  // EN: Structure delimiter for a block or scope.

        System.out.println("輸入向量組：");  // EN: Execute a statement: System.out.println("輸入向量組：");.
        for (int i = 0; i < A.length; i++)  // EN: Loop control flow: for (int i = 0; i < A.length; i++).
            printVector("a" + (i+1), A[i]);  // EN: Execute a statement: printVector("a" + (i+1), A[i]);.

        double[][] Q = modifiedGramSchmidt(A);  // EN: Execute a statement: double[][] Q = modifiedGramSchmidt(A);.

        System.out.println("\n正交化結果（MGS）：");  // EN: Execute a statement: System.out.println("\n正交化結果（MGS）：");.
        for (int i = 0; i < Q.length; i++) {  // EN: Loop control flow: for (int i = 0; i < Q.length; i++) {.
            printVector("q" + (i+1), Q[i]);  // EN: Execute a statement: printVector("q" + (i+1), Q[i]);.
            System.out.printf("    ‖q%d‖ = %.4f%n", i+1, vectorNorm(Q[i]));  // EN: Execute a statement: System.out.printf(" ‖q%d‖ = %.4f%n", i+1, vectorNorm(Q[i]));.
        }  // EN: Structure delimiter for a block or scope.

        System.out.println("\n正交？ " + verifyOrthogonality(Q));  // EN: Execute a statement: System.out.println("\n正交？ " + verifyOrthogonality(Q));.

        System.out.println("\n內積驗證：");  // EN: Execute a statement: System.out.println("\n內積驗證：");.
        System.out.printf("q₁ · q₂ = %.6f%n", dotProduct(Q[0], Q[1]));  // EN: Execute a statement: System.out.printf("q₁ · q₂ = %.6f%n", dotProduct(Q[0], Q[1]));.
        System.out.printf("q₁ · q₃ = %.6f%n", dotProduct(Q[0], Q[2]));  // EN: Execute a statement: System.out.printf("q₁ · q₃ = %.6f%n", dotProduct(Q[0], Q[2]));.
        System.out.printf("q₂ · q₃ = %.6f%n", dotProduct(Q[1], Q[2]));  // EN: Execute a statement: System.out.printf("q₂ · q₃ = %.6f%n", dotProduct(Q[1], Q[2]));.

        printSeparator("標準正交化");  // EN: Execute a statement: printSeparator("標準正交化");.

        double[][] E = gramSchmidtNormalized(A);  // EN: Execute a statement: double[][] E = gramSchmidtNormalized(A);.

        System.out.println("標準正交向量組：");  // EN: Execute a statement: System.out.println("標準正交向量組：");.
        for (int i = 0; i < E.length; i++) {  // EN: Loop control flow: for (int i = 0; i < E.length; i++) {.
            printVector("e" + (i+1), E[i]);  // EN: Execute a statement: printVector("e" + (i+1), E[i]);.
            System.out.printf("    ‖e%d‖ = %.4f%n", i+1, vectorNorm(E[i]));  // EN: Execute a statement: System.out.printf(" ‖e%d‖ = %.4f%n", i+1, vectorNorm(E[i]));.
        }  // EN: Structure delimiter for a block or scope.

        printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
        System.out.println("""  // EN: Execute line: System.out.println(""".

Gram-Schmidt 核心公式：  // EN: Execute line: Gram-Schmidt 核心公式：.

proj_q(a) = (qᵀa / qᵀq) q  // EN: Execute line: proj_q(a) = (qᵀa / qᵀq) q.

q₁ = a₁  // EN: Execute line: q₁ = a₁.
qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ)  // EN: Execute line: qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ).

eᵢ = qᵢ / ‖qᵢ‖  // EN: Execute line: eᵢ = qᵢ / ‖qᵢ‖.
""");  // EN: Execute a statement: """);.

        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
        System.out.println("示範完成！");  // EN: Execute a statement: System.out.println("示範完成！");.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
