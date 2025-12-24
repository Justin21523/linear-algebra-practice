/**
 * 內積與正交性 (Inner Product and Orthogonality)
 *
 * 本程式示範：
 * 1. 向量內積計算
 * 2. 向量長度（範數）
 * 3. 向量夾角
 * 4. 正交性判斷
 * 5. 正交矩陣驗證
 *
 * 編譯：javac InnerProduct.java
 * 執行：java InnerProduct
 */

import java.util.Arrays;  // EN: Execute a statement: import java.util.Arrays;.

public class InnerProduct {  // EN: Execute line: public class InnerProduct {.

    private static final double EPSILON = 1e-10;  // EN: Execute a statement: private static final double EPSILON = 1e-10;.

    // ========================================
    // 輔助方法
    // ========================================

    public static void printSeparator(String title) {  // EN: Execute line: public static void printSeparator(String title) {.
        System.out.println();  // EN: Execute a statement: System.out.println();.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
        System.out.println(title);  // EN: Execute a statement: System.out.println(title);.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
    }  // EN: Structure delimiter for a block or scope.

    public static void printVector(String name, double[] v) {  // EN: Execute line: public static void printVector(String name, double[] v) {.
        StringBuilder sb = new StringBuilder();  // EN: Execute a statement: StringBuilder sb = new StringBuilder();.
        sb.append(name).append(" = [");  // EN: Execute a statement: sb.append(name).append(" = [");.
        for (int i = 0; i < v.length; i++) {  // EN: Loop control flow: for (int i = 0; i < v.length; i++) {.
            sb.append(String.format("%.4f", v[i]));  // EN: Execute a statement: sb.append(String.format("%.4f", v[i]));.
            if (i < v.length - 1) sb.append(", ");  // EN: Conditional control flow: if (i < v.length - 1) sb.append(", ");.
        }  // EN: Structure delimiter for a block or scope.
        sb.append("]");  // EN: Execute a statement: sb.append("]");.
        System.out.println(sb.toString());  // EN: Execute a statement: System.out.println(sb.toString());.
    }  // EN: Structure delimiter for a block or scope.

    public static void printMatrix(String name, double[][] M) {  // EN: Execute line: public static void printMatrix(String name, double[][] M) {.
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
    // 向量運算
    // ========================================

    /**
     * 計算兩向量的內積 (Dot Product)
     * x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ
     */
    public static double dotProduct(double[] x, double[] y) {  // EN: Execute line: public static double dotProduct(double[] x, double[] y) {.
        if (x.length != y.length) {  // EN: Conditional control flow: if (x.length != y.length) {.
            throw new IllegalArgumentException("向量維度必須相同");  // EN: Execute a statement: throw new IllegalArgumentException("向量維度必須相同");.
        }  // EN: Structure delimiter for a block or scope.

        double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
        for (int i = 0; i < x.length; i++) {  // EN: Loop control flow: for (int i = 0; i < x.length; i++) {.
            result += x[i] * y[i];  // EN: Execute a statement: result += x[i] * y[i];.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    /**
     * 計算向量的長度（L2 範數）
     * ‖x‖ = √(x · x)
     */
    public static double vectorNorm(double[] x) {  // EN: Execute line: public static double vectorNorm(double[] x) {.
        return Math.sqrt(dotProduct(x, x));  // EN: Return from the current function: return Math.sqrt(dotProduct(x, x));.
    }  // EN: Structure delimiter for a block or scope.

    /**
     * 正規化向量為單位向量
     * û = x / ‖x‖
     */
    public static double[] normalize(double[] x) {  // EN: Execute line: public static double[] normalize(double[] x) {.
        double norm = vectorNorm(x);  // EN: Execute a statement: double norm = vectorNorm(x);.
        if (norm < EPSILON) {  // EN: Conditional control flow: if (norm < EPSILON) {.
            throw new IllegalArgumentException("零向量無法正規化");  // EN: Execute a statement: throw new IllegalArgumentException("零向量無法正規化");.
        }  // EN: Structure delimiter for a block or scope.

        double[] result = new double[x.length];  // EN: Execute a statement: double[] result = new double[x.length];.
        for (int i = 0; i < x.length; i++) {  // EN: Loop control flow: for (int i = 0; i < x.length; i++) {.
            result[i] = x[i] / norm;  // EN: Execute a statement: result[i] = x[i] / norm;.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    /**
     * 計算兩向量的夾角（弧度）
     * cos θ = (x · y) / (‖x‖ ‖y‖)
     */
    public static double vectorAngle(double[] x, double[] y) {  // EN: Execute line: public static double vectorAngle(double[] x, double[] y) {.
        double dot = dotProduct(x, y);  // EN: Execute a statement: double dot = dotProduct(x, y);.
        double normX = vectorNorm(x);  // EN: Execute a statement: double normX = vectorNorm(x);.
        double normY = vectorNorm(y);  // EN: Execute a statement: double normY = vectorNorm(y);.

        if (normX < EPSILON || normY < EPSILON) {  // EN: Conditional control flow: if (normX < EPSILON || normY < EPSILON) {.
            throw new IllegalArgumentException("零向量沒有定義夾角");  // EN: Execute a statement: throw new IllegalArgumentException("零向量沒有定義夾角");.
        }  // EN: Structure delimiter for a block or scope.

        double cosTheta = dot / (normX * normY);  // EN: Execute a statement: double cosTheta = dot / (normX * normY);.
        // 處理浮點數誤差
        cosTheta = Math.max(-1.0, Math.min(1.0, cosTheta));  // EN: Execute a statement: cosTheta = Math.max(-1.0, Math.min(1.0, cosTheta));.
        return Math.acos(cosTheta);  // EN: Return from the current function: return Math.acos(cosTheta);.
    }  // EN: Structure delimiter for a block or scope.

    /**
     * 判斷兩向量是否正交
     * x ⊥ y ⟺ x · y = 0
     */
    public static boolean isOrthogonal(double[] x, double[] y) {  // EN: Execute line: public static boolean isOrthogonal(double[] x, double[] y) {.
        return Math.abs(dotProduct(x, y)) < EPSILON;  // EN: Return from the current function: return Math.abs(dotProduct(x, y)) < EPSILON;.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 矩陣運算
    // ========================================

    /**
     * 矩陣轉置
     */
    public static double[][] transpose(double[][] A) {  // EN: Execute line: public static double[][] transpose(double[][] A) {.
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

    /**
     * 矩陣乘法
     */
    public static double[][] matrixMultiply(double[][] A, double[][] B) {  // EN: Execute line: public static double[][] matrixMultiply(double[][] A, double[][] B) {.
        int m = A.length;  // EN: Execute a statement: int m = A.length;.
        int n = B[0].length;  // EN: Execute a statement: int n = B[0].length;.
        int k = B.length;  // EN: Execute a statement: int k = B.length;.

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

    /**
     * 矩陣乘向量
     */
    public static double[] matrixVectorMultiply(double[][] A, double[] x) {  // EN: Execute line: public static double[] matrixVectorMultiply(double[][] A, double[] x) {.
        int m = A.length;  // EN: Execute a statement: int m = A.length;.
        int n = A[0].length;  // EN: Execute a statement: int n = A[0].length;.

        double[] result = new double[m];  // EN: Execute a statement: double[] result = new double[m];.
        for (int i = 0; i < m; i++) {  // EN: Loop control flow: for (int i = 0; i < m; i++) {.
            for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
                result[i] += A[i][j] * x[j];  // EN: Execute a statement: result[i] += A[i][j] * x[j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    /**
     * 判斷是否為單位矩陣
     */
    public static boolean isIdentity(double[][] A) {  // EN: Execute line: public static boolean isIdentity(double[][] A) {.
        int n = A.length;  // EN: Execute a statement: int n = A.length;.
        if (A[0].length != n) return false;  // EN: Conditional control flow: if (A[0].length != n) return false;.

        for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
            for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
                double expected = (i == j) ? 1.0 : 0.0;  // EN: Execute a statement: double expected = (i == j) ? 1.0 : 0.0;.
                if (Math.abs(A[i][j] - expected) > EPSILON) {  // EN: Conditional control flow: if (Math.abs(A[i][j] - expected) > EPSILON) {.
                    return false;  // EN: Return from the current function: return false;.
                }  // EN: Structure delimiter for a block or scope.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return true;  // EN: Return from the current function: return true;.
    }  // EN: Structure delimiter for a block or scope.

    /**
     * 判斷矩陣是否為正交矩陣
     * QᵀQ = I
     */
    public static boolean isOrthogonalMatrix(double[][] Q) {  // EN: Execute line: public static boolean isOrthogonalMatrix(double[][] Q) {.
        double[][] QT = transpose(Q);  // EN: Execute a statement: double[][] QT = transpose(Q);.
        double[][] product = matrixMultiply(QT, Q);  // EN: Execute a statement: double[][] product = matrixMultiply(QT, Q);.
        return isIdentity(product);  // EN: Return from the current function: return isIdentity(product);.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 主程式
    // ========================================

    public static void main(String[] args) {  // EN: Execute line: public static void main(String[] args) {.
        printSeparator("內積與正交性示範 (Java)\nInner Product & Orthogonality Demo");  // EN: Execute a statement: printSeparator("內積與正交性示範 (Java)\nInner Product & Orthogonality Demo");.

        // 1. 內積計算
        printSeparator("1. 內積計算 (Dot Product)");  // EN: Execute a statement: printSeparator("1. 內積計算 (Dot Product)");.

        double[] x = {1.0, 2.0, 3.0};  // EN: Execute a statement: double[] x = {1.0, 2.0, 3.0};.
        double[] y = {4.0, 5.0, 6.0};  // EN: Execute a statement: double[] y = {4.0, 5.0, 6.0};.

        printVector("x", x);  // EN: Execute a statement: printVector("x", x);.
        printVector("y", y);  // EN: Execute a statement: printVector("y", y);.
        System.out.println("\nx · y = " + dotProduct(x, y));  // EN: Execute a statement: System.out.println("\nx · y = " + dotProduct(x, y));.
        System.out.println("計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32");  // EN: Execute a statement: System.out.println("計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32");.

        // 2. 向量長度
        printSeparator("2. 向量長度 (Vector Norm)");  // EN: Execute a statement: printSeparator("2. 向量長度 (Vector Norm)");.

        double[] v = {3.0, 4.0};  // EN: Execute a statement: double[] v = {3.0, 4.0};.
        printVector("v", v);  // EN: Execute a statement: printVector("v", v);.
        System.out.println("‖v‖ = " + vectorNorm(v));  // EN: Execute a statement: System.out.println("‖v‖ = " + vectorNorm(v));.
        System.out.println("計算：√(3² + 4²) = √25 = 5");  // EN: Execute a statement: System.out.println("計算：√(3² + 4²) = √25 = 5");.

        // 正規化
        double[] vNormalized = normalize(v);  // EN: Execute a statement: double[] vNormalized = normalize(v);.
        System.out.println("\n單位向量：");  // EN: Execute a statement: System.out.println("\n單位向量：");.
        printVector("v̂ = v/‖v‖", vNormalized);  // EN: Execute a statement: printVector("v̂ = v/‖v‖", vNormalized);.
        System.out.println("‖v̂‖ = " + vectorNorm(vNormalized));  // EN: Execute a statement: System.out.println("‖v̂‖ = " + vectorNorm(vNormalized));.

        // 3. 向量夾角
        printSeparator("3. 向量夾角 (Vector Angle)");  // EN: Execute a statement: printSeparator("3. 向量夾角 (Vector Angle)");.

        double[] a = {1.0, 0.0};  // EN: Execute a statement: double[] a = {1.0, 0.0};.
        double[] b = {1.0, 1.0};  // EN: Execute a statement: double[] b = {1.0, 1.0};.

        printVector("a", a);  // EN: Execute a statement: printVector("a", a);.
        printVector("b", b);  // EN: Execute a statement: printVector("b", b);.

        double theta = vectorAngle(a, b);  // EN: Execute a statement: double theta = vectorAngle(a, b);.
        System.out.printf("\n夾角 θ = %.4f rad = %.2f°%n", theta, Math.toDegrees(theta));  // EN: Execute a statement: System.out.printf("\n夾角 θ = %.4f rad = %.2f°%n", theta, Math.toDegrees(….
        System.out.printf("cos θ = %.4f%n", Math.cos(theta));  // EN: Execute a statement: System.out.printf("cos θ = %.4f%n", Math.cos(theta));.
        System.out.println("預期：cos 45° = 1/√2 ≈ 0.7071");  // EN: Execute a statement: System.out.println("預期：cos 45° = 1/√2 ≈ 0.7071");.

        // 4. 正交性判斷
        printSeparator("4. 正交性判斷 (Orthogonality Check)");  // EN: Execute a statement: printSeparator("4. 正交性判斷 (Orthogonality Check)");.

        double[] u1 = {1.0, 2.0};  // EN: Execute a statement: double[] u1 = {1.0, 2.0};.
        double[] u2 = {-2.0, 1.0};  // EN: Execute a statement: double[] u2 = {-2.0, 1.0};.

        printVector("u₁", u1);  // EN: Execute a statement: printVector("u₁", u1);.
        printVector("u₂", u2);  // EN: Execute a statement: printVector("u₂", u2);.
        System.out.println("\nu₁ · u₂ = " + dotProduct(u1, u2));  // EN: Execute a statement: System.out.println("\nu₁ · u₂ = " + dotProduct(u1, u2));.
        System.out.println("u₁ ⊥ u₂？ " + isOrthogonal(u1, u2));  // EN: Execute a statement: System.out.println("u₁ ⊥ u₂？ " + isOrthogonal(u1, u2));.

        // 非正交
        double[] w1 = {1.0, 1.0};  // EN: Execute a statement: double[] w1 = {1.0, 1.0};.
        double[] w2 = {1.0, 2.0};  // EN: Execute a statement: double[] w2 = {1.0, 2.0};.

        System.out.println("\n另一組：");  // EN: Execute a statement: System.out.println("\n另一組：");.
        printVector("w₁", w1);  // EN: Execute a statement: printVector("w₁", w1);.
        printVector("w₂", w2);  // EN: Execute a statement: printVector("w₂", w2);.
        System.out.println("w₁ · w₂ = " + dotProduct(w1, w2));  // EN: Execute a statement: System.out.println("w₁ · w₂ = " + dotProduct(w1, w2));.
        System.out.println("w₁ ⊥ w₂？ " + isOrthogonal(w1, w2));  // EN: Execute a statement: System.out.println("w₁ ⊥ w₂？ " + isOrthogonal(w1, w2));.

        // 5. 正交矩陣
        printSeparator("5. 正交矩陣 (Orthogonal Matrix)");  // EN: Execute a statement: printSeparator("5. 正交矩陣 (Orthogonal Matrix)");.

        double angle = Math.PI / 4;  // EN: Execute a statement: double angle = Math.PI / 4;.
        double[][] Q = {  // EN: Execute line: double[][] Q = {.
            {Math.cos(angle), -Math.sin(angle)},  // EN: Execute line: {Math.cos(angle), -Math.sin(angle)},.
            {Math.sin(angle), Math.cos(angle)}  // EN: Execute line: {Math.sin(angle), Math.cos(angle)}.
        };  // EN: Structure delimiter for a block or scope.

        System.out.println("旋轉矩陣（θ = 45°）：");  // EN: Execute a statement: System.out.println("旋轉矩陣（θ = 45°）：");.
        printMatrix("Q", Q);  // EN: Execute a statement: printMatrix("Q", Q);.

        double[][] QT = transpose(Q);  // EN: Execute a statement: double[][] QT = transpose(Q);.
        printMatrix("\nQᵀ", QT);  // EN: Execute a statement: printMatrix("\nQᵀ", QT);.

        double[][] QTQ = matrixMultiply(QT, Q);  // EN: Execute a statement: double[][] QTQ = matrixMultiply(QT, Q);.
        printMatrix("\nQᵀQ", QTQ);  // EN: Execute a statement: printMatrix("\nQᵀQ", QTQ);.

        System.out.println("\nQ 是正交矩陣？ " + isOrthogonalMatrix(Q));  // EN: Execute a statement: System.out.println("\nQ 是正交矩陣？ " + isOrthogonalMatrix(Q));.

        // 驗證保長度
        double[] xVec = {3.0, 4.0};  // EN: Execute a statement: double[] xVec = {3.0, 4.0};.
        double[] Qx = matrixVectorMultiply(Q, xVec);  // EN: Execute a statement: double[] Qx = matrixVectorMultiply(Q, xVec);.

        System.out.println("\n保長度驗證：");  // EN: Execute a statement: System.out.println("\n保長度驗證：");.
        printVector("x", xVec);  // EN: Execute a statement: printVector("x", xVec);.
        printVector("Qx", Qx);  // EN: Execute a statement: printVector("Qx", Qx);.
        System.out.printf("‖x‖ = %.4f%n", vectorNorm(xVec));  // EN: Execute a statement: System.out.printf("‖x‖ = %.4f%n", vectorNorm(xVec));.
        System.out.printf("‖Qx‖ = %.4f%n", vectorNorm(Qx));  // EN: Execute a statement: System.out.printf("‖Qx‖ = %.4f%n", vectorNorm(Qx));.

        // 6. Cauchy-Schwarz 不等式
        printSeparator("6. Cauchy-Schwarz 不等式");  // EN: Execute a statement: printSeparator("6. Cauchy-Schwarz 不等式");.

        double[] csX = {1.0, 2.0, 3.0};  // EN: Execute a statement: double[] csX = {1.0, 2.0, 3.0};.
        double[] csY = {4.0, 5.0, 6.0};  // EN: Execute a statement: double[] csY = {4.0, 5.0, 6.0};.

        printVector("x", csX);  // EN: Execute a statement: printVector("x", csX);.
        printVector("y", csY);  // EN: Execute a statement: printVector("y", csY);.

        double leftSide = Math.abs(dotProduct(csX, csY));  // EN: Execute a statement: double leftSide = Math.abs(dotProduct(csX, csY));.
        double rightSide = vectorNorm(csX) * vectorNorm(csY);  // EN: Execute a statement: double rightSide = vectorNorm(csX) * vectorNorm(csY);.

        System.out.printf("\n|x · y| = %.4f%n", leftSide);  // EN: Execute a statement: System.out.printf("\n|x · y| = %.4f%n", leftSide);.
        System.out.printf("‖x‖ ‖y‖ = %.4f%n", rightSide);  // EN: Execute a statement: System.out.printf("‖x‖ ‖y‖ = %.4f%n", rightSide);.
        System.out.println("|x · y| ≤ ‖x‖ ‖y‖？ " + (leftSide <= rightSide + EPSILON));  // EN: Execute a statement: System.out.println("|x · y| ≤ ‖x‖ ‖y‖？ " + (leftSide <= rightSide + EPS….

        // 總結
        printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
        System.out.println("""  // EN: Execute line: System.out.println(""".

內積與正交性的核心公式：  // EN: Execute line: 內積與正交性的核心公式：.

1. 內積：x · y = Σ xᵢyᵢ  // EN: Execute line: 1. 內積：x · y = Σ xᵢyᵢ.

2. 長度：‖x‖ = √(x · x)  // EN: Execute line: 2. 長度：‖x‖ = √(x · x).

3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖)  // EN: Execute line: 3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖).

4. 正交：x ⊥ y ⟺ x · y = 0  // EN: Execute line: 4. 正交：x ⊥ y ⟺ x · y = 0.

5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ  // EN: Execute line: 5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ.
            """);  // EN: Execute a statement: """);.

        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
        System.out.println("示範完成！");  // EN: Execute a statement: System.out.println("示範完成！");.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
