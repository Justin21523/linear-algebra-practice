/**
 * 克萊姆法則 (Cramer's Rule)
 *
 * 編譯：javac CramersRule.java
 * 執行：java CramersRule
 */

public class CramersRule {  // EN: Execute line: public class CramersRule {.

    static void printSeparator(String title) {  // EN: Execute line: static void printSeparator(String title) {.
        System.out.println();  // EN: Execute a statement: System.out.println();.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
        System.out.println(title);  // EN: Execute a statement: System.out.println(title);.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
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

    static void printVector(String name, double[] v) {  // EN: Execute line: static void printVector(String name, double[] v) {.
        StringBuilder sb = new StringBuilder(name + " = [");  // EN: Execute a statement: StringBuilder sb = new StringBuilder(name + " = [");.
        for (int i = 0; i < v.length; i++) {  // EN: Loop control flow: for (int i = 0; i < v.length; i++) {.
            sb.append(String.format("%.4f", v[i]));  // EN: Execute a statement: sb.append(String.format("%.4f", v[i]));.
            if (i < v.length - 1) sb.append(", ");  // EN: Conditional control flow: if (i < v.length - 1) sb.append(", ");.
        }  // EN: Structure delimiter for a block or scope.
        sb.append("]");  // EN: Execute a statement: sb.append("]");.
        System.out.println(sb.toString());  // EN: Execute a statement: System.out.println(sb.toString());.
    }  // EN: Structure delimiter for a block or scope.

    // 2×2 行列式
    static double det2x2(double[][] A) {  // EN: Execute line: static double det2x2(double[][] A) {.
        return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Return from the current function: return A[0][0] * A[1][1] - A[0][1] * A[1][0];.
    }  // EN: Structure delimiter for a block or scope.

    // 3×3 行列式
    static double det3x3(double[][] A) {  // EN: Execute line: static double det3x3(double[][] A) {.
        return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])  // EN: Return from the current function: return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]).
             - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])  // EN: Execute line: - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]).
             + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);  // EN: Execute a statement: + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);.
    }  // EN: Structure delimiter for a block or scope.

    static double determinant(double[][] A) {  // EN: Execute line: static double determinant(double[][] A) {.
        int n = A.length;  // EN: Execute a statement: int n = A.length;.
        if (n == 2) return det2x2(A);  // EN: Conditional control flow: if (n == 2) return det2x2(A);.
        if (n == 3) return det3x3(A);  // EN: Conditional control flow: if (n == 3) return det3x3(A);.
        throw new RuntimeException("僅支援 2×2 和 3×3 矩陣");  // EN: Execute a statement: throw new RuntimeException("僅支援 2×2 和 3×3 矩陣");.
    }  // EN: Structure delimiter for a block or scope.

    // 替換第 j 行
    static double[][] replaceColumn(double[][] A, double[] b, int j) {  // EN: Execute line: static double[][] replaceColumn(double[][] A, double[] b, int j) {.
        int n = A.length;  // EN: Execute a statement: int n = A.length;.
        double[][] Aj = new double[n][n];  // EN: Execute a statement: double[][] Aj = new double[n][n];.
        for (int i = 0; i < n; i++) {  // EN: Loop control flow: for (int i = 0; i < n; i++) {.
            for (int k = 0; k < n; k++) {  // EN: Loop control flow: for (int k = 0; k < n; k++) {.
                Aj[i][k] = (k == j) ? b[i] : A[i][k];  // EN: Execute a statement: Aj[i][k] = (k == j) ? b[i] : A[i][k];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return Aj;  // EN: Return from the current function: return Aj;.
    }  // EN: Structure delimiter for a block or scope.

    // 克萊姆法則
    static double[] cramersRule(double[][] A, double[] b) {  // EN: Execute line: static double[] cramersRule(double[][] A, double[] b) {.
        int n = A.length;  // EN: Execute a statement: int n = A.length;.
        double detA = determinant(A);  // EN: Execute a statement: double detA = determinant(A);.

        if (Math.abs(detA) < 1e-10) {  // EN: Conditional control flow: if (Math.abs(detA) < 1e-10) {.
            throw new RuntimeException("矩陣奇異");  // EN: Execute a statement: throw new RuntimeException("矩陣奇異");.
        }  // EN: Structure delimiter for a block or scope.

        double[] x = new double[n];  // EN: Execute a statement: double[] x = new double[n];.
        for (int j = 0; j < n; j++) {  // EN: Loop control flow: for (int j = 0; j < n; j++) {.
            double[][] Aj = replaceColumn(A, b, j);  // EN: Execute a statement: double[][] Aj = replaceColumn(A, b, j);.
            x[j] = determinant(Aj) / detA;  // EN: Execute a statement: x[j] = determinant(Aj) / detA;.
        }  // EN: Structure delimiter for a block or scope.
        return x;  // EN: Return from the current function: return x;.
    }  // EN: Structure delimiter for a block or scope.

    public static void main(String[] args) {  // EN: Execute line: public static void main(String[] args) {.
        printSeparator("克萊姆法則示範 (Java)");  // EN: Execute a statement: printSeparator("克萊姆法則示範 (Java)");.

        // ========================================
        // 1. 2×2 系統
        // ========================================
        printSeparator("1. 2×2 系統");  // EN: Execute a statement: printSeparator("1. 2×2 系統");.

        double[][] A2 = {{2, 3}, {4, 5}};  // EN: Execute a statement: double[][] A2 = {{2, 3}, {4, 5}};.
        double[] b2 = {8, 14};  // EN: Execute a statement: double[] b2 = {8, 14};.

        System.out.println("方程組：");  // EN: Execute a statement: System.out.println("方程組：");.
        System.out.println("  2x + 3y = 8");  // EN: Execute a statement: System.out.println(" 2x + 3y = 8");.
        System.out.println("  4x + 5y = 14");  // EN: Execute a statement: System.out.println(" 4x + 5y = 14");.

        printMatrix("\nA", A2);  // EN: Execute a statement: printMatrix("\nA", A2);.
        printVector("b", b2);  // EN: Execute a statement: printVector("b", b2);.

        double detA2 = determinant(A2);  // EN: Execute a statement: double detA2 = determinant(A2);.
        System.out.printf("%ndet(A) = %.4f%n", detA2);  // EN: Execute a statement: System.out.printf("%ndet(A) = %.4f%n", detA2);.

        double[] x2 = cramersRule(A2, b2);  // EN: Execute a statement: double[] x2 = cramersRule(A2, b2);.

        for (int j = 0; j < 2; j++) {  // EN: Loop control flow: for (int j = 0; j < 2; j++) {.
            double[][] Aj = replaceColumn(A2, b2, j);  // EN: Execute a statement: double[][] Aj = replaceColumn(A2, b2, j);.
            double detAj = determinant(Aj);  // EN: Execute a statement: double detAj = determinant(Aj);.
            System.out.printf("%nA%d（第 %d 行換成 b）：%n", j+1, j+1);  // EN: Execute a statement: System.out.printf("%nA%d（第 %d 行換成 b）：%n", j+1, j+1);.
            printMatrix("", Aj);  // EN: Execute a statement: printMatrix("", Aj);.
            System.out.printf("det(A%d) = %.4f%n", j+1, detAj);  // EN: Execute a statement: System.out.printf("det(A%d) = %.4f%n", j+1, detAj);.
            System.out.printf("x%d = %.4f%n", j+1, x2[j]);  // EN: Execute a statement: System.out.printf("x%d = %.4f%n", j+1, x2[j]);.
        }  // EN: Structure delimiter for a block or scope.

        System.out.printf("%n解：x = %.4f, y = %.4f%n", x2[0], x2[1]);  // EN: Execute a statement: System.out.printf("%n解：x = %.4f, y = %.4f%n", x2[0], x2[1]);.

        // ========================================
        // 2. 3×3 系統
        // ========================================
        printSeparator("2. 3×3 系統");  // EN: Execute a statement: printSeparator("2. 3×3 系統");.

        double[][] A3 = {  // EN: Execute line: double[][] A3 = {.
            {2, 1, -1},  // EN: Execute line: {2, 1, -1},.
            {-3, -1, 2},  // EN: Execute line: {-3, -1, 2},.
            {-2, 1, 2}  // EN: Execute line: {-2, 1, 2}.
        };  // EN: Structure delimiter for a block or scope.
        double[] b3 = {8, -11, -3};  // EN: Execute a statement: double[] b3 = {8, -11, -3};.

        System.out.println("方程組：");  // EN: Execute a statement: System.out.println("方程組：");.
        System.out.println("   2x +  y -  z =  8");  // EN: Execute a statement: System.out.println(" 2x + y - z = 8");.
        System.out.println("  -3x -  y + 2z = -11");  // EN: Execute a statement: System.out.println(" -3x - y + 2z = -11");.
        System.out.println("  -2x +  y + 2z = -3");  // EN: Execute a statement: System.out.println(" -2x + y + 2z = -3");.

        printMatrix("\nA", A3);  // EN: Execute a statement: printMatrix("\nA", A3);.
        printVector("b", b3);  // EN: Execute a statement: printVector("b", b3);.

        double[] x3 = cramersRule(A3, b3);  // EN: Execute a statement: double[] x3 = cramersRule(A3, b3);.

        System.out.printf("%n解：x = %.4f, y = %.4f, z = %.4f%n", x3[0], x3[1], x3[2]);  // EN: Execute a statement: System.out.printf("%n解：x = %.4f, y = %.4f, z = %.4f%n", x3[0], x3[1], x….

        // 驗證
        System.out.println("\n驗證：");  // EN: Execute a statement: System.out.println("\n驗證：");.
        System.out.printf("  2(%.0f) + (%.0f) - (%.0f) = %.4f%n",  // EN: Execute line: System.out.printf(" 2(%.0f) + (%.0f) - (%.0f) = %.4f%n",.
            x3[0], x3[1], x3[2], 2*x3[0] + x3[1] - x3[2]);  // EN: Execute a statement: x3[0], x3[1], x3[2], 2*x3[0] + x3[1] - x3[2]);.
        System.out.printf("  -3(%.0f) - (%.0f) + 2(%.0f) = %.4f%n",  // EN: Execute line: System.out.printf(" -3(%.0f) - (%.0f) + 2(%.0f) = %.4f%n",.
            x3[0], x3[1], x3[2], -3*x3[0] - x3[1] + 2*x3[2]);  // EN: Execute a statement: x3[0], x3[1], x3[2], -3*x3[0] - x3[1] + 2*x3[2]);.
        System.out.printf("  -2(%.0f) + (%.0f) + 2(%.0f) = %.4f%n",  // EN: Execute line: System.out.printf(" -2(%.0f) + (%.0f) + 2(%.0f) = %.4f%n",.
            x3[0], x3[1], x3[2], -2*x3[0] + x3[1] + 2*x3[2]);  // EN: Execute a statement: x3[0], x3[1], x3[2], -2*x3[0] + x3[1] + 2*x3[2]);.

        // 總結
        printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
        System.out.println("""  // EN: Execute line: System.out.println(""".

克萊姆法則：  // EN: Execute line: 克萊姆法則：.
  xⱼ = det(Aⱼ) / det(A)  // EN: Execute line: xⱼ = det(Aⱼ) / det(A).
  Aⱼ = A 的第 j 行換成 b  // EN: Execute line: Aⱼ = A 的第 j 行換成 b.

適用條件：  // EN: Execute line: 適用條件：.
  - det(A) ≠ 0  // EN: Execute line: - det(A) ≠ 0.
  - 方陣系統  // EN: Execute line: - 方陣系統.

時間複雜度：O(n! × n)  // EN: Execute line: 時間複雜度：O(n! × n).
            """);  // EN: Execute a statement: """);.

        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
        System.out.println("示範完成！");  // EN: Execute a statement: System.out.println("示範完成！");.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
