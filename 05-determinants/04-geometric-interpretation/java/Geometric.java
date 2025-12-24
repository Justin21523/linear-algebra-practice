/**
 * 行列式的幾何解釋 (Geometric Interpretation)
 *
 * 編譯：javac Geometric.java
 * 執行：java Geometric
 */

public class Geometric {  // EN: Execute line: public class Geometric {.

    static void printSeparator(String title) {  // EN: Execute line: static void printSeparator(String title) {.
        System.out.println();  // EN: Execute a statement: System.out.println();.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
        System.out.println(title);  // EN: Execute a statement: System.out.println(title);.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
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

    // 2D 叉積（純量）
    static double cross2D(double[] a, double[] b) {  // EN: Execute line: static double cross2D(double[] a, double[] b) {.
        return a[0] * b[1] - a[1] * b[0];  // EN: Return from the current function: return a[0] * b[1] - a[1] * b[0];.
    }  // EN: Structure delimiter for a block or scope.

    // 3D 叉積
    static double[] cross3D(double[] a, double[] b) {  // EN: Execute line: static double[] cross3D(double[] a, double[] b) {.
        return new double[] {  // EN: Return from the current function: return new double[] {.
            a[1] * b[2] - a[2] * b[1],  // EN: Execute line: a[1] * b[2] - a[2] * b[1],.
            a[2] * b[0] - a[0] * b[2],  // EN: Execute line: a[2] * b[0] - a[0] * b[2],.
            a[0] * b[1] - a[1] * b[0]  // EN: Execute line: a[0] * b[1] - a[1] * b[0].
        };  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    // 內積
    static double dot(double[] a, double[] b) {  // EN: Execute line: static double dot(double[] a, double[] b) {.
        double result = 0;  // EN: Execute a statement: double result = 0;.
        for (int i = 0; i < a.length; i++) {  // EN: Loop control flow: for (int i = 0; i < a.length; i++) {.
            result += a[i] * b[i];  // EN: Execute a statement: result += a[i] * b[i];.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    // 2×2 行列式
    static double det2x2(double[][] A) {  // EN: Execute line: static double det2x2(double[][] A) {.
        return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Return from the current function: return A[0][0] * A[1][1] - A[0][1] * A[1][0];.
    }  // EN: Structure delimiter for a block or scope.

    // 平行四邊形面積
    static double parallelogramArea(double[] a, double[] b) {  // EN: Execute line: static double parallelogramArea(double[] a, double[] b) {.
        return Math.abs(cross2D(a, b));  // EN: Return from the current function: return Math.abs(cross2D(a, b));.
    }  // EN: Structure delimiter for a block or scope.

    // 平行六面體體積
    static double parallelepipedVolume(double[] a, double[] b, double[] c) {  // EN: Execute line: static double parallelepipedVolume(double[] a, double[] b, double[] c) {.
        double[] bxc = cross3D(b, c);  // EN: Execute a statement: double[] bxc = cross3D(b, c);.
        return Math.abs(dot(a, bxc));  // EN: Return from the current function: return Math.abs(dot(a, bxc));.
    }  // EN: Structure delimiter for a block or scope.

    // 三角形面積
    static double triangleArea(double x1, double y1,  // EN: Execute line: static double triangleArea(double x1, double y1,.
                               double x2, double y2,  // EN: Execute line: double x2, double y2,.
                               double x3, double y3) {  // EN: Execute line: double x3, double y3) {.
        return Math.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;  // EN: Return from the current function: return Math.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;.
    }  // EN: Structure delimiter for a block or scope.

    public static void main(String[] args) {  // EN: Execute line: public static void main(String[] args) {.
        printSeparator("行列式幾何解釋示範 (Java)");  // EN: Execute a statement: printSeparator("行列式幾何解釋示範 (Java)");.

        // ========================================
        // 1. 平行四邊形面積
        // ========================================
        printSeparator("1. 平行四邊形面積");  // EN: Execute a statement: printSeparator("1. 平行四邊形面積");.

        double[] a = {3, 0};  // EN: Execute a statement: double[] a = {3, 0};.
        double[] b = {1, 2};  // EN: Execute a statement: double[] b = {1, 2};.

        printVector("a", a);  // EN: Execute a statement: printVector("a", a);.
        printVector("b", b);  // EN: Execute a statement: printVector("b", b);.

        double area = parallelogramArea(a, b);  // EN: Execute a statement: double area = parallelogramArea(a, b);.
        double signedArea = cross2D(a, b);  // EN: Execute a statement: double signedArea = cross2D(a, b);.

        System.out.println("\n平行四邊形：");  // EN: Execute a statement: System.out.println("\n平行四邊形：");.
        System.out.printf("  有號面積 = a × b = %.4f%n", signedArea);  // EN: Execute a statement: System.out.printf(" 有號面積 = a × b = %.4f%n", signedArea);.
        System.out.printf("  面積 = |a × b| = %.4f%n", area);  // EN: Execute a statement: System.out.printf(" 面積 = |a × b| = %.4f%n", area);.

        // ========================================
        // 2. 定向判斷
        // ========================================
        printSeparator("2. 定向判斷");  // EN: Execute a statement: printSeparator("2. 定向判斷");.

        a = new double[]{1, 0};  // EN: Execute a statement: a = new double[]{1, 0};.
        b = new double[]{0, 1};  // EN: Execute a statement: b = new double[]{0, 1};.
        double signedVal = cross2D(a, b);  // EN: Execute a statement: double signedVal = cross2D(a, b);.

        printVector("a", a);  // EN: Execute a statement: printVector("a", a);.
        printVector("b", b);  // EN: Execute a statement: printVector("b", b);.
        System.out.printf("有號面積 = %.4f%n", signedVal);  // EN: Execute a statement: System.out.printf("有號面積 = %.4f%n", signedVal);.
        System.out.println("定向：" + (signedVal > 0 ? "逆時針（正向）" : "順時針（負向）"));  // EN: Execute a statement: System.out.println("定向：" + (signedVal > 0 ? "逆時針（正向）" : "順時針（負向）"));.

        System.out.println("\n交換 a, b 順序：");  // EN: Execute a statement: System.out.println("\n交換 a, b 順序：");.
        signedVal = cross2D(b, a);  // EN: Execute a statement: signedVal = cross2D(b, a);.
        System.out.printf("有號面積 = %.4f%n", signedVal);  // EN: Execute a statement: System.out.printf("有號面積 = %.4f%n", signedVal);.
        System.out.println("定向：" + (signedVal > 0 ? "逆時針（正向）" : "順時針（負向）"));  // EN: Execute a statement: System.out.println("定向：" + (signedVal > 0 ? "逆時針（正向）" : "順時針（負向）"));.

        // ========================================
        // 3. 平行六面體體積
        // ========================================
        printSeparator("3. 平行六面體體積");  // EN: Execute a statement: printSeparator("3. 平行六面體體積");.

        double[] v1 = {1, 0, 0};  // EN: Execute a statement: double[] v1 = {1, 0, 0};.
        double[] v2 = {0, 2, 0};  // EN: Execute a statement: double[] v2 = {0, 2, 0};.
        double[] v3 = {0, 0, 3};  // EN: Execute a statement: double[] v3 = {0, 0, 3};.

        printVector("a", v1);  // EN: Execute a statement: printVector("a", v1);.
        printVector("b", v2);  // EN: Execute a statement: printVector("b", v2);.
        printVector("c", v3);  // EN: Execute a statement: printVector("c", v3);.

        double vol = parallelepipedVolume(v1, v2, v3);  // EN: Execute a statement: double vol = parallelepipedVolume(v1, v2, v3);.
        System.out.printf("%n體積 = |a · (b × c)| = %.4f%n", vol);  // EN: Execute a statement: System.out.printf("%n體積 = |a · (b × c)| = %.4f%n", vol);.

        // ========================================
        // 4. 三角形面積
        // ========================================
        printSeparator("4. 三角形面積");  // EN: Execute a statement: printSeparator("4. 三角形面積");.

        double x1 = 0, y1 = 0;  // EN: Execute a statement: double x1 = 0, y1 = 0;.
        double x2 = 4, y2 = 0;  // EN: Execute a statement: double x2 = 4, y2 = 0;.
        double x3 = 0, y3 = 3;  // EN: Execute a statement: double x3 = 0, y3 = 3;.

        System.out.println("三角形頂點：");  // EN: Execute a statement: System.out.println("三角形頂點：");.
        System.out.printf("  P1 = (%.0f, %.0f)%n", x1, y1);  // EN: Execute a statement: System.out.printf(" P1 = (%.0f, %.0f)%n", x1, y1);.
        System.out.printf("  P2 = (%.0f, %.0f)%n", x2, y2);  // EN: Execute a statement: System.out.printf(" P2 = (%.0f, %.0f)%n", x2, y2);.
        System.out.printf("  P3 = (%.0f, %.0f)%n", x3, y3);  // EN: Execute a statement: System.out.printf(" P3 = (%.0f, %.0f)%n", x3, y3);.

        double triArea = triangleArea(x1, y1, x2, y2, x3, y3);  // EN: Execute a statement: double triArea = triangleArea(x1, y1, x2, y2, x3, y3);.
        System.out.printf("%n面積 = %.4f%n", triArea);  // EN: Execute a statement: System.out.printf("%n面積 = %.4f%n", triArea);.

        // ========================================
        // 5. 線性變換的體積縮放
        // ========================================
        printSeparator("5. 線性變換的體積縮放");  // EN: Execute a statement: printSeparator("5. 線性變換的體積縮放");.

        double[][] A = {{2, 0}, {0, 3}};  // EN: Execute a statement: double[][] A = {{2, 0}, {0, 3}};.
        printMatrix("縮放矩陣 A", A);  // EN: Execute a statement: printMatrix("縮放矩陣 A", A);.
        System.out.printf("det(A) = %.4f%n", det2x2(A));  // EN: Execute a statement: System.out.printf("det(A) = %.4f%n", det2x2(A));.
        System.out.println("\n單位正方形 → 2×3 長方形");  // EN: Execute a statement: System.out.println("\n單位正方形 → 2×3 長方形");.
        System.out.printf("面積從 1 變成 %.4f%n", Math.abs(det2x2(A)));  // EN: Execute a statement: System.out.printf("面積從 1 變成 %.4f%n", Math.abs(det2x2(A)));.

        double theta = Math.PI / 4;  // EN: Execute a statement: double theta = Math.PI / 4;.
        double[][] R = {  // EN: Execute line: double[][] R = {.
            {Math.cos(theta), -Math.sin(theta)},  // EN: Execute line: {Math.cos(theta), -Math.sin(theta)},.
            {Math.sin(theta), Math.cos(theta)}  // EN: Execute line: {Math.sin(theta), Math.cos(theta)}.
        };  // EN: Structure delimiter for a block or scope.
        System.out.printf("%n旋轉矩陣：det(R) = %.4f（面積不變）%n", det2x2(R));  // EN: Execute a statement: System.out.printf("%n旋轉矩陣：det(R) = %.4f（面積不變）%n", det2x2(R));.

        double[][] H = {{1, 0}, {0, -1}};  // EN: Execute a statement: double[][] H = {{1, 0}, {0, -1}};.
        System.out.printf("反射矩陣：det(H) = %.4f（面積不變，定向反轉）%n", det2x2(H));  // EN: Execute a statement: System.out.printf("反射矩陣：det(H) = %.4f（面積不變，定向反轉）%n", det2x2(H));.

        double[][] S = {{1, 2}, {0, 1}};  // EN: Execute a statement: double[][] S = {{1, 2}, {0, 1}};.
        System.out.printf("剪切矩陣：det(S) = %.4f（面積不變）%n", det2x2(S));  // EN: Execute a statement: System.out.printf("剪切矩陣：det(S) = %.4f（面積不變）%n", det2x2(S));.

        // 總結
        printSeparator("總結");  // EN: Execute a statement: printSeparator("總結");.
        System.out.println("""  // EN: Execute line: System.out.println(""".

行列式的幾何意義：  // EN: Execute line: 行列式的幾何意義：.

1. |det| = 體積/面積的縮放因子  // EN: Execute line: 1. |det| = 體積/面積的縮放因子.
2. sign(det) = 定向保持/反轉  // EN: Execute line: 2. sign(det) = 定向保持/反轉.
3. det = 0 → 降維  // EN: Execute line: 3. det = 0 → 降維.

特殊矩陣：  // EN: Execute line: 特殊矩陣：.
   - 旋轉：det = 1  // EN: Execute line: - 旋轉：det = 1.
   - 反射：det = -1  // EN: Execute line: - 反射：det = -1.
   - 剪切：det = 1  // EN: Execute line: - 剪切：det = 1.
            """);  // EN: Execute a statement: """);.

        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
        System.out.println("示範完成！");  // EN: Execute a statement: System.out.println("示範完成！");.
        System.out.println("=".repeat(60));  // EN: Execute a statement: System.out.println("=".repeat(60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
