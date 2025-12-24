/**
 * 行列式的幾何解釋 (Geometric Interpretation)
 *
 * 編譯：javac Geometric.java
 * 執行：java Geometric
 */

public class Geometric {

    static void printSeparator(String title) {
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println(title);
        System.out.println("=".repeat(60));
    }

    static void printVector(String name, double[] v) {
        StringBuilder sb = new StringBuilder(name + " = [");
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

    // 2D 叉積（純量）
    static double cross2D(double[] a, double[] b) {
        return a[0] * b[1] - a[1] * b[0];
    }

    // 3D 叉積
    static double[] cross3D(double[] a, double[] b) {
        return new double[] {
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        };
    }

    // 內積
    static double dot(double[] a, double[] b) {
        double result = 0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    // 2×2 行列式
    static double det2x2(double[][] A) {
        return A[0][0] * A[1][1] - A[0][1] * A[1][0];
    }

    // 平行四邊形面積
    static double parallelogramArea(double[] a, double[] b) {
        return Math.abs(cross2D(a, b));
    }

    // 平行六面體體積
    static double parallelepipedVolume(double[] a, double[] b, double[] c) {
        double[] bxc = cross3D(b, c);
        return Math.abs(dot(a, bxc));
    }

    // 三角形面積
    static double triangleArea(double x1, double y1,
                               double x2, double y2,
                               double x3, double y3) {
        return Math.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;
    }

    public static void main(String[] args) {
        printSeparator("行列式幾何解釋示範 (Java)");

        // ========================================
        // 1. 平行四邊形面積
        // ========================================
        printSeparator("1. 平行四邊形面積");

        double[] a = {3, 0};
        double[] b = {1, 2};

        printVector("a", a);
        printVector("b", b);

        double area = parallelogramArea(a, b);
        double signedArea = cross2D(a, b);

        System.out.println("\n平行四邊形：");
        System.out.printf("  有號面積 = a × b = %.4f%n", signedArea);
        System.out.printf("  面積 = |a × b| = %.4f%n", area);

        // ========================================
        // 2. 定向判斷
        // ========================================
        printSeparator("2. 定向判斷");

        a = new double[]{1, 0};
        b = new double[]{0, 1};
        double signedVal = cross2D(a, b);

        printVector("a", a);
        printVector("b", b);
        System.out.printf("有號面積 = %.4f%n", signedVal);
        System.out.println("定向：" + (signedVal > 0 ? "逆時針（正向）" : "順時針（負向）"));

        System.out.println("\n交換 a, b 順序：");
        signedVal = cross2D(b, a);
        System.out.printf("有號面積 = %.4f%n", signedVal);
        System.out.println("定向：" + (signedVal > 0 ? "逆時針（正向）" : "順時針（負向）"));

        // ========================================
        // 3. 平行六面體體積
        // ========================================
        printSeparator("3. 平行六面體體積");

        double[] v1 = {1, 0, 0};
        double[] v2 = {0, 2, 0};
        double[] v3 = {0, 0, 3};

        printVector("a", v1);
        printVector("b", v2);
        printVector("c", v3);

        double vol = parallelepipedVolume(v1, v2, v3);
        System.out.printf("%n體積 = |a · (b × c)| = %.4f%n", vol);

        // ========================================
        // 4. 三角形面積
        // ========================================
        printSeparator("4. 三角形面積");

        double x1 = 0, y1 = 0;
        double x2 = 4, y2 = 0;
        double x3 = 0, y3 = 3;

        System.out.println("三角形頂點：");
        System.out.printf("  P1 = (%.0f, %.0f)%n", x1, y1);
        System.out.printf("  P2 = (%.0f, %.0f)%n", x2, y2);
        System.out.printf("  P3 = (%.0f, %.0f)%n", x3, y3);

        double triArea = triangleArea(x1, y1, x2, y2, x3, y3);
        System.out.printf("%n面積 = %.4f%n", triArea);

        // ========================================
        // 5. 線性變換的體積縮放
        // ========================================
        printSeparator("5. 線性變換的體積縮放");

        double[][] A = {{2, 0}, {0, 3}};
        printMatrix("縮放矩陣 A", A);
        System.out.printf("det(A) = %.4f%n", det2x2(A));
        System.out.println("\n單位正方形 → 2×3 長方形");
        System.out.printf("面積從 1 變成 %.4f%n", Math.abs(det2x2(A)));

        double theta = Math.PI / 4;
        double[][] R = {
            {Math.cos(theta), -Math.sin(theta)},
            {Math.sin(theta), Math.cos(theta)}
        };
        System.out.printf("%n旋轉矩陣：det(R) = %.4f（面積不變）%n", det2x2(R));

        double[][] H = {{1, 0}, {0, -1}};
        System.out.printf("反射矩陣：det(H) = %.4f（面積不變，定向反轉）%n", det2x2(H));

        double[][] S = {{1, 2}, {0, 1}};
        System.out.printf("剪切矩陣：det(S) = %.4f（面積不變）%n", det2x2(S));

        // 總結
        printSeparator("總結");
        System.out.println("""

行列式的幾何意義：

1. |det| = 體積/面積的縮放因子
2. sign(det) = 定向保持/反轉
3. det = 0 → 降維

特殊矩陣：
   - 旋轉：det = 1
   - 反射：det = -1
   - 剪切：det = 1
            """);

        System.out.println("=".repeat(60));
        System.out.println("示範完成！");
        System.out.println("=".repeat(60));
    }
}
