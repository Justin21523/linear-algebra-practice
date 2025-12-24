/**
 * 行列式的幾何解釋 (Geometric Interpretation)
 *
 * 編譯：dotnet build 或 csc Geometric.cs
 * 執行：dotnet run 或 ./Geometric.exe
 */

using System;  // EN: Execute a statement: using System;.

class Geometric  // EN: Execute line: class Geometric.
{  // EN: Structure delimiter for a block or scope.
    static void PrintSeparator(string title)  // EN: Execute line: static void PrintSeparator(string title).
    {  // EN: Structure delimiter for a block or scope.
        Console.WriteLine();  // EN: Execute a statement: Console.WriteLine();.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine(title);  // EN: Execute a statement: Console.WriteLine(title);.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
    }  // EN: Structure delimiter for a block or scope.

    static void PrintVector(string name, double[] v)  // EN: Execute line: static void PrintVector(string name, double[] v).
    {  // EN: Structure delimiter for a block or scope.
        Console.WriteLine($"{name} = [{string.Join(", ", Array.ConvertAll(v, x => x.ToString("F4")))}]");  // EN: Execute a statement: Console.WriteLine($"{name} = [{string.Join(", ", Array.ConvertAll(v, x ….
    }  // EN: Structure delimiter for a block or scope.

    static void PrintMatrix(string name, double[,] M)  // EN: Execute line: static void PrintMatrix(string name, double[,] M).
    {  // EN: Structure delimiter for a block or scope.
        Console.WriteLine($"{name} =");  // EN: Execute a statement: Console.WriteLine($"{name} =");.
        for (int i = 0; i < M.GetLength(0); i++)  // EN: Loop control flow: for (int i = 0; i < M.GetLength(0); i++).
        {  // EN: Structure delimiter for a block or scope.
            Console.Write("  [");  // EN: Execute a statement: Console.Write(" [");.
            for (int j = 0; j < M.GetLength(1); j++)  // EN: Loop control flow: for (int j = 0; j < M.GetLength(1); j++).
            {  // EN: Structure delimiter for a block or scope.
                Console.Write(M[i, j].ToString("F4").PadLeft(8));  // EN: Execute a statement: Console.Write(M[i, j].ToString("F4").PadLeft(8));.
                if (j < M.GetLength(1) - 1) Console.Write(", ");  // EN: Conditional control flow: if (j < M.GetLength(1) - 1) Console.Write(", ");.
            }  // EN: Structure delimiter for a block or scope.
            Console.WriteLine("]");  // EN: Execute a statement: Console.WriteLine("]");.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    // 2D 叉積（純量）
    static double Cross2D(double[] a, double[] b)  // EN: Execute line: static double Cross2D(double[] a, double[] b).
    {  // EN: Structure delimiter for a block or scope.
        return a[0] * b[1] - a[1] * b[0];  // EN: Return from the current function: return a[0] * b[1] - a[1] * b[0];.
    }  // EN: Structure delimiter for a block or scope.

    // 3D 叉積
    static double[] Cross3D(double[] a, double[] b)  // EN: Execute line: static double[] Cross3D(double[] a, double[] b).
    {  // EN: Structure delimiter for a block or scope.
        return new double[] {  // EN: Return from the current function: return new double[] {.
            a[1] * b[2] - a[2] * b[1],  // EN: Execute line: a[1] * b[2] - a[2] * b[1],.
            a[2] * b[0] - a[0] * b[2],  // EN: Execute line: a[2] * b[0] - a[0] * b[2],.
            a[0] * b[1] - a[1] * b[0]  // EN: Execute line: a[0] * b[1] - a[1] * b[0].
        };  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    // 3D 內積
    static double Dot3D(double[] a, double[] b)  // EN: Execute line: static double Dot3D(double[] a, double[] b).
    {  // EN: Structure delimiter for a block or scope.
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];  // EN: Return from the current function: return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];.
    }  // EN: Structure delimiter for a block or scope.

    // 2×2 行列式
    static double Det2x2(double[,] A)  // EN: Execute line: static double Det2x2(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];  // EN: Return from the current function: return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];.
    }  // EN: Structure delimiter for a block or scope.

    // 平行四邊形面積
    static double ParallelogramArea(double[] a, double[] b)  // EN: Execute line: static double ParallelogramArea(double[] a, double[] b).
    {  // EN: Structure delimiter for a block or scope.
        return Math.Abs(Cross2D(a, b));  // EN: Return from the current function: return Math.Abs(Cross2D(a, b));.
    }  // EN: Structure delimiter for a block or scope.

    // 平行六面體體積
    static double ParallelepipedVolume(double[] a, double[] b, double[] c)  // EN: Execute line: static double ParallelepipedVolume(double[] a, double[] b, double[] c).
    {  // EN: Structure delimiter for a block or scope.
        double[] bxc = Cross3D(b, c);  // EN: Execute a statement: double[] bxc = Cross3D(b, c);.
        return Math.Abs(Dot3D(a, bxc));  // EN: Return from the current function: return Math.Abs(Dot3D(a, bxc));.
    }  // EN: Structure delimiter for a block or scope.

    // 三角形面積
    static double TriangleArea(double x1, double y1,  // EN: Execute line: static double TriangleArea(double x1, double y1,.
                               double x2, double y2,  // EN: Execute line: double x2, double y2,.
                               double x3, double y3)  // EN: Execute line: double x3, double y3).
    {  // EN: Structure delimiter for a block or scope.
        return Math.Abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;  // EN: Return from the current function: return Math.Abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;.
    }  // EN: Structure delimiter for a block or scope.

    static void Main(string[] args)  // EN: Execute line: static void Main(string[] args).
    {  // EN: Structure delimiter for a block or scope.
        PrintSeparator("行列式幾何解釋示範 (C#)");  // EN: Execute a statement: PrintSeparator("行列式幾何解釋示範 (C#)");.

        // ========================================
        // 1. 平行四邊形面積
        // ========================================
        PrintSeparator("1. 平行四邊形面積");  // EN: Execute a statement: PrintSeparator("1. 平行四邊形面積");.

        double[] a = {3, 0};  // EN: Execute a statement: double[] a = {3, 0};.
        double[] b = {1, 2};  // EN: Execute a statement: double[] b = {1, 2};.

        PrintVector("a", a);  // EN: Execute a statement: PrintVector("a", a);.
        PrintVector("b", b);  // EN: Execute a statement: PrintVector("b", b);.

        double area = ParallelogramArea(a, b);  // EN: Execute a statement: double area = ParallelogramArea(a, b);.
        double signedArea = Cross2D(a, b);  // EN: Execute a statement: double signedArea = Cross2D(a, b);.

        Console.WriteLine("\n平行四邊形：");  // EN: Execute a statement: Console.WriteLine("\n平行四邊形：");.
        Console.WriteLine($"  有號面積 = a × b = {signedArea:F4}");  // EN: Execute a statement: Console.WriteLine($" 有號面積 = a × b = {signedArea:F4}");.
        Console.WriteLine($"  面積 = |a × b| = {area:F4}");  // EN: Execute a statement: Console.WriteLine($" 面積 = |a × b| = {area:F4}");.

        // ========================================
        // 2. 定向判斷
        // ========================================
        PrintSeparator("2. 定向判斷");  // EN: Execute a statement: PrintSeparator("2. 定向判斷");.

        a = new double[]{1, 0};  // EN: Execute a statement: a = new double[]{1, 0};.
        b = new double[]{0, 1};  // EN: Execute a statement: b = new double[]{0, 1};.
        double signedVal = Cross2D(a, b);  // EN: Execute a statement: double signedVal = Cross2D(a, b);.

        PrintVector("a", a);  // EN: Execute a statement: PrintVector("a", a);.
        PrintVector("b", b);  // EN: Execute a statement: PrintVector("b", b);.
        Console.WriteLine($"有號面積 = {signedVal:F4}");  // EN: Execute a statement: Console.WriteLine($"有號面積 = {signedVal:F4}");.
        Console.WriteLine($"定向：{(signedVal > 0 ? "逆時針（正向）" : "順時針（負向）")}");  // EN: Execute a statement: Console.WriteLine($"定向：{(signedVal > 0 ? "逆時針（正向）" : "順時針（負向）")}");.

        Console.WriteLine("\n交換 a, b 順序：");  // EN: Execute a statement: Console.WriteLine("\n交換 a, b 順序：");.
        signedVal = Cross2D(b, a);  // EN: Execute a statement: signedVal = Cross2D(b, a);.
        Console.WriteLine($"有號面積 = {signedVal:F4}");  // EN: Execute a statement: Console.WriteLine($"有號面積 = {signedVal:F4}");.
        Console.WriteLine($"定向：{(signedVal > 0 ? "逆時針（正向）" : "順時針（負向）")}");  // EN: Execute a statement: Console.WriteLine($"定向：{(signedVal > 0 ? "逆時針（正向）" : "順時針（負向）")}");.

        // ========================================
        // 3. 平行六面體體積
        // ========================================
        PrintSeparator("3. 平行六面體體積");  // EN: Execute a statement: PrintSeparator("3. 平行六面體體積");.

        double[] v1 = {1, 0, 0};  // EN: Execute a statement: double[] v1 = {1, 0, 0};.
        double[] v2 = {0, 2, 0};  // EN: Execute a statement: double[] v2 = {0, 2, 0};.
        double[] v3 = {0, 0, 3};  // EN: Execute a statement: double[] v3 = {0, 0, 3};.

        PrintVector("a", v1);  // EN: Execute a statement: PrintVector("a", v1);.
        PrintVector("b", v2);  // EN: Execute a statement: PrintVector("b", v2);.
        PrintVector("c", v3);  // EN: Execute a statement: PrintVector("c", v3);.

        double vol = ParallelepipedVolume(v1, v2, v3);  // EN: Execute a statement: double vol = ParallelepipedVolume(v1, v2, v3);.
        Console.WriteLine($"\n體積 = |a · (b × c)| = {vol:F4}");  // EN: Execute a statement: Console.WriteLine($"\n體積 = |a · (b × c)| = {vol:F4}");.

        // ========================================
        // 4. 三角形面積
        // ========================================
        PrintSeparator("4. 三角形面積");  // EN: Execute a statement: PrintSeparator("4. 三角形面積");.

        double x1 = 0, y1 = 0;  // EN: Execute a statement: double x1 = 0, y1 = 0;.
        double x2 = 4, y2 = 0;  // EN: Execute a statement: double x2 = 4, y2 = 0;.
        double x3 = 0, y3 = 3;  // EN: Execute a statement: double x3 = 0, y3 = 3;.

        Console.WriteLine("三角形頂點：");  // EN: Execute a statement: Console.WriteLine("三角形頂點：");.
        Console.WriteLine($"  P1 = ({x1}, {y1})");  // EN: Execute a statement: Console.WriteLine($" P1 = ({x1}, {y1})");.
        Console.WriteLine($"  P2 = ({x2}, {y2})");  // EN: Execute a statement: Console.WriteLine($" P2 = ({x2}, {y2})");.
        Console.WriteLine($"  P3 = ({x3}, {y3})");  // EN: Execute a statement: Console.WriteLine($" P3 = ({x3}, {y3})");.

        double triArea = TriangleArea(x1, y1, x2, y2, x3, y3);  // EN: Execute a statement: double triArea = TriangleArea(x1, y1, x2, y2, x3, y3);.
        Console.WriteLine($"\n面積 = {triArea:F4}");  // EN: Execute a statement: Console.WriteLine($"\n面積 = {triArea:F4}");.

        // ========================================
        // 5. 線性變換的體積縮放
        // ========================================
        PrintSeparator("5. 線性變換的體積縮放");  // EN: Execute a statement: PrintSeparator("5. 線性變換的體積縮放");.

        double[,] A = {{2, 0}, {0, 3}};  // EN: Execute a statement: double[,] A = {{2, 0}, {0, 3}};.
        PrintMatrix("縮放矩陣 A", A);  // EN: Execute a statement: PrintMatrix("縮放矩陣 A", A);.
        Console.WriteLine($"det(A) = {Det2x2(A):F4}");  // EN: Execute a statement: Console.WriteLine($"det(A) = {Det2x2(A):F4}");.
        Console.WriteLine("\n單位正方形 → 2×3 長方形");  // EN: Execute a statement: Console.WriteLine("\n單位正方形 → 2×3 長方形");.
        Console.WriteLine($"面積從 1 變成 {Math.Abs(Det2x2(A)):F4}");  // EN: Execute a statement: Console.WriteLine($"面積從 1 變成 {Math.Abs(Det2x2(A)):F4}");.

        double theta = Math.PI / 4;  // EN: Execute a statement: double theta = Math.PI / 4;.
        double[,] R = {  // EN: Execute line: double[,] R = {.
            {Math.Cos(theta), -Math.Sin(theta)},  // EN: Execute line: {Math.Cos(theta), -Math.Sin(theta)},.
            {Math.Sin(theta), Math.Cos(theta)}  // EN: Execute line: {Math.Sin(theta), Math.Cos(theta)}.
        };  // EN: Structure delimiter for a block or scope.
        Console.WriteLine($"\n旋轉矩陣：det(R) = {Det2x2(R):F4}（面積不變）");  // EN: Execute a statement: Console.WriteLine($"\n旋轉矩陣：det(R) = {Det2x2(R):F4}（面積不變）");.

        double[,] H = {{1, 0}, {0, -1}};  // EN: Execute a statement: double[,] H = {{1, 0}, {0, -1}};.
        Console.WriteLine($"反射矩陣：det(H) = {Det2x2(H):F4}（面積不變，定向反轉）");  // EN: Execute a statement: Console.WriteLine($"反射矩陣：det(H) = {Det2x2(H):F4}（面積不變，定向反轉）");.

        double[,] S = {{1, 2}, {0, 1}};  // EN: Execute a statement: double[,] S = {{1, 2}, {0, 1}};.
        Console.WriteLine($"剪切矩陣：det(S) = {Det2x2(S):F4}（面積不變）");  // EN: Execute a statement: Console.WriteLine($"剪切矩陣：det(S) = {Det2x2(S):F4}（面積不變）");.

        // 總結
        PrintSeparator("總結");  // EN: Execute a statement: PrintSeparator("總結");.
        Console.WriteLine(@"  // EN: Execute line: Console.WriteLine(@".
行列式的幾何意義：  // EN: Execute line: 行列式的幾何意義：.

1. |det| = 體積/面積的縮放因子  // EN: Execute line: 1. |det| = 體積/面積的縮放因子.
2. sign(det) = 定向保持/反轉  // EN: Execute line: 2. sign(det) = 定向保持/反轉.
3. det = 0 → 降維  // EN: Execute line: 3. det = 0 → 降維.

特殊矩陣：  // EN: Execute line: 特殊矩陣：.
   - 旋轉：det = 1  // EN: Execute line: - 旋轉：det = 1.
   - 反射：det = -1  // EN: Execute line: - 反射：det = -1.
   - 剪切：det = 1  // EN: Execute line: - 剪切：det = 1.
");  // EN: Execute a statement: ");.

        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine("示範完成！");  // EN: Execute a statement: Console.WriteLine("示範完成！");.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
