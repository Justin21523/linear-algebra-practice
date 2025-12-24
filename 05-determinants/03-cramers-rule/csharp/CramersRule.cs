/**
 * 克萊姆法則 (Cramer's Rule)
 *
 * 編譯：dotnet build 或 csc CramersRule.cs
 * 執行：dotnet run 或 ./CramersRule.exe
 */

using System;  // EN: Execute a statement: using System;.
using System.Text;  // EN: Execute a statement: using System.Text;.

class CramersRule  // EN: Execute line: class CramersRule.
{  // EN: Structure delimiter for a block or scope.
    static void PrintSeparator(string title)  // EN: Execute line: static void PrintSeparator(string title).
    {  // EN: Structure delimiter for a block or scope.
        Console.WriteLine();  // EN: Execute a statement: Console.WriteLine();.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine(title);  // EN: Execute a statement: Console.WriteLine(title);.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
    }  // EN: Structure delimiter for a block or scope.

    static void PrintMatrix(string name, double[,] M)  // EN: Execute line: static void PrintMatrix(string name, double[,] M).
    {  // EN: Structure delimiter for a block or scope.
        int n = M.GetLength(0);  // EN: Execute a statement: int n = M.GetLength(0);.
        Console.WriteLine($"{name} =");  // EN: Execute a statement: Console.WriteLine($"{name} =");.
        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
        {  // EN: Structure delimiter for a block or scope.
            var sb = new StringBuilder("  [");  // EN: Execute a statement: var sb = new StringBuilder(" [");.
            for (int j = 0; j < M.GetLength(1); j++)  // EN: Loop control flow: for (int j = 0; j < M.GetLength(1); j++).
            {  // EN: Structure delimiter for a block or scope.
                sb.Append(M[i, j].ToString("F4").PadLeft(8));  // EN: Execute a statement: sb.Append(M[i, j].ToString("F4").PadLeft(8));.
                if (j < M.GetLength(1) - 1) sb.Append(", ");  // EN: Conditional control flow: if (j < M.GetLength(1) - 1) sb.Append(", ");.
            }  // EN: Structure delimiter for a block or scope.
            sb.Append("]");  // EN: Execute a statement: sb.Append("]");.
            Console.WriteLine(sb.ToString());  // EN: Execute a statement: Console.WriteLine(sb.ToString());.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    static void PrintVector(string name, double[] v)  // EN: Execute line: static void PrintVector(string name, double[] v).
    {  // EN: Structure delimiter for a block or scope.
        var sb = new StringBuilder($"{name} = [");  // EN: Execute a statement: var sb = new StringBuilder($"{name} = [");.
        for (int i = 0; i < v.Length; i++)  // EN: Loop control flow: for (int i = 0; i < v.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            sb.Append(v[i].ToString("F4"));  // EN: Execute a statement: sb.Append(v[i].ToString("F4"));.
            if (i < v.Length - 1) sb.Append(", ");  // EN: Conditional control flow: if (i < v.Length - 1) sb.Append(", ");.
        }  // EN: Structure delimiter for a block or scope.
        sb.Append("]");  // EN: Execute a statement: sb.Append("]");.
        Console.WriteLine(sb.ToString());  // EN: Execute a statement: Console.WriteLine(sb.ToString());.
    }  // EN: Structure delimiter for a block or scope.

    // 2×2 行列式
    static double Det2x2(double[,] A)  // EN: Execute line: static double Det2x2(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];  // EN: Return from the current function: return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];.
    }  // EN: Structure delimiter for a block or scope.

    // 3×3 行列式
    static double Det3x3(double[,] A)  // EN: Execute line: static double Det3x3(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        return A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])  // EN: Return from the current function: return A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]).
             - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])  // EN: Execute line: - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]).
             + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]);  // EN: Execute a statement: + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]);.
    }  // EN: Structure delimiter for a block or scope.

    static double Determinant(double[,] A)  // EN: Execute line: static double Determinant(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int n = A.GetLength(0);  // EN: Execute a statement: int n = A.GetLength(0);.
        if (n == 2) return Det2x2(A);  // EN: Conditional control flow: if (n == 2) return Det2x2(A);.
        if (n == 3) return Det3x3(A);  // EN: Conditional control flow: if (n == 3) return Det3x3(A);.
        throw new Exception("僅支援 2×2 和 3×3 矩陣");  // EN: Execute a statement: throw new Exception("僅支援 2×2 和 3×3 矩陣");.
    }  // EN: Structure delimiter for a block or scope.

    // 替換第 j 行
    static double[,] ReplaceColumn(double[,] A, double[] b, int j)  // EN: Execute line: static double[,] ReplaceColumn(double[,] A, double[] b, int j).
    {  // EN: Structure delimiter for a block or scope.
        int n = A.GetLength(0);  // EN: Execute a statement: int n = A.GetLength(0);.
        double[,] Aj = new double[n, n];  // EN: Execute a statement: double[,] Aj = new double[n, n];.
        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int k = 0; k < n; k++)  // EN: Loop control flow: for (int k = 0; k < n; k++).
            {  // EN: Structure delimiter for a block or scope.
                Aj[i, k] = (k == j) ? b[i] : A[i, k];  // EN: Execute a statement: Aj[i, k] = (k == j) ? b[i] : A[i, k];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return Aj;  // EN: Return from the current function: return Aj;.
    }  // EN: Structure delimiter for a block or scope.

    // 克萊姆法則
    static double[] CramersRuleSolve(double[,] A, double[] b)  // EN: Execute line: static double[] CramersRuleSolve(double[,] A, double[] b).
    {  // EN: Structure delimiter for a block or scope.
        int n = A.GetLength(0);  // EN: Execute a statement: int n = A.GetLength(0);.
        double detA = Determinant(A);  // EN: Execute a statement: double detA = Determinant(A);.

        if (Math.Abs(detA) < 1e-10)  // EN: Conditional control flow: if (Math.Abs(detA) < 1e-10).
        {  // EN: Structure delimiter for a block or scope.
            throw new Exception("矩陣奇異");  // EN: Execute a statement: throw new Exception("矩陣奇異");.
        }  // EN: Structure delimiter for a block or scope.

        double[] x = new double[n];  // EN: Execute a statement: double[] x = new double[n];.
        for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
        {  // EN: Structure delimiter for a block or scope.
            double[,] Aj = ReplaceColumn(A, b, j);  // EN: Execute a statement: double[,] Aj = ReplaceColumn(A, b, j);.
            x[j] = Determinant(Aj) / detA;  // EN: Execute a statement: x[j] = Determinant(Aj) / detA;.
        }  // EN: Structure delimiter for a block or scope.
        return x;  // EN: Return from the current function: return x;.
    }  // EN: Structure delimiter for a block or scope.

    static void Main(string[] args)  // EN: Execute line: static void Main(string[] args).
    {  // EN: Structure delimiter for a block or scope.
        PrintSeparator("克萊姆法則示範 (C#)");  // EN: Execute a statement: PrintSeparator("克萊姆法則示範 (C#)");.

        // ========================================
        // 1. 2×2 系統
        // ========================================
        PrintSeparator("1. 2×2 系統");  // EN: Execute a statement: PrintSeparator("1. 2×2 系統");.

        double[,] A2 = {{2, 3}, {4, 5}};  // EN: Execute a statement: double[,] A2 = {{2, 3}, {4, 5}};.
        double[] b2 = {8, 14};  // EN: Execute a statement: double[] b2 = {8, 14};.

        Console.WriteLine("方程組：");  // EN: Execute a statement: Console.WriteLine("方程組：");.
        Console.WriteLine("  2x + 3y = 8");  // EN: Execute a statement: Console.WriteLine(" 2x + 3y = 8");.
        Console.WriteLine("  4x + 5y = 14");  // EN: Execute a statement: Console.WriteLine(" 4x + 5y = 14");.

        PrintMatrix("\nA", A2);  // EN: Execute a statement: PrintMatrix("\nA", A2);.
        PrintVector("b", b2);  // EN: Execute a statement: PrintVector("b", b2);.

        double detA2 = Determinant(A2);  // EN: Execute a statement: double detA2 = Determinant(A2);.
        Console.WriteLine($"\ndet(A) = {detA2:F4}");  // EN: Execute a statement: Console.WriteLine($"\ndet(A) = {detA2:F4}");.

        double[] x2 = CramersRuleSolve(A2, b2);  // EN: Execute a statement: double[] x2 = CramersRuleSolve(A2, b2);.

        for (int j = 0; j < 2; j++)  // EN: Loop control flow: for (int j = 0; j < 2; j++).
        {  // EN: Structure delimiter for a block or scope.
            double[,] Aj = ReplaceColumn(A2, b2, j);  // EN: Execute a statement: double[,] Aj = ReplaceColumn(A2, b2, j);.
            double detAj = Determinant(Aj);  // EN: Execute a statement: double detAj = Determinant(Aj);.
            Console.WriteLine($"\nA{j+1}（第 {j+1} 行換成 b）：");  // EN: Execute a statement: Console.WriteLine($"\nA{j+1}（第 {j+1} 行換成 b）：");.
            PrintMatrix("", Aj);  // EN: Execute a statement: PrintMatrix("", Aj);.
            Console.WriteLine($"det(A{j+1}) = {detAj:F4}");  // EN: Execute a statement: Console.WriteLine($"det(A{j+1}) = {detAj:F4}");.
            Console.WriteLine($"x{j+1} = {x2[j]:F4}");  // EN: Execute a statement: Console.WriteLine($"x{j+1} = {x2[j]:F4}");.
        }  // EN: Structure delimiter for a block or scope.

        Console.WriteLine($"\n解：x = {x2[0]:F4}, y = {x2[1]:F4}");  // EN: Execute a statement: Console.WriteLine($"\n解：x = {x2[0]:F4}, y = {x2[1]:F4}");.

        // ========================================
        // 2. 3×3 系統
        // ========================================
        PrintSeparator("2. 3×3 系統");  // EN: Execute a statement: PrintSeparator("2. 3×3 系統");.

        double[,] A3 = {  // EN: Execute line: double[,] A3 = {.
            {2, 1, -1},  // EN: Execute line: {2, 1, -1},.
            {-3, -1, 2},  // EN: Execute line: {-3, -1, 2},.
            {-2, 1, 2}  // EN: Execute line: {-2, 1, 2}.
        };  // EN: Structure delimiter for a block or scope.
        double[] b3 = {8, -11, -3};  // EN: Execute a statement: double[] b3 = {8, -11, -3};.

        Console.WriteLine("方程組：");  // EN: Execute a statement: Console.WriteLine("方程組：");.
        Console.WriteLine("   2x +  y -  z =  8");  // EN: Execute a statement: Console.WriteLine(" 2x + y - z = 8");.
        Console.WriteLine("  -3x -  y + 2z = -11");  // EN: Execute a statement: Console.WriteLine(" -3x - y + 2z = -11");.
        Console.WriteLine("  -2x +  y + 2z = -3");  // EN: Execute a statement: Console.WriteLine(" -2x + y + 2z = -3");.

        PrintMatrix("\nA", A3);  // EN: Execute a statement: PrintMatrix("\nA", A3);.
        PrintVector("b", b3);  // EN: Execute a statement: PrintVector("b", b3);.

        double[] x3 = CramersRuleSolve(A3, b3);  // EN: Execute a statement: double[] x3 = CramersRuleSolve(A3, b3);.

        Console.WriteLine($"\n解：x = {x3[0]:F4}, y = {x3[1]:F4}, z = {x3[2]:F4}");  // EN: Execute a statement: Console.WriteLine($"\n解：x = {x3[0]:F4}, y = {x3[1]:F4}, z = {x3[2]:F4}"….

        // 驗證
        Console.WriteLine("\n驗證：");  // EN: Execute a statement: Console.WriteLine("\n驗證：");.
        Console.WriteLine($"  2({x3[0]:F0}) + ({x3[1]:F0}) - ({x3[2]:F0}) = {2*x3[0] + x3[1] - x3[2]:F4}");  // EN: Execute a statement: Console.WriteLine($" 2({x3[0]:F0}) + ({x3[1]:F0}) - ({x3[2]:F0}) = {2*x….
        Console.WriteLine($"  -3({x3[0]:F0}) - ({x3[1]:F0}) + 2({x3[2]:F0}) = {-3*x3[0] - x3[1] + 2*x3[2]:F4}");  // EN: Execute a statement: Console.WriteLine($" -3({x3[0]:F0}) - ({x3[1]:F0}) + 2({x3[2]:F0}) = {-….
        Console.WriteLine($"  -2({x3[0]:F0}) + ({x3[1]:F0}) + 2({x3[2]:F0}) = {-2*x3[0] + x3[1] + 2*x3[2]:F4}");  // EN: Execute a statement: Console.WriteLine($" -2({x3[0]:F0}) + ({x3[1]:F0}) + 2({x3[2]:F0}) = {-….

        // 總結
        PrintSeparator("總結");  // EN: Execute a statement: PrintSeparator("總結");.
        Console.WriteLine(@"  // EN: Execute line: Console.WriteLine(@".
克萊姆法則：  // EN: Execute line: 克萊姆法則：.
  xⱼ = det(Aⱼ) / det(A)  // EN: Execute line: xⱼ = det(Aⱼ) / det(A).
  Aⱼ = A 的第 j 行換成 b  // EN: Execute line: Aⱼ = A 的第 j 行換成 b.

適用條件：  // EN: Execute line: 適用條件：.
  - det(A) ≠ 0  // EN: Execute line: - det(A) ≠ 0.
  - 方陣系統  // EN: Execute line: - 方陣系統.

時間複雜度：O(n! × n)  // EN: Execute line: 時間複雜度：O(n! × n).
");  // EN: Execute a statement: ");.

        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine("示範完成！");  // EN: Execute a statement: Console.WriteLine("示範完成！");.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
