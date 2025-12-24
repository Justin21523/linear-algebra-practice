/**
 * 克萊姆法則 (Cramer's Rule)
 *
 * 編譯：dotnet build 或 csc CramersRule.cs
 * 執行：dotnet run 或 ./CramersRule.exe
 */

using System;
using System.Text;

class CramersRule
{
    static void PrintSeparator(string title)
    {
        Console.WriteLine();
        Console.WriteLine(new string('=', 60));
        Console.WriteLine(title);
        Console.WriteLine(new string('=', 60));
    }

    static void PrintMatrix(string name, double[,] M)
    {
        int n = M.GetLength(0);
        Console.WriteLine($"{name} =");
        for (int i = 0; i < n; i++)
        {
            var sb = new StringBuilder("  [");
            for (int j = 0; j < M.GetLength(1); j++)
            {
                sb.Append(M[i, j].ToString("F4").PadLeft(8));
                if (j < M.GetLength(1) - 1) sb.Append(", ");
            }
            sb.Append("]");
            Console.WriteLine(sb.ToString());
        }
    }

    static void PrintVector(string name, double[] v)
    {
        var sb = new StringBuilder($"{name} = [");
        for (int i = 0; i < v.Length; i++)
        {
            sb.Append(v[i].ToString("F4"));
            if (i < v.Length - 1) sb.Append(", ");
        }
        sb.Append("]");
        Console.WriteLine(sb.ToString());
    }

    // 2×2 行列式
    static double Det2x2(double[,] A)
    {
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];
    }

    // 3×3 行列式
    static double Det3x3(double[,] A)
    {
        return A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
             - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
             + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]);
    }

    static double Determinant(double[,] A)
    {
        int n = A.GetLength(0);
        if (n == 2) return Det2x2(A);
        if (n == 3) return Det3x3(A);
        throw new Exception("僅支援 2×2 和 3×3 矩陣");
    }

    // 替換第 j 行
    static double[,] ReplaceColumn(double[,] A, double[] b, int j)
    {
        int n = A.GetLength(0);
        double[,] Aj = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < n; k++)
            {
                Aj[i, k] = (k == j) ? b[i] : A[i, k];
            }
        }
        return Aj;
    }

    // 克萊姆法則
    static double[] CramersRuleSolve(double[,] A, double[] b)
    {
        int n = A.GetLength(0);
        double detA = Determinant(A);

        if (Math.Abs(detA) < 1e-10)
        {
            throw new Exception("矩陣奇異");
        }

        double[] x = new double[n];
        for (int j = 0; j < n; j++)
        {
            double[,] Aj = ReplaceColumn(A, b, j);
            x[j] = Determinant(Aj) / detA;
        }
        return x;
    }

    static void Main(string[] args)
    {
        PrintSeparator("克萊姆法則示範 (C#)");

        // ========================================
        // 1. 2×2 系統
        // ========================================
        PrintSeparator("1. 2×2 系統");

        double[,] A2 = {{2, 3}, {4, 5}};
        double[] b2 = {8, 14};

        Console.WriteLine("方程組：");
        Console.WriteLine("  2x + 3y = 8");
        Console.WriteLine("  4x + 5y = 14");

        PrintMatrix("\nA", A2);
        PrintVector("b", b2);

        double detA2 = Determinant(A2);
        Console.WriteLine($"\ndet(A) = {detA2:F4}");

        double[] x2 = CramersRuleSolve(A2, b2);

        for (int j = 0; j < 2; j++)
        {
            double[,] Aj = ReplaceColumn(A2, b2, j);
            double detAj = Determinant(Aj);
            Console.WriteLine($"\nA{j+1}（第 {j+1} 行換成 b）：");
            PrintMatrix("", Aj);
            Console.WriteLine($"det(A{j+1}) = {detAj:F4}");
            Console.WriteLine($"x{j+1} = {x2[j]:F4}");
        }

        Console.WriteLine($"\n解：x = {x2[0]:F4}, y = {x2[1]:F4}");

        // ========================================
        // 2. 3×3 系統
        // ========================================
        PrintSeparator("2. 3×3 系統");

        double[,] A3 = {
            {2, 1, -1},
            {-3, -1, 2},
            {-2, 1, 2}
        };
        double[] b3 = {8, -11, -3};

        Console.WriteLine("方程組：");
        Console.WriteLine("   2x +  y -  z =  8");
        Console.WriteLine("  -3x -  y + 2z = -11");
        Console.WriteLine("  -2x +  y + 2z = -3");

        PrintMatrix("\nA", A3);
        PrintVector("b", b3);

        double[] x3 = CramersRuleSolve(A3, b3);

        Console.WriteLine($"\n解：x = {x3[0]:F4}, y = {x3[1]:F4}, z = {x3[2]:F4}");

        // 驗證
        Console.WriteLine("\n驗證：");
        Console.WriteLine($"  2({x3[0]:F0}) + ({x3[1]:F0}) - ({x3[2]:F0}) = {2*x3[0] + x3[1] - x3[2]:F4}");
        Console.WriteLine($"  -3({x3[0]:F0}) - ({x3[1]:F0}) + 2({x3[2]:F0}) = {-3*x3[0] - x3[1] + 2*x3[2]:F4}");
        Console.WriteLine($"  -2({x3[0]:F0}) + ({x3[1]:F0}) + 2({x3[2]:F0}) = {-2*x3[0] + x3[1] + 2*x3[2]:F4}");

        // 總結
        PrintSeparator("總結");
        Console.WriteLine(@"
克萊姆法則：
  xⱼ = det(Aⱼ) / det(A)
  Aⱼ = A 的第 j 行換成 b

適用條件：
  - det(A) ≠ 0
  - 方陣系統

時間複雜度：O(n! × n)
");

        Console.WriteLine(new string('=', 60));
        Console.WriteLine("示範完成！");
        Console.WriteLine(new string('=', 60));
    }
}
