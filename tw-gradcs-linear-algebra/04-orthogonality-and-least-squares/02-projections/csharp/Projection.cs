/**
 * 投影 (Projections)
 *
 * 本程式示範：
 * 1. 投影到直線
 * 2. 投影矩陣及其性質
 * 3. 誤差向量的正交性驗證
 *
 * 編譯：dotnet build 或 csc Projection.cs
 * 執行：dotnet run 或 ./Projection.exe
 */

using System;
using System.Text;

class Projection
{
    private const double EPSILON = 1e-10;

    // ========================================
    // 輔助方法
    // ========================================

    static void PrintSeparator(string title)
    {
        Console.WriteLine();
        Console.WriteLine(new string('=', 60));
        Console.WriteLine(title);
        Console.WriteLine(new string('=', 60));
    }

    static void PrintVector(string name, double[] v)
    {
        var sb = new StringBuilder();
        sb.Append(name).Append(" = [");
        for (int i = 0; i < v.Length; i++)
        {
            sb.Append(v[i].ToString("F4"));
            if (i < v.Length - 1) sb.Append(", ");
        }
        sb.Append("]");
        Console.WriteLine(sb.ToString());
    }

    static void PrintMatrix(string name, double[,] M)
    {
        int rows = M.GetLength(0);
        int cols = M.GetLength(1);

        Console.WriteLine($"{name} =");
        for (int i = 0; i < rows; i++)
        {
            var sb = new StringBuilder("  [");
            for (int j = 0; j < cols; j++)
            {
                sb.Append(M[i, j].ToString("F4").PadLeft(8));
                if (j < cols - 1) sb.Append(", ");
            }
            sb.Append("]");
            Console.WriteLine(sb.ToString());
        }
    }

    // ========================================
    // 基本運算
    // ========================================

    static double DotProduct(double[] x, double[] y)
    {
        double result = 0.0;
        for (int i = 0; i < x.Length; i++)
        {
            result += x[i] * y[i];
        }
        return result;
    }

    static double VectorNorm(double[] x)
    {
        return Math.Sqrt(DotProduct(x, x));
    }

    static double[] ScalarMultiply(double c, double[] x)
    {
        double[] result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            result[i] = c * x[i];
        }
        return result;
    }

    static double[] VectorSubtract(double[] x, double[] y)
    {
        double[] result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            result[i] = x[i] - y[i];
        }
        return result;
    }

    static double[,] OuterProduct(double[] x, double[] y)
    {
        double[,] result = new double[x.Length, y.Length];
        for (int i = 0; i < x.Length; i++)
        {
            for (int j = 0; j < y.Length; j++)
            {
                result[i, j] = x[i] * y[j];
            }
        }
        return result;
    }

    static double[,] MatrixScalarMultiply(double c, double[,] A)
    {
        int m = A.GetLength(0), n = A.GetLength(1);
        double[,] result = new double[m, n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = c * A[i, j];
            }
        }
        return result;
    }

    static double[] MatrixVectorMultiply(double[,] A, double[] x)
    {
        int m = A.GetLength(0), n = A.GetLength(1);
        double[] result = new double[m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i] += A[i, j] * x[j];
            }
        }
        return result;
    }

    static double[,] MatrixMultiply(double[,] A, double[,] B)
    {
        int m = A.GetLength(0), k = A.GetLength(1), n = B.GetLength(1);
        double[,] result = new double[m, n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int p = 0; p < k; p++)
                {
                    result[i, j] += A[i, p] * B[p, j];
                }
            }
        }
        return result;
    }

    // ========================================
    // 投影函數
    // ========================================

    class ProjectionResult
    {
        public double XHat { get; set; }
        public double[] Proj { get; set; }
        public double[] Error { get; set; }
        public double ErrorNorm { get; set; }
    }

    /// <summary>
    /// 投影到直線
    /// p = (aᵀb / aᵀa) * a
    /// </summary>
    static ProjectionResult ProjectOntoLine(double[] b, double[] a)
    {
        double aTb = DotProduct(a, b);
        double aTa = DotProduct(a, a);

        double xHat = aTb / aTa;
        double[] p = ScalarMultiply(xHat, a);
        double[] e = VectorSubtract(b, p);

        return new ProjectionResult
        {
            XHat = xHat,
            Proj = p,
            Error = e,
            ErrorNorm = VectorNorm(e)
        };
    }

    /// <summary>
    /// 投影到直線的投影矩陣
    /// P = aaᵀ / (aᵀa)
    /// </summary>
    static double[,] ProjectionMatrixLine(double[] a)
    {
        double aTa = DotProduct(a, a);
        double[,] aaT = OuterProduct(a, a);
        return MatrixScalarMultiply(1.0 / aTa, aaT);
    }

    /// <summary>
    /// 驗證投影矩陣的性質
    /// </summary>
    static void VerifyProjectionMatrix(double[,] P, string name)
    {
        int n = P.GetLength(0);

        Console.WriteLine($"\n驗證 {name} 的性質：");

        // 對稱性
        bool isSymmetric = true;
        for (int i = 0; i < n && isSymmetric; i++)
        {
            for (int j = 0; j < n && isSymmetric; j++)
            {
                if (Math.Abs(P[i, j] - P[j, i]) > EPSILON)
                    isSymmetric = false;
            }
        }
        Console.WriteLine($"  對稱性 ({name}ᵀ = {name})：{isSymmetric}");

        // 冪等性
        double[,] P2 = MatrixMultiply(P, P);
        bool isIdempotent = true;
        for (int i = 0; i < n && isIdempotent; i++)
        {
            for (int j = 0; j < n && isIdempotent; j++)
            {
                if (Math.Abs(P[i, j] - P2[i, j]) > EPSILON)
                    isIdempotent = false;
            }
        }
        Console.WriteLine($"  冪等性 ({name}² = {name})：{isIdempotent}");
    }

    // ========================================
    // 主程式
    // ========================================

    static void Main(string[] args)
    {
        PrintSeparator("投影示範 (C#)\nProjection Demo");

        // 1. 投影到直線
        PrintSeparator("1. 投影到直線");

        double[] a = { 1.0, 1.0 };
        double[] b = { 2.0, 0.0 };

        PrintVector("方向 a", a);
        PrintVector("向量 b", b);

        var result = ProjectOntoLine(b, a);

        Console.WriteLine($"\n投影係數 x̂ = (aᵀb)/(aᵀa) = {result.XHat:F4}");
        PrintVector("投影 p = x̂a", result.Proj);
        PrintVector("誤差 e = b - p", result.Error);

        // 驗證正交性
        double eDotA = DotProduct(result.Error, a);
        Console.WriteLine($"\n驗證 e ⊥ a：e · a = {eDotA:F6}");
        Console.WriteLine($"正交？ {Math.Abs(eDotA) < EPSILON}");

        // 2. 投影矩陣
        PrintSeparator("2. 投影矩陣（到直線）");

        double[] a2 = { 1.0, 2.0 };
        PrintVector("方向 a", a2);

        double[,] P = ProjectionMatrixLine(a2);
        PrintMatrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P);

        VerifyProjectionMatrix(P, "P");

        // 用投影矩陣計算投影
        double[] b2 = { 3.0, 4.0 };
        PrintVector("\n向量 b", b2);

        double[] p = MatrixVectorMultiply(P, b2);
        PrintVector("投影 p = Pb", p);

        // 3. 多個向量的投影
        PrintSeparator("3. 批次投影");

        double[][] vectors = {
            new[] { 1.0, 0.0 },
            new[] { 0.0, 1.0 },
            new[] { 2.0, 2.0 },
            new[] { 3.0, -1.0 }
        };

        Console.WriteLine("方向 a = [1, 2]");
        Console.WriteLine("\n各向量投影結果：");

        foreach (var v in vectors)
        {
            var proj = ProjectOntoLine(v, a2);
            Console.WriteLine($"  [{v[0]:F1}, {v[1]:F1}] -> [{proj.Proj[0]:F4}, {proj.Proj[1]:F4}]");
        }

        // 總結
        PrintSeparator("總結");
        Console.WriteLine(@"
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
");

        Console.WriteLine(new string('=', 60));
        Console.WriteLine("示範完成！");
        Console.WriteLine(new string('=', 60));
    }
}
