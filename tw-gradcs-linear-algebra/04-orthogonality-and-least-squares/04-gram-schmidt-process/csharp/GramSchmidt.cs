/**
 * Gram-Schmidt 正交化 (Gram-Schmidt Process)
 *
 * 編譯：dotnet build 或 csc GramSchmidt.cs
 * 執行：dotnet run 或 ./GramSchmidt.exe
 */

using System;
using System.Linq;
using System.Text;

class GramSchmidt
{
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

    static double DotProduct(double[] x, double[] y)
    {
        double result = 0.0;
        for (int i = 0; i < x.Length; i++) result += x[i] * y[i];
        return result;
    }

    static double VectorNorm(double[] x) => Math.Sqrt(DotProduct(x, x));

    static double[] ScalarMultiply(double c, double[] x)
    {
        return x.Select(xi => c * xi).ToArray();
    }

    static double[] VectorSubtract(double[] x, double[] y)
    {
        return x.Select((xi, i) => xi - y[i]).ToArray();
    }

    static double[] Normalize(double[] x)
    {
        return ScalarMultiply(1.0 / VectorNorm(x), x);
    }

    static double[][] ModifiedGramSchmidt(double[][] A)
    {
        int n = A.Length;
        double[][] Q = new double[n][];

        for (int i = 0; i < n; i++)
            Q[i] = (double[])A[i].Clone();

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < j; i++)
            {
                double coeff = DotProduct(Q[i], Q[j]) / DotProduct(Q[i], Q[i]);
                Q[j] = VectorSubtract(Q[j], ScalarMultiply(coeff, Q[i]));
            }
        }

        return Q;
    }

    static double[][] GramSchmidtNormalized(double[][] A)
    {
        double[][] Q = ModifiedGramSchmidt(A);
        for (int i = 0; i < Q.Length; i++)
            Q[i] = Normalize(Q[i]);
        return Q;
    }

    static bool VerifyOrthogonality(double[][] Q)
    {
        for (int i = 0; i < Q.Length; i++)
            for (int j = i + 1; j < Q.Length; j++)
                if (Math.Abs(DotProduct(Q[i], Q[j])) > 1e-10) return false;
        return true;
    }

    static void Main(string[] args)
    {
        PrintSeparator("Gram-Schmidt 正交化示範 (C#)");

        double[][] A = {
            new[] { 1.0, 1.0, 0.0 },
            new[] { 1.0, 0.0, 1.0 },
            new[] { 0.0, 1.0, 1.0 }
        };

        Console.WriteLine("輸入向量組：");
        for (int i = 0; i < A.Length; i++)
            PrintVector($"a{i+1}", A[i]);

        double[][] Q = ModifiedGramSchmidt(A);

        Console.WriteLine("\n正交化結果（MGS）：");
        for (int i = 0; i < Q.Length; i++)
        {
            PrintVector($"q{i+1}", Q[i]);
            Console.WriteLine($"    ‖q{i+1}‖ = {VectorNorm(Q[i]):F4}");
        }

        Console.WriteLine($"\n正交？ {VerifyOrthogonality(Q)}");

        Console.WriteLine("\n內積驗證：");
        Console.WriteLine($"q₁ · q₂ = {DotProduct(Q[0], Q[1]):F6}");
        Console.WriteLine($"q₁ · q₃ = {DotProduct(Q[0], Q[2]):F6}");
        Console.WriteLine($"q₂ · q₃ = {DotProduct(Q[1], Q[2]):F6}");

        PrintSeparator("標準正交化");

        double[][] E = GramSchmidtNormalized(A);

        Console.WriteLine("標準正交向量組：");
        for (int i = 0; i < E.Length; i++)
        {
            PrintVector($"e{i+1}", E[i]);
            Console.WriteLine($"    ‖e{i+1}‖ = {VectorNorm(E[i]):F4}");
        }

        PrintSeparator("總結");
        Console.WriteLine(@"
Gram-Schmidt 核心公式：

proj_q(a) = (qᵀa / qᵀq) q

q₁ = a₁
qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ)

eᵢ = qᵢ / ‖qᵢ‖
");

        Console.WriteLine(new string('=', 60));
        Console.WriteLine("示範完成！");
        Console.WriteLine(new string('=', 60));
    }
}
