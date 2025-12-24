/**
 * Gram-Schmidt 正交化 (Gram-Schmidt Process)
 *
 * 編譯：dotnet build 或 csc GramSchmidt.cs
 * 執行：dotnet run 或 ./GramSchmidt.exe
 */

using System;  // EN: Execute a statement: using System;.
using System.Linq;  // EN: Execute a statement: using System.Linq;.
using System.Text;  // EN: Execute a statement: using System.Text;.

class GramSchmidt  // EN: Execute line: class GramSchmidt.
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
        var sb = new StringBuilder();  // EN: Execute a statement: var sb = new StringBuilder();.
        sb.Append(name).Append(" = [");  // EN: Execute a statement: sb.Append(name).Append(" = [");.
        for (int i = 0; i < v.Length; i++)  // EN: Loop control flow: for (int i = 0; i < v.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            sb.Append(v[i].ToString("F4"));  // EN: Execute a statement: sb.Append(v[i].ToString("F4"));.
            if (i < v.Length - 1) sb.Append(", ");  // EN: Conditional control flow: if (i < v.Length - 1) sb.Append(", ");.
        }  // EN: Structure delimiter for a block or scope.
        sb.Append("]");  // EN: Execute a statement: sb.Append("]");.
        Console.WriteLine(sb.ToString());  // EN: Execute a statement: Console.WriteLine(sb.ToString());.
    }  // EN: Structure delimiter for a block or scope.

    static double DotProduct(double[] x, double[] y)  // EN: Execute line: static double DotProduct(double[] x, double[] y).
    {  // EN: Structure delimiter for a block or scope.
        double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
        for (int i = 0; i < x.Length; i++) result += x[i] * y[i];  // EN: Loop control flow: for (int i = 0; i < x.Length; i++) result += x[i] * y[i];.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double VectorNorm(double[] x) => Math.Sqrt(DotProduct(x, x));  // EN: Execute a statement: static double VectorNorm(double[] x) => Math.Sqrt(DotProduct(x, x));.

    static double[] ScalarMultiply(double c, double[] x)  // EN: Execute line: static double[] ScalarMultiply(double c, double[] x).
    {  // EN: Structure delimiter for a block or scope.
        return x.Select(xi => c * xi).ToArray();  // EN: Return from the current function: return x.Select(xi => c * xi).ToArray();.
    }  // EN: Structure delimiter for a block or scope.

    static double[] VectorSubtract(double[] x, double[] y)  // EN: Execute line: static double[] VectorSubtract(double[] x, double[] y).
    {  // EN: Structure delimiter for a block or scope.
        return x.Select((xi, i) => xi - y[i]).ToArray();  // EN: Return from the current function: return x.Select((xi, i) => xi - y[i]).ToArray();.
    }  // EN: Structure delimiter for a block or scope.

    static double[] Normalize(double[] x)  // EN: Execute line: static double[] Normalize(double[] x).
    {  // EN: Structure delimiter for a block or scope.
        return ScalarMultiply(1.0 / VectorNorm(x), x);  // EN: Return from the current function: return ScalarMultiply(1.0 / VectorNorm(x), x);.
    }  // EN: Structure delimiter for a block or scope.

    static double[][] ModifiedGramSchmidt(double[][] A)  // EN: Execute line: static double[][] ModifiedGramSchmidt(double[][] A).
    {  // EN: Structure delimiter for a block or scope.
        int n = A.Length;  // EN: Execute a statement: int n = A.Length;.
        double[][] Q = new double[n][];  // EN: Execute a statement: double[][] Q = new double[n][];.

        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
            Q[i] = (double[])A[i].Clone();  // EN: Execute a statement: Q[i] = (double[])A[i].Clone();.

        for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
        {  // EN: Structure delimiter for a block or scope.
            for (int i = 0; i < j; i++)  // EN: Loop control flow: for (int i = 0; i < j; i++).
            {  // EN: Structure delimiter for a block or scope.
                double coeff = DotProduct(Q[i], Q[j]) / DotProduct(Q[i], Q[i]);  // EN: Execute a statement: double coeff = DotProduct(Q[i], Q[j]) / DotProduct(Q[i], Q[i]);.
                Q[j] = VectorSubtract(Q[j], ScalarMultiply(coeff, Q[i]));  // EN: Execute a statement: Q[j] = VectorSubtract(Q[j], ScalarMultiply(coeff, Q[i]));.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.

        return Q;  // EN: Return from the current function: return Q;.
    }  // EN: Structure delimiter for a block or scope.

    static double[][] GramSchmidtNormalized(double[][] A)  // EN: Execute line: static double[][] GramSchmidtNormalized(double[][] A).
    {  // EN: Structure delimiter for a block or scope.
        double[][] Q = ModifiedGramSchmidt(A);  // EN: Execute a statement: double[][] Q = ModifiedGramSchmidt(A);.
        for (int i = 0; i < Q.Length; i++)  // EN: Loop control flow: for (int i = 0; i < Q.Length; i++).
            Q[i] = Normalize(Q[i]);  // EN: Execute a statement: Q[i] = Normalize(Q[i]);.
        return Q;  // EN: Return from the current function: return Q;.
    }  // EN: Structure delimiter for a block or scope.

    static bool VerifyOrthogonality(double[][] Q)  // EN: Execute line: static bool VerifyOrthogonality(double[][] Q).
    {  // EN: Structure delimiter for a block or scope.
        for (int i = 0; i < Q.Length; i++)  // EN: Loop control flow: for (int i = 0; i < Q.Length; i++).
            for (int j = i + 1; j < Q.Length; j++)  // EN: Loop control flow: for (int j = i + 1; j < Q.Length; j++).
                if (Math.Abs(DotProduct(Q[i], Q[j])) > 1e-10) return false;  // EN: Conditional control flow: if (Math.Abs(DotProduct(Q[i], Q[j])) > 1e-10) return false;.
        return true;  // EN: Return from the current function: return true;.
    }  // EN: Structure delimiter for a block or scope.

    static void Main(string[] args)  // EN: Execute line: static void Main(string[] args).
    {  // EN: Structure delimiter for a block or scope.
        PrintSeparator("Gram-Schmidt 正交化示範 (C#)");  // EN: Execute a statement: PrintSeparator("Gram-Schmidt 正交化示範 (C#)");.

        double[][] A = {  // EN: Execute line: double[][] A = {.
            new[] { 1.0, 1.0, 0.0 },  // EN: Execute line: new[] { 1.0, 1.0, 0.0 },.
            new[] { 1.0, 0.0, 1.0 },  // EN: Execute line: new[] { 1.0, 0.0, 1.0 },.
            new[] { 0.0, 1.0, 1.0 }  // EN: Execute line: new[] { 0.0, 1.0, 1.0 }.
        };  // EN: Structure delimiter for a block or scope.

        Console.WriteLine("輸入向量組：");  // EN: Execute a statement: Console.WriteLine("輸入向量組：");.
        for (int i = 0; i < A.Length; i++)  // EN: Loop control flow: for (int i = 0; i < A.Length; i++).
            PrintVector($"a{i+1}", A[i]);  // EN: Execute a statement: PrintVector($"a{i+1}", A[i]);.

        double[][] Q = ModifiedGramSchmidt(A);  // EN: Execute a statement: double[][] Q = ModifiedGramSchmidt(A);.

        Console.WriteLine("\n正交化結果（MGS）：");  // EN: Execute a statement: Console.WriteLine("\n正交化結果（MGS）：");.
        for (int i = 0; i < Q.Length; i++)  // EN: Loop control flow: for (int i = 0; i < Q.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            PrintVector($"q{i+1}", Q[i]);  // EN: Execute a statement: PrintVector($"q{i+1}", Q[i]);.
            Console.WriteLine($"    ‖q{i+1}‖ = {VectorNorm(Q[i]):F4}");  // EN: Execute a statement: Console.WriteLine($" ‖q{i+1}‖ = {VectorNorm(Q[i]):F4}");.
        }  // EN: Structure delimiter for a block or scope.

        Console.WriteLine($"\n正交？ {VerifyOrthogonality(Q)}");  // EN: Execute a statement: Console.WriteLine($"\n正交？ {VerifyOrthogonality(Q)}");.

        Console.WriteLine("\n內積驗證：");  // EN: Execute a statement: Console.WriteLine("\n內積驗證：");.
        Console.WriteLine($"q₁ · q₂ = {DotProduct(Q[0], Q[1]):F6}");  // EN: Execute a statement: Console.WriteLine($"q₁ · q₂ = {DotProduct(Q[0], Q[1]):F6}");.
        Console.WriteLine($"q₁ · q₃ = {DotProduct(Q[0], Q[2]):F6}");  // EN: Execute a statement: Console.WriteLine($"q₁ · q₃ = {DotProduct(Q[0], Q[2]):F6}");.
        Console.WriteLine($"q₂ · q₃ = {DotProduct(Q[1], Q[2]):F6}");  // EN: Execute a statement: Console.WriteLine($"q₂ · q₃ = {DotProduct(Q[1], Q[2]):F6}");.

        PrintSeparator("標準正交化");  // EN: Execute a statement: PrintSeparator("標準正交化");.

        double[][] E = GramSchmidtNormalized(A);  // EN: Execute a statement: double[][] E = GramSchmidtNormalized(A);.

        Console.WriteLine("標準正交向量組：");  // EN: Execute a statement: Console.WriteLine("標準正交向量組：");.
        for (int i = 0; i < E.Length; i++)  // EN: Loop control flow: for (int i = 0; i < E.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            PrintVector($"e{i+1}", E[i]);  // EN: Execute a statement: PrintVector($"e{i+1}", E[i]);.
            Console.WriteLine($"    ‖e{i+1}‖ = {VectorNorm(E[i]):F4}");  // EN: Execute a statement: Console.WriteLine($" ‖e{i+1}‖ = {VectorNorm(E[i]):F4}");.
        }  // EN: Structure delimiter for a block or scope.

        PrintSeparator("總結");  // EN: Execute a statement: PrintSeparator("總結");.
        Console.WriteLine(@"  // EN: Execute line: Console.WriteLine(@".
Gram-Schmidt 核心公式：  // EN: Execute line: Gram-Schmidt 核心公式：.

proj_q(a) = (qᵀa / qᵀq) q  // EN: Execute line: proj_q(a) = (qᵀa / qᵀq) q.

q₁ = a₁  // EN: Execute line: q₁ = a₁.
qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ)  // EN: Execute line: qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ).

eᵢ = qᵢ / ‖qᵢ‖  // EN: Execute line: eᵢ = qᵢ / ‖qᵢ‖.
");  // EN: Execute a statement: ");.

        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine("示範完成！");  // EN: Execute a statement: Console.WriteLine("示範完成！");.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
