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

using System;  // EN: Execute a statement: using System;.
using System.Text;  // EN: Execute a statement: using System.Text;.

class Projection  // EN: Execute line: class Projection.
{  // EN: Structure delimiter for a block or scope.
    private const double EPSILON = 1e-10;  // EN: Execute a statement: private const double EPSILON = 1e-10;.

    // ========================================
    // 輔助方法
    // ========================================

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

    static void PrintMatrix(string name, double[,] M)  // EN: Execute line: static void PrintMatrix(string name, double[,] M).
    {  // EN: Structure delimiter for a block or scope.
        int rows = M.GetLength(0);  // EN: Execute a statement: int rows = M.GetLength(0);.
        int cols = M.GetLength(1);  // EN: Execute a statement: int cols = M.GetLength(1);.

        Console.WriteLine($"{name} =");  // EN: Execute a statement: Console.WriteLine($"{name} =");.
        for (int i = 0; i < rows; i++)  // EN: Loop control flow: for (int i = 0; i < rows; i++).
        {  // EN: Structure delimiter for a block or scope.
            var sb = new StringBuilder("  [");  // EN: Execute a statement: var sb = new StringBuilder(" [");.
            for (int j = 0; j < cols; j++)  // EN: Loop control flow: for (int j = 0; j < cols; j++).
            {  // EN: Structure delimiter for a block or scope.
                sb.Append(M[i, j].ToString("F4").PadLeft(8));  // EN: Execute a statement: sb.Append(M[i, j].ToString("F4").PadLeft(8));.
                if (j < cols - 1) sb.Append(", ");  // EN: Conditional control flow: if (j < cols - 1) sb.Append(", ");.
            }  // EN: Structure delimiter for a block or scope.
            sb.Append("]");  // EN: Execute a statement: sb.Append("]");.
            Console.WriteLine(sb.ToString());  // EN: Execute a statement: Console.WriteLine(sb.ToString());.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 基本運算
    // ========================================

    static double DotProduct(double[] x, double[] y)  // EN: Execute line: static double DotProduct(double[] x, double[] y).
    {  // EN: Structure delimiter for a block or scope.
        double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
        for (int i = 0; i < x.Length; i++)  // EN: Loop control flow: for (int i = 0; i < x.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            result += x[i] * y[i];  // EN: Execute a statement: result += x[i] * y[i];.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double VectorNorm(double[] x)  // EN: Execute line: static double VectorNorm(double[] x).
    {  // EN: Structure delimiter for a block or scope.
        return Math.Sqrt(DotProduct(x, x));  // EN: Return from the current function: return Math.Sqrt(DotProduct(x, x));.
    }  // EN: Structure delimiter for a block or scope.

    static double[] ScalarMultiply(double c, double[] x)  // EN: Execute line: static double[] ScalarMultiply(double c, double[] x).
    {  // EN: Structure delimiter for a block or scope.
        double[] result = new double[x.Length];  // EN: Execute a statement: double[] result = new double[x.Length];.
        for (int i = 0; i < x.Length; i++)  // EN: Loop control flow: for (int i = 0; i < x.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            result[i] = c * x[i];  // EN: Execute a statement: result[i] = c * x[i];.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[] VectorSubtract(double[] x, double[] y)  // EN: Execute line: static double[] VectorSubtract(double[] x, double[] y).
    {  // EN: Structure delimiter for a block or scope.
        double[] result = new double[x.Length];  // EN: Execute a statement: double[] result = new double[x.Length];.
        for (int i = 0; i < x.Length; i++)  // EN: Loop control flow: for (int i = 0; i < x.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            result[i] = x[i] - y[i];  // EN: Execute a statement: result[i] = x[i] - y[i];.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[,] OuterProduct(double[] x, double[] y)  // EN: Execute line: static double[,] OuterProduct(double[] x, double[] y).
    {  // EN: Structure delimiter for a block or scope.
        double[,] result = new double[x.Length, y.Length];  // EN: Execute a statement: double[,] result = new double[x.Length, y.Length];.
        for (int i = 0; i < x.Length; i++)  // EN: Loop control flow: for (int i = 0; i < x.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < y.Length; j++)  // EN: Loop control flow: for (int j = 0; j < y.Length; j++).
            {  // EN: Structure delimiter for a block or scope.
                result[i, j] = x[i] * y[j];  // EN: Execute a statement: result[i, j] = x[i] * y[j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[,] MatrixScalarMultiply(double c, double[,] A)  // EN: Execute line: static double[,] MatrixScalarMultiply(double c, double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0), n = A.GetLength(1);  // EN: Execute a statement: int m = A.GetLength(0), n = A.GetLength(1);.
        double[,] result = new double[m, n];  // EN: Execute a statement: double[,] result = new double[m, n];.
        for (int i = 0; i < m; i++)  // EN: Loop control flow: for (int i = 0; i < m; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            {  // EN: Structure delimiter for a block or scope.
                result[i, j] = c * A[i, j];  // EN: Execute a statement: result[i, j] = c * A[i, j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[] MatrixVectorMultiply(double[,] A, double[] x)  // EN: Execute line: static double[] MatrixVectorMultiply(double[,] A, double[] x).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0), n = A.GetLength(1);  // EN: Execute a statement: int m = A.GetLength(0), n = A.GetLength(1);.
        double[] result = new double[m];  // EN: Execute a statement: double[] result = new double[m];.
        for (int i = 0; i < m; i++)  // EN: Loop control flow: for (int i = 0; i < m; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            {  // EN: Structure delimiter for a block or scope.
                result[i] += A[i, j] * x[j];  // EN: Execute a statement: result[i] += A[i, j] * x[j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[,] MatrixMultiply(double[,] A, double[,] B)  // EN: Execute line: static double[,] MatrixMultiply(double[,] A, double[,] B).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0), k = A.GetLength(1), n = B.GetLength(1);  // EN: Execute a statement: int m = A.GetLength(0), k = A.GetLength(1), n = B.GetLength(1);.
        double[,] result = new double[m, n];  // EN: Execute a statement: double[,] result = new double[m, n];.
        for (int i = 0; i < m; i++)  // EN: Loop control flow: for (int i = 0; i < m; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            {  // EN: Structure delimiter for a block or scope.
                for (int p = 0; p < k; p++)  // EN: Loop control flow: for (int p = 0; p < k; p++).
                {  // EN: Structure delimiter for a block or scope.
                    result[i, j] += A[i, p] * B[p, j];  // EN: Execute a statement: result[i, j] += A[i, p] * B[p, j];.
                }  // EN: Structure delimiter for a block or scope.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 投影函數
    // ========================================

    class ProjectionResult  // EN: Execute line: class ProjectionResult.
    {  // EN: Structure delimiter for a block or scope.
        public double XHat { get; set; }  // EN: Execute line: public double XHat { get; set; }.
        public double[] Proj { get; set; }  // EN: Execute line: public double[] Proj { get; set; }.
        public double[] Error { get; set; }  // EN: Execute line: public double[] Error { get; set; }.
        public double ErrorNorm { get; set; }  // EN: Execute line: public double ErrorNorm { get; set; }.
    }  // EN: Structure delimiter for a block or scope.

    /// <summary>
    /// 投影到直線
    /// p = (aᵀb / aᵀa) * a
    /// </summary>
    static ProjectionResult ProjectOntoLine(double[] b, double[] a)  // EN: Execute line: static ProjectionResult ProjectOntoLine(double[] b, double[] a).
    {  // EN: Structure delimiter for a block or scope.
        double aTb = DotProduct(a, b);  // EN: Execute a statement: double aTb = DotProduct(a, b);.
        double aTa = DotProduct(a, a);  // EN: Execute a statement: double aTa = DotProduct(a, a);.

        double xHat = aTb / aTa;  // EN: Execute a statement: double xHat = aTb / aTa;.
        double[] p = ScalarMultiply(xHat, a);  // EN: Execute a statement: double[] p = ScalarMultiply(xHat, a);.
        double[] e = VectorSubtract(b, p);  // EN: Execute a statement: double[] e = VectorSubtract(b, p);.

        return new ProjectionResult  // EN: Return from the current function: return new ProjectionResult.
        {  // EN: Structure delimiter for a block or scope.
            XHat = xHat,  // EN: Execute line: XHat = xHat,.
            Proj = p,  // EN: Execute line: Proj = p,.
            Error = e,  // EN: Execute line: Error = e,.
            ErrorNorm = VectorNorm(e)  // EN: Execute line: ErrorNorm = VectorNorm(e).
        };  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    /// <summary>
    /// 投影到直線的投影矩陣
    /// P = aaᵀ / (aᵀa)
    /// </summary>
    static double[,] ProjectionMatrixLine(double[] a)  // EN: Execute line: static double[,] ProjectionMatrixLine(double[] a).
    {  // EN: Structure delimiter for a block or scope.
        double aTa = DotProduct(a, a);  // EN: Execute a statement: double aTa = DotProduct(a, a);.
        double[,] aaT = OuterProduct(a, a);  // EN: Execute a statement: double[,] aaT = OuterProduct(a, a);.
        return MatrixScalarMultiply(1.0 / aTa, aaT);  // EN: Return from the current function: return MatrixScalarMultiply(1.0 / aTa, aaT);.
    }  // EN: Structure delimiter for a block or scope.

    /// <summary>
    /// 驗證投影矩陣的性質
    /// </summary>
    static void VerifyProjectionMatrix(double[,] P, string name)  // EN: Execute line: static void VerifyProjectionMatrix(double[,] P, string name).
    {  // EN: Structure delimiter for a block or scope.
        int n = P.GetLength(0);  // EN: Execute a statement: int n = P.GetLength(0);.

        Console.WriteLine($"\n驗證 {name} 的性質：");  // EN: Execute a statement: Console.WriteLine($"\n驗證 {name} 的性質：");.

        // 對稱性
        bool isSymmetric = true;  // EN: Execute a statement: bool isSymmetric = true;.
        for (int i = 0; i < n && isSymmetric; i++)  // EN: Loop control flow: for (int i = 0; i < n && isSymmetric; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < n && isSymmetric; j++)  // EN: Loop control flow: for (int j = 0; j < n && isSymmetric; j++).
            {  // EN: Structure delimiter for a block or scope.
                if (Math.Abs(P[i, j] - P[j, i]) > EPSILON)  // EN: Conditional control flow: if (Math.Abs(P[i, j] - P[j, i]) > EPSILON).
                    isSymmetric = false;  // EN: Execute a statement: isSymmetric = false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        Console.WriteLine($"  對稱性 ({name}ᵀ = {name})：{isSymmetric}");  // EN: Execute a statement: Console.WriteLine($" 對稱性 ({name}ᵀ = {name})：{isSymmetric}");.

        // 冪等性
        double[,] P2 = MatrixMultiply(P, P);  // EN: Execute a statement: double[,] P2 = MatrixMultiply(P, P);.
        bool isIdempotent = true;  // EN: Execute a statement: bool isIdempotent = true;.
        for (int i = 0; i < n && isIdempotent; i++)  // EN: Loop control flow: for (int i = 0; i < n && isIdempotent; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < n && isIdempotent; j++)  // EN: Loop control flow: for (int j = 0; j < n && isIdempotent; j++).
            {  // EN: Structure delimiter for a block or scope.
                if (Math.Abs(P[i, j] - P2[i, j]) > EPSILON)  // EN: Conditional control flow: if (Math.Abs(P[i, j] - P2[i, j]) > EPSILON).
                    isIdempotent = false;  // EN: Execute a statement: isIdempotent = false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        Console.WriteLine($"  冪等性 ({name}² = {name})：{isIdempotent}");  // EN: Execute a statement: Console.WriteLine($" 冪等性 ({name}² = {name})：{isIdempotent}");.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 主程式
    // ========================================

    static void Main(string[] args)  // EN: Execute line: static void Main(string[] args).
    {  // EN: Structure delimiter for a block or scope.
        PrintSeparator("投影示範 (C#)\nProjection Demo");  // EN: Execute a statement: PrintSeparator("投影示範 (C#)\nProjection Demo");.

        // 1. 投影到直線
        PrintSeparator("1. 投影到直線");  // EN: Execute a statement: PrintSeparator("1. 投影到直線");.

        double[] a = { 1.0, 1.0 };  // EN: Execute a statement: double[] a = { 1.0, 1.0 };.
        double[] b = { 2.0, 0.0 };  // EN: Execute a statement: double[] b = { 2.0, 0.0 };.

        PrintVector("方向 a", a);  // EN: Execute a statement: PrintVector("方向 a", a);.
        PrintVector("向量 b", b);  // EN: Execute a statement: PrintVector("向量 b", b);.

        var result = ProjectOntoLine(b, a);  // EN: Execute a statement: var result = ProjectOntoLine(b, a);.

        Console.WriteLine($"\n投影係數 x̂ = (aᵀb)/(aᵀa) = {result.XHat:F4}");  // EN: Execute a statement: Console.WriteLine($"\n投影係數 x̂ = (aᵀb)/(aᵀa) = {result.XHat:F4}");.
        PrintVector("投影 p = x̂a", result.Proj);  // EN: Execute a statement: PrintVector("投影 p = x̂a", result.Proj);.
        PrintVector("誤差 e = b - p", result.Error);  // EN: Execute a statement: PrintVector("誤差 e = b - p", result.Error);.

        // 驗證正交性
        double eDotA = DotProduct(result.Error, a);  // EN: Execute a statement: double eDotA = DotProduct(result.Error, a);.
        Console.WriteLine($"\n驗證 e ⊥ a：e · a = {eDotA:F6}");  // EN: Execute a statement: Console.WriteLine($"\n驗證 e ⊥ a：e · a = {eDotA:F6}");.
        Console.WriteLine($"正交？ {Math.Abs(eDotA) < EPSILON}");  // EN: Execute a statement: Console.WriteLine($"正交？ {Math.Abs(eDotA) < EPSILON}");.

        // 2. 投影矩陣
        PrintSeparator("2. 投影矩陣（到直線）");  // EN: Execute a statement: PrintSeparator("2. 投影矩陣（到直線）");.

        double[] a2 = { 1.0, 2.0 };  // EN: Execute a statement: double[] a2 = { 1.0, 2.0 };.
        PrintVector("方向 a", a2);  // EN: Execute a statement: PrintVector("方向 a", a2);.

        double[,] P = ProjectionMatrixLine(a2);  // EN: Execute a statement: double[,] P = ProjectionMatrixLine(a2);.
        PrintMatrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P);  // EN: Execute a statement: PrintMatrix("\n投影矩陣 P = aaᵀ/(aᵀa)", P);.

        VerifyProjectionMatrix(P, "P");  // EN: Execute a statement: VerifyProjectionMatrix(P, "P");.

        // 用投影矩陣計算投影
        double[] b2 = { 3.0, 4.0 };  // EN: Execute a statement: double[] b2 = { 3.0, 4.0 };.
        PrintVector("\n向量 b", b2);  // EN: Execute a statement: PrintVector("\n向量 b", b2);.

        double[] p = MatrixVectorMultiply(P, b2);  // EN: Execute a statement: double[] p = MatrixVectorMultiply(P, b2);.
        PrintVector("投影 p = Pb", p);  // EN: Execute a statement: PrintVector("投影 p = Pb", p);.

        // 3. 多個向量的投影
        PrintSeparator("3. 批次投影");  // EN: Execute a statement: PrintSeparator("3. 批次投影");.

        double[][] vectors = {  // EN: Execute line: double[][] vectors = {.
            new[] { 1.0, 0.0 },  // EN: Execute line: new[] { 1.0, 0.0 },.
            new[] { 0.0, 1.0 },  // EN: Execute line: new[] { 0.0, 1.0 },.
            new[] { 2.0, 2.0 },  // EN: Execute line: new[] { 2.0, 2.0 },.
            new[] { 3.0, -1.0 }  // EN: Execute line: new[] { 3.0, -1.0 }.
        };  // EN: Structure delimiter for a block or scope.

        Console.WriteLine("方向 a = [1, 2]");  // EN: Execute a statement: Console.WriteLine("方向 a = [1, 2]");.
        Console.WriteLine("\n各向量投影結果：");  // EN: Execute a statement: Console.WriteLine("\n各向量投影結果：");.

        foreach (var v in vectors)  // EN: Execute line: foreach (var v in vectors).
        {  // EN: Structure delimiter for a block or scope.
            var proj = ProjectOntoLine(v, a2);  // EN: Execute a statement: var proj = ProjectOntoLine(v, a2);.
            Console.WriteLine($"  [{v[0]:F1}, {v[1]:F1}] -> [{proj.Proj[0]:F4}, {proj.Proj[1]:F4}]");  // EN: Execute a statement: Console.WriteLine($" [{v[0]:F1}, {v[1]:F1}] -> [{proj.Proj[0]:F4}, {pro….
        }  // EN: Structure delimiter for a block or scope.

        // 總結
        PrintSeparator("總結");  // EN: Execute a statement: PrintSeparator("總結");.
        Console.WriteLine(@"  // EN: Execute line: Console.WriteLine(@".
投影公式：  // EN: Execute line: 投影公式：.

1. 投影到直線：  // EN: Execute line: 1. 投影到直線：.
   p = (aᵀb / aᵀa) a  // EN: Execute line: p = (aᵀb / aᵀa) a.
   P = aaᵀ / (aᵀa)  // EN: Execute line: P = aaᵀ / (aᵀa).

2. 投影到子空間：  // EN: Execute line: 2. 投影到子空間：.
   p = A(AᵀA)⁻¹Aᵀb  // EN: Execute line: p = A(AᵀA)⁻¹Aᵀb.
   P = A(AᵀA)⁻¹Aᵀ  // EN: Execute line: P = A(AᵀA)⁻¹Aᵀ.

3. 投影矩陣性質：  // EN: Execute line: 3. 投影矩陣性質：.
   Pᵀ = P（對稱）  // EN: Execute line: Pᵀ = P（對稱）.
   P² = P（冪等）  // EN: Execute line: P² = P（冪等）.
");  // EN: Execute a statement: ");.

        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine("示範完成！");  // EN: Execute a statement: Console.WriteLine("示範完成！");.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
