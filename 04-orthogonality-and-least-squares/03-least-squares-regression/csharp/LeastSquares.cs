/**
 * 最小平方回歸 (Least Squares Regression)
 *
 * 編譯：dotnet build 或 csc LeastSquares.cs
 * 執行：dotnet run 或 ./LeastSquares.exe
 */

using System;  // EN: Execute a statement: using System;.
using System.Linq;  // EN: Execute a statement: using System.Linq;.
using System.Text;  // EN: Execute a statement: using System.Text;.

class LeastSquares  // EN: Execute line: class LeastSquares.
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

    static void PrintMatrix(string name, double[,] M)  // EN: Execute line: static void PrintMatrix(string name, double[,] M).
    {  // EN: Structure delimiter for a block or scope.
        int rows = M.GetLength(0), cols = M.GetLength(1);  // EN: Execute a statement: int rows = M.GetLength(0), cols = M.GetLength(1);.
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

    static double DotProduct(double[] x, double[] y)  // EN: Execute line: static double DotProduct(double[] x, double[] y).
    {  // EN: Structure delimiter for a block or scope.
        double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
        for (int i = 0; i < x.Length; i++)  // EN: Loop control flow: for (int i = 0; i < x.Length; i++).
            result += x[i] * y[i];  // EN: Execute a statement: result += x[i] * y[i];.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double VectorNorm(double[] x) => Math.Sqrt(DotProduct(x, x));  // EN: Execute a statement: static double VectorNorm(double[] x) => Math.Sqrt(DotProduct(x, x));.

    static double[,] Transpose(double[,] A)  // EN: Execute line: static double[,] Transpose(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0), n = A.GetLength(1);  // EN: Execute a statement: int m = A.GetLength(0), n = A.GetLength(1);.
        double[,] result = new double[n, m];  // EN: Execute a statement: double[,] result = new double[n, m];.
        for (int i = 0; i < m; i++)  // EN: Loop control flow: for (int i = 0; i < m; i++).
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
                result[j, i] = A[i, j];  // EN: Execute a statement: result[j, i] = A[i, j];.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[,] MatrixMultiply(double[,] A, double[,] B)  // EN: Execute line: static double[,] MatrixMultiply(double[,] A, double[,] B).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0), k = A.GetLength(1), n = B.GetLength(1);  // EN: Execute a statement: int m = A.GetLength(0), k = A.GetLength(1), n = B.GetLength(1);.
        double[,] result = new double[m, n];  // EN: Execute a statement: double[,] result = new double[m, n];.
        for (int i = 0; i < m; i++)  // EN: Loop control flow: for (int i = 0; i < m; i++).
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
                for (int p = 0; p < k; p++)  // EN: Loop control flow: for (int p = 0; p < k; p++).
                    result[i, j] += A[i, p] * B[p, j];  // EN: Execute a statement: result[i, j] += A[i, p] * B[p, j];.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[] MatrixVectorMultiply(double[,] A, double[] x)  // EN: Execute line: static double[] MatrixVectorMultiply(double[,] A, double[] x).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0), n = A.GetLength(1);  // EN: Execute a statement: int m = A.GetLength(0), n = A.GetLength(1);.
        double[] result = new double[m];  // EN: Execute a statement: double[] result = new double[m];.
        for (int i = 0; i < m; i++)  // EN: Loop control flow: for (int i = 0; i < m; i++).
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
                result[i] += A[i, j] * x[j];  // EN: Execute a statement: result[i] += A[i, j] * x[j];.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static double[] Solve2x2(double[,] A, double[] b)  // EN: Execute line: static double[] Solve2x2(double[,] A, double[] b).
    {  // EN: Structure delimiter for a block or scope.
        double det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];  // EN: Execute a statement: double det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];.
        return new double[]  // EN: Return from the current function: return new double[].
        {  // EN: Structure delimiter for a block or scope.
            (A[1, 1] * b[0] - A[0, 1] * b[1]) / det,  // EN: Execute line: (A[1, 1] * b[0] - A[0, 1] * b[1]) / det,.
            (-A[1, 0] * b[0] + A[0, 0] * b[1]) / det  // EN: Execute line: (-A[1, 0] * b[0] + A[0, 0] * b[1]) / det.
        };  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    class LeastSquaresResult  // EN: Execute line: class LeastSquaresResult.
    {  // EN: Structure delimiter for a block or scope.
        public double[] Coefficients { get; set; }  // EN: Execute line: public double[] Coefficients { get; set; }.
        public double[] Fitted { get; set; }  // EN: Execute line: public double[] Fitted { get; set; }.
        public double[] Residual { get; set; }  // EN: Execute line: public double[] Residual { get; set; }.
        public double ResidualNorm { get; set; }  // EN: Execute line: public double ResidualNorm { get; set; }.
        public double RSquared { get; set; }  // EN: Execute line: public double RSquared { get; set; }.
    }  // EN: Structure delimiter for a block or scope.

    static double[,] CreateDesignMatrixLinear(double[] t)  // EN: Execute line: static double[,] CreateDesignMatrixLinear(double[] t).
    {  // EN: Structure delimiter for a block or scope.
        double[,] A = new double[t.Length, 2];  // EN: Execute a statement: double[,] A = new double[t.Length, 2];.
        for (int i = 0; i < t.Length; i++)  // EN: Loop control flow: for (int i = 0; i < t.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            A[i, 0] = 1.0;  // EN: Execute a statement: A[i, 0] = 1.0;.
            A[i, 1] = t[i];  // EN: Execute a statement: A[i, 1] = t[i];.
        }  // EN: Structure delimiter for a block or scope.
        return A;  // EN: Return from the current function: return A;.
    }  // EN: Structure delimiter for a block or scope.

    static LeastSquaresResult LeastSquaresSolve(double[,] A, double[] b)  // EN: Execute line: static LeastSquaresResult LeastSquaresSolve(double[,] A, double[] b).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0), n = A.GetLength(1);  // EN: Execute a statement: int m = A.GetLength(0), n = A.GetLength(1);.

        var AT = Transpose(A);  // EN: Execute a statement: var AT = Transpose(A);.
        var ATA = MatrixMultiply(AT, A);  // EN: Execute a statement: var ATA = MatrixMultiply(AT, A);.
        var ATb = MatrixVectorMultiply(AT, b);  // EN: Execute a statement: var ATb = MatrixVectorMultiply(AT, b);.

        var coefficients = Solve2x2(ATA, ATb);  // EN: Execute a statement: var coefficients = Solve2x2(ATA, ATb);.
        var fitted = MatrixVectorMultiply(A, coefficients);  // EN: Execute a statement: var fitted = MatrixVectorMultiply(A, coefficients);.

        double[] residual = new double[m];  // EN: Execute a statement: double[] residual = new double[m];.
        for (int i = 0; i < m; i++)  // EN: Loop control flow: for (int i = 0; i < m; i++).
            residual[i] = b[i] - fitted[i];  // EN: Execute a statement: residual[i] = b[i] - fitted[i];.

        double residualNorm = VectorNorm(residual);  // EN: Execute a statement: double residualNorm = VectorNorm(residual);.

        double bMean = b.Average();  // EN: Execute a statement: double bMean = b.Average();.
        double tss = b.Sum(bi => (bi - bMean) * (bi - bMean));  // EN: Execute a statement: double tss = b.Sum(bi => (bi - bMean) * (bi - bMean));.
        double rss = residual.Sum(ei => ei * ei);  // EN: Execute a statement: double rss = residual.Sum(ei => ei * ei);.
        double rSquared = tss > 0 ? 1 - rss / tss : 0;  // EN: Execute a statement: double rSquared = tss > 0 ? 1 - rss / tss : 0;.

        return new LeastSquaresResult  // EN: Return from the current function: return new LeastSquaresResult.
        {  // EN: Structure delimiter for a block or scope.
            Coefficients = coefficients,  // EN: Execute line: Coefficients = coefficients,.
            Fitted = fitted,  // EN: Execute line: Fitted = fitted,.
            Residual = residual,  // EN: Execute line: Residual = residual,.
            ResidualNorm = residualNorm,  // EN: Execute line: ResidualNorm = residualNorm,.
            RSquared = rSquared  // EN: Execute line: RSquared = rSquared.
        };  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    static void Main(string[] args)  // EN: Execute line: static void Main(string[] args).
    {  // EN: Structure delimiter for a block or scope.
        PrintSeparator("最小平方回歸示範 (C#)\nLeast Squares Regression Demo");  // EN: Execute a statement: PrintSeparator("最小平方回歸示範 (C#)\nLeast Squares Regression Demo");.

        // 1. 簡單線性迴歸
        PrintSeparator("1. 簡單線性迴歸：y = C + Dt");  // EN: Execute a statement: PrintSeparator("1. 簡單線性迴歸：y = C + Dt");.

        double[] t = { 0.0, 1.0, 2.0 };  // EN: Execute a statement: double[] t = { 0.0, 1.0, 2.0 };.
        double[] b = { 1.0, 3.0, 4.0 };  // EN: Execute a statement: double[] b = { 1.0, 3.0, 4.0 };.

        Console.WriteLine("數據點：");  // EN: Execute a statement: Console.WriteLine("數據點：");.
        for (int i = 0; i < t.Length; i++)  // EN: Loop control flow: for (int i = 0; i < t.Length; i++).
            Console.WriteLine($"  t = {t[i]}, b = {b[i]}");  // EN: Execute a statement: Console.WriteLine($" t = {t[i]}, b = {b[i]}");.

        var A = CreateDesignMatrixLinear(t);  // EN: Execute a statement: var A = CreateDesignMatrixLinear(t);.
        PrintMatrix("\n設計矩陣 A [1, t]", A);  // EN: Execute a statement: PrintMatrix("\n設計矩陣 A [1, t]", A);.
        PrintVector("觀測值 b", b);  // EN: Execute a statement: PrintVector("觀測值 b", b);.

        var result = LeastSquaresSolve(A, b);  // EN: Execute a statement: var result = LeastSquaresSolve(A, b);.

        Console.WriteLine("\n【解】");  // EN: Execute a statement: Console.WriteLine("\n【解】");.
        Console.WriteLine($"C（截距）= {result.Coefficients[0]:F4}");  // EN: Execute a statement: Console.WriteLine($"C（截距）= {result.Coefficients[0]:F4}");.
        Console.WriteLine($"D（斜率）= {result.Coefficients[1]:F4}");  // EN: Execute a statement: Console.WriteLine($"D（斜率）= {result.Coefficients[1]:F4}");.
        Console.WriteLine($"\n最佳直線：y = {result.Coefficients[0]:F4} + {result.Coefficients[1]:F4}t");  // EN: Execute a statement: Console.WriteLine($"\n最佳直線：y = {result.Coefficients[0]:F4} + {result.Co….

        PrintVector("\n擬合值 ŷ", result.Fitted);  // EN: Execute a statement: PrintVector("\n擬合值 ŷ", result.Fitted);.
        PrintVector("殘差 e", result.Residual);  // EN: Execute a statement: PrintVector("殘差 e", result.Residual);.
        Console.WriteLine($"殘差範數 ‖e‖ = {result.ResidualNorm:F4}");  // EN: Execute a statement: Console.WriteLine($"殘差範數 ‖e‖ = {result.ResidualNorm:F4}");.
        Console.WriteLine($"R² = {result.RSquared:F4}");  // EN: Execute a statement: Console.WriteLine($"R² = {result.RSquared:F4}");.

        // 總結
        PrintSeparator("總結");  // EN: Execute a statement: PrintSeparator("總結");.
        Console.WriteLine(@"  // EN: Execute line: Console.WriteLine(@".
最小平方法核心公式：  // EN: Execute line: 最小平方法核心公式：.

1. 正規方程：AᵀA x̂ = Aᵀb  // EN: Execute line: 1. 正規方程：AᵀA x̂ = Aᵀb.

2. 解：x̂ = (AᵀA)⁻¹Aᵀb  // EN: Execute line: 2. 解：x̂ = (AᵀA)⁻¹Aᵀb.

3. R² = 1 - RSS/TSS  // EN: Execute line: 3. R² = 1 - RSS/TSS.
");  // EN: Execute a statement: ");.

        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine("示範完成！");  // EN: Execute a statement: Console.WriteLine("示範完成！");.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
