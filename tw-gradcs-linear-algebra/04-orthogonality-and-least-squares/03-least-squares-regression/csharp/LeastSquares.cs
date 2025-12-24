/**
 * 最小平方回歸 (Least Squares Regression)
 *
 * 編譯：dotnet build 或 csc LeastSquares.cs
 * 執行：dotnet run 或 ./LeastSquares.exe
 */

using System;
using System.Linq;
using System.Text;

class LeastSquares
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

    static void PrintMatrix(string name, double[,] M)
    {
        int rows = M.GetLength(0), cols = M.GetLength(1);
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

    static double DotProduct(double[] x, double[] y)
    {
        double result = 0.0;
        for (int i = 0; i < x.Length; i++)
            result += x[i] * y[i];
        return result;
    }

    static double VectorNorm(double[] x) => Math.Sqrt(DotProduct(x, x));

    static double[,] Transpose(double[,] A)
    {
        int m = A.GetLength(0), n = A.GetLength(1);
        double[,] result = new double[n, m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[j, i] = A[i, j];
        return result;
    }

    static double[,] MatrixMultiply(double[,] A, double[,] B)
    {
        int m = A.GetLength(0), k = A.GetLength(1), n = B.GetLength(1);
        double[,] result = new double[m, n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int p = 0; p < k; p++)
                    result[i, j] += A[i, p] * B[p, j];
        return result;
    }

    static double[] MatrixVectorMultiply(double[,] A, double[] x)
    {
        int m = A.GetLength(0), n = A.GetLength(1);
        double[] result = new double[m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[i] += A[i, j] * x[j];
        return result;
    }

    static double[] Solve2x2(double[,] A, double[] b)
    {
        double det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];
        return new double[]
        {
            (A[1, 1] * b[0] - A[0, 1] * b[1]) / det,
            (-A[1, 0] * b[0] + A[0, 0] * b[1]) / det
        };
    }

    class LeastSquaresResult
    {
        public double[] Coefficients { get; set; }
        public double[] Fitted { get; set; }
        public double[] Residual { get; set; }
        public double ResidualNorm { get; set; }
        public double RSquared { get; set; }
    }

    static double[,] CreateDesignMatrixLinear(double[] t)
    {
        double[,] A = new double[t.Length, 2];
        for (int i = 0; i < t.Length; i++)
        {
            A[i, 0] = 1.0;
            A[i, 1] = t[i];
        }
        return A;
    }

    static LeastSquaresResult LeastSquaresSolve(double[,] A, double[] b)
    {
        int m = A.GetLength(0), n = A.GetLength(1);

        var AT = Transpose(A);
        var ATA = MatrixMultiply(AT, A);
        var ATb = MatrixVectorMultiply(AT, b);

        var coefficients = Solve2x2(ATA, ATb);
        var fitted = MatrixVectorMultiply(A, coefficients);

        double[] residual = new double[m];
        for (int i = 0; i < m; i++)
            residual[i] = b[i] - fitted[i];

        double residualNorm = VectorNorm(residual);

        double bMean = b.Average();
        double tss = b.Sum(bi => (bi - bMean) * (bi - bMean));
        double rss = residual.Sum(ei => ei * ei);
        double rSquared = tss > 0 ? 1 - rss / tss : 0;

        return new LeastSquaresResult
        {
            Coefficients = coefficients,
            Fitted = fitted,
            Residual = residual,
            ResidualNorm = residualNorm,
            RSquared = rSquared
        };
    }

    static void Main(string[] args)
    {
        PrintSeparator("最小平方回歸示範 (C#)\nLeast Squares Regression Demo");

        // 1. 簡單線性迴歸
        PrintSeparator("1. 簡單線性迴歸：y = C + Dt");

        double[] t = { 0.0, 1.0, 2.0 };
        double[] b = { 1.0, 3.0, 4.0 };

        Console.WriteLine("數據點：");
        for (int i = 0; i < t.Length; i++)
            Console.WriteLine($"  t = {t[i]}, b = {b[i]}");

        var A = CreateDesignMatrixLinear(t);
        PrintMatrix("\n設計矩陣 A [1, t]", A);
        PrintVector("觀測值 b", b);

        var result = LeastSquaresSolve(A, b);

        Console.WriteLine("\n【解】");
        Console.WriteLine($"C（截距）= {result.Coefficients[0]:F4}");
        Console.WriteLine($"D（斜率）= {result.Coefficients[1]:F4}");
        Console.WriteLine($"\n最佳直線：y = {result.Coefficients[0]:F4} + {result.Coefficients[1]:F4}t");

        PrintVector("\n擬合值 ŷ", result.Fitted);
        PrintVector("殘差 e", result.Residual);
        Console.WriteLine($"殘差範數 ‖e‖ = {result.ResidualNorm:F4}");
        Console.WriteLine($"R² = {result.RSquared:F4}");

        // 總結
        PrintSeparator("總結");
        Console.WriteLine(@"
最小平方法核心公式：

1. 正規方程：AᵀA x̂ = Aᵀb

2. 解：x̂ = (AᵀA)⁻¹Aᵀb

3. R² = 1 - RSS/TSS
");

        Console.WriteLine(new string('=', 60));
        Console.WriteLine("示範完成！");
        Console.WriteLine(new string('=', 60));
    }
}
