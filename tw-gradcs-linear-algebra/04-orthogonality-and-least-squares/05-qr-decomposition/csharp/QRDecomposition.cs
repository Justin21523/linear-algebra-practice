/**
 * QR 分解 (QR Decomposition)
 *
 * 編譯：dotnet build 或 csc QRDecomposition.cs
 * 執行：dotnet run 或 ./QRDecomposition.exe
 */

using System;
using System.Linq;
using System.Text;

class QRDecomposition
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
        Console.WriteLine($"{name} =");
        for (int i = 0; i < M.GetLength(0); i++)
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

    // 基本向量運算
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

    // 取得矩陣的第 j 行（column）
    static double[] GetColumn(double[,] A, int j)
    {
        int m = A.GetLength(0);
        double[] col = new double[m];
        for (int i = 0; i < m; i++) col[i] = A[i, j];
        return col;
    }

    // Gram-Schmidt QR 分解
    static (double[,] Q, double[,] R) QrDecomposition(double[,] A)
    {
        int m = A.GetLength(0);
        int n = A.GetLength(1);

        // Q: m×n, R: n×n
        double[,] Q = new double[m, n];
        double[,] R = new double[n, n];

        for (int j = 0; j < n; j++)
        {
            // 取得 A 的第 j 行
            double[] v = GetColumn(A, j);

            // 減去前面所有 q 向量的投影
            for (int i = 0; i < j; i++)
            {
                double[] qi = GetColumn(Q, i);
                R[i, j] = DotProduct(qi, GetColumn(A, j));
                double[] proj = ScalarMultiply(R[i, j], qi);
                v = VectorSubtract(v, proj);
            }

            // 標準化
            R[j, j] = VectorNorm(v);

            if (R[j, j] > 1e-10)
            {
                for (int i = 0; i < m; i++)
                {
                    Q[i, j] = v[i] / R[j, j];
                }
            }
        }

        return (Q, R);
    }

    // 回代法解上三角方程組 Rx = b
    static double[] SolveUpperTriangular(double[,] R, double[] b)
    {
        int n = b.Length;
        double[] x = new double[n];

        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = b[i];
            for (int j = i + 1; j < n; j++)
            {
                x[i] -= R[i, j] * x[j];
            }
            x[i] /= R[i, i];
        }

        return x;
    }

    // 用 QR 分解解最小平方問題
    static double[] QrLeastSquares(double[,] A, double[] b)
    {
        var (Q, R) = QrDecomposition(A);

        // 計算 Qᵀb
        int n = Q.GetLength(1);
        double[] Qt_b = new double[n];
        for (int j = 0; j < n; j++)
        {
            double[] qj = GetColumn(Q, j);
            Qt_b[j] = DotProduct(qj, b);
        }

        // 解 Rx = Qᵀb
        return SolveUpperTriangular(R, Qt_b);
    }

    // 矩陣乘法
    static double[,] MatrixMultiply(double[,] A, double[,] B)
    {
        int m = A.GetLength(0);
        int k = A.GetLength(1);
        int n = B.GetLength(1);

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

    // 矩陣轉置
    static double[,] Transpose(double[,] A)
    {
        int m = A.GetLength(0);
        int n = A.GetLength(1);
        double[,] result = new double[n, m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[j, i] = A[i, j];
            }
        }
        return result;
    }

    static void Main(string[] args)
    {
        PrintSeparator("QR 分解示範 (C#)");

        // ========================================
        // 1. 基本 QR 分解
        // ========================================
        PrintSeparator("1. 基本 QR 分解");

        double[,] A = {
            {1.0, 1.0},
            {1.0, 0.0},
            {0.0, 1.0}
        };

        Console.WriteLine("輸入矩陣 A：");
        PrintMatrix("A", A);

        var (Q, R) = QrDecomposition(A);

        Console.WriteLine("\nQR 分解結果：");
        PrintMatrix("Q", Q);
        PrintMatrix("\nR", R);

        // 驗證 QᵀQ = I
        var QT = Transpose(Q);
        var QTQ = MatrixMultiply(QT, Q);
        Console.WriteLine("\n驗證 QᵀQ = I：");
        PrintMatrix("QᵀQ", QTQ);

        // 驗證 A = QR
        var QR_result = MatrixMultiply(Q, R);
        Console.WriteLine("\n驗證 A = QR：");
        PrintMatrix("QR", QR_result);

        // ========================================
        // 2. 用 QR 解最小平方
        // ========================================
        PrintSeparator("2. 用 QR 解最小平方");

        // 數據
        double[] t = {0.0, 1.0, 2.0};
        double[] b = {1.0, 3.0, 4.0};

        Console.WriteLine("數據點：");
        for (int i = 0; i < t.Length; i++)
        {
            Console.WriteLine($"  ({t[i]}, {b[i]})");
        }

        // 設計矩陣
        double[,] A_ls = new double[t.Length, 2];
        for (int i = 0; i < t.Length; i++)
        {
            A_ls[i, 0] = 1.0;
            A_ls[i, 1] = t[i];
        }

        Console.WriteLine("\n設計矩陣 A：");
        PrintMatrix("A", A_ls);
        PrintVector("觀測值 b", b);

        // QR 分解
        var (Q_ls, R_ls) = QrDecomposition(A_ls);
        PrintMatrix("\nQ", Q_ls);
        PrintMatrix("R", R_ls);

        // 解最小平方
        double[] x = QrLeastSquares(A_ls, b);
        PrintVector("\n解 x", x);

        Console.WriteLine($"\n最佳直線：y = {x[0]:F4} + {x[1]:F4}t");

        // ========================================
        // 3. 3×3 矩陣的 QR 分解
        // ========================================
        PrintSeparator("3. 3×3 矩陣的 QR 分解");

        double[,] A2 = {
            {1.0, 1.0, 0.0},
            {1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0}
        };

        Console.WriteLine("輸入矩陣 A：");
        PrintMatrix("A", A2);

        var (Q2, R2) = QrDecomposition(A2);

        Console.WriteLine("\nQR 分解結果：");
        PrintMatrix("Q", Q2);
        PrintMatrix("\nR", R2);

        // 總結
        PrintSeparator("總結");
        Console.WriteLine(@"
QR 分解核心：

1. A = QR
   - Q: 標準正交矩陣 (QᵀQ = I)
   - R: 上三角矩陣

2. Gram-Schmidt 演算法：
   - 對 A 的行向量正交化得到 Q
   - R 的元素是投影係數

3. 用 QR 解最小平方：
   min ‖Ax - b‖²
   → Rx = Qᵀb

4. 優勢：
   - 比正規方程更穩定
   - 避免計算 AᵀA
");

        Console.WriteLine(new string('=', 60));
        Console.WriteLine("示範完成！");
        Console.WriteLine(new string('=', 60));
    }
}
