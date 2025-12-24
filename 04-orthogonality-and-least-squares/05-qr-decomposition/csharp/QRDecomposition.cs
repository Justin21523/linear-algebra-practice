/**
 * QR 分解 (QR Decomposition)
 *
 * 編譯：dotnet build 或 csc QRDecomposition.cs
 * 執行：dotnet run 或 ./QRDecomposition.exe
 */

using System;  // EN: Execute a statement: using System;.
using System.Linq;  // EN: Execute a statement: using System.Linq;.
using System.Text;  // EN: Execute a statement: using System.Text;.

class QRDecomposition  // EN: Execute line: class QRDecomposition.
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
        Console.WriteLine($"{name} =");  // EN: Execute a statement: Console.WriteLine($"{name} =");.
        for (int i = 0; i < M.GetLength(0); i++)  // EN: Loop control flow: for (int i = 0; i < M.GetLength(0); i++).
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

    // 基本向量運算
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

    // 取得矩陣的第 j 行（column）
    static double[] GetColumn(double[,] A, int j)  // EN: Execute line: static double[] GetColumn(double[,] A, int j).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0);  // EN: Execute a statement: int m = A.GetLength(0);.
        double[] col = new double[m];  // EN: Execute a statement: double[] col = new double[m];.
        for (int i = 0; i < m; i++) col[i] = A[i, j];  // EN: Loop control flow: for (int i = 0; i < m; i++) col[i] = A[i, j];.
        return col;  // EN: Return from the current function: return col;.
    }  // EN: Structure delimiter for a block or scope.

    // Gram-Schmidt QR 分解
    static (double[,] Q, double[,] R) QrDecomposition(double[,] A)  // EN: Execute line: static (double[,] Q, double[,] R) QrDecomposition(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0);  // EN: Execute a statement: int m = A.GetLength(0);.
        int n = A.GetLength(1);  // EN: Execute a statement: int n = A.GetLength(1);.

        // Q: m×n, R: n×n
        double[,] Q = new double[m, n];  // EN: Execute a statement: double[,] Q = new double[m, n];.
        double[,] R = new double[n, n];  // EN: Execute a statement: double[,] R = new double[n, n];.

        for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
        {  // EN: Structure delimiter for a block or scope.
            // 取得 A 的第 j 行
            double[] v = GetColumn(A, j);  // EN: Execute a statement: double[] v = GetColumn(A, j);.

            // 減去前面所有 q 向量的投影
            for (int i = 0; i < j; i++)  // EN: Loop control flow: for (int i = 0; i < j; i++).
            {  // EN: Structure delimiter for a block or scope.
                double[] qi = GetColumn(Q, i);  // EN: Execute a statement: double[] qi = GetColumn(Q, i);.
                R[i, j] = DotProduct(qi, GetColumn(A, j));  // EN: Execute a statement: R[i, j] = DotProduct(qi, GetColumn(A, j));.
                double[] proj = ScalarMultiply(R[i, j], qi);  // EN: Execute a statement: double[] proj = ScalarMultiply(R[i, j], qi);.
                v = VectorSubtract(v, proj);  // EN: Execute a statement: v = VectorSubtract(v, proj);.
            }  // EN: Structure delimiter for a block or scope.

            // 標準化
            R[j, j] = VectorNorm(v);  // EN: Execute a statement: R[j, j] = VectorNorm(v);.

            if (R[j, j] > 1e-10)  // EN: Conditional control flow: if (R[j, j] > 1e-10).
            {  // EN: Structure delimiter for a block or scope.
                for (int i = 0; i < m; i++)  // EN: Loop control flow: for (int i = 0; i < m; i++).
                {  // EN: Structure delimiter for a block or scope.
                    Q[i, j] = v[i] / R[j, j];  // EN: Execute a statement: Q[i, j] = v[i] / R[j, j];.
                }  // EN: Structure delimiter for a block or scope.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.

        return (Q, R);  // EN: Return from the current function: return (Q, R);.
    }  // EN: Structure delimiter for a block or scope.

    // 回代法解上三角方程組 Rx = b
    static double[] SolveUpperTriangular(double[,] R, double[] b)  // EN: Execute line: static double[] SolveUpperTriangular(double[,] R, double[] b).
    {  // EN: Structure delimiter for a block or scope.
        int n = b.Length;  // EN: Execute a statement: int n = b.Length;.
        double[] x = new double[n];  // EN: Execute a statement: double[] x = new double[n];.

        for (int i = n - 1; i >= 0; i--)  // EN: Loop control flow: for (int i = n - 1; i >= 0; i--).
        {  // EN: Structure delimiter for a block or scope.
            x[i] = b[i];  // EN: Execute a statement: x[i] = b[i];.
            for (int j = i + 1; j < n; j++)  // EN: Loop control flow: for (int j = i + 1; j < n; j++).
            {  // EN: Structure delimiter for a block or scope.
                x[i] -= R[i, j] * x[j];  // EN: Execute a statement: x[i] -= R[i, j] * x[j];.
            }  // EN: Structure delimiter for a block or scope.
            x[i] /= R[i, i];  // EN: Execute a statement: x[i] /= R[i, i];.
        }  // EN: Structure delimiter for a block or scope.

        return x;  // EN: Return from the current function: return x;.
    }  // EN: Structure delimiter for a block or scope.

    // 用 QR 分解解最小平方問題
    static double[] QrLeastSquares(double[,] A, double[] b)  // EN: Execute line: static double[] QrLeastSquares(double[,] A, double[] b).
    {  // EN: Structure delimiter for a block or scope.
        var (Q, R) = QrDecomposition(A);  // EN: Execute a statement: var (Q, R) = QrDecomposition(A);.

        // 計算 Qᵀb
        int n = Q.GetLength(1);  // EN: Execute a statement: int n = Q.GetLength(1);.
        double[] Qt_b = new double[n];  // EN: Execute a statement: double[] Qt_b = new double[n];.
        for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
        {  // EN: Structure delimiter for a block or scope.
            double[] qj = GetColumn(Q, j);  // EN: Execute a statement: double[] qj = GetColumn(Q, j);.
            Qt_b[j] = DotProduct(qj, b);  // EN: Execute a statement: Qt_b[j] = DotProduct(qj, b);.
        }  // EN: Structure delimiter for a block or scope.

        // 解 Rx = Qᵀb
        return SolveUpperTriangular(R, Qt_b);  // EN: Return from the current function: return SolveUpperTriangular(R, Qt_b);.
    }  // EN: Structure delimiter for a block or scope.

    // 矩陣乘法
    static double[,] MatrixMultiply(double[,] A, double[,] B)  // EN: Execute line: static double[,] MatrixMultiply(double[,] A, double[,] B).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0);  // EN: Execute a statement: int m = A.GetLength(0);.
        int k = A.GetLength(1);  // EN: Execute a statement: int k = A.GetLength(1);.
        int n = B.GetLength(1);  // EN: Execute a statement: int n = B.GetLength(1);.

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

    // 矩陣轉置
    static double[,] Transpose(double[,] A)  // EN: Execute line: static double[,] Transpose(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0);  // EN: Execute a statement: int m = A.GetLength(0);.
        int n = A.GetLength(1);  // EN: Execute a statement: int n = A.GetLength(1);.
        double[,] result = new double[n, m];  // EN: Execute a statement: double[,] result = new double[n, m];.
        for (int i = 0; i < m; i++)  // EN: Loop control flow: for (int i = 0; i < m; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            {  // EN: Structure delimiter for a block or scope.
                result[j, i] = A[i, j];  // EN: Execute a statement: result[j, i] = A[i, j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    static void Main(string[] args)  // EN: Execute line: static void Main(string[] args).
    {  // EN: Structure delimiter for a block or scope.
        PrintSeparator("QR 分解示範 (C#)");  // EN: Execute a statement: PrintSeparator("QR 分解示範 (C#)");.

        // ========================================
        // 1. 基本 QR 分解
        // ========================================
        PrintSeparator("1. 基本 QR 分解");  // EN: Execute a statement: PrintSeparator("1. 基本 QR 分解");.

        double[,] A = {  // EN: Execute line: double[,] A = {.
            {1.0, 1.0},  // EN: Execute line: {1.0, 1.0},.
            {1.0, 0.0},  // EN: Execute line: {1.0, 0.0},.
            {0.0, 1.0}  // EN: Execute line: {0.0, 1.0}.
        };  // EN: Structure delimiter for a block or scope.

        Console.WriteLine("輸入矩陣 A：");  // EN: Execute a statement: Console.WriteLine("輸入矩陣 A：");.
        PrintMatrix("A", A);  // EN: Execute a statement: PrintMatrix("A", A);.

        var (Q, R) = QrDecomposition(A);  // EN: Execute a statement: var (Q, R) = QrDecomposition(A);.

        Console.WriteLine("\nQR 分解結果：");  // EN: Execute a statement: Console.WriteLine("\nQR 分解結果：");.
        PrintMatrix("Q", Q);  // EN: Execute a statement: PrintMatrix("Q", Q);.
        PrintMatrix("\nR", R);  // EN: Execute a statement: PrintMatrix("\nR", R);.

        // 驗證 QᵀQ = I
        var QT = Transpose(Q);  // EN: Execute a statement: var QT = Transpose(Q);.
        var QTQ = MatrixMultiply(QT, Q);  // EN: Execute a statement: var QTQ = MatrixMultiply(QT, Q);.
        Console.WriteLine("\n驗證 QᵀQ = I：");  // EN: Execute a statement: Console.WriteLine("\n驗證 QᵀQ = I：");.
        PrintMatrix("QᵀQ", QTQ);  // EN: Execute a statement: PrintMatrix("QᵀQ", QTQ);.

        // 驗證 A = QR
        var QR_result = MatrixMultiply(Q, R);  // EN: Execute a statement: var QR_result = MatrixMultiply(Q, R);.
        Console.WriteLine("\n驗證 A = QR：");  // EN: Execute a statement: Console.WriteLine("\n驗證 A = QR：");.
        PrintMatrix("QR", QR_result);  // EN: Execute a statement: PrintMatrix("QR", QR_result);.

        // ========================================
        // 2. 用 QR 解最小平方
        // ========================================
        PrintSeparator("2. 用 QR 解最小平方");  // EN: Execute a statement: PrintSeparator("2. 用 QR 解最小平方");.

        // 數據
        double[] t = {0.0, 1.0, 2.0};  // EN: Execute a statement: double[] t = {0.0, 1.0, 2.0};.
        double[] b = {1.0, 3.0, 4.0};  // EN: Execute a statement: double[] b = {1.0, 3.0, 4.0};.

        Console.WriteLine("數據點：");  // EN: Execute a statement: Console.WriteLine("數據點：");.
        for (int i = 0; i < t.Length; i++)  // EN: Loop control flow: for (int i = 0; i < t.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            Console.WriteLine($"  ({t[i]}, {b[i]})");  // EN: Execute a statement: Console.WriteLine($" ({t[i]}, {b[i]})");.
        }  // EN: Structure delimiter for a block or scope.

        // 設計矩陣
        double[,] A_ls = new double[t.Length, 2];  // EN: Execute a statement: double[,] A_ls = new double[t.Length, 2];.
        for (int i = 0; i < t.Length; i++)  // EN: Loop control flow: for (int i = 0; i < t.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            A_ls[i, 0] = 1.0;  // EN: Execute a statement: A_ls[i, 0] = 1.0;.
            A_ls[i, 1] = t[i];  // EN: Execute a statement: A_ls[i, 1] = t[i];.
        }  // EN: Structure delimiter for a block or scope.

        Console.WriteLine("\n設計矩陣 A：");  // EN: Execute a statement: Console.WriteLine("\n設計矩陣 A：");.
        PrintMatrix("A", A_ls);  // EN: Execute a statement: PrintMatrix("A", A_ls);.
        PrintVector("觀測值 b", b);  // EN: Execute a statement: PrintVector("觀測值 b", b);.

        // QR 分解
        var (Q_ls, R_ls) = QrDecomposition(A_ls);  // EN: Execute a statement: var (Q_ls, R_ls) = QrDecomposition(A_ls);.
        PrintMatrix("\nQ", Q_ls);  // EN: Execute a statement: PrintMatrix("\nQ", Q_ls);.
        PrintMatrix("R", R_ls);  // EN: Execute a statement: PrintMatrix("R", R_ls);.

        // 解最小平方
        double[] x = QrLeastSquares(A_ls, b);  // EN: Execute a statement: double[] x = QrLeastSquares(A_ls, b);.
        PrintVector("\n解 x", x);  // EN: Execute a statement: PrintVector("\n解 x", x);.

        Console.WriteLine($"\n最佳直線：y = {x[0]:F4} + {x[1]:F4}t");  // EN: Execute a statement: Console.WriteLine($"\n最佳直線：y = {x[0]:F4} + {x[1]:F4}t");.

        // ========================================
        // 3. 3×3 矩陣的 QR 分解
        // ========================================
        PrintSeparator("3. 3×3 矩陣的 QR 分解");  // EN: Execute a statement: PrintSeparator("3. 3×3 矩陣的 QR 分解");.

        double[,] A2 = {  // EN: Execute line: double[,] A2 = {.
            {1.0, 1.0, 0.0},  // EN: Execute line: {1.0, 1.0, 0.0},.
            {1.0, 0.0, 1.0},  // EN: Execute line: {1.0, 0.0, 1.0},.
            {0.0, 1.0, 1.0}  // EN: Execute line: {0.0, 1.0, 1.0}.
        };  // EN: Structure delimiter for a block or scope.

        Console.WriteLine("輸入矩陣 A：");  // EN: Execute a statement: Console.WriteLine("輸入矩陣 A：");.
        PrintMatrix("A", A2);  // EN: Execute a statement: PrintMatrix("A", A2);.

        var (Q2, R2) = QrDecomposition(A2);  // EN: Execute a statement: var (Q2, R2) = QrDecomposition(A2);.

        Console.WriteLine("\nQR 分解結果：");  // EN: Execute a statement: Console.WriteLine("\nQR 分解結果：");.
        PrintMatrix("Q", Q2);  // EN: Execute a statement: PrintMatrix("Q", Q2);.
        PrintMatrix("\nR", R2);  // EN: Execute a statement: PrintMatrix("\nR", R2);.

        // 總結
        PrintSeparator("總結");  // EN: Execute a statement: PrintSeparator("總結");.
        Console.WriteLine(@"  // EN: Execute line: Console.WriteLine(@".
QR 分解核心：  // EN: Execute line: QR 分解核心：.

1. A = QR  // EN: Execute line: 1. A = QR.
   - Q: 標準正交矩陣 (QᵀQ = I)  // EN: Execute line: - Q: 標準正交矩陣 (QᵀQ = I).
   - R: 上三角矩陣  // EN: Execute line: - R: 上三角矩陣.

2. Gram-Schmidt 演算法：  // EN: Execute line: 2. Gram-Schmidt 演算法：.
   - 對 A 的行向量正交化得到 Q  // EN: Execute line: - 對 A 的行向量正交化得到 Q.
   - R 的元素是投影係數  // EN: Execute line: - R 的元素是投影係數.

3. 用 QR 解最小平方：  // EN: Execute line: 3. 用 QR 解最小平方：.
   min ‖Ax - b‖²  // EN: Execute line: min ‖Ax - b‖².
   → Rx = Qᵀb  // EN: Execute line: → Rx = Qᵀb.

4. 優勢：  // EN: Execute line: 4. 優勢：.
   - 比正規方程更穩定  // EN: Execute line: - 比正規方程更穩定.
   - 避免計算 AᵀA  // EN: Execute line: - 避免計算 AᵀA.
");  // EN: Execute a statement: ");.

        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine("示範完成！");  // EN: Execute a statement: Console.WriteLine("示範完成！");.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
