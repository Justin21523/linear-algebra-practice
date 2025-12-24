/**
 * 行列式的性質 (Determinant Properties)
 *
 * 編譯：dotnet build 或 csc Determinant.cs
 * 執行：dotnet run 或 ./Determinant.exe
 */

using System;
using System.Text;

class Determinant
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

    // 2×2 行列式
    static double Det2x2(double[,] A)
    {
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];
    }

    // 3×3 行列式
    static double Det3x3(double[,] A)
    {
        double a = A[0, 0], b = A[0, 1], c = A[0, 2];
        double d = A[1, 0], e = A[1, 1], f = A[1, 2];
        double g = A[2, 0], h = A[2, 1], i = A[2, 2];

        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    }

    // n×n 行列式（列運算化為上三角）
    static double DetNxN(double[,] A)
    {
        int n = A.GetLength(0);
        double[,] M = new double[n, n];

        // 複製矩陣
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                M[i, j] = A[i, j];

        int sign = 1;

        for (int col = 0; col < n; col++)
        {
            // 找主元
            int pivotRow = -1;
            for (int row = col; row < n; row++)
            {
                if (Math.Abs(M[row, col]) > 1e-10)
                {
                    pivotRow = row;
                    break;
                }
            }

            if (pivotRow == -1) return 0.0;

            // 列交換
            if (pivotRow != col)
            {
                for (int j = 0; j < n; j++)
                {
                    double temp = M[col, j];
                    M[col, j] = M[pivotRow, j];
                    M[pivotRow, j] = temp;
                }
                sign *= -1;
            }

            // 消去
            for (int row = col + 1; row < n; row++)
            {
                double factor = M[row, col] / M[col, col];
                for (int j = col; j < n; j++)
                {
                    M[row, j] -= factor * M[col, j];
                }
            }
        }

        double det = sign;
        for (int i = 0; i < n; i++)
        {
            det *= M[i, i];
        }

        return det;
    }

    // 矩陣乘法
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

    // 矩陣轉置
    static double[,] Transpose(double[,] A)
    {
        int m = A.GetLength(0), n = A.GetLength(1);
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

    // 純量乘矩陣
    static double[,] ScalarMultiply(double c, double[,] A)
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

    static void Main(string[] args)
    {
        PrintSeparator("行列式性質示範 (C#)");

        // ========================================
        // 1. 基本計算
        // ========================================
        PrintSeparator("1. 基本行列式計算");

        double[,] A2 = {{3, 8}, {4, 6}};
        PrintMatrix("A (2×2)", A2);
        Console.WriteLine($"det(A) = {Det2x2(A2):F4}");

        double[,] A3 = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 10}
        };
        PrintMatrix("\nA (3×3)", A3);
        Console.WriteLine($"det(A) = {Det3x3(A3):F4}");

        // ========================================
        // 2. 性質 1：det(I) = 1
        // ========================================
        PrintSeparator("2. 性質 1：det(I) = 1");

        double[,] I3 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        PrintMatrix("I₃", I3);
        Console.WriteLine($"det(I₃) = {Det3x3(I3):F4}");

        // ========================================
        // 3. 性質 2：列交換變號
        // ========================================
        PrintSeparator("3. 性質 2：列交換變號");

        double[,] A = {{1, 2}, {3, 4}};
        PrintMatrix("A", A);
        Console.WriteLine($"det(A) = {Det2x2(A):F4}");

        double[,] A_swap = {{3, 4}, {1, 2}};
        PrintMatrix("\nA（交換列）", A_swap);
        Console.WriteLine($"det(交換後) = {Det2x2(A_swap):F4}");
        Console.WriteLine("驗證：變號 ✓");

        // ========================================
        // 4. 乘積公式
        // ========================================
        PrintSeparator("4. 乘積公式：det(AB) = det(A)·det(B)");

        A = new double[,] {{1, 2}, {3, 4}};
        double[,] B = {{5, 6}, {7, 8}};
        double[,] AB = MatrixMultiply(A, B);

        PrintMatrix("A", A);
        PrintMatrix("B", B);
        PrintMatrix("AB", AB);

        double detA = Det2x2(A);
        double detB = Det2x2(B);
        double detAB = Det2x2(AB);

        Console.WriteLine($"\ndet(A) = {detA:F4}");
        Console.WriteLine($"det(B) = {detB:F4}");
        Console.WriteLine($"det(A)·det(B) = {detA * detB:F4}");
        Console.WriteLine($"det(AB) = {detAB:F4}");

        // ========================================
        // 5. 轉置公式
        // ========================================
        PrintSeparator("5. 轉置公式：det(Aᵀ) = det(A)");

        A3 = new double[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
        double[,] AT = Transpose(A3);

        PrintMatrix("A", A3);
        PrintMatrix("Aᵀ", AT);

        Console.WriteLine($"\ndet(A) = {Det3x3(A3):F4}");
        Console.WriteLine($"det(Aᵀ) = {Det3x3(AT):F4}");

        // ========================================
        // 6. 純量乘法
        // ========================================
        PrintSeparator("6. 純量乘法：det(cA) = cⁿ·det(A)");

        A = new double[,] {{1, 2}, {3, 4}};
        double c = 2;
        double[,] cA = ScalarMultiply(c, A);

        PrintMatrix("A (2×2)", A);
        Console.WriteLine($"c = {c}");
        PrintMatrix("cA", cA);

        detA = Det2x2(A);
        double detcA = Det2x2(cA);
        int n = 2;

        Console.WriteLine($"\ndet(A) = {detA:F4}");
        Console.WriteLine($"cⁿ·det(A) = {c}² × {detA:F4} = {Math.Pow(c, n) * detA:F4}");
        Console.WriteLine($"det(cA) = {detcA:F4}");

        // ========================================
        // 7. 上三角矩陣
        // ========================================
        PrintSeparator("7. 上三角矩陣：det = 對角線乘積");

        double[,] U = {{2, 3, 1}, {0, 4, 5}, {0, 0, 6}};
        PrintMatrix("U（上三角）", U);
        Console.WriteLine($"對角線乘積：2 × 4 × 6 = {2 * 4 * 6}");
        Console.WriteLine($"det(U) = {Det3x3(U):F4}");

        // ========================================
        // 8. 奇異矩陣
        // ========================================
        PrintSeparator("8. 奇異矩陣：det(A) = 0");

        double[,] A_singular = {{1, 2}, {2, 4}};
        PrintMatrix("A（列成比例）", A_singular);
        Console.WriteLine($"det(A) = {Det2x2(A_singular):F4}");
        Console.WriteLine("此矩陣不可逆");

        // 總結
        PrintSeparator("總結");
        Console.WriteLine(@"
行列式三大性質：
1. det(I) = 1
2. 列交換 → det 變號
3. 對單列線性

重要公式：
- det(AB) = det(A)·det(B)
- det(Aᵀ) = det(A)
- det(A⁻¹) = 1/det(A)
- det(cA) = cⁿ·det(A)
");

        Console.WriteLine(new string('=', 60));
        Console.WriteLine("示範完成！");
        Console.WriteLine(new string('=', 60));
    }
}
