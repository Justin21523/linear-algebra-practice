/**
 * 行列式的性質 (Determinant Properties)
 *
 * 編譯：dotnet build 或 csc Determinant.cs
 * 執行：dotnet run 或 ./Determinant.exe
 */

using System;  // EN: Execute a statement: using System;.
using System.Text;  // EN: Execute a statement: using System.Text;.

class Determinant  // EN: Execute line: class Determinant.
{  // EN: Structure delimiter for a block or scope.
    static void PrintSeparator(string title)  // EN: Execute line: static void PrintSeparator(string title).
    {  // EN: Structure delimiter for a block or scope.
        Console.WriteLine();  // EN: Execute a statement: Console.WriteLine();.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine(title);  // EN: Execute a statement: Console.WriteLine(title);.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
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

    // 2×2 行列式
    static double Det2x2(double[,] A)  // EN: Execute line: static double Det2x2(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];  // EN: Return from the current function: return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];.
    }  // EN: Structure delimiter for a block or scope.

    // 3×3 行列式
    static double Det3x3(double[,] A)  // EN: Execute line: static double Det3x3(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        double a = A[0, 0], b = A[0, 1], c = A[0, 2];  // EN: Execute a statement: double a = A[0, 0], b = A[0, 1], c = A[0, 2];.
        double d = A[1, 0], e = A[1, 1], f = A[1, 2];  // EN: Execute a statement: double d = A[1, 0], e = A[1, 1], f = A[1, 2];.
        double g = A[2, 0], h = A[2, 1], i = A[2, 2];  // EN: Execute a statement: double g = A[2, 0], h = A[2, 1], i = A[2, 2];.

        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);  // EN: Return from the current function: return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);.
    }  // EN: Structure delimiter for a block or scope.

    // n×n 行列式（列運算化為上三角）
    static double DetNxN(double[,] A)  // EN: Execute line: static double DetNxN(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int n = A.GetLength(0);  // EN: Execute a statement: int n = A.GetLength(0);.
        double[,] M = new double[n, n];  // EN: Execute a statement: double[,] M = new double[n, n];.

        // 複製矩陣
        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
                M[i, j] = A[i, j];  // EN: Execute a statement: M[i, j] = A[i, j];.

        int sign = 1;  // EN: Execute a statement: int sign = 1;.

        for (int col = 0; col < n; col++)  // EN: Loop control flow: for (int col = 0; col < n; col++).
        {  // EN: Structure delimiter for a block or scope.
            // 找主元
            int pivotRow = -1;  // EN: Execute a statement: int pivotRow = -1;.
            for (int row = col; row < n; row++)  // EN: Loop control flow: for (int row = col; row < n; row++).
            {  // EN: Structure delimiter for a block or scope.
                if (Math.Abs(M[row, col]) > 1e-10)  // EN: Conditional control flow: if (Math.Abs(M[row, col]) > 1e-10).
                {  // EN: Structure delimiter for a block or scope.
                    pivotRow = row;  // EN: Execute a statement: pivotRow = row;.
                    break;  // EN: Execute a statement: break;.
                }  // EN: Structure delimiter for a block or scope.
            }  // EN: Structure delimiter for a block or scope.

            if (pivotRow == -1) return 0.0;  // EN: Conditional control flow: if (pivotRow == -1) return 0.0;.

            // 列交換
            if (pivotRow != col)  // EN: Conditional control flow: if (pivotRow != col).
            {  // EN: Structure delimiter for a block or scope.
                for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
                {  // EN: Structure delimiter for a block or scope.
                    double temp = M[col, j];  // EN: Execute a statement: double temp = M[col, j];.
                    M[col, j] = M[pivotRow, j];  // EN: Execute a statement: M[col, j] = M[pivotRow, j];.
                    M[pivotRow, j] = temp;  // EN: Execute a statement: M[pivotRow, j] = temp;.
                }  // EN: Structure delimiter for a block or scope.
                sign *= -1;  // EN: Execute a statement: sign *= -1;.
            }  // EN: Structure delimiter for a block or scope.

            // 消去
            for (int row = col + 1; row < n; row++)  // EN: Loop control flow: for (int row = col + 1; row < n; row++).
            {  // EN: Structure delimiter for a block or scope.
                double factor = M[row, col] / M[col, col];  // EN: Execute a statement: double factor = M[row, col] / M[col, col];.
                for (int j = col; j < n; j++)  // EN: Loop control flow: for (int j = col; j < n; j++).
                {  // EN: Structure delimiter for a block or scope.
                    M[row, j] -= factor * M[col, j];  // EN: Execute a statement: M[row, j] -= factor * M[col, j];.
                }  // EN: Structure delimiter for a block or scope.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.

        double det = sign;  // EN: Execute a statement: double det = sign;.
        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
        {  // EN: Structure delimiter for a block or scope.
            det *= M[i, i];  // EN: Execute a statement: det *= M[i, i];.
        }  // EN: Structure delimiter for a block or scope.

        return det;  // EN: Return from the current function: return det;.
    }  // EN: Structure delimiter for a block or scope.

    // 矩陣乘法
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

    // 矩陣轉置
    static double[,] Transpose(double[,] A)  // EN: Execute line: static double[,] Transpose(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0), n = A.GetLength(1);  // EN: Execute a statement: int m = A.GetLength(0), n = A.GetLength(1);.
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

    // 純量乘矩陣
    static double[,] ScalarMultiply(double c, double[,] A)  // EN: Execute line: static double[,] ScalarMultiply(double c, double[,] A).
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

    static void Main(string[] args)  // EN: Execute line: static void Main(string[] args).
    {  // EN: Structure delimiter for a block or scope.
        PrintSeparator("行列式性質示範 (C#)");  // EN: Execute a statement: PrintSeparator("行列式性質示範 (C#)");.

        // ========================================
        // 1. 基本計算
        // ========================================
        PrintSeparator("1. 基本行列式計算");  // EN: Execute a statement: PrintSeparator("1. 基本行列式計算");.

        double[,] A2 = {{3, 8}, {4, 6}};  // EN: Execute a statement: double[,] A2 = {{3, 8}, {4, 6}};.
        PrintMatrix("A (2×2)", A2);  // EN: Execute a statement: PrintMatrix("A (2×2)", A2);.
        Console.WriteLine($"det(A) = {Det2x2(A2):F4}");  // EN: Execute a statement: Console.WriteLine($"det(A) = {Det2x2(A2):F4}");.

        double[,] A3 = {  // EN: Execute line: double[,] A3 = {.
            {1, 2, 3},  // EN: Execute line: {1, 2, 3},.
            {4, 5, 6},  // EN: Execute line: {4, 5, 6},.
            {7, 8, 10}  // EN: Execute line: {7, 8, 10}.
        };  // EN: Structure delimiter for a block or scope.
        PrintMatrix("\nA (3×3)", A3);  // EN: Execute a statement: PrintMatrix("\nA (3×3)", A3);.
        Console.WriteLine($"det(A) = {Det3x3(A3):F4}");  // EN: Execute a statement: Console.WriteLine($"det(A) = {Det3x3(A3):F4}");.

        // ========================================
        // 2. 性質 1：det(I) = 1
        // ========================================
        PrintSeparator("2. 性質 1：det(I) = 1");  // EN: Execute a statement: PrintSeparator("2. 性質 1：det(I) = 1");.

        double[,] I3 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};  // EN: Execute a statement: double[,] I3 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};.
        PrintMatrix("I₃", I3);  // EN: Execute a statement: PrintMatrix("I₃", I3);.
        Console.WriteLine($"det(I₃) = {Det3x3(I3):F4}");  // EN: Execute a statement: Console.WriteLine($"det(I₃) = {Det3x3(I3):F4}");.

        // ========================================
        // 3. 性質 2：列交換變號
        // ========================================
        PrintSeparator("3. 性質 2：列交換變號");  // EN: Execute a statement: PrintSeparator("3. 性質 2：列交換變號");.

        double[,] A = {{1, 2}, {3, 4}};  // EN: Execute a statement: double[,] A = {{1, 2}, {3, 4}};.
        PrintMatrix("A", A);  // EN: Execute a statement: PrintMatrix("A", A);.
        Console.WriteLine($"det(A) = {Det2x2(A):F4}");  // EN: Execute a statement: Console.WriteLine($"det(A) = {Det2x2(A):F4}");.

        double[,] A_swap = {{3, 4}, {1, 2}};  // EN: Execute a statement: double[,] A_swap = {{3, 4}, {1, 2}};.
        PrintMatrix("\nA（交換列）", A_swap);  // EN: Execute a statement: PrintMatrix("\nA（交換列）", A_swap);.
        Console.WriteLine($"det(交換後) = {Det2x2(A_swap):F4}");  // EN: Execute a statement: Console.WriteLine($"det(交換後) = {Det2x2(A_swap):F4}");.
        Console.WriteLine("驗證：變號 ✓");  // EN: Execute a statement: Console.WriteLine("驗證：變號 ✓");.

        // ========================================
        // 4. 乘積公式
        // ========================================
        PrintSeparator("4. 乘積公式：det(AB) = det(A)·det(B)");  // EN: Execute a statement: PrintSeparator("4. 乘積公式：det(AB) = det(A)·det(B)");.

        A = new double[,] {{1, 2}, {3, 4}};  // EN: Execute a statement: A = new double[,] {{1, 2}, {3, 4}};.
        double[,] B = {{5, 6}, {7, 8}};  // EN: Execute a statement: double[,] B = {{5, 6}, {7, 8}};.
        double[,] AB = MatrixMultiply(A, B);  // EN: Execute a statement: double[,] AB = MatrixMultiply(A, B);.

        PrintMatrix("A", A);  // EN: Execute a statement: PrintMatrix("A", A);.
        PrintMatrix("B", B);  // EN: Execute a statement: PrintMatrix("B", B);.
        PrintMatrix("AB", AB);  // EN: Execute a statement: PrintMatrix("AB", AB);.

        double detA = Det2x2(A);  // EN: Execute a statement: double detA = Det2x2(A);.
        double detB = Det2x2(B);  // EN: Execute a statement: double detB = Det2x2(B);.
        double detAB = Det2x2(AB);  // EN: Execute a statement: double detAB = Det2x2(AB);.

        Console.WriteLine($"\ndet(A) = {detA:F4}");  // EN: Execute a statement: Console.WriteLine($"\ndet(A) = {detA:F4}");.
        Console.WriteLine($"det(B) = {detB:F4}");  // EN: Execute a statement: Console.WriteLine($"det(B) = {detB:F4}");.
        Console.WriteLine($"det(A)·det(B) = {detA * detB:F4}");  // EN: Execute a statement: Console.WriteLine($"det(A)·det(B) = {detA * detB:F4}");.
        Console.WriteLine($"det(AB) = {detAB:F4}");  // EN: Execute a statement: Console.WriteLine($"det(AB) = {detAB:F4}");.

        // ========================================
        // 5. 轉置公式
        // ========================================
        PrintSeparator("5. 轉置公式：det(Aᵀ) = det(A)");  // EN: Execute a statement: PrintSeparator("5. 轉置公式：det(Aᵀ) = det(A)");.

        A3 = new double[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};  // EN: Execute a statement: A3 = new double[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};.
        double[,] AT = Transpose(A3);  // EN: Execute a statement: double[,] AT = Transpose(A3);.

        PrintMatrix("A", A3);  // EN: Execute a statement: PrintMatrix("A", A3);.
        PrintMatrix("Aᵀ", AT);  // EN: Execute a statement: PrintMatrix("Aᵀ", AT);.

        Console.WriteLine($"\ndet(A) = {Det3x3(A3):F4}");  // EN: Execute a statement: Console.WriteLine($"\ndet(A) = {Det3x3(A3):F4}");.
        Console.WriteLine($"det(Aᵀ) = {Det3x3(AT):F4}");  // EN: Execute a statement: Console.WriteLine($"det(Aᵀ) = {Det3x3(AT):F4}");.

        // ========================================
        // 6. 純量乘法
        // ========================================
        PrintSeparator("6. 純量乘法：det(cA) = cⁿ·det(A)");  // EN: Execute a statement: PrintSeparator("6. 純量乘法：det(cA) = cⁿ·det(A)");.

        A = new double[,] {{1, 2}, {3, 4}};  // EN: Execute a statement: A = new double[,] {{1, 2}, {3, 4}};.
        double c = 2;  // EN: Execute a statement: double c = 2;.
        double[,] cA = ScalarMultiply(c, A);  // EN: Execute a statement: double[,] cA = ScalarMultiply(c, A);.

        PrintMatrix("A (2×2)", A);  // EN: Execute a statement: PrintMatrix("A (2×2)", A);.
        Console.WriteLine($"c = {c}");  // EN: Execute a statement: Console.WriteLine($"c = {c}");.
        PrintMatrix("cA", cA);  // EN: Execute a statement: PrintMatrix("cA", cA);.

        detA = Det2x2(A);  // EN: Execute a statement: detA = Det2x2(A);.
        double detcA = Det2x2(cA);  // EN: Execute a statement: double detcA = Det2x2(cA);.
        int n = 2;  // EN: Execute a statement: int n = 2;.

        Console.WriteLine($"\ndet(A) = {detA:F4}");  // EN: Execute a statement: Console.WriteLine($"\ndet(A) = {detA:F4}");.
        Console.WriteLine($"cⁿ·det(A) = {c}² × {detA:F4} = {Math.Pow(c, n) * detA:F4}");  // EN: Execute a statement: Console.WriteLine($"cⁿ·det(A) = {c}² × {detA:F4} = {Math.Pow(c, n) * de….
        Console.WriteLine($"det(cA) = {detcA:F4}");  // EN: Execute a statement: Console.WriteLine($"det(cA) = {detcA:F4}");.

        // ========================================
        // 7. 上三角矩陣
        // ========================================
        PrintSeparator("7. 上三角矩陣：det = 對角線乘積");  // EN: Execute a statement: PrintSeparator("7. 上三角矩陣：det = 對角線乘積");.

        double[,] U = {{2, 3, 1}, {0, 4, 5}, {0, 0, 6}};  // EN: Execute a statement: double[,] U = {{2, 3, 1}, {0, 4, 5}, {0, 0, 6}};.
        PrintMatrix("U（上三角）", U);  // EN: Execute a statement: PrintMatrix("U（上三角）", U);.
        Console.WriteLine($"對角線乘積：2 × 4 × 6 = {2 * 4 * 6}");  // EN: Execute a statement: Console.WriteLine($"對角線乘積：2 × 4 × 6 = {2 * 4 * 6}");.
        Console.WriteLine($"det(U) = {Det3x3(U):F4}");  // EN: Execute a statement: Console.WriteLine($"det(U) = {Det3x3(U):F4}");.

        // ========================================
        // 8. 奇異矩陣
        // ========================================
        PrintSeparator("8. 奇異矩陣：det(A) = 0");  // EN: Execute a statement: PrintSeparator("8. 奇異矩陣：det(A) = 0");.

        double[,] A_singular = {{1, 2}, {2, 4}};  // EN: Execute a statement: double[,] A_singular = {{1, 2}, {2, 4}};.
        PrintMatrix("A（列成比例）", A_singular);  // EN: Execute a statement: PrintMatrix("A（列成比例）", A_singular);.
        Console.WriteLine($"det(A) = {Det2x2(A_singular):F4}");  // EN: Execute a statement: Console.WriteLine($"det(A) = {Det2x2(A_singular):F4}");.
        Console.WriteLine("此矩陣不可逆");  // EN: Execute a statement: Console.WriteLine("此矩陣不可逆");.

        // 總結
        PrintSeparator("總結");  // EN: Execute a statement: PrintSeparator("總結");.
        Console.WriteLine(@"  // EN: Execute line: Console.WriteLine(@".
行列式三大性質：  // EN: Execute line: 行列式三大性質：.
1. det(I) = 1  // EN: Execute line: 1. det(I) = 1.
2. 列交換 → det 變號  // EN: Execute line: 2. 列交換 → det 變號.
3. 對單列線性  // EN: Execute line: 3. 對單列線性.

重要公式：  // EN: Execute line: 重要公式：.
- det(AB) = det(A)·det(B)  // EN: Execute line: - det(AB) = det(A)·det(B).
- det(Aᵀ) = det(A)  // EN: Execute line: - det(Aᵀ) = det(A).
- det(A⁻¹) = 1/det(A)  // EN: Execute line: - det(A⁻¹) = 1/det(A).
- det(cA) = cⁿ·det(A)  // EN: Execute line: - det(cA) = cⁿ·det(A).
");  // EN: Execute a statement: ");.

        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine("示範完成！");  // EN: Execute a statement: Console.WriteLine("示範完成！");.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
