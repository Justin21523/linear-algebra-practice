/**
 * 餘因子展開 (Cofactor Expansion)
 *
 * 編譯：dotnet build 或 csc Cofactor.cs
 * 執行：dotnet run 或 ./Cofactor.exe
 */

using System;  // EN: Execute a statement: using System;.
using System.Text;  // EN: Execute a statement: using System.Text;.

class Cofactor  // EN: Execute line: class Cofactor.
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
        int n = M.GetLength(0);  // EN: Execute a statement: int n = M.GetLength(0);.
        Console.WriteLine($"{name} =");  // EN: Execute a statement: Console.WriteLine($"{name} =");.
        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
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

    // 取得子矩陣
    static double[,] GetMinorMatrix(double[,] A, int row, int col)  // EN: Execute line: static double[,] GetMinorMatrix(double[,] A, int row, int col).
    {  // EN: Structure delimiter for a block or scope.
        int n = A.GetLength(0);  // EN: Execute a statement: int n = A.GetLength(0);.
        double[,] sub = new double[n - 1, n - 1];  // EN: Execute a statement: double[,] sub = new double[n - 1, n - 1];.
        int si = 0;  // EN: Execute a statement: int si = 0;.
        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
        {  // EN: Structure delimiter for a block or scope.
            if (i == row) continue;  // EN: Conditional control flow: if (i == row) continue;.
            int sj = 0;  // EN: Execute a statement: int sj = 0;.
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            {  // EN: Structure delimiter for a block or scope.
                if (j == col) continue;  // EN: Conditional control flow: if (j == col) continue;.
                sub[si, sj] = A[i, j];  // EN: Execute a statement: sub[si, sj] = A[i, j];.
                sj++;  // EN: Execute a statement: sj++;.
            }  // EN: Structure delimiter for a block or scope.
            si++;  // EN: Execute a statement: si++;.
        }  // EN: Structure delimiter for a block or scope.
        return sub;  // EN: Return from the current function: return sub;.
    }  // EN: Structure delimiter for a block or scope.

    // 行列式（遞迴餘因子展開）
    static double Determinant(double[,] A)  // EN: Execute line: static double Determinant(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int n = A.GetLength(0);  // EN: Execute a statement: int n = A.GetLength(0);.
        if (n == 1) return A[0, 0];  // EN: Conditional control flow: if (n == 1) return A[0, 0];.
        if (n == 2) return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];  // EN: Conditional control flow: if (n == 2) return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];.

        double det = 0.0;  // EN: Execute a statement: double det = 0.0;.
        for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
        {  // EN: Structure delimiter for a block or scope.
            double[,] sub = GetMinorMatrix(A, 0, j);  // EN: Execute a statement: double[,] sub = GetMinorMatrix(A, 0, j);.
            double sign = (j % 2 == 0) ? 1.0 : -1.0;  // EN: Execute a statement: double sign = (j % 2 == 0) ? 1.0 : -1.0;.
            det += sign * A[0, j] * Determinant(sub);  // EN: Execute a statement: det += sign * A[0, j] * Determinant(sub);.
        }  // EN: Structure delimiter for a block or scope.
        return det;  // EN: Return from the current function: return det;.
    }  // EN: Structure delimiter for a block or scope.

    // 子行列式
    static double Minor(double[,] A, int i, int j)  // EN: Execute line: static double Minor(double[,] A, int i, int j).
    {  // EN: Structure delimiter for a block or scope.
        return Determinant(GetMinorMatrix(A, i, j));  // EN: Return from the current function: return Determinant(GetMinorMatrix(A, i, j));.
    }  // EN: Structure delimiter for a block or scope.

    // 餘因子
    static double CofactorValue(double[,] A, int i, int j)  // EN: Execute line: static double CofactorValue(double[,] A, int i, int j).
    {  // EN: Structure delimiter for a block or scope.
        double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;  // EN: Execute a statement: double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;.
        return sign * Minor(A, i, j);  // EN: Return from the current function: return sign * Minor(A, i, j);.
    }  // EN: Structure delimiter for a block or scope.

    // 餘因子矩陣
    static double[,] CofactorMatrix(double[,] A)  // EN: Execute line: static double[,] CofactorMatrix(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int n = A.GetLength(0);  // EN: Execute a statement: int n = A.GetLength(0);.
        double[,] C = new double[n, n];  // EN: Execute a statement: double[,] C = new double[n, n];.
        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            {  // EN: Structure delimiter for a block or scope.
                C[i, j] = CofactorValue(A, i, j);  // EN: Execute a statement: C[i, j] = CofactorValue(A, i, j);.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return C;  // EN: Return from the current function: return C;.
    }  // EN: Structure delimiter for a block or scope.

    // 轉置
    static double[,] Transpose(double[,] A)  // EN: Execute line: static double[,] Transpose(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int n = A.GetLength(0);  // EN: Execute a statement: int n = A.GetLength(0);.
        double[,] T = new double[n, n];  // EN: Execute a statement: double[,] T = new double[n, n];.
        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            {  // EN: Structure delimiter for a block or scope.
                T[j, i] = A[i, j];  // EN: Execute a statement: T[j, i] = A[i, j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return T;  // EN: Return from the current function: return T;.
    }  // EN: Structure delimiter for a block or scope.

    // 伴隨矩陣
    static double[,] Adjugate(double[,] A)  // EN: Execute line: static double[,] Adjugate(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        return Transpose(CofactorMatrix(A));  // EN: Return from the current function: return Transpose(CofactorMatrix(A));.
    }  // EN: Structure delimiter for a block or scope.

    // 逆矩陣
    static double[,] Inverse(double[,] A)  // EN: Execute line: static double[,] Inverse(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        double det = Determinant(A);  // EN: Execute a statement: double det = Determinant(A);.
        double[,] adj = Adjugate(A);  // EN: Execute a statement: double[,] adj = Adjugate(A);.
        int n = A.GetLength(0);  // EN: Execute a statement: int n = A.GetLength(0);.
        double[,] inv = new double[n, n];  // EN: Execute a statement: double[,] inv = new double[n, n];.
        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            {  // EN: Structure delimiter for a block or scope.
                inv[i, j] = adj[i, j] / det;  // EN: Execute a statement: inv[i, j] = adj[i, j] / det;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return inv;  // EN: Return from the current function: return inv;.
    }  // EN: Structure delimiter for a block or scope.

    // 矩陣乘法
    static double[,] Multiply(double[,] A, double[,] B)  // EN: Execute line: static double[,] Multiply(double[,] A, double[,] B).
    {  // EN: Structure delimiter for a block or scope.
        int n = A.GetLength(0);  // EN: Execute a statement: int n = A.GetLength(0);.
        double[,] C = new double[n, n];  // EN: Execute a statement: double[,] C = new double[n, n];.
        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            {  // EN: Structure delimiter for a block or scope.
                for (int k = 0; k < n; k++)  // EN: Loop control flow: for (int k = 0; k < n; k++).
                {  // EN: Structure delimiter for a block or scope.
                    C[i, j] += A[i, k] * B[k, j];  // EN: Execute a statement: C[i, j] += A[i, k] * B[k, j];.
                }  // EN: Structure delimiter for a block or scope.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return C;  // EN: Return from the current function: return C;.
    }  // EN: Structure delimiter for a block or scope.

    static void Main(string[] args)  // EN: Execute line: static void Main(string[] args).
    {  // EN: Structure delimiter for a block or scope.
        PrintSeparator("餘因子展開示範 (C#)");  // EN: Execute a statement: PrintSeparator("餘因子展開示範 (C#)");.

        // ========================================
        // 1. 子行列式與餘因子
        // ========================================
        PrintSeparator("1. 子行列式與餘因子");  // EN: Execute a statement: PrintSeparator("1. 子行列式與餘因子");.

        double[,] A = {  // EN: Execute line: double[,] A = {.
            {1, 2, 3},  // EN: Execute line: {1, 2, 3},.
            {4, 5, 6},  // EN: Execute line: {4, 5, 6},.
            {7, 8, 9}  // EN: Execute line: {7, 8, 9}.
        };  // EN: Structure delimiter for a block or scope.

        PrintMatrix("A", A);  // EN: Execute a statement: PrintMatrix("A", A);.

        Console.WriteLine("\n所有餘因子 Cᵢⱼ：");  // EN: Execute a statement: Console.WriteLine("\n所有餘因子 Cᵢⱼ：");.
        for (int i = 0; i < 3; i++)  // EN: Loop control flow: for (int i = 0; i < 3; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < 3; j++)  // EN: Loop control flow: for (int j = 0; j < 3; j++).
            {  // EN: Structure delimiter for a block or scope.
                Console.Write($"  C{i+1}{j+1} = {CofactorValue(A, i, j),8:F4}");  // EN: Execute a statement: Console.Write($" C{i+1}{j+1} = {CofactorValue(A, i, j),8:F4}");.
            }  // EN: Structure delimiter for a block or scope.
            Console.WriteLine();  // EN: Execute a statement: Console.WriteLine();.
        }  // EN: Structure delimiter for a block or scope.

        // ========================================
        // 2. 餘因子展開
        // ========================================
        PrintSeparator("2. 餘因子展開計算行列式");  // EN: Execute a statement: PrintSeparator("2. 餘因子展開計算行列式");.

        Console.WriteLine("沿第一列展開：");  // EN: Execute a statement: Console.WriteLine("沿第一列展開：");.
        Console.WriteLine("det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃");  // EN: Execute a statement: Console.WriteLine("det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃");.
        Console.WriteLine($"       = {A[0,0]}×{CofactorValue(A, 0, 0)} + {A[0,1]}×{CofactorValue(A, 0, 1)} + {A[0,2]}×{CofactorValue(A, 0, 2)}");  // EN: Execute a statement: Console.WriteLine($" = {A[0,0]}×{CofactorValue(A, 0, 0)} + {A[0,1]}×{Co….
        Console.WriteLine($"       = {Determinant(A):F4}");  // EN: Execute a statement: Console.WriteLine($" = {Determinant(A):F4}");.

        // ========================================
        // 3. 餘因子矩陣與伴隨矩陣
        // ========================================
        PrintSeparator("3. 餘因子矩陣與伴隨矩陣");  // EN: Execute a statement: PrintSeparator("3. 餘因子矩陣與伴隨矩陣");.

        double[,] B = {  // EN: Execute line: double[,] B = {.
            {2, 1, 3},  // EN: Execute line: {2, 1, 3},.
            {1, 0, 2},  // EN: Execute line: {1, 0, 2},.
            {4, 1, 5}  // EN: Execute line: {4, 1, 5}.
        };  // EN: Structure delimiter for a block or scope.

        PrintMatrix("A", B);  // EN: Execute a statement: PrintMatrix("A", B);.
        Console.WriteLine($"\ndet(A) = {Determinant(B):F4}");  // EN: Execute a statement: Console.WriteLine($"\ndet(A) = {Determinant(B):F4}");.

        double[,] C = CofactorMatrix(B);  // EN: Execute a statement: double[,] C = CofactorMatrix(B);.
        PrintMatrix("\n餘因子矩陣 C", C);  // EN: Execute a statement: PrintMatrix("\n餘因子矩陣 C", C);.

        double[,] adj = Adjugate(B);  // EN: Execute a statement: double[,] adj = Adjugate(B);.
        PrintMatrix("\n伴隨矩陣 adj(A) = Cᵀ", adj);  // EN: Execute a statement: PrintMatrix("\n伴隨矩陣 adj(A) = Cᵀ", adj);.

        // ========================================
        // 4. 用伴隨矩陣求逆矩陣
        // ========================================
        PrintSeparator("4. 用伴隨矩陣求逆矩陣");  // EN: Execute a statement: PrintSeparator("4. 用伴隨矩陣求逆矩陣");.

        Console.WriteLine("A⁻¹ = adj(A) / det(A)");  // EN: Execute a statement: Console.WriteLine("A⁻¹ = adj(A) / det(A)");.

        double[,] B_inv = Inverse(B);  // EN: Execute a statement: double[,] B_inv = Inverse(B);.
        PrintMatrix("\nA⁻¹", B_inv);  // EN: Execute a statement: PrintMatrix("\nA⁻¹", B_inv);.

        // 驗證
        double[,] I = Multiply(B, B_inv);  // EN: Execute a statement: double[,] I = Multiply(B, B_inv);.
        PrintMatrix("\n驗證 A × A⁻¹", I);  // EN: Execute a statement: PrintMatrix("\n驗證 A × A⁻¹", I);.

        // ========================================
        // 5. 2×2 特例
        // ========================================
        PrintSeparator("5. 2×2 伴隨矩陣公式");  // EN: Execute a statement: PrintSeparator("5. 2×2 伴隨矩陣公式");.

        double[,] A2 = {{3, 4}, {5, 6}};  // EN: Execute a statement: double[,] A2 = {{3, 4}, {5, 6}};.
        PrintMatrix("A", A2);  // EN: Execute a statement: PrintMatrix("A", A2);.

        Console.WriteLine("\n對於 [[a,b],[c,d]]:");  // EN: Execute a statement: Console.WriteLine("\n對於 [[a,b],[c,d]]:");.
        Console.WriteLine($"adj(A) = [[d,-b],[-c,a]] = [[{A2[1,1]},{-A2[0,1]}],[{-A2[1,0]},{A2[0,0]}]]");  // EN: Execute a statement: Console.WriteLine($"adj(A) = [[d,-b],[-c,a]] = [[{A2[1,1]},{-A2[0,1]}],….

        double[,] adj2 = Adjugate(A2);  // EN: Execute a statement: double[,] adj2 = Adjugate(A2);.
        PrintMatrix("\n計算得到的 adj(A)", adj2);  // EN: Execute a statement: PrintMatrix("\n計算得到的 adj(A)", adj2);.

        // 總結
        PrintSeparator("總結");  // EN: Execute a statement: PrintSeparator("總結");.
        Console.WriteLine(@"  // EN: Execute line: Console.WriteLine(@".
餘因子展開公式：  // EN: Execute line: 餘因子展開公式：.
  Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ  // EN: Execute line: Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ.
  det(A) = Σⱼ aᵢⱼ Cᵢⱼ  // EN: Execute line: det(A) = Σⱼ aᵢⱼ Cᵢⱼ.

伴隨矩陣：  // EN: Execute line: 伴隨矩陣：.
  adj(A) = Cᵀ  // EN: Execute line: adj(A) = Cᵀ.

逆矩陣：  // EN: Execute line: 逆矩陣：.
  A⁻¹ = adj(A) / det(A)  // EN: Execute line: A⁻¹ = adj(A) / det(A).

時間複雜度：O(n!)  // EN: Execute line: 時間複雜度：O(n!).
");  // EN: Execute a statement: ");.

        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine("示範完成！");  // EN: Execute a statement: Console.WriteLine("示範完成！");.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
