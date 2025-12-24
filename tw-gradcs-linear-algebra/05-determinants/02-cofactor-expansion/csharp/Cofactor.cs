/**
 * 餘因子展開 (Cofactor Expansion)
 *
 * 編譯：dotnet build 或 csc Cofactor.cs
 * 執行：dotnet run 或 ./Cofactor.exe
 */

using System;
using System.Text;

class Cofactor
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
        int n = M.GetLength(0);
        Console.WriteLine($"{name} =");
        for (int i = 0; i < n; i++)
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

    // 取得子矩陣
    static double[,] GetMinorMatrix(double[,] A, int row, int col)
    {
        int n = A.GetLength(0);
        double[,] sub = new double[n - 1, n - 1];
        int si = 0;
        for (int i = 0; i < n; i++)
        {
            if (i == row) continue;
            int sj = 0;
            for (int j = 0; j < n; j++)
            {
                if (j == col) continue;
                sub[si, sj] = A[i, j];
                sj++;
            }
            si++;
        }
        return sub;
    }

    // 行列式（遞迴餘因子展開）
    static double Determinant(double[,] A)
    {
        int n = A.GetLength(0);
        if (n == 1) return A[0, 0];
        if (n == 2) return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];

        double det = 0.0;
        for (int j = 0; j < n; j++)
        {
            double[,] sub = GetMinorMatrix(A, 0, j);
            double sign = (j % 2 == 0) ? 1.0 : -1.0;
            det += sign * A[0, j] * Determinant(sub);
        }
        return det;
    }

    // 子行列式
    static double Minor(double[,] A, int i, int j)
    {
        return Determinant(GetMinorMatrix(A, i, j));
    }

    // 餘因子
    static double CofactorValue(double[,] A, int i, int j)
    {
        double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        return sign * Minor(A, i, j);
    }

    // 餘因子矩陣
    static double[,] CofactorMatrix(double[,] A)
    {
        int n = A.GetLength(0);
        double[,] C = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                C[i, j] = CofactorValue(A, i, j);
            }
        }
        return C;
    }

    // 轉置
    static double[,] Transpose(double[,] A)
    {
        int n = A.GetLength(0);
        double[,] T = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T[j, i] = A[i, j];
            }
        }
        return T;
    }

    // 伴隨矩陣
    static double[,] Adjugate(double[,] A)
    {
        return Transpose(CofactorMatrix(A));
    }

    // 逆矩陣
    static double[,] Inverse(double[,] A)
    {
        double det = Determinant(A);
        double[,] adj = Adjugate(A);
        int n = A.GetLength(0);
        double[,] inv = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                inv[i, j] = adj[i, j] / det;
            }
        }
        return inv;
    }

    // 矩陣乘法
    static double[,] Multiply(double[,] A, double[,] B)
    {
        int n = A.GetLength(0);
        double[,] C = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    C[i, j] += A[i, k] * B[k, j];
                }
            }
        }
        return C;
    }

    static void Main(string[] args)
    {
        PrintSeparator("餘因子展開示範 (C#)");

        // ========================================
        // 1. 子行列式與餘因子
        // ========================================
        PrintSeparator("1. 子行列式與餘因子");

        double[,] A = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        PrintMatrix("A", A);

        Console.WriteLine("\n所有餘因子 Cᵢⱼ：");
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Console.Write($"  C{i+1}{j+1} = {CofactorValue(A, i, j),8:F4}");
            }
            Console.WriteLine();
        }

        // ========================================
        // 2. 餘因子展開
        // ========================================
        PrintSeparator("2. 餘因子展開計算行列式");

        Console.WriteLine("沿第一列展開：");
        Console.WriteLine("det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃");
        Console.WriteLine($"       = {A[0,0]}×{CofactorValue(A, 0, 0)} + {A[0,1]}×{CofactorValue(A, 0, 1)} + {A[0,2]}×{CofactorValue(A, 0, 2)}");
        Console.WriteLine($"       = {Determinant(A):F4}");

        // ========================================
        // 3. 餘因子矩陣與伴隨矩陣
        // ========================================
        PrintSeparator("3. 餘因子矩陣與伴隨矩陣");

        double[,] B = {
            {2, 1, 3},
            {1, 0, 2},
            {4, 1, 5}
        };

        PrintMatrix("A", B);
        Console.WriteLine($"\ndet(A) = {Determinant(B):F4}");

        double[,] C = CofactorMatrix(B);
        PrintMatrix("\n餘因子矩陣 C", C);

        double[,] adj = Adjugate(B);
        PrintMatrix("\n伴隨矩陣 adj(A) = Cᵀ", adj);

        // ========================================
        // 4. 用伴隨矩陣求逆矩陣
        // ========================================
        PrintSeparator("4. 用伴隨矩陣求逆矩陣");

        Console.WriteLine("A⁻¹ = adj(A) / det(A)");

        double[,] B_inv = Inverse(B);
        PrintMatrix("\nA⁻¹", B_inv);

        // 驗證
        double[,] I = Multiply(B, B_inv);
        PrintMatrix("\n驗證 A × A⁻¹", I);

        // ========================================
        // 5. 2×2 特例
        // ========================================
        PrintSeparator("5. 2×2 伴隨矩陣公式");

        double[,] A2 = {{3, 4}, {5, 6}};
        PrintMatrix("A", A2);

        Console.WriteLine("\n對於 [[a,b],[c,d]]:");
        Console.WriteLine($"adj(A) = [[d,-b],[-c,a]] = [[{A2[1,1]},{-A2[0,1]}],[{-A2[1,0]},{A2[0,0]}]]");

        double[,] adj2 = Adjugate(A2);
        PrintMatrix("\n計算得到的 adj(A)", adj2);

        // 總結
        PrintSeparator("總結");
        Console.WriteLine(@"
餘因子展開公式：
  Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ
  det(A) = Σⱼ aᵢⱼ Cᵢⱼ

伴隨矩陣：
  adj(A) = Cᵀ

逆矩陣：
  A⁻¹ = adj(A) / det(A)

時間複雜度：O(n!)
");

        Console.WriteLine(new string('=', 60));
        Console.WriteLine("示範完成！");
        Console.WriteLine(new string('=', 60));
    }
}
