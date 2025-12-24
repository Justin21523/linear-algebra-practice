/**
 * 內積與正交性 (Inner Product and Orthogonality)
 *
 * 本程式示範：
 * 1. 向量內積計算
 * 2. 向量長度（範數）
 * 3. 向量夾角
 * 4. 正交性判斷
 * 5. 正交矩陣驗證
 *
 * 編譯：dotnet build 或 csc InnerProduct.cs
 * 執行：dotnet run 或 ./InnerProduct.exe
 */

using System;
using System.Text;

class InnerProduct
{
    private const double EPSILON = 1e-10;

    // ========================================
    // 輔助方法
    // ========================================

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
        int rows = M.GetLength(0);
        int cols = M.GetLength(1);

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

    // ========================================
    // 向量運算
    // ========================================

    /// <summary>
    /// 計算兩向量的內積 (Dot Product)
    /// x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ
    /// </summary>
    static double DotProduct(double[] x, double[] y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("向量維度必須相同");

        double result = 0.0;
        for (int i = 0; i < x.Length; i++)
        {
            result += x[i] * y[i];
        }
        return result;
    }

    /// <summary>
    /// 計算向量的長度（L2 範數）
    /// ‖x‖ = √(x · x)
    /// </summary>
    static double VectorNorm(double[] x)
    {
        return Math.Sqrt(DotProduct(x, x));
    }

    /// <summary>
    /// 正規化向量為單位向量
    /// û = x / ‖x‖
    /// </summary>
    static double[] Normalize(double[] x)
    {
        double norm = VectorNorm(x);
        if (norm < EPSILON)
            throw new ArgumentException("零向量無法正規化");

        double[] result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            result[i] = x[i] / norm;
        }
        return result;
    }

    /// <summary>
    /// 計算兩向量的夾角（弧度）
    /// cos θ = (x · y) / (‖x‖ ‖y‖)
    /// </summary>
    static double VectorAngle(double[] x, double[] y)
    {
        double dot = DotProduct(x, y);
        double normX = VectorNorm(x);
        double normY = VectorNorm(y);

        if (normX < EPSILON || normY < EPSILON)
            throw new ArgumentException("零向量沒有定義夾角");

        double cosTheta = dot / (normX * normY);
        // 處理浮點數誤差
        cosTheta = Math.Max(-1.0, Math.Min(1.0, cosTheta));
        return Math.Acos(cosTheta);
    }

    /// <summary>
    /// 判斷兩向量是否正交
    /// x ⊥ y ⟺ x · y = 0
    /// </summary>
    static bool IsOrthogonal(double[] x, double[] y)
    {
        return Math.Abs(DotProduct(x, y)) < EPSILON;
    }

    // ========================================
    // 矩陣運算
    // ========================================

    /// <summary>
    /// 矩陣轉置
    /// </summary>
    static double[,] Transpose(double[,] A)
    {
        int rows = A.GetLength(0);
        int cols = A.GetLength(1);

        double[,] result = new double[cols, rows];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[j, i] = A[i, j];
            }
        }
        return result;
    }

    /// <summary>
    /// 矩陣乘法
    /// </summary>
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

    /// <summary>
    /// 矩陣乘向量
    /// </summary>
    static double[] MatrixVectorMultiply(double[,] A, double[] x)
    {
        int m = A.GetLength(0);
        int n = A.GetLength(1);

        double[] result = new double[m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i] += A[i, j] * x[j];
            }
        }
        return result;
    }

    /// <summary>
    /// 判斷是否為單位矩陣
    /// </summary>
    static bool IsIdentity(double[,] A)
    {
        int n = A.GetLength(0);
        if (A.GetLength(1) != n) return false;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                if (Math.Abs(A[i, j] - expected) > EPSILON)
                    return false;
            }
        }
        return true;
    }

    /// <summary>
    /// 判斷矩陣是否為正交矩陣
    /// QᵀQ = I
    /// </summary>
    static bool IsOrthogonalMatrix(double[,] Q)
    {
        double[,] QT = Transpose(Q);
        double[,] product = MatrixMultiply(QT, Q);
        return IsIdentity(product);
    }

    // ========================================
    // 主程式
    // ========================================

    static void Main(string[] args)
    {
        PrintSeparator("內積與正交性示範 (C#)\nInner Product & Orthogonality Demo");

        // 1. 內積計算
        PrintSeparator("1. 內積計算 (Dot Product)");

        double[] x = { 1.0, 2.0, 3.0 };
        double[] y = { 4.0, 5.0, 6.0 };

        PrintVector("x", x);
        PrintVector("y", y);
        Console.WriteLine($"\nx · y = {DotProduct(x, y)}");
        Console.WriteLine("計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32");

        // 2. 向量長度
        PrintSeparator("2. 向量長度 (Vector Norm)");

        double[] v = { 3.0, 4.0 };
        PrintVector("v", v);
        Console.WriteLine($"‖v‖ = {VectorNorm(v)}");
        Console.WriteLine("計算：√(3² + 4²) = √25 = 5");

        // 正規化
        double[] vNormalized = Normalize(v);
        Console.WriteLine("\n單位向量：");
        PrintVector("v̂ = v/‖v‖", vNormalized);
        Console.WriteLine($"‖v̂‖ = {VectorNorm(vNormalized)}");

        // 3. 向量夾角
        PrintSeparator("3. 向量夾角 (Vector Angle)");

        double[] a = { 1.0, 0.0 };
        double[] b = { 1.0, 1.0 };

        PrintVector("a", a);
        PrintVector("b", b);

        double theta = VectorAngle(a, b);
        Console.WriteLine($"\n夾角 θ = {theta:F4} rad = {theta * 180 / Math.PI:F2}°");
        Console.WriteLine($"cos θ = {Math.Cos(theta):F4}");
        Console.WriteLine("預期：cos 45° = 1/√2 ≈ 0.7071");

        // 4. 正交性判斷
        PrintSeparator("4. 正交性判斷 (Orthogonality Check)");

        double[] u1 = { 1.0, 2.0 };
        double[] u2 = { -2.0, 1.0 };

        PrintVector("u₁", u1);
        PrintVector("u₂", u2);
        Console.WriteLine($"\nu₁ · u₂ = {DotProduct(u1, u2)}");
        Console.WriteLine($"u₁ ⊥ u₂？ {IsOrthogonal(u1, u2)}");

        // 非正交
        double[] w1 = { 1.0, 1.0 };
        double[] w2 = { 1.0, 2.0 };

        Console.WriteLine("\n另一組：");
        PrintVector("w₁", w1);
        PrintVector("w₂", w2);
        Console.WriteLine($"w₁ · w₂ = {DotProduct(w1, w2)}");
        Console.WriteLine($"w₁ ⊥ w₂？ {IsOrthogonal(w1, w2)}");

        // 5. 正交矩陣
        PrintSeparator("5. 正交矩陣 (Orthogonal Matrix)");

        double angle = Math.PI / 4;
        double[,] Q = {
            { Math.Cos(angle), -Math.Sin(angle) },
            { Math.Sin(angle), Math.Cos(angle) }
        };

        Console.WriteLine("旋轉矩陣（θ = 45°）：");
        PrintMatrix("Q", Q);

        double[,] QT = Transpose(Q);
        PrintMatrix("\nQᵀ", QT);

        double[,] QTQ = MatrixMultiply(QT, Q);
        PrintMatrix("\nQᵀQ", QTQ);

        Console.WriteLine($"\nQ 是正交矩陣？ {IsOrthogonalMatrix(Q)}");

        // 驗證保長度
        double[] xVec = { 3.0, 4.0 };
        double[] Qx = MatrixVectorMultiply(Q, xVec);

        Console.WriteLine("\n保長度驗證：");
        PrintVector("x", xVec);
        PrintVector("Qx", Qx);
        Console.WriteLine($"‖x‖ = {VectorNorm(xVec):F4}");
        Console.WriteLine($"‖Qx‖ = {VectorNorm(Qx):F4}");

        // 6. Cauchy-Schwarz 不等式
        PrintSeparator("6. Cauchy-Schwarz 不等式");

        double[] csX = { 1.0, 2.0, 3.0 };
        double[] csY = { 4.0, 5.0, 6.0 };

        PrintVector("x", csX);
        PrintVector("y", csY);

        double leftSide = Math.Abs(DotProduct(csX, csY));
        double rightSide = VectorNorm(csX) * VectorNorm(csY);

        Console.WriteLine($"\n|x · y| = {leftSide:F4}");
        Console.WriteLine($"‖x‖ ‖y‖ = {rightSide:F4}");
        Console.WriteLine($"|x · y| ≤ ‖x‖ ‖y‖？ {leftSide <= rightSide + EPSILON}");

        // 總結
        PrintSeparator("總結");
        Console.WriteLine(@"
內積與正交性的核心公式：

1. 內積：x · y = Σ xᵢyᵢ

2. 長度：‖x‖ = √(x · x)

3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖)

4. 正交：x ⊥ y ⟺ x · y = 0

5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ
");

        Console.WriteLine(new string('=', 60));
        Console.WriteLine("示範完成！");
        Console.WriteLine(new string('=', 60));
    }
}
