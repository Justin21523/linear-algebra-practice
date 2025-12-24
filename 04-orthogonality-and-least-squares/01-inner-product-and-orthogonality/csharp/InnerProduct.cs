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

using System;  // EN: Execute a statement: using System;.
using System.Text;  // EN: Execute a statement: using System.Text;.

class InnerProduct  // EN: Execute line: class InnerProduct.
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
    // 向量運算
    // ========================================

    /// <summary>
    /// 計算兩向量的內積 (Dot Product)
    /// x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ
    /// </summary>
    static double DotProduct(double[] x, double[] y)  // EN: Execute line: static double DotProduct(double[] x, double[] y).
    {  // EN: Structure delimiter for a block or scope.
        if (x.Length != y.Length)  // EN: Conditional control flow: if (x.Length != y.Length).
            throw new ArgumentException("向量維度必須相同");  // EN: Execute a statement: throw new ArgumentException("向量維度必須相同");.

        double result = 0.0;  // EN: Execute a statement: double result = 0.0;.
        for (int i = 0; i < x.Length; i++)  // EN: Loop control flow: for (int i = 0; i < x.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            result += x[i] * y[i];  // EN: Execute a statement: result += x[i] * y[i];.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    /// <summary>
    /// 計算向量的長度（L2 範數）
    /// ‖x‖ = √(x · x)
    /// </summary>
    static double VectorNorm(double[] x)  // EN: Execute line: static double VectorNorm(double[] x).
    {  // EN: Structure delimiter for a block or scope.
        return Math.Sqrt(DotProduct(x, x));  // EN: Return from the current function: return Math.Sqrt(DotProduct(x, x));.
    }  // EN: Structure delimiter for a block or scope.

    /// <summary>
    /// 正規化向量為單位向量
    /// û = x / ‖x‖
    /// </summary>
    static double[] Normalize(double[] x)  // EN: Execute line: static double[] Normalize(double[] x).
    {  // EN: Structure delimiter for a block or scope.
        double norm = VectorNorm(x);  // EN: Execute a statement: double norm = VectorNorm(x);.
        if (norm < EPSILON)  // EN: Conditional control flow: if (norm < EPSILON).
            throw new ArgumentException("零向量無法正規化");  // EN: Execute a statement: throw new ArgumentException("零向量無法正規化");.

        double[] result = new double[x.Length];  // EN: Execute a statement: double[] result = new double[x.Length];.
        for (int i = 0; i < x.Length; i++)  // EN: Loop control flow: for (int i = 0; i < x.Length; i++).
        {  // EN: Structure delimiter for a block or scope.
            result[i] = x[i] / norm;  // EN: Execute a statement: result[i] = x[i] / norm;.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    /// <summary>
    /// 計算兩向量的夾角（弧度）
    /// cos θ = (x · y) / (‖x‖ ‖y‖)
    /// </summary>
    static double VectorAngle(double[] x, double[] y)  // EN: Execute line: static double VectorAngle(double[] x, double[] y).
    {  // EN: Structure delimiter for a block or scope.
        double dot = DotProduct(x, y);  // EN: Execute a statement: double dot = DotProduct(x, y);.
        double normX = VectorNorm(x);  // EN: Execute a statement: double normX = VectorNorm(x);.
        double normY = VectorNorm(y);  // EN: Execute a statement: double normY = VectorNorm(y);.

        if (normX < EPSILON || normY < EPSILON)  // EN: Conditional control flow: if (normX < EPSILON || normY < EPSILON).
            throw new ArgumentException("零向量沒有定義夾角");  // EN: Execute a statement: throw new ArgumentException("零向量沒有定義夾角");.

        double cosTheta = dot / (normX * normY);  // EN: Execute a statement: double cosTheta = dot / (normX * normY);.
        // 處理浮點數誤差
        cosTheta = Math.Max(-1.0, Math.Min(1.0, cosTheta));  // EN: Execute a statement: cosTheta = Math.Max(-1.0, Math.Min(1.0, cosTheta));.
        return Math.Acos(cosTheta);  // EN: Return from the current function: return Math.Acos(cosTheta);.
    }  // EN: Structure delimiter for a block or scope.

    /// <summary>
    /// 判斷兩向量是否正交
    /// x ⊥ y ⟺ x · y = 0
    /// </summary>
    static bool IsOrthogonal(double[] x, double[] y)  // EN: Execute line: static bool IsOrthogonal(double[] x, double[] y).
    {  // EN: Structure delimiter for a block or scope.
        return Math.Abs(DotProduct(x, y)) < EPSILON;  // EN: Return from the current function: return Math.Abs(DotProduct(x, y)) < EPSILON;.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 矩陣運算
    // ========================================

    /// <summary>
    /// 矩陣轉置
    /// </summary>
    static double[,] Transpose(double[,] A)  // EN: Execute line: static double[,] Transpose(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int rows = A.GetLength(0);  // EN: Execute a statement: int rows = A.GetLength(0);.
        int cols = A.GetLength(1);  // EN: Execute a statement: int cols = A.GetLength(1);.

        double[,] result = new double[cols, rows];  // EN: Execute a statement: double[,] result = new double[cols, rows];.
        for (int i = 0; i < rows; i++)  // EN: Loop control flow: for (int i = 0; i < rows; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < cols; j++)  // EN: Loop control flow: for (int j = 0; j < cols; j++).
            {  // EN: Structure delimiter for a block or scope.
                result[j, i] = A[i, j];  // EN: Execute a statement: result[j, i] = A[i, j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return result;  // EN: Return from the current function: return result;.
    }  // EN: Structure delimiter for a block or scope.

    /// <summary>
    /// 矩陣乘法
    /// </summary>
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

    /// <summary>
    /// 矩陣乘向量
    /// </summary>
    static double[] MatrixVectorMultiply(double[,] A, double[] x)  // EN: Execute line: static double[] MatrixVectorMultiply(double[,] A, double[] x).
    {  // EN: Structure delimiter for a block or scope.
        int m = A.GetLength(0);  // EN: Execute a statement: int m = A.GetLength(0);.
        int n = A.GetLength(1);  // EN: Execute a statement: int n = A.GetLength(1);.

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

    /// <summary>
    /// 判斷是否為單位矩陣
    /// </summary>
    static bool IsIdentity(double[,] A)  // EN: Execute line: static bool IsIdentity(double[,] A).
    {  // EN: Structure delimiter for a block or scope.
        int n = A.GetLength(0);  // EN: Execute a statement: int n = A.GetLength(0);.
        if (A.GetLength(1) != n) return false;  // EN: Conditional control flow: if (A.GetLength(1) != n) return false;.

        for (int i = 0; i < n; i++)  // EN: Loop control flow: for (int i = 0; i < n; i++).
        {  // EN: Structure delimiter for a block or scope.
            for (int j = 0; j < n; j++)  // EN: Loop control flow: for (int j = 0; j < n; j++).
            {  // EN: Structure delimiter for a block or scope.
                double expected = (i == j) ? 1.0 : 0.0;  // EN: Execute a statement: double expected = (i == j) ? 1.0 : 0.0;.
                if (Math.Abs(A[i, j] - expected) > EPSILON)  // EN: Conditional control flow: if (Math.Abs(A[i, j] - expected) > EPSILON).
                    return false;  // EN: Return from the current function: return false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
        return true;  // EN: Return from the current function: return true;.
    }  // EN: Structure delimiter for a block or scope.

    /// <summary>
    /// 判斷矩陣是否為正交矩陣
    /// QᵀQ = I
    /// </summary>
    static bool IsOrthogonalMatrix(double[,] Q)  // EN: Execute line: static bool IsOrthogonalMatrix(double[,] Q).
    {  // EN: Structure delimiter for a block or scope.
        double[,] QT = Transpose(Q);  // EN: Execute a statement: double[,] QT = Transpose(Q);.
        double[,] product = MatrixMultiply(QT, Q);  // EN: Execute a statement: double[,] product = MatrixMultiply(QT, Q);.
        return IsIdentity(product);  // EN: Return from the current function: return IsIdentity(product);.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 主程式
    // ========================================

    static void Main(string[] args)  // EN: Execute line: static void Main(string[] args).
    {  // EN: Structure delimiter for a block or scope.
        PrintSeparator("內積與正交性示範 (C#)\nInner Product & Orthogonality Demo");  // EN: Execute a statement: PrintSeparator("內積與正交性示範 (C#)\nInner Product & Orthogonality Demo");.

        // 1. 內積計算
        PrintSeparator("1. 內積計算 (Dot Product)");  // EN: Execute a statement: PrintSeparator("1. 內積計算 (Dot Product)");.

        double[] x = { 1.0, 2.0, 3.0 };  // EN: Execute a statement: double[] x = { 1.0, 2.0, 3.0 };.
        double[] y = { 4.0, 5.0, 6.0 };  // EN: Execute a statement: double[] y = { 4.0, 5.0, 6.0 };.

        PrintVector("x", x);  // EN: Execute a statement: PrintVector("x", x);.
        PrintVector("y", y);  // EN: Execute a statement: PrintVector("y", y);.
        Console.WriteLine($"\nx · y = {DotProduct(x, y)}");  // EN: Execute a statement: Console.WriteLine($"\nx · y = {DotProduct(x, y)}");.
        Console.WriteLine("計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32");  // EN: Execute a statement: Console.WriteLine("計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32");.

        // 2. 向量長度
        PrintSeparator("2. 向量長度 (Vector Norm)");  // EN: Execute a statement: PrintSeparator("2. 向量長度 (Vector Norm)");.

        double[] v = { 3.0, 4.0 };  // EN: Execute a statement: double[] v = { 3.0, 4.0 };.
        PrintVector("v", v);  // EN: Execute a statement: PrintVector("v", v);.
        Console.WriteLine($"‖v‖ = {VectorNorm(v)}");  // EN: Execute a statement: Console.WriteLine($"‖v‖ = {VectorNorm(v)}");.
        Console.WriteLine("計算：√(3² + 4²) = √25 = 5");  // EN: Execute a statement: Console.WriteLine("計算：√(3² + 4²) = √25 = 5");.

        // 正規化
        double[] vNormalized = Normalize(v);  // EN: Execute a statement: double[] vNormalized = Normalize(v);.
        Console.WriteLine("\n單位向量：");  // EN: Execute a statement: Console.WriteLine("\n單位向量：");.
        PrintVector("v̂ = v/‖v‖", vNormalized);  // EN: Execute a statement: PrintVector("v̂ = v/‖v‖", vNormalized);.
        Console.WriteLine($"‖v̂‖ = {VectorNorm(vNormalized)}");  // EN: Execute a statement: Console.WriteLine($"‖v̂‖ = {VectorNorm(vNormalized)}");.

        // 3. 向量夾角
        PrintSeparator("3. 向量夾角 (Vector Angle)");  // EN: Execute a statement: PrintSeparator("3. 向量夾角 (Vector Angle)");.

        double[] a = { 1.0, 0.0 };  // EN: Execute a statement: double[] a = { 1.0, 0.0 };.
        double[] b = { 1.0, 1.0 };  // EN: Execute a statement: double[] b = { 1.0, 1.0 };.

        PrintVector("a", a);  // EN: Execute a statement: PrintVector("a", a);.
        PrintVector("b", b);  // EN: Execute a statement: PrintVector("b", b);.

        double theta = VectorAngle(a, b);  // EN: Execute a statement: double theta = VectorAngle(a, b);.
        Console.WriteLine($"\n夾角 θ = {theta:F4} rad = {theta * 180 / Math.PI:F2}°");  // EN: Execute a statement: Console.WriteLine($"\n夾角 θ = {theta:F4} rad = {theta * 180 / Math.PI:F2….
        Console.WriteLine($"cos θ = {Math.Cos(theta):F4}");  // EN: Execute a statement: Console.WriteLine($"cos θ = {Math.Cos(theta):F4}");.
        Console.WriteLine("預期：cos 45° = 1/√2 ≈ 0.7071");  // EN: Execute a statement: Console.WriteLine("預期：cos 45° = 1/√2 ≈ 0.7071");.

        // 4. 正交性判斷
        PrintSeparator("4. 正交性判斷 (Orthogonality Check)");  // EN: Execute a statement: PrintSeparator("4. 正交性判斷 (Orthogonality Check)");.

        double[] u1 = { 1.0, 2.0 };  // EN: Execute a statement: double[] u1 = { 1.0, 2.0 };.
        double[] u2 = { -2.0, 1.0 };  // EN: Execute a statement: double[] u2 = { -2.0, 1.0 };.

        PrintVector("u₁", u1);  // EN: Execute a statement: PrintVector("u₁", u1);.
        PrintVector("u₂", u2);  // EN: Execute a statement: PrintVector("u₂", u2);.
        Console.WriteLine($"\nu₁ · u₂ = {DotProduct(u1, u2)}");  // EN: Execute a statement: Console.WriteLine($"\nu₁ · u₂ = {DotProduct(u1, u2)}");.
        Console.WriteLine($"u₁ ⊥ u₂？ {IsOrthogonal(u1, u2)}");  // EN: Execute a statement: Console.WriteLine($"u₁ ⊥ u₂？ {IsOrthogonal(u1, u2)}");.

        // 非正交
        double[] w1 = { 1.0, 1.0 };  // EN: Execute a statement: double[] w1 = { 1.0, 1.0 };.
        double[] w2 = { 1.0, 2.0 };  // EN: Execute a statement: double[] w2 = { 1.0, 2.0 };.

        Console.WriteLine("\n另一組：");  // EN: Execute a statement: Console.WriteLine("\n另一組：");.
        PrintVector("w₁", w1);  // EN: Execute a statement: PrintVector("w₁", w1);.
        PrintVector("w₂", w2);  // EN: Execute a statement: PrintVector("w₂", w2);.
        Console.WriteLine($"w₁ · w₂ = {DotProduct(w1, w2)}");  // EN: Execute a statement: Console.WriteLine($"w₁ · w₂ = {DotProduct(w1, w2)}");.
        Console.WriteLine($"w₁ ⊥ w₂？ {IsOrthogonal(w1, w2)}");  // EN: Execute a statement: Console.WriteLine($"w₁ ⊥ w₂？ {IsOrthogonal(w1, w2)}");.

        // 5. 正交矩陣
        PrintSeparator("5. 正交矩陣 (Orthogonal Matrix)");  // EN: Execute a statement: PrintSeparator("5. 正交矩陣 (Orthogonal Matrix)");.

        double angle = Math.PI / 4;  // EN: Execute a statement: double angle = Math.PI / 4;.
        double[,] Q = {  // EN: Execute line: double[,] Q = {.
            { Math.Cos(angle), -Math.Sin(angle) },  // EN: Execute line: { Math.Cos(angle), -Math.Sin(angle) },.
            { Math.Sin(angle), Math.Cos(angle) }  // EN: Execute line: { Math.Sin(angle), Math.Cos(angle) }.
        };  // EN: Structure delimiter for a block or scope.

        Console.WriteLine("旋轉矩陣（θ = 45°）：");  // EN: Execute a statement: Console.WriteLine("旋轉矩陣（θ = 45°）：");.
        PrintMatrix("Q", Q);  // EN: Execute a statement: PrintMatrix("Q", Q);.

        double[,] QT = Transpose(Q);  // EN: Execute a statement: double[,] QT = Transpose(Q);.
        PrintMatrix("\nQᵀ", QT);  // EN: Execute a statement: PrintMatrix("\nQᵀ", QT);.

        double[,] QTQ = MatrixMultiply(QT, Q);  // EN: Execute a statement: double[,] QTQ = MatrixMultiply(QT, Q);.
        PrintMatrix("\nQᵀQ", QTQ);  // EN: Execute a statement: PrintMatrix("\nQᵀQ", QTQ);.

        Console.WriteLine($"\nQ 是正交矩陣？ {IsOrthogonalMatrix(Q)}");  // EN: Execute a statement: Console.WriteLine($"\nQ 是正交矩陣？ {IsOrthogonalMatrix(Q)}");.

        // 驗證保長度
        double[] xVec = { 3.0, 4.0 };  // EN: Execute a statement: double[] xVec = { 3.0, 4.0 };.
        double[] Qx = MatrixVectorMultiply(Q, xVec);  // EN: Execute a statement: double[] Qx = MatrixVectorMultiply(Q, xVec);.

        Console.WriteLine("\n保長度驗證：");  // EN: Execute a statement: Console.WriteLine("\n保長度驗證：");.
        PrintVector("x", xVec);  // EN: Execute a statement: PrintVector("x", xVec);.
        PrintVector("Qx", Qx);  // EN: Execute a statement: PrintVector("Qx", Qx);.
        Console.WriteLine($"‖x‖ = {VectorNorm(xVec):F4}");  // EN: Execute a statement: Console.WriteLine($"‖x‖ = {VectorNorm(xVec):F4}");.
        Console.WriteLine($"‖Qx‖ = {VectorNorm(Qx):F4}");  // EN: Execute a statement: Console.WriteLine($"‖Qx‖ = {VectorNorm(Qx):F4}");.

        // 6. Cauchy-Schwarz 不等式
        PrintSeparator("6. Cauchy-Schwarz 不等式");  // EN: Execute a statement: PrintSeparator("6. Cauchy-Schwarz 不等式");.

        double[] csX = { 1.0, 2.0, 3.0 };  // EN: Execute a statement: double[] csX = { 1.0, 2.0, 3.0 };.
        double[] csY = { 4.0, 5.0, 6.0 };  // EN: Execute a statement: double[] csY = { 4.0, 5.0, 6.0 };.

        PrintVector("x", csX);  // EN: Execute a statement: PrintVector("x", csX);.
        PrintVector("y", csY);  // EN: Execute a statement: PrintVector("y", csY);.

        double leftSide = Math.Abs(DotProduct(csX, csY));  // EN: Execute a statement: double leftSide = Math.Abs(DotProduct(csX, csY));.
        double rightSide = VectorNorm(csX) * VectorNorm(csY);  // EN: Execute a statement: double rightSide = VectorNorm(csX) * VectorNorm(csY);.

        Console.WriteLine($"\n|x · y| = {leftSide:F4}");  // EN: Execute a statement: Console.WriteLine($"\n|x · y| = {leftSide:F4}");.
        Console.WriteLine($"‖x‖ ‖y‖ = {rightSide:F4}");  // EN: Execute a statement: Console.WriteLine($"‖x‖ ‖y‖ = {rightSide:F4}");.
        Console.WriteLine($"|x · y| ≤ ‖x‖ ‖y‖？ {leftSide <= rightSide + EPSILON}");  // EN: Execute a statement: Console.WriteLine($"|x · y| ≤ ‖x‖ ‖y‖？ {leftSide <= rightSide + EPSILON….

        // 總結
        PrintSeparator("總結");  // EN: Execute a statement: PrintSeparator("總結");.
        Console.WriteLine(@"  // EN: Execute line: Console.WriteLine(@".
內積與正交性的核心公式：  // EN: Execute line: 內積與正交性的核心公式：.

1. 內積：x · y = Σ xᵢyᵢ  // EN: Execute line: 1. 內積：x · y = Σ xᵢyᵢ.

2. 長度：‖x‖ = √(x · x)  // EN: Execute line: 2. 長度：‖x‖ = √(x · x).

3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖)  // EN: Execute line: 3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖).

4. 正交：x ⊥ y ⟺ x · y = 0  // EN: Execute line: 4. 正交：x ⊥ y ⟺ x · y = 0.

5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ  // EN: Execute line: 5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ.
");  // EN: Execute a statement: ");.

        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
        Console.WriteLine("示範完成！");  // EN: Execute a statement: Console.WriteLine("示範完成！");.
        Console.WriteLine(new string('=', 60));  // EN: Execute a statement: Console.WriteLine(new string('=', 60));.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.
