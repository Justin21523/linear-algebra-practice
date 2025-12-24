/**
 * 最小平方回歸 (Least Squares Regression)
 *
 * 本程式示範：
 * 1. 正規方程求解最小平方問題
 * 2. 簡單線性迴歸
 * 3. 殘差分析
 *
 * 執行：node least_squares.js
 */

// ========================================
// 輔助函數
// ========================================

function printSeparator(title) {  // EN: Execute line: function printSeparator(title) {.
    console.log();  // EN: Execute a statement: console.log();.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log(title);  // EN: Execute a statement: console.log(title);.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

function formatNumber(n) {  // EN: Execute line: function formatNumber(n) {.
    return n.toFixed(4);  // EN: Return from the current function: return n.toFixed(4);.
}  // EN: Structure delimiter for a block or scope.

function printVector(name, v) {  // EN: Execute line: function printVector(name, v) {.
    const formatted = v.map(x => formatNumber(x)).join(', ');  // EN: Execute a statement: const formatted = v.map(x => formatNumber(x)).join(', ');.
    console.log(`${name} = [${formatted}]`);  // EN: Execute a statement: console.log(`${name} = [${formatted}]`);.
}  // EN: Structure delimiter for a block or scope.

function printMatrix(name, M) {  // EN: Execute line: function printMatrix(name, M) {.
    console.log(`${name} =`);  // EN: Execute a statement: console.log(`${name} =`);.
    for (const row of M) {  // EN: Loop control flow: for (const row of M) {.
        const formatted = row.map(x => formatNumber(x).padStart(8)).join(', ');  // EN: Execute a statement: const formatted = row.map(x => formatNumber(x).padStart(8)).join(', ');.
        console.log(`  [${formatted}]`);  // EN: Execute a statement: console.log(` [${formatted}]`);.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 基本運算
// ========================================

function dotProduct(x, y) {  // EN: Execute line: function dotProduct(x, y) {.
    let result = 0;  // EN: Execute a statement: let result = 0;.
    for (let i = 0; i < x.length; i++) {  // EN: Loop control flow: for (let i = 0; i < x.length; i++) {.
        result += x[i] * y[i];  // EN: Execute a statement: result += x[i] * y[i];.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

function vectorNorm(x) {  // EN: Execute line: function vectorNorm(x) {.
    return Math.sqrt(dotProduct(x, x));  // EN: Return from the current function: return Math.sqrt(dotProduct(x, x));.
}  // EN: Structure delimiter for a block or scope.

function vectorSubtract(x, y) {  // EN: Execute line: function vectorSubtract(x, y) {.
    return x.map((xi, i) => xi - y[i]);  // EN: Return from the current function: return x.map((xi, i) => xi - y[i]);.
}  // EN: Structure delimiter for a block or scope.

function transpose(A) {  // EN: Execute line: function transpose(A) {.
    const m = A.length, n = A[0].length;  // EN: Execute a statement: const m = A.length, n = A[0].length;.
    const result = [];  // EN: Execute a statement: const result = [];.
    for (let j = 0; j < n; j++) {  // EN: Loop control flow: for (let j = 0; j < n; j++) {.
        result[j] = [];  // EN: Execute a statement: result[j] = [];.
        for (let i = 0; i < m; i++) {  // EN: Loop control flow: for (let i = 0; i < m; i++) {.
            result[j][i] = A[i][j];  // EN: Execute a statement: result[j][i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

function matrixMultiply(A, B) {  // EN: Execute line: function matrixMultiply(A, B) {.
    const m = A.length, k = B.length, n = B[0].length;  // EN: Execute a statement: const m = A.length, k = B.length, n = B[0].length;.
    const result = [];  // EN: Execute a statement: const result = [];.
    for (let i = 0; i < m; i++) {  // EN: Loop control flow: for (let i = 0; i < m; i++) {.
        result[i] = [];  // EN: Execute a statement: result[i] = [];.
        for (let j = 0; j < n; j++) {  // EN: Loop control flow: for (let j = 0; j < n; j++) {.
            result[i][j] = 0;  // EN: Execute a statement: result[i][j] = 0;.
            for (let p = 0; p < k; p++) {  // EN: Loop control flow: for (let p = 0; p < k; p++) {.
                result[i][j] += A[i][p] * B[p][j];  // EN: Execute a statement: result[i][j] += A[i][p] * B[p][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

function matrixVectorMultiply(A, x) {  // EN: Execute line: function matrixVectorMultiply(A, x) {.
    return A.map(row => dotProduct(row, x));  // EN: Return from the current function: return A.map(row => dotProduct(row, x));.
}  // EN: Structure delimiter for a block or scope.

function solve2x2(A, b) {  // EN: Execute line: function solve2x2(A, b) {.
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Execute a statement: const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];.
    return [  // EN: Return from the current function: return [.
        (A[1][1] * b[0] - A[0][1] * b[1]) / det,  // EN: Execute line: (A[1][1] * b[0] - A[0][1] * b[1]) / det,.
        (-A[1][0] * b[0] + A[0][0] * b[1]) / det  // EN: Execute line: (-A[1][0] * b[0] + A[0][0] * b[1]) / det.
    ];  // EN: Execute a statement: ];.
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 最小平方求解
// ========================================

function createDesignMatrixLinear(t) {  // EN: Execute line: function createDesignMatrixLinear(t) {.
    return t.map(ti => [1, ti]);  // EN: Return from the current function: return t.map(ti => [1, ti]);.
}  // EN: Structure delimiter for a block or scope.

function leastSquaresSolve(A, b) {  // EN: Execute line: function leastSquaresSolve(A, b) {.
    // AᵀA
    const AT = transpose(A);  // EN: Execute a statement: const AT = transpose(A);.
    const ATA = matrixMultiply(AT, A);  // EN: Execute a statement: const ATA = matrixMultiply(AT, A);.

    // Aᵀb
    const ATb = matrixVectorMultiply(AT, b);  // EN: Execute a statement: const ATb = matrixVectorMultiply(AT, b);.

    // 解
    const coefficients = solve2x2(ATA, ATb);  // EN: Execute a statement: const coefficients = solve2x2(ATA, ATb);.

    // 擬合值和殘差
    const fitted = matrixVectorMultiply(A, coefficients);  // EN: Execute a statement: const fitted = matrixVectorMultiply(A, coefficients);.
    const residual = vectorSubtract(b, fitted);  // EN: Execute a statement: const residual = vectorSubtract(b, fitted);.
    const residualNorm = vectorNorm(residual);  // EN: Execute a statement: const residualNorm = vectorNorm(residual);.

    // R²
    const bMean = b.reduce((a, c) => a + c, 0) / b.length;  // EN: Execute a statement: const bMean = b.reduce((a, c) => a + c, 0) / b.length;.
    const tss = b.reduce((a, bi) => a + (bi - bMean) ** 2, 0);  // EN: Execute a statement: const tss = b.reduce((a, bi) => a + (bi - bMean) ** 2, 0);.
    const rss = residual.reduce((a, ei) => a + ei ** 2, 0);  // EN: Execute a statement: const rss = residual.reduce((a, ei) => a + ei ** 2, 0);.
    const rSquared = tss > 0 ? 1 - rss / tss : 0;  // EN: Execute a statement: const rSquared = tss > 0 ? 1 - rss / tss : 0;.

    return { coefficients, fitted, residual, residualNorm, rSquared };  // EN: Return from the current function: return { coefficients, fitted, residual, residualNorm, rSquared };.
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 主程式
// ========================================

function main() {  // EN: Execute line: function main() {.
    printSeparator('最小平方回歸示範 (JavaScript)\nLeast Squares Regression Demo');  // EN: Execute a statement: printSeparator('最小平方回歸示範 (JavaScript)\nLeast Squares Regression Demo');.

    // 1. 簡單線性迴歸
    printSeparator('1. 簡單線性迴歸：y = C + Dt');  // EN: Execute a statement: printSeparator('1. 簡單線性迴歸：y = C + Dt');.

    const t = [0, 1, 2];  // EN: Execute a statement: const t = [0, 1, 2];.
    const b = [1, 3, 4];  // EN: Execute a statement: const b = [1, 3, 4];.

    console.log('數據點：');  // EN: Execute a statement: console.log('數據點：');.
    for (let i = 0; i < t.length; i++) {  // EN: Loop control flow: for (let i = 0; i < t.length; i++) {.
        console.log(`  t = ${t[i]}, b = ${b[i]}`);  // EN: Execute a statement: console.log(` t = ${t[i]}, b = ${b[i]}`);.
    }  // EN: Structure delimiter for a block or scope.

    const A = createDesignMatrixLinear(t);  // EN: Execute a statement: const A = createDesignMatrixLinear(t);.
    printMatrix('\n設計矩陣 A [1, t]', A);  // EN: Execute a statement: printMatrix('\n設計矩陣 A [1, t]', A);.
    printVector('觀測值 b', b);  // EN: Execute a statement: printVector('觀測值 b', b);.

    const result = leastSquaresSolve(A, b);  // EN: Execute a statement: const result = leastSquaresSolve(A, b);.

    console.log('\n【解】');  // EN: Execute a statement: console.log('\n【解】');.
    console.log(`C（截距）= ${formatNumber(result.coefficients[0])}`);  // EN: Execute a statement: console.log(`C（截距）= ${formatNumber(result.coefficients[0])}`);.
    console.log(`D（斜率）= ${formatNumber(result.coefficients[1])}`);  // EN: Execute a statement: console.log(`D（斜率）= ${formatNumber(result.coefficients[1])}`);.
    console.log(`\n最佳直線：y = ${formatNumber(result.coefficients[0])} + ${formatNumber(result.coefficients[1])}t`);  // EN: Execute a statement: console.log(`\n最佳直線：y = ${formatNumber(result.coefficients[0])} + ${for….

    printVector('\n擬合值 ŷ', result.fitted);  // EN: Execute a statement: printVector('\n擬合值 ŷ', result.fitted);.
    printVector('殘差 e', result.residual);  // EN: Execute a statement: printVector('殘差 e', result.residual);.
    console.log(`殘差範數 ‖e‖ = ${formatNumber(result.residualNorm)}`);  // EN: Execute a statement: console.log(`殘差範數 ‖e‖ = ${formatNumber(result.residualNorm)}`);.
    console.log(`R² = ${formatNumber(result.rSquared)}`);  // EN: Execute a statement: console.log(`R² = ${formatNumber(result.rSquared)}`);.

    // 2. 更多數據
    printSeparator('2. 更多數據點');  // EN: Execute a statement: printSeparator('2. 更多數據點');.

    const t2 = [0, 1, 2, 3, 4];  // EN: Execute a statement: const t2 = [0, 1, 2, 3, 4];.
    const b2 = [1, 2.5, 3.5, 5, 6.5];  // EN: Execute a statement: const b2 = [1, 2.5, 3.5, 5, 6.5];.

    console.log('數據點：');  // EN: Execute a statement: console.log('數據點：');.
    for (let i = 0; i < t2.length; i++) {  // EN: Loop control flow: for (let i = 0; i < t2.length; i++) {.
        console.log(`  (${t2[i]}, ${b2[i]})`);  // EN: Execute a statement: console.log(` (${t2[i]}, ${b2[i]})`);.
    }  // EN: Structure delimiter for a block or scope.

    const A2 = createDesignMatrixLinear(t2);  // EN: Execute a statement: const A2 = createDesignMatrixLinear(t2);.
    const result2 = leastSquaresSolve(A2, b2);  // EN: Execute a statement: const result2 = leastSquaresSolve(A2, b2);.

    console.log(`\n最佳直線：y = ${formatNumber(result2.coefficients[0])} + ${formatNumber(result2.coefficients[1])}t`);  // EN: Execute a statement: console.log(`\n最佳直線：y = ${formatNumber(result2.coefficients[0])} + ${fo….
    console.log(`R² = ${formatNumber(result2.rSquared)}`);  // EN: Execute a statement: console.log(`R² = ${formatNumber(result2.rSquared)}`);.

    // 總結
    printSeparator('總結');  // EN: Execute a statement: printSeparator('總結');.
    console.log(`  // EN: Execute line: console.log(`.
最小平方法核心公式：  // EN: Execute line: 最小平方法核心公式：.

1. 正規方程：AᵀA x̂ = Aᵀb  // EN: Execute line: 1. 正規方程：AᵀA x̂ = Aᵀb.

2. 解：x̂ = (AᵀA)⁻¹Aᵀb  // EN: Execute line: 2. 解：x̂ = (AᵀA)⁻¹Aᵀb.

3. 幾何意義：Ax̂ 是 b 在 C(A) 上的投影  // EN: Execute line: 3. 幾何意義：Ax̂ 是 b 在 C(A) 上的投影.

4. R² = 1 - RSS/TSS（越接近 1 越好）  // EN: Execute line: 4. R² = 1 - RSS/TSS（越接近 1 越好）.
`);  // EN: Execute a statement: `);.

    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log('示範完成！');  // EN: Execute a statement: console.log('示範完成！');.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

main();  // EN: Execute a statement: main();.
