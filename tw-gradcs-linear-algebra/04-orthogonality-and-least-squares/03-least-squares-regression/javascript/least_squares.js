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

function printSeparator(title) {
    console.log();
    console.log('='.repeat(60));
    console.log(title);
    console.log('='.repeat(60));
}

function formatNumber(n) {
    return n.toFixed(4);
}

function printVector(name, v) {
    const formatted = v.map(x => formatNumber(x)).join(', ');
    console.log(`${name} = [${formatted}]`);
}

function printMatrix(name, M) {
    console.log(`${name} =`);
    for (const row of M) {
        const formatted = row.map(x => formatNumber(x).padStart(8)).join(', ');
        console.log(`  [${formatted}]`);
    }
}

// ========================================
// 基本運算
// ========================================

function dotProduct(x, y) {
    let result = 0;
    for (let i = 0; i < x.length; i++) {
        result += x[i] * y[i];
    }
    return result;
}

function vectorNorm(x) {
    return Math.sqrt(dotProduct(x, x));
}

function vectorSubtract(x, y) {
    return x.map((xi, i) => xi - y[i]);
}

function transpose(A) {
    const m = A.length, n = A[0].length;
    const result = [];
    for (let j = 0; j < n; j++) {
        result[j] = [];
        for (let i = 0; i < m; i++) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

function matrixMultiply(A, B) {
    const m = A.length, k = B.length, n = B[0].length;
    const result = [];
    for (let i = 0; i < m; i++) {
        result[i] = [];
        for (let j = 0; j < n; j++) {
            result[i][j] = 0;
            for (let p = 0; p < k; p++) {
                result[i][j] += A[i][p] * B[p][j];
            }
        }
    }
    return result;
}

function matrixVectorMultiply(A, x) {
    return A.map(row => dotProduct(row, x));
}

function solve2x2(A, b) {
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    return [
        (A[1][1] * b[0] - A[0][1] * b[1]) / det,
        (-A[1][0] * b[0] + A[0][0] * b[1]) / det
    ];
}

// ========================================
// 最小平方求解
// ========================================

function createDesignMatrixLinear(t) {
    return t.map(ti => [1, ti]);
}

function leastSquaresSolve(A, b) {
    // AᵀA
    const AT = transpose(A);
    const ATA = matrixMultiply(AT, A);

    // Aᵀb
    const ATb = matrixVectorMultiply(AT, b);

    // 解
    const coefficients = solve2x2(ATA, ATb);

    // 擬合值和殘差
    const fitted = matrixVectorMultiply(A, coefficients);
    const residual = vectorSubtract(b, fitted);
    const residualNorm = vectorNorm(residual);

    // R²
    const bMean = b.reduce((a, c) => a + c, 0) / b.length;
    const tss = b.reduce((a, bi) => a + (bi - bMean) ** 2, 0);
    const rss = residual.reduce((a, ei) => a + ei ** 2, 0);
    const rSquared = tss > 0 ? 1 - rss / tss : 0;

    return { coefficients, fitted, residual, residualNorm, rSquared };
}

// ========================================
// 主程式
// ========================================

function main() {
    printSeparator('最小平方回歸示範 (JavaScript)\nLeast Squares Regression Demo');

    // 1. 簡單線性迴歸
    printSeparator('1. 簡單線性迴歸：y = C + Dt');

    const t = [0, 1, 2];
    const b = [1, 3, 4];

    console.log('數據點：');
    for (let i = 0; i < t.length; i++) {
        console.log(`  t = ${t[i]}, b = ${b[i]}`);
    }

    const A = createDesignMatrixLinear(t);
    printMatrix('\n設計矩陣 A [1, t]', A);
    printVector('觀測值 b', b);

    const result = leastSquaresSolve(A, b);

    console.log('\n【解】');
    console.log(`C（截距）= ${formatNumber(result.coefficients[0])}`);
    console.log(`D（斜率）= ${formatNumber(result.coefficients[1])}`);
    console.log(`\n最佳直線：y = ${formatNumber(result.coefficients[0])} + ${formatNumber(result.coefficients[1])}t`);

    printVector('\n擬合值 ŷ', result.fitted);
    printVector('殘差 e', result.residual);
    console.log(`殘差範數 ‖e‖ = ${formatNumber(result.residualNorm)}`);
    console.log(`R² = ${formatNumber(result.rSquared)}`);

    // 2. 更多數據
    printSeparator('2. 更多數據點');

    const t2 = [0, 1, 2, 3, 4];
    const b2 = [1, 2.5, 3.5, 5, 6.5];

    console.log('數據點：');
    for (let i = 0; i < t2.length; i++) {
        console.log(`  (${t2[i]}, ${b2[i]})`);
    }

    const A2 = createDesignMatrixLinear(t2);
    const result2 = leastSquaresSolve(A2, b2);

    console.log(`\n最佳直線：y = ${formatNumber(result2.coefficients[0])} + ${formatNumber(result2.coefficients[1])}t`);
    console.log(`R² = ${formatNumber(result2.rSquared)}`);

    // 總結
    printSeparator('總結');
    console.log(`
最小平方法核心公式：

1. 正規方程：AᵀA x̂ = Aᵀb

2. 解：x̂ = (AᵀA)⁻¹Aᵀb

3. 幾何意義：Ax̂ 是 b 在 C(A) 上的投影

4. R² = 1 - RSS/TSS（越接近 1 越好）
`);

    console.log('='.repeat(60));
    console.log('示範完成！');
    console.log('='.repeat(60));
}

main();
