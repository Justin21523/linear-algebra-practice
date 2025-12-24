/**
 * 投影 (Projections)
 *
 * 本程式示範：
 * 1. 投影到直線
 * 2. 投影矩陣及其性質
 * 3. 誤差向量的正交性驗證
 *
 * 執行：node projection.js
 */

const EPSILON = 1e-10;

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

function scalarMultiply(c, x) {
    return x.map(xi => c * xi);
}

function vectorSubtract(x, y) {
    return x.map((xi, i) => xi - y[i]);
}

function outerProduct(x, y) {
    return x.map(xi => y.map(yj => xi * yj));
}

function matrixScalarMultiply(c, A) {
    return A.map(row => row.map(val => c * val));
}

function matrixVectorMultiply(A, x) {
    return A.map(row => dotProduct(row, x));
}

function matrixMultiply(A, B) {
    const m = A.length, n = B[0].length, k = B.length;
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

// ========================================
// 投影函數
// ========================================

/**
 * 投影到直線
 * p = (aᵀb / aᵀa) * a
 */
function projectOntoLine(b, a) {
    const aTb = dotProduct(a, b);
    const aTa = dotProduct(a, a);

    const xHat = aTb / aTa;
    const p = scalarMultiply(xHat, a);
    const e = vectorSubtract(b, p);

    return {
        xHat,
        projection: p,
        error: e,
        errorNorm: vectorNorm(e)
    };
}

/**
 * 投影到直線的投影矩陣
 * P = aaᵀ / (aᵀa)
 */
function projectionMatrixLine(a) {
    const aTa = dotProduct(a, a);
    const aaT = outerProduct(a, a);
    return matrixScalarMultiply(1.0 / aTa, aaT);
}

/**
 * 驗證投影矩陣的性質
 */
function verifyProjectionMatrix(P, name = 'P') {
    const n = P.length;

    console.log(`\n驗證 ${name} 的性質：`);

    // 對稱性
    let isSymmetric = true;
    for (let i = 0; i < n && isSymmetric; i++) {
        for (let j = 0; j < n && isSymmetric; j++) {
            if (Math.abs(P[i][j] - P[j][i]) > EPSILON) {
                isSymmetric = false;
            }
        }
    }
    console.log(`  對稱性 (${name}ᵀ = ${name})：${isSymmetric}`);

    // 冪等性
    const P2 = matrixMultiply(P, P);
    let isIdempotent = true;
    for (let i = 0; i < n && isIdempotent; i++) {
        for (let j = 0; j < n && isIdempotent; j++) {
            if (Math.abs(P[i][j] - P2[i][j]) > EPSILON) {
                isIdempotent = false;
            }
        }
    }
    console.log(`  冪等性 (${name}² = ${name})：${isIdempotent}`);
}

// ========================================
// 主程式
// ========================================

function main() {
    printSeparator('投影示範 (JavaScript)\nProjection Demo');

    // 1. 投影到直線
    printSeparator('1. 投影到直線');

    const a = [1, 1];
    const b = [2, 0];

    printVector('方向 a', a);
    printVector('向量 b', b);

    const result = projectOntoLine(b, a);

    console.log(`\n投影係數 x̂ = (aᵀb)/(aᵀa) = ${formatNumber(result.xHat)}`);
    printVector('投影 p = x̂a', result.projection);
    printVector('誤差 e = b - p', result.error);

    // 驗證正交性
    const eDotA = dotProduct(result.error, a);
    console.log(`\n驗證 e ⊥ a：e · a = ${eDotA.toFixed(6)}`);
    console.log(`正交？ ${Math.abs(eDotA) < EPSILON}`);

    // 2. 投影矩陣
    printSeparator('2. 投影矩陣（到直線）');

    const a2 = [1, 2];
    printVector('方向 a', a2);

    const P = projectionMatrixLine(a2);
    printMatrix('\n投影矩陣 P = aaᵀ/(aᵀa)', P);

    verifyProjectionMatrix(P);

    // 用投影矩陣計算投影
    const b2 = [3, 4];
    printVector('\n向量 b', b2);

    const p = matrixVectorMultiply(P, b2);
    printVector('投影 p = Pb', p);

    // 3. 多個向量的投影
    printSeparator('3. 批次投影');

    const vectors = [[1, 0], [0, 1], [2, 2], [3, -1]];

    console.log('方向 a = [1, 2]');
    console.log('\n各向量投影結果：');

    for (const v of vectors) {
        const proj = projectOntoLine(v, a2);
        console.log(`  [${v[0]}, ${v[1]}] -> [${formatNumber(proj.projection[0])}, ${formatNumber(proj.projection[1])}]`);
    }

    // 總結
    printSeparator('總結');
    console.log(`
投影公式：

1. 投影到直線：
   p = (aᵀb / aᵀa) a
   P = aaᵀ / (aᵀa)

2. 投影到子空間：
   p = A(AᵀA)⁻¹Aᵀb
   P = A(AᵀA)⁻¹Aᵀ

3. 投影矩陣性質：
   Pᵀ = P（對稱）
   P² = P（冪等）
`);

    console.log('='.repeat(60));
    console.log('示範完成！');
    console.log('='.repeat(60));
}

main();
