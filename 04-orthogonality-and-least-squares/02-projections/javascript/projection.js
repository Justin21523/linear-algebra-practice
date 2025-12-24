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

const EPSILON = 1e-10;  // EN: Execute a statement: const EPSILON = 1e-10;.

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

function scalarMultiply(c, x) {  // EN: Execute line: function scalarMultiply(c, x) {.
    return x.map(xi => c * xi);  // EN: Return from the current function: return x.map(xi => c * xi);.
}  // EN: Structure delimiter for a block or scope.

function vectorSubtract(x, y) {  // EN: Execute line: function vectorSubtract(x, y) {.
    return x.map((xi, i) => xi - y[i]);  // EN: Return from the current function: return x.map((xi, i) => xi - y[i]);.
}  // EN: Structure delimiter for a block or scope.

function outerProduct(x, y) {  // EN: Execute line: function outerProduct(x, y) {.
    return x.map(xi => y.map(yj => xi * yj));  // EN: Return from the current function: return x.map(xi => y.map(yj => xi * yj));.
}  // EN: Structure delimiter for a block or scope.

function matrixScalarMultiply(c, A) {  // EN: Execute line: function matrixScalarMultiply(c, A) {.
    return A.map(row => row.map(val => c * val));  // EN: Return from the current function: return A.map(row => row.map(val => c * val));.
}  // EN: Structure delimiter for a block or scope.

function matrixVectorMultiply(A, x) {  // EN: Execute line: function matrixVectorMultiply(A, x) {.
    return A.map(row => dotProduct(row, x));  // EN: Return from the current function: return A.map(row => dotProduct(row, x));.
}  // EN: Structure delimiter for a block or scope.

function matrixMultiply(A, B) {  // EN: Execute line: function matrixMultiply(A, B) {.
    const m = A.length, n = B[0].length, k = B.length;  // EN: Execute a statement: const m = A.length, n = B[0].length, k = B.length;.
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

// ========================================
// 投影函數
// ========================================

/**
 * 投影到直線
 * p = (aᵀb / aᵀa) * a
 */
function projectOntoLine(b, a) {  // EN: Execute line: function projectOntoLine(b, a) {.
    const aTb = dotProduct(a, b);  // EN: Execute a statement: const aTb = dotProduct(a, b);.
    const aTa = dotProduct(a, a);  // EN: Execute a statement: const aTa = dotProduct(a, a);.

    const xHat = aTb / aTa;  // EN: Execute a statement: const xHat = aTb / aTa;.
    const p = scalarMultiply(xHat, a);  // EN: Execute a statement: const p = scalarMultiply(xHat, a);.
    const e = vectorSubtract(b, p);  // EN: Execute a statement: const e = vectorSubtract(b, p);.

    return {  // EN: Return from the current function: return {.
        xHat,  // EN: Execute line: xHat,.
        projection: p,  // EN: Execute line: projection: p,.
        error: e,  // EN: Execute line: error: e,.
        errorNorm: vectorNorm(e)  // EN: Execute line: errorNorm: vectorNorm(e).
    };  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

/**
 * 投影到直線的投影矩陣
 * P = aaᵀ / (aᵀa)
 */
function projectionMatrixLine(a) {  // EN: Execute line: function projectionMatrixLine(a) {.
    const aTa = dotProduct(a, a);  // EN: Execute a statement: const aTa = dotProduct(a, a);.
    const aaT = outerProduct(a, a);  // EN: Execute a statement: const aaT = outerProduct(a, a);.
    return matrixScalarMultiply(1.0 / aTa, aaT);  // EN: Return from the current function: return matrixScalarMultiply(1.0 / aTa, aaT);.
}  // EN: Structure delimiter for a block or scope.

/**
 * 驗證投影矩陣的性質
 */
function verifyProjectionMatrix(P, name = 'P') {  // EN: Execute line: function verifyProjectionMatrix(P, name = 'P') {.
    const n = P.length;  // EN: Execute a statement: const n = P.length;.

    console.log(`\n驗證 ${name} 的性質：`);  // EN: Execute a statement: console.log(`\n驗證 ${name} 的性質：`);.

    // 對稱性
    let isSymmetric = true;  // EN: Execute a statement: let isSymmetric = true;.
    for (let i = 0; i < n && isSymmetric; i++) {  // EN: Loop control flow: for (let i = 0; i < n && isSymmetric; i++) {.
        for (let j = 0; j < n && isSymmetric; j++) {  // EN: Loop control flow: for (let j = 0; j < n && isSymmetric; j++) {.
            if (Math.abs(P[i][j] - P[j][i]) > EPSILON) {  // EN: Conditional control flow: if (Math.abs(P[i][j] - P[j][i]) > EPSILON) {.
                isSymmetric = false;  // EN: Execute a statement: isSymmetric = false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    console.log(`  對稱性 (${name}ᵀ = ${name})：${isSymmetric}`);  // EN: Execute a statement: console.log(` 對稱性 (${name}ᵀ = ${name})：${isSymmetric}`);.

    // 冪等性
    const P2 = matrixMultiply(P, P);  // EN: Execute a statement: const P2 = matrixMultiply(P, P);.
    let isIdempotent = true;  // EN: Execute a statement: let isIdempotent = true;.
    for (let i = 0; i < n && isIdempotent; i++) {  // EN: Loop control flow: for (let i = 0; i < n && isIdempotent; i++) {.
        for (let j = 0; j < n && isIdempotent; j++) {  // EN: Loop control flow: for (let j = 0; j < n && isIdempotent; j++) {.
            if (Math.abs(P[i][j] - P2[i][j]) > EPSILON) {  // EN: Conditional control flow: if (Math.abs(P[i][j] - P2[i][j]) > EPSILON) {.
                isIdempotent = false;  // EN: Execute a statement: isIdempotent = false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    console.log(`  冪等性 (${name}² = ${name})：${isIdempotent}`);  // EN: Execute a statement: console.log(` 冪等性 (${name}² = ${name})：${isIdempotent}`);.
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 主程式
// ========================================

function main() {  // EN: Execute line: function main() {.
    printSeparator('投影示範 (JavaScript)\nProjection Demo');  // EN: Execute a statement: printSeparator('投影示範 (JavaScript)\nProjection Demo');.

    // 1. 投影到直線
    printSeparator('1. 投影到直線');  // EN: Execute a statement: printSeparator('1. 投影到直線');.

    const a = [1, 1];  // EN: Execute a statement: const a = [1, 1];.
    const b = [2, 0];  // EN: Execute a statement: const b = [2, 0];.

    printVector('方向 a', a);  // EN: Execute a statement: printVector('方向 a', a);.
    printVector('向量 b', b);  // EN: Execute a statement: printVector('向量 b', b);.

    const result = projectOntoLine(b, a);  // EN: Execute a statement: const result = projectOntoLine(b, a);.

    console.log(`\n投影係數 x̂ = (aᵀb)/(aᵀa) = ${formatNumber(result.xHat)}`);  // EN: Execute a statement: console.log(`\n投影係數 x̂ = (aᵀb)/(aᵀa) = ${formatNumber(result.xHat)}`);.
    printVector('投影 p = x̂a', result.projection);  // EN: Execute a statement: printVector('投影 p = x̂a', result.projection);.
    printVector('誤差 e = b - p', result.error);  // EN: Execute a statement: printVector('誤差 e = b - p', result.error);.

    // 驗證正交性
    const eDotA = dotProduct(result.error, a);  // EN: Execute a statement: const eDotA = dotProduct(result.error, a);.
    console.log(`\n驗證 e ⊥ a：e · a = ${eDotA.toFixed(6)}`);  // EN: Execute a statement: console.log(`\n驗證 e ⊥ a：e · a = ${eDotA.toFixed(6)}`);.
    console.log(`正交？ ${Math.abs(eDotA) < EPSILON}`);  // EN: Execute a statement: console.log(`正交？ ${Math.abs(eDotA) < EPSILON}`);.

    // 2. 投影矩陣
    printSeparator('2. 投影矩陣（到直線）');  // EN: Execute a statement: printSeparator('2. 投影矩陣（到直線）');.

    const a2 = [1, 2];  // EN: Execute a statement: const a2 = [1, 2];.
    printVector('方向 a', a2);  // EN: Execute a statement: printVector('方向 a', a2);.

    const P = projectionMatrixLine(a2);  // EN: Execute a statement: const P = projectionMatrixLine(a2);.
    printMatrix('\n投影矩陣 P = aaᵀ/(aᵀa)', P);  // EN: Execute a statement: printMatrix('\n投影矩陣 P = aaᵀ/(aᵀa)', P);.

    verifyProjectionMatrix(P);  // EN: Execute a statement: verifyProjectionMatrix(P);.

    // 用投影矩陣計算投影
    const b2 = [3, 4];  // EN: Execute a statement: const b2 = [3, 4];.
    printVector('\n向量 b', b2);  // EN: Execute a statement: printVector('\n向量 b', b2);.

    const p = matrixVectorMultiply(P, b2);  // EN: Execute a statement: const p = matrixVectorMultiply(P, b2);.
    printVector('投影 p = Pb', p);  // EN: Execute a statement: printVector('投影 p = Pb', p);.

    // 3. 多個向量的投影
    printSeparator('3. 批次投影');  // EN: Execute a statement: printSeparator('3. 批次投影');.

    const vectors = [[1, 0], [0, 1], [2, 2], [3, -1]];  // EN: Execute a statement: const vectors = [[1, 0], [0, 1], [2, 2], [3, -1]];.

    console.log('方向 a = [1, 2]');  // EN: Execute a statement: console.log('方向 a = [1, 2]');.
    console.log('\n各向量投影結果：');  // EN: Execute a statement: console.log('\n各向量投影結果：');.

    for (const v of vectors) {  // EN: Loop control flow: for (const v of vectors) {.
        const proj = projectOntoLine(v, a2);  // EN: Execute a statement: const proj = projectOntoLine(v, a2);.
        console.log(`  [${v[0]}, ${v[1]}] -> [${formatNumber(proj.projection[0])}, ${formatNumber(proj.projection[1])}]`);  // EN: Execute a statement: console.log(` [${v[0]}, ${v[1]}] -> [${formatNumber(proj.projection[0])….
    }  // EN: Structure delimiter for a block or scope.

    // 總結
    printSeparator('總結');  // EN: Execute a statement: printSeparator('總結');.
    console.log(`  // EN: Execute line: console.log(`.
投影公式：  // EN: Execute line: 投影公式：.

1. 投影到直線：  // EN: Execute line: 1. 投影到直線：.
   p = (aᵀb / aᵀa) a  // EN: Execute line: p = (aᵀb / aᵀa) a.
   P = aaᵀ / (aᵀa)  // EN: Execute line: P = aaᵀ / (aᵀa).

2. 投影到子空間：  // EN: Execute line: 2. 投影到子空間：.
   p = A(AᵀA)⁻¹Aᵀb  // EN: Execute line: p = A(AᵀA)⁻¹Aᵀb.
   P = A(AᵀA)⁻¹Aᵀ  // EN: Execute line: P = A(AᵀA)⁻¹Aᵀ.

3. 投影矩陣性質：  // EN: Execute line: 3. 投影矩陣性質：.
   Pᵀ = P（對稱）  // EN: Execute line: Pᵀ = P（對稱）.
   P² = P（冪等）  // EN: Execute line: P² = P（冪等）.
`);  // EN: Execute a statement: `);.

    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log('示範完成！');  // EN: Execute a statement: console.log('示範完成！');.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

main();  // EN: Execute a statement: main();.
