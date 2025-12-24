/**
 * Gram-Schmidt 正交化 (Gram-Schmidt Process)
 *
 * 執行：node gram_schmidt.js
 */

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

function dotProduct(x, y) {  // EN: Execute line: function dotProduct(x, y) {.
    return x.reduce((sum, xi, i) => sum + xi * y[i], 0);  // EN: Return from the current function: return x.reduce((sum, xi, i) => sum + xi * y[i], 0);.
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

function normalize(x) {  // EN: Execute line: function normalize(x) {.
    return scalarMultiply(1 / vectorNorm(x), x);  // EN: Return from the current function: return scalarMultiply(1 / vectorNorm(x), x);.
}  // EN: Structure delimiter for a block or scope.

/**
 * Modified Gram-Schmidt
 */
function modifiedGramSchmidt(A) {  // EN: Execute line: function modifiedGramSchmidt(A) {.
    const Q = A.map(row => [...row]);  // EN: Execute a statement: const Q = A.map(row => [...row]);.

    for (let j = 0; j < Q.length; j++) {  // EN: Loop control flow: for (let j = 0; j < Q.length; j++) {.
        for (let i = 0; i < j; i++) {  // EN: Loop control flow: for (let i = 0; i < j; i++) {.
            const coeff = dotProduct(Q[i], Q[j]) / dotProduct(Q[i], Q[i]);  // EN: Execute a statement: const coeff = dotProduct(Q[i], Q[j]) / dotProduct(Q[i], Q[i]);.
            Q[j] = vectorSubtract(Q[j], scalarMultiply(coeff, Q[i]));  // EN: Execute a statement: Q[j] = vectorSubtract(Q[j], scalarMultiply(coeff, Q[i]));.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    return Q;  // EN: Return from the current function: return Q;.
}  // EN: Structure delimiter for a block or scope.

function gramSchmidtNormalized(A) {  // EN: Execute line: function gramSchmidtNormalized(A) {.
    return modifiedGramSchmidt(A).map(q => normalize(q));  // EN: Return from the current function: return modifiedGramSchmidt(A).map(q => normalize(q));.
}  // EN: Structure delimiter for a block or scope.

function verifyOrthogonality(Q) {  // EN: Execute line: function verifyOrthogonality(Q) {.
    for (let i = 0; i < Q.length; i++) {  // EN: Loop control flow: for (let i = 0; i < Q.length; i++) {.
        for (let j = i + 1; j < Q.length; j++) {  // EN: Loop control flow: for (let j = i + 1; j < Q.length; j++) {.
            if (Math.abs(dotProduct(Q[i], Q[j])) > 1e-10) return false;  // EN: Conditional control flow: if (Math.abs(dotProduct(Q[i], Q[j])) > 1e-10) return false;.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return true;  // EN: Return from the current function: return true;.
}  // EN: Structure delimiter for a block or scope.

function main() {  // EN: Execute line: function main() {.
    printSeparator('Gram-Schmidt 正交化示範 (JavaScript)');  // EN: Execute a statement: printSeparator('Gram-Schmidt 正交化示範 (JavaScript)');.

    const A = [  // EN: Execute line: const A = [.
        [1, 1, 0],  // EN: Execute line: [1, 1, 0],.
        [1, 0, 1],  // EN: Execute line: [1, 0, 1],.
        [0, 1, 1]  // EN: Execute line: [0, 1, 1].
    ];  // EN: Execute a statement: ];.

    console.log('輸入向量組：');  // EN: Execute a statement: console.log('輸入向量組：');.
    A.forEach((a, i) => printVector(`a${i+1}`, a));  // EN: Execute a statement: A.forEach((a, i) => printVector(`a${i+1}`, a));.

    const Q = modifiedGramSchmidt(A);  // EN: Execute a statement: const Q = modifiedGramSchmidt(A);.

    console.log('\n正交化結果（MGS）：');  // EN: Execute a statement: console.log('\n正交化結果（MGS）：');.
    Q.forEach((q, i) => {  // EN: Execute line: Q.forEach((q, i) => {.
        printVector(`q${i+1}`, q);  // EN: Execute a statement: printVector(`q${i+1}`, q);.
        console.log(`    ‖q${i+1}‖ = ${formatNumber(vectorNorm(q))}`);  // EN: Execute a statement: console.log(` ‖q${i+1}‖ = ${formatNumber(vectorNorm(q))}`);.
    });  // EN: Execute a statement: });.

    console.log(`\n正交？ ${verifyOrthogonality(Q)}`);  // EN: Execute a statement: console.log(`\n正交？ ${verifyOrthogonality(Q)}`);.

    console.log('\n內積驗證：');  // EN: Execute a statement: console.log('\n內積驗證：');.
    console.log(`q₁ · q₂ = ${dotProduct(Q[0], Q[1]).toFixed(6)}`);  // EN: Execute a statement: console.log(`q₁ · q₂ = ${dotProduct(Q[0], Q[1]).toFixed(6)}`);.
    console.log(`q₁ · q₃ = ${dotProduct(Q[0], Q[2]).toFixed(6)}`);  // EN: Execute a statement: console.log(`q₁ · q₃ = ${dotProduct(Q[0], Q[2]).toFixed(6)}`);.
    console.log(`q₂ · q₃ = ${dotProduct(Q[1], Q[2]).toFixed(6)}`);  // EN: Execute a statement: console.log(`q₂ · q₃ = ${dotProduct(Q[1], Q[2]).toFixed(6)}`);.

    printSeparator('標準正交化');  // EN: Execute a statement: printSeparator('標準正交化');.

    const E = gramSchmidtNormalized(A);  // EN: Execute a statement: const E = gramSchmidtNormalized(A);.

    console.log('標準正交向量組：');  // EN: Execute a statement: console.log('標準正交向量組：');.
    E.forEach((e, i) => {  // EN: Execute line: E.forEach((e, i) => {.
        printVector(`e${i+1}`, e);  // EN: Execute a statement: printVector(`e${i+1}`, e);.
        console.log(`    ‖e${i+1}‖ = ${formatNumber(vectorNorm(e))}`);  // EN: Execute a statement: console.log(` ‖e${i+1}‖ = ${formatNumber(vectorNorm(e))}`);.
    });  // EN: Execute a statement: });.

    printSeparator('總結');  // EN: Execute a statement: printSeparator('總結');.
    console.log(`  // EN: Execute line: console.log(`.
Gram-Schmidt 核心公式：  // EN: Execute line: Gram-Schmidt 核心公式：.

proj_q(a) = (qᵀa / qᵀq) q  // EN: Execute line: proj_q(a) = (qᵀa / qᵀq) q.

q₁ = a₁  // EN: Execute line: q₁ = a₁.
qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ)  // EN: Execute line: qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ).

eᵢ = qᵢ / ‖qᵢ‖  // EN: Execute line: eᵢ = qᵢ / ‖qᵢ‖.
`);  // EN: Execute a statement: `);.

    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log('示範完成！');  // EN: Execute a statement: console.log('示範完成！');.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

main();  // EN: Execute a statement: main();.
