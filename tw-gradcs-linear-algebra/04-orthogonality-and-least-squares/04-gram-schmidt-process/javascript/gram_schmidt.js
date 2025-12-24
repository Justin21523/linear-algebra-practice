/**
 * Gram-Schmidt 正交化 (Gram-Schmidt Process)
 *
 * 執行：node gram_schmidt.js
 */

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

function dotProduct(x, y) {
    return x.reduce((sum, xi, i) => sum + xi * y[i], 0);
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

function normalize(x) {
    return scalarMultiply(1 / vectorNorm(x), x);
}

/**
 * Modified Gram-Schmidt
 */
function modifiedGramSchmidt(A) {
    const Q = A.map(row => [...row]);

    for (let j = 0; j < Q.length; j++) {
        for (let i = 0; i < j; i++) {
            const coeff = dotProduct(Q[i], Q[j]) / dotProduct(Q[i], Q[i]);
            Q[j] = vectorSubtract(Q[j], scalarMultiply(coeff, Q[i]));
        }
    }

    return Q;
}

function gramSchmidtNormalized(A) {
    return modifiedGramSchmidt(A).map(q => normalize(q));
}

function verifyOrthogonality(Q) {
    for (let i = 0; i < Q.length; i++) {
        for (let j = i + 1; j < Q.length; j++) {
            if (Math.abs(dotProduct(Q[i], Q[j])) > 1e-10) return false;
        }
    }
    return true;
}

function main() {
    printSeparator('Gram-Schmidt 正交化示範 (JavaScript)');

    const A = [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ];

    console.log('輸入向量組：');
    A.forEach((a, i) => printVector(`a${i+1}`, a));

    const Q = modifiedGramSchmidt(A);

    console.log('\n正交化結果（MGS）：');
    Q.forEach((q, i) => {
        printVector(`q${i+1}`, q);
        console.log(`    ‖q${i+1}‖ = ${formatNumber(vectorNorm(q))}`);
    });

    console.log(`\n正交？ ${verifyOrthogonality(Q)}`);

    console.log('\n內積驗證：');
    console.log(`q₁ · q₂ = ${dotProduct(Q[0], Q[1]).toFixed(6)}`);
    console.log(`q₁ · q₃ = ${dotProduct(Q[0], Q[2]).toFixed(6)}`);
    console.log(`q₂ · q₃ = ${dotProduct(Q[1], Q[2]).toFixed(6)}`);

    printSeparator('標準正交化');

    const E = gramSchmidtNormalized(A);

    console.log('標準正交向量組：');
    E.forEach((e, i) => {
        printVector(`e${i+1}`, e);
        console.log(`    ‖e${i+1}‖ = ${formatNumber(vectorNorm(e))}`);
    });

    printSeparator('總結');
    console.log(`
Gram-Schmidt 核心公式：

proj_q(a) = (qᵀa / qᵀq) q

q₁ = a₁
qₖ = aₖ - Σᵢ proj_{qᵢ}(aₖ)

eᵢ = qᵢ / ‖qᵢ‖
`);

    console.log('='.repeat(60));
    console.log('示範完成！');
    console.log('='.repeat(60));
}

main();
