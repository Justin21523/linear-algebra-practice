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
 * 執行：node inner_product.js
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
// 向量運算
// ========================================

/**
 * 計算兩向量的內積 (Dot Product)
 * x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ
 */
function dotProduct(x, y) {  // EN: Execute line: function dotProduct(x, y) {.
    if (x.length !== y.length) {  // EN: Conditional control flow: if (x.length !== y.length) {.
        throw new Error('向量維度必須相同');  // EN: Execute a statement: throw new Error('向量維度必須相同');.
    }  // EN: Structure delimiter for a block or scope.

    let result = 0;  // EN: Execute a statement: let result = 0;.
    for (let i = 0; i < x.length; i++) {  // EN: Loop control flow: for (let i = 0; i < x.length; i++) {.
        result += x[i] * y[i];  // EN: Execute a statement: result += x[i] * y[i];.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

/**
 * 計算向量的長度（L2 範數）
 * ‖x‖ = √(x · x)
 */
function vectorNorm(x) {  // EN: Execute line: function vectorNorm(x) {.
    return Math.sqrt(dotProduct(x, x));  // EN: Return from the current function: return Math.sqrt(dotProduct(x, x));.
}  // EN: Structure delimiter for a block or scope.

/**
 * 正規化向量為單位向量
 * û = x / ‖x‖
 */
function normalize(x) {  // EN: Execute line: function normalize(x) {.
    const norm = vectorNorm(x);  // EN: Execute a statement: const norm = vectorNorm(x);.
    if (norm < EPSILON) {  // EN: Conditional control flow: if (norm < EPSILON) {.
        throw new Error('零向量無法正規化');  // EN: Execute a statement: throw new Error('零向量無法正規化');.
    }  // EN: Structure delimiter for a block or scope.
    return x.map(xi => xi / norm);  // EN: Return from the current function: return x.map(xi => xi / norm);.
}  // EN: Structure delimiter for a block or scope.

/**
 * 計算兩向量的夾角（弧度）
 * cos θ = (x · y) / (‖x‖ ‖y‖)
 */
function vectorAngle(x, y) {  // EN: Execute line: function vectorAngle(x, y) {.
    const dot = dotProduct(x, y);  // EN: Execute a statement: const dot = dotProduct(x, y);.
    const normX = vectorNorm(x);  // EN: Execute a statement: const normX = vectorNorm(x);.
    const normY = vectorNorm(y);  // EN: Execute a statement: const normY = vectorNorm(y);.

    if (normX < EPSILON || normY < EPSILON) {  // EN: Conditional control flow: if (normX < EPSILON || normY < EPSILON) {.
        throw new Error('零向量沒有定義夾角');  // EN: Execute a statement: throw new Error('零向量沒有定義夾角');.
    }  // EN: Structure delimiter for a block or scope.

    let cosTheta = dot / (normX * normY);  // EN: Execute a statement: let cosTheta = dot / (normX * normY);.
    // 處理浮點數誤差
    cosTheta = Math.max(-1, Math.min(1, cosTheta));  // EN: Execute a statement: cosTheta = Math.max(-1, Math.min(1, cosTheta));.
    return Math.acos(cosTheta);  // EN: Return from the current function: return Math.acos(cosTheta);.
}  // EN: Structure delimiter for a block or scope.

/**
 * 判斷兩向量是否正交
 * x ⊥ y ⟺ x · y = 0
 */
function isOrthogonal(x, y) {  // EN: Execute line: function isOrthogonal(x, y) {.
    return Math.abs(dotProduct(x, y)) < EPSILON;  // EN: Return from the current function: return Math.abs(dotProduct(x, y)) < EPSILON;.
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 矩陣運算
// ========================================

/**
 * 矩陣轉置
 */
function transpose(A) {  // EN: Execute line: function transpose(A) {.
    const m = A.length;  // EN: Execute a statement: const m = A.length;.
    const n = A[0].length;  // EN: Execute a statement: const n = A[0].length;.

    const result = [];  // EN: Execute a statement: const result = [];.
    for (let j = 0; j < n; j++) {  // EN: Loop control flow: for (let j = 0; j < n; j++) {.
        result[j] = [];  // EN: Execute a statement: result[j] = [];.
        for (let i = 0; i < m; i++) {  // EN: Loop control flow: for (let i = 0; i < m; i++) {.
            result[j][i] = A[i][j];  // EN: Execute a statement: result[j][i] = A[i][j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

/**
 * 矩陣乘法
 */
function matrixMultiply(A, B) {  // EN: Execute line: function matrixMultiply(A, B) {.
    const m = A.length;  // EN: Execute a statement: const m = A.length;.
    const n = B[0].length;  // EN: Execute a statement: const n = B[0].length;.
    const k = B.length;  // EN: Execute a statement: const k = B.length;.

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

/**
 * 矩陣乘向量
 */
function matrixVectorMultiply(A, x) {  // EN: Execute line: function matrixVectorMultiply(A, x) {.
    const m = A.length;  // EN: Execute a statement: const m = A.length;.
    const n = A[0].length;  // EN: Execute a statement: const n = A[0].length;.

    const result = [];  // EN: Execute a statement: const result = [];.
    for (let i = 0; i < m; i++) {  // EN: Loop control flow: for (let i = 0; i < m; i++) {.
        result[i] = 0;  // EN: Execute a statement: result[i] = 0;.
        for (let j = 0; j < n; j++) {  // EN: Loop control flow: for (let j = 0; j < n; j++) {.
            result[i] += A[i][j] * x[j];  // EN: Execute a statement: result[i] += A[i][j] * x[j];.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

/**
 * 判斷是否為單位矩陣
 */
function isIdentity(A) {  // EN: Execute line: function isIdentity(A) {.
    const n = A.length;  // EN: Execute a statement: const n = A.length;.
    if (A[0].length !== n) return false;  // EN: Conditional control flow: if (A[0].length !== n) return false;.

    for (let i = 0; i < n; i++) {  // EN: Loop control flow: for (let i = 0; i < n; i++) {.
        for (let j = 0; j < n; j++) {  // EN: Loop control flow: for (let j = 0; j < n; j++) {.
            const expected = (i === j) ? 1 : 0;  // EN: Execute a statement: const expected = (i === j) ? 1 : 0;.
            if (Math.abs(A[i][j] - expected) > EPSILON) {  // EN: Conditional control flow: if (Math.abs(A[i][j] - expected) > EPSILON) {.
                return false;  // EN: Return from the current function: return false;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return true;  // EN: Return from the current function: return true;.
}  // EN: Structure delimiter for a block or scope.

/**
 * 判斷矩陣是否為正交矩陣
 * QᵀQ = I
 */
function isOrthogonalMatrix(Q) {  // EN: Execute line: function isOrthogonalMatrix(Q) {.
    const QT = transpose(Q);  // EN: Execute a statement: const QT = transpose(Q);.
    const product = matrixMultiply(QT, Q);  // EN: Execute a statement: const product = matrixMultiply(QT, Q);.
    return isIdentity(product);  // EN: Return from the current function: return isIdentity(product);.
}  // EN: Structure delimiter for a block or scope.

// ========================================
// 主程式
// ========================================

function main() {  // EN: Execute line: function main() {.
    printSeparator('內積與正交性示範 (JavaScript)\nInner Product & Orthogonality Demo');  // EN: Execute a statement: printSeparator('內積與正交性示範 (JavaScript)\nInner Product & Orthogonality De….

    // 1. 內積計算
    printSeparator('1. 內積計算 (Dot Product)');  // EN: Execute a statement: printSeparator('1. 內積計算 (Dot Product)');.

    const x = [1, 2, 3];  // EN: Execute a statement: const x = [1, 2, 3];.
    const y = [4, 5, 6];  // EN: Execute a statement: const y = [4, 5, 6];.

    printVector('x', x);  // EN: Execute a statement: printVector('x', x);.
    printVector('y', y);  // EN: Execute a statement: printVector('y', y);.
    console.log(`\nx · y = ${dotProduct(x, y)}`);  // EN: Execute a statement: console.log(`\nx · y = ${dotProduct(x, y)}`);.
    console.log('計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32');  // EN: Execute a statement: console.log('計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32');.

    // 2. 向量長度
    printSeparator('2. 向量長度 (Vector Norm)');  // EN: Execute a statement: printSeparator('2. 向量長度 (Vector Norm)');.

    const v = [3, 4];  // EN: Execute a statement: const v = [3, 4];.
    printVector('v', v);  // EN: Execute a statement: printVector('v', v);.
    console.log(`‖v‖ = ${vectorNorm(v)}`);  // EN: Execute a statement: console.log(`‖v‖ = ${vectorNorm(v)}`);.
    console.log('計算：√(3² + 4²) = √25 = 5');  // EN: Execute a statement: console.log('計算：√(3² + 4²) = √25 = 5');.

    // 正規化
    const vNormalized = normalize(v);  // EN: Execute a statement: const vNormalized = normalize(v);.
    console.log('\n單位向量：');  // EN: Execute a statement: console.log('\n單位向量：');.
    printVector('v̂ = v/‖v‖', vNormalized);  // EN: Execute a statement: printVector('v̂ = v/‖v‖', vNormalized);.
    console.log(`‖v̂‖ = ${vectorNorm(vNormalized)}`);  // EN: Execute a statement: console.log(`‖v̂‖ = ${vectorNorm(vNormalized)}`);.

    // 3. 向量夾角
    printSeparator('3. 向量夾角 (Vector Angle)');  // EN: Execute a statement: printSeparator('3. 向量夾角 (Vector Angle)');.

    const a = [1, 0];  // EN: Execute a statement: const a = [1, 0];.
    const b = [1, 1];  // EN: Execute a statement: const b = [1, 1];.

    printVector('a', a);  // EN: Execute a statement: printVector('a', a);.
    printVector('b', b);  // EN: Execute a statement: printVector('b', b);.

    const theta = vectorAngle(a, b);  // EN: Execute a statement: const theta = vectorAngle(a, b);.
    console.log(`\n夾角 θ = ${formatNumber(theta)} rad = ${formatNumber(theta * 180 / Math.PI)}°`);  // EN: Execute a statement: console.log(`\n夾角 θ = ${formatNumber(theta)} rad = ${formatNumber(theta….
    console.log(`cos θ = ${formatNumber(Math.cos(theta))}`);  // EN: Execute a statement: console.log(`cos θ = ${formatNumber(Math.cos(theta))}`);.
    console.log('預期：cos 45° = 1/√2 ≈ 0.7071');  // EN: Execute a statement: console.log('預期：cos 45° = 1/√2 ≈ 0.7071');.

    // 4. 正交性判斷
    printSeparator('4. 正交性判斷 (Orthogonality Check)');  // EN: Execute a statement: printSeparator('4. 正交性判斷 (Orthogonality Check)');.

    const u1 = [1, 2];  // EN: Execute a statement: const u1 = [1, 2];.
    const u2 = [-2, 1];  // EN: Execute a statement: const u2 = [-2, 1];.

    printVector('u₁', u1);  // EN: Execute a statement: printVector('u₁', u1);.
    printVector('u₂', u2);  // EN: Execute a statement: printVector('u₂', u2);.
    console.log(`\nu₁ · u₂ = ${dotProduct(u1, u2)}`);  // EN: Execute a statement: console.log(`\nu₁ · u₂ = ${dotProduct(u1, u2)}`);.
    console.log(`u₁ ⊥ u₂？ ${isOrthogonal(u1, u2)}`);  // EN: Execute a statement: console.log(`u₁ ⊥ u₂？ ${isOrthogonal(u1, u2)}`);.

    // 非正交
    const w1 = [1, 1];  // EN: Execute a statement: const w1 = [1, 1];.
    const w2 = [1, 2];  // EN: Execute a statement: const w2 = [1, 2];.

    console.log('\n另一組：');  // EN: Execute a statement: console.log('\n另一組：');.
    printVector('w₁', w1);  // EN: Execute a statement: printVector('w₁', w1);.
    printVector('w₂', w2);  // EN: Execute a statement: printVector('w₂', w2);.
    console.log(`w₁ · w₂ = ${dotProduct(w1, w2)}`);  // EN: Execute a statement: console.log(`w₁ · w₂ = ${dotProduct(w1, w2)}`);.
    console.log(`w₁ ⊥ w₂？ ${isOrthogonal(w1, w2)}`);  // EN: Execute a statement: console.log(`w₁ ⊥ w₂？ ${isOrthogonal(w1, w2)}`);.

    // 5. 正交矩陣
    printSeparator('5. 正交矩陣 (Orthogonal Matrix)');  // EN: Execute a statement: printSeparator('5. 正交矩陣 (Orthogonal Matrix)');.

    const angle = Math.PI / 4;  // EN: Execute a statement: const angle = Math.PI / 4;.
    const Q = [  // EN: Execute line: const Q = [.
        [Math.cos(angle), -Math.sin(angle)],  // EN: Execute line: [Math.cos(angle), -Math.sin(angle)],.
        [Math.sin(angle), Math.cos(angle)]  // EN: Execute line: [Math.sin(angle), Math.cos(angle)].
    ];  // EN: Execute a statement: ];.

    console.log('旋轉矩陣（θ = 45°）：');  // EN: Execute a statement: console.log('旋轉矩陣（θ = 45°）：');.
    printMatrix('Q', Q);  // EN: Execute a statement: printMatrix('Q', Q);.

    const QT = transpose(Q);  // EN: Execute a statement: const QT = transpose(Q);.
    printMatrix('\nQᵀ', QT);  // EN: Execute a statement: printMatrix('\nQᵀ', QT);.

    const QTQ = matrixMultiply(QT, Q);  // EN: Execute a statement: const QTQ = matrixMultiply(QT, Q);.
    printMatrix('\nQᵀQ', QTQ);  // EN: Execute a statement: printMatrix('\nQᵀQ', QTQ);.

    console.log(`\nQ 是正交矩陣？ ${isOrthogonalMatrix(Q)}`);  // EN: Execute a statement: console.log(`\nQ 是正交矩陣？ ${isOrthogonalMatrix(Q)}`);.

    // 驗證保長度
    const xVec = [3, 4];  // EN: Execute a statement: const xVec = [3, 4];.
    const Qx = matrixVectorMultiply(Q, xVec);  // EN: Execute a statement: const Qx = matrixVectorMultiply(Q, xVec);.

    console.log('\n保長度驗證：');  // EN: Execute a statement: console.log('\n保長度驗證：');.
    printVector('x', xVec);  // EN: Execute a statement: printVector('x', xVec);.
    printVector('Qx', Qx);  // EN: Execute a statement: printVector('Qx', Qx);.
    console.log(`‖x‖ = ${formatNumber(vectorNorm(xVec))}`);  // EN: Execute a statement: console.log(`‖x‖ = ${formatNumber(vectorNorm(xVec))}`);.
    console.log(`‖Qx‖ = ${formatNumber(vectorNorm(Qx))}`);  // EN: Execute a statement: console.log(`‖Qx‖ = ${formatNumber(vectorNorm(Qx))}`);.

    // 6. Cauchy-Schwarz 不等式
    printSeparator('6. Cauchy-Schwarz 不等式');  // EN: Execute a statement: printSeparator('6. Cauchy-Schwarz 不等式');.

    const csX = [1, 2, 3];  // EN: Execute a statement: const csX = [1, 2, 3];.
    const csY = [4, 5, 6];  // EN: Execute a statement: const csY = [4, 5, 6];.

    printVector('x', csX);  // EN: Execute a statement: printVector('x', csX);.
    printVector('y', csY);  // EN: Execute a statement: printVector('y', csY);.

    const leftSide = Math.abs(dotProduct(csX, csY));  // EN: Execute a statement: const leftSide = Math.abs(dotProduct(csX, csY));.
    const rightSide = vectorNorm(csX) * vectorNorm(csY);  // EN: Execute a statement: const rightSide = vectorNorm(csX) * vectorNorm(csY);.

    console.log(`\n|x · y| = ${formatNumber(leftSide)}`);  // EN: Execute a statement: console.log(`\n|x · y| = ${formatNumber(leftSide)}`);.
    console.log(`‖x‖ ‖y‖ = ${formatNumber(rightSide)}`);  // EN: Execute a statement: console.log(`‖x‖ ‖y‖ = ${formatNumber(rightSide)}`);.
    console.log(`|x · y| ≤ ‖x‖ ‖y‖？ ${leftSide <= rightSide + EPSILON}`);  // EN: Execute a statement: console.log(`|x · y| ≤ ‖x‖ ‖y‖？ ${leftSide <= rightSide + EPSILON}`);.

    // 總結
    printSeparator('總結');  // EN: Execute a statement: printSeparator('總結');.
    console.log(`  // EN: Execute line: console.log(`.
內積與正交性的核心公式：  // EN: Execute line: 內積與正交性的核心公式：.

1. 內積：x · y = Σ xᵢyᵢ  // EN: Execute line: 1. 內積：x · y = Σ xᵢyᵢ.

2. 長度：‖x‖ = √(x · x)  // EN: Execute line: 2. 長度：‖x‖ = √(x · x).

3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖)  // EN: Execute line: 3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖).

4. 正交：x ⊥ y ⟺ x · y = 0  // EN: Execute line: 4. 正交：x ⊥ y ⟺ x · y = 0.

5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ  // EN: Execute line: 5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ.
`);  // EN: Execute a statement: `);.

    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log('示範完成！');  // EN: Execute a statement: console.log('示範完成！');.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

main();  // EN: Execute a statement: main();.
