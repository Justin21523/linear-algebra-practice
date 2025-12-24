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
// 向量運算
// ========================================

/**
 * 計算兩向量的內積 (Dot Product)
 * x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ
 */
function dotProduct(x, y) {
    if (x.length !== y.length) {
        throw new Error('向量維度必須相同');
    }

    let result = 0;
    for (let i = 0; i < x.length; i++) {
        result += x[i] * y[i];
    }
    return result;
}

/**
 * 計算向量的長度（L2 範數）
 * ‖x‖ = √(x · x)
 */
function vectorNorm(x) {
    return Math.sqrt(dotProduct(x, x));
}

/**
 * 正規化向量為單位向量
 * û = x / ‖x‖
 */
function normalize(x) {
    const norm = vectorNorm(x);
    if (norm < EPSILON) {
        throw new Error('零向量無法正規化');
    }
    return x.map(xi => xi / norm);
}

/**
 * 計算兩向量的夾角（弧度）
 * cos θ = (x · y) / (‖x‖ ‖y‖)
 */
function vectorAngle(x, y) {
    const dot = dotProduct(x, y);
    const normX = vectorNorm(x);
    const normY = vectorNorm(y);

    if (normX < EPSILON || normY < EPSILON) {
        throw new Error('零向量沒有定義夾角');
    }

    let cosTheta = dot / (normX * normY);
    // 處理浮點數誤差
    cosTheta = Math.max(-1, Math.min(1, cosTheta));
    return Math.acos(cosTheta);
}

/**
 * 判斷兩向量是否正交
 * x ⊥ y ⟺ x · y = 0
 */
function isOrthogonal(x, y) {
    return Math.abs(dotProduct(x, y)) < EPSILON;
}

// ========================================
// 矩陣運算
// ========================================

/**
 * 矩陣轉置
 */
function transpose(A) {
    const m = A.length;
    const n = A[0].length;

    const result = [];
    for (let j = 0; j < n; j++) {
        result[j] = [];
        for (let i = 0; i < m; i++) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

/**
 * 矩陣乘法
 */
function matrixMultiply(A, B) {
    const m = A.length;
    const n = B[0].length;
    const k = B.length;

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

/**
 * 矩陣乘向量
 */
function matrixVectorMultiply(A, x) {
    const m = A.length;
    const n = A[0].length;

    const result = [];
    for (let i = 0; i < m; i++) {
        result[i] = 0;
        for (let j = 0; j < n; j++) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

/**
 * 判斷是否為單位矩陣
 */
function isIdentity(A) {
    const n = A.length;
    if (A[0].length !== n) return false;

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const expected = (i === j) ? 1 : 0;
            if (Math.abs(A[i][j] - expected) > EPSILON) {
                return false;
            }
        }
    }
    return true;
}

/**
 * 判斷矩陣是否為正交矩陣
 * QᵀQ = I
 */
function isOrthogonalMatrix(Q) {
    const QT = transpose(Q);
    const product = matrixMultiply(QT, Q);
    return isIdentity(product);
}

// ========================================
// 主程式
// ========================================

function main() {
    printSeparator('內積與正交性示範 (JavaScript)\nInner Product & Orthogonality Demo');

    // 1. 內積計算
    printSeparator('1. 內積計算 (Dot Product)');

    const x = [1, 2, 3];
    const y = [4, 5, 6];

    printVector('x', x);
    printVector('y', y);
    console.log(`\nx · y = ${dotProduct(x, y)}`);
    console.log('計算：1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32');

    // 2. 向量長度
    printSeparator('2. 向量長度 (Vector Norm)');

    const v = [3, 4];
    printVector('v', v);
    console.log(`‖v‖ = ${vectorNorm(v)}`);
    console.log('計算：√(3² + 4²) = √25 = 5');

    // 正規化
    const vNormalized = normalize(v);
    console.log('\n單位向量：');
    printVector('v̂ = v/‖v‖', vNormalized);
    console.log(`‖v̂‖ = ${vectorNorm(vNormalized)}`);

    // 3. 向量夾角
    printSeparator('3. 向量夾角 (Vector Angle)');

    const a = [1, 0];
    const b = [1, 1];

    printVector('a', a);
    printVector('b', b);

    const theta = vectorAngle(a, b);
    console.log(`\n夾角 θ = ${formatNumber(theta)} rad = ${formatNumber(theta * 180 / Math.PI)}°`);
    console.log(`cos θ = ${formatNumber(Math.cos(theta))}`);
    console.log('預期：cos 45° = 1/√2 ≈ 0.7071');

    // 4. 正交性判斷
    printSeparator('4. 正交性判斷 (Orthogonality Check)');

    const u1 = [1, 2];
    const u2 = [-2, 1];

    printVector('u₁', u1);
    printVector('u₂', u2);
    console.log(`\nu₁ · u₂ = ${dotProduct(u1, u2)}`);
    console.log(`u₁ ⊥ u₂？ ${isOrthogonal(u1, u2)}`);

    // 非正交
    const w1 = [1, 1];
    const w2 = [1, 2];

    console.log('\n另一組：');
    printVector('w₁', w1);
    printVector('w₂', w2);
    console.log(`w₁ · w₂ = ${dotProduct(w1, w2)}`);
    console.log(`w₁ ⊥ w₂？ ${isOrthogonal(w1, w2)}`);

    // 5. 正交矩陣
    printSeparator('5. 正交矩陣 (Orthogonal Matrix)');

    const angle = Math.PI / 4;
    const Q = [
        [Math.cos(angle), -Math.sin(angle)],
        [Math.sin(angle), Math.cos(angle)]
    ];

    console.log('旋轉矩陣（θ = 45°）：');
    printMatrix('Q', Q);

    const QT = transpose(Q);
    printMatrix('\nQᵀ', QT);

    const QTQ = matrixMultiply(QT, Q);
    printMatrix('\nQᵀQ', QTQ);

    console.log(`\nQ 是正交矩陣？ ${isOrthogonalMatrix(Q)}`);

    // 驗證保長度
    const xVec = [3, 4];
    const Qx = matrixVectorMultiply(Q, xVec);

    console.log('\n保長度驗證：');
    printVector('x', xVec);
    printVector('Qx', Qx);
    console.log(`‖x‖ = ${formatNumber(vectorNorm(xVec))}`);
    console.log(`‖Qx‖ = ${formatNumber(vectorNorm(Qx))}`);

    // 6. Cauchy-Schwarz 不等式
    printSeparator('6. Cauchy-Schwarz 不等式');

    const csX = [1, 2, 3];
    const csY = [4, 5, 6];

    printVector('x', csX);
    printVector('y', csY);

    const leftSide = Math.abs(dotProduct(csX, csY));
    const rightSide = vectorNorm(csX) * vectorNorm(csY);

    console.log(`\n|x · y| = ${formatNumber(leftSide)}`);
    console.log(`‖x‖ ‖y‖ = ${formatNumber(rightSide)}`);
    console.log(`|x · y| ≤ ‖x‖ ‖y‖？ ${leftSide <= rightSide + EPSILON}`);

    // 總結
    printSeparator('總結');
    console.log(`
內積與正交性的核心公式：

1. 內積：x · y = Σ xᵢyᵢ

2. 長度：‖x‖ = √(x · x)

3. 夾角：cos θ = (x · y) / (‖x‖ ‖y‖)

4. 正交：x ⊥ y ⟺ x · y = 0

5. 正交矩陣：QᵀQ = I, Q⁻¹ = Qᵀ
`);

    console.log('='.repeat(60));
    console.log('示範完成！');
    console.log('='.repeat(60));
}

main();
