/**
 * QR 分解 (QR Decomposition)
 *
 * 執行：node qr_decomposition.js
 */

function printSeparator(title) {  // EN: Execute line: function printSeparator(title) {.
    console.log();  // EN: Execute a statement: console.log();.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log(title);  // EN: Execute a statement: console.log(title);.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

function printVector(name, v) {  // EN: Execute line: function printVector(name, v) {.
    const formatted = v.map(x => x.toFixed(4)).join(', ');  // EN: Execute a statement: const formatted = v.map(x => x.toFixed(4)).join(', ');.
    console.log(`${name} = [${formatted}]`);  // EN: Execute a statement: console.log(`${name} = [${formatted}]`);.
}  // EN: Structure delimiter for a block or scope.

function printMatrix(name, M) {  // EN: Execute line: function printMatrix(name, M) {.
    console.log(`${name} =`);  // EN: Execute a statement: console.log(`${name} =`);.
    for (const row of M) {  // EN: Loop control flow: for (const row of M) {.
        const formatted = row.map(x => x.toFixed(4).padStart(8)).join(', ');  // EN: Execute a statement: const formatted = row.map(x => x.toFixed(4).padStart(8)).join(', ');.
        console.log(`  [${formatted}]`);  // EN: Execute a statement: console.log(` [${formatted}]`);.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// 基本向量運算
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

// 取得矩陣的第 j 行（column）
function getColumn(A, j) {  // EN: Execute line: function getColumn(A, j) {.
    return A.map(row => row[j]);  // EN: Return from the current function: return A.map(row => row[j]);.
}  // EN: Structure delimiter for a block or scope.

// Gram-Schmidt QR 分解
function qrDecomposition(A) {  // EN: Execute line: function qrDecomposition(A) {.
    const m = A.length;  // EN: Execute a statement: const m = A.length;.
    const n = A[0].length;  // EN: Execute a statement: const n = A[0].length;.

    // Q: m×n, R: n×n
    const Q = Array.from({length: m}, () => Array(n).fill(0));  // EN: Execute a statement: const Q = Array.from({length: m}, () => Array(n).fill(0));.
    const R = Array.from({length: n}, () => Array(n).fill(0));  // EN: Execute a statement: const R = Array.from({length: n}, () => Array(n).fill(0));.

    for (let j = 0; j < n; j++) {  // EN: Loop control flow: for (let j = 0; j < n; j++) {.
        // 取得 A 的第 j 行
        let v = getColumn(A, j);  // EN: Execute a statement: let v = getColumn(A, j);.

        // 減去前面所有 q 向量的投影
        for (let i = 0; i < j; i++) {  // EN: Loop control flow: for (let i = 0; i < j; i++) {.
            const qi = getColumn(Q, i);  // EN: Execute a statement: const qi = getColumn(Q, i);.
            R[i][j] = dotProduct(qi, getColumn(A, j));  // EN: Execute a statement: R[i][j] = dotProduct(qi, getColumn(A, j));.
            const proj = scalarMultiply(R[i][j], qi);  // EN: Execute a statement: const proj = scalarMultiply(R[i][j], qi);.
            v = vectorSubtract(v, proj);  // EN: Execute a statement: v = vectorSubtract(v, proj);.
        }  // EN: Structure delimiter for a block or scope.

        // 標準化
        R[j][j] = vectorNorm(v);  // EN: Execute a statement: R[j][j] = vectorNorm(v);.

        if (R[j][j] > 1e-10) {  // EN: Conditional control flow: if (R[j][j] > 1e-10) {.
            for (let i = 0; i < m; i++) {  // EN: Loop control flow: for (let i = 0; i < m; i++) {.
                Q[i][j] = v[i] / R[j][j];  // EN: Execute a statement: Q[i][j] = v[i] / R[j][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    return [Q, R];  // EN: Return from the current function: return [Q, R];.
}  // EN: Structure delimiter for a block or scope.

// 回代法解上三角方程組 Rx = b
function solveUpperTriangular(R, b) {  // EN: Execute line: function solveUpperTriangular(R, b) {.
    const n = b.length;  // EN: Execute a statement: const n = b.length;.
    const x = Array(n).fill(0);  // EN: Execute a statement: const x = Array(n).fill(0);.

    for (let i = n - 1; i >= 0; i--) {  // EN: Loop control flow: for (let i = n - 1; i >= 0; i--) {.
        x[i] = b[i];  // EN: Execute a statement: x[i] = b[i];.
        for (let j = i + 1; j < n; j++) {  // EN: Loop control flow: for (let j = i + 1; j < n; j++) {.
            x[i] -= R[i][j] * x[j];  // EN: Execute a statement: x[i] -= R[i][j] * x[j];.
        }  // EN: Structure delimiter for a block or scope.
        x[i] /= R[i][i];  // EN: Execute a statement: x[i] /= R[i][i];.
    }  // EN: Structure delimiter for a block or scope.

    return x;  // EN: Return from the current function: return x;.
}  // EN: Structure delimiter for a block or scope.

// 用 QR 分解解最小平方問題
function qrLeastSquares(A, b) {  // EN: Execute line: function qrLeastSquares(A, b) {.
    const [Q, R] = qrDecomposition(A);  // EN: Execute a statement: const [Q, R] = qrDecomposition(A);.

    // 計算 Qᵀb
    const n = Q[0].length;  // EN: Execute a statement: const n = Q[0].length;.
    const Qt_b = [];  // EN: Execute a statement: const Qt_b = [];.
    for (let j = 0; j < n; j++) {  // EN: Loop control flow: for (let j = 0; j < n; j++) {.
        const qj = getColumn(Q, j);  // EN: Execute a statement: const qj = getColumn(Q, j);.
        Qt_b.push(dotProduct(qj, b));  // EN: Execute a statement: Qt_b.push(dotProduct(qj, b));.
    }  // EN: Structure delimiter for a block or scope.

    // 解 Rx = Qᵀb
    return solveUpperTriangular(R, Qt_b);  // EN: Return from the current function: return solveUpperTriangular(R, Qt_b);.
}  // EN: Structure delimiter for a block or scope.

// 矩陣乘法
function matrixMultiply(A, B) {  // EN: Execute line: function matrixMultiply(A, B) {.
    const m = A.length;  // EN: Execute a statement: const m = A.length;.
    const k = B.length;  // EN: Execute a statement: const k = B.length;.
    const n = B[0].length;  // EN: Execute a statement: const n = B[0].length;.

    const result = Array.from({length: m}, () => Array(n).fill(0));  // EN: Execute a statement: const result = Array.from({length: m}, () => Array(n).fill(0));.
    for (let i = 0; i < m; i++) {  // EN: Loop control flow: for (let i = 0; i < m; i++) {.
        for (let j = 0; j < n; j++) {  // EN: Loop control flow: for (let j = 0; j < n; j++) {.
            for (let p = 0; p < k; p++) {  // EN: Loop control flow: for (let p = 0; p < k; p++) {.
                result[i][j] += A[i][p] * B[p][j];  // EN: Execute a statement: result[i][j] += A[i][p] * B[p][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.
    return result;  // EN: Return from the current function: return result;.
}  // EN: Structure delimiter for a block or scope.

// 矩陣轉置
function transpose(A) {  // EN: Execute line: function transpose(A) {.
    const m = A.length;  // EN: Execute a statement: const m = A.length;.
    const n = A[0].length;  // EN: Execute a statement: const n = A[0].length;.
    return Array.from({length: n}, (_, j) =>  // EN: Return from the current function: return Array.from({length: n}, (_, j) =>.
        Array.from({length: m}, (_, i) => A[i][j])  // EN: Execute line: Array.from({length: m}, (_, i) => A[i][j]).
    );  // EN: Execute a statement: );.
}  // EN: Structure delimiter for a block or scope.

function main() {  // EN: Execute line: function main() {.
    printSeparator('QR 分解示範 (JavaScript)');  // EN: Execute a statement: printSeparator('QR 分解示範 (JavaScript)');.

    // ========================================
    // 1. 基本 QR 分解
    // ========================================
    printSeparator('1. 基本 QR 分解');  // EN: Execute a statement: printSeparator('1. 基本 QR 分解');.

    const A = [  // EN: Execute line: const A = [.
        [1, 1],  // EN: Execute line: [1, 1],.
        [1, 0],  // EN: Execute line: [1, 0],.
        [0, 1]  // EN: Execute line: [0, 1].
    ];  // EN: Execute a statement: ];.

    console.log('輸入矩陣 A：');  // EN: Execute a statement: console.log('輸入矩陣 A：');.
    printMatrix('A', A);  // EN: Execute a statement: printMatrix('A', A);.

    const [Q, R] = qrDecomposition(A);  // EN: Execute a statement: const [Q, R] = qrDecomposition(A);.

    console.log('\nQR 分解結果：');  // EN: Execute a statement: console.log('\nQR 分解結果：');.
    printMatrix('Q', Q);  // EN: Execute a statement: printMatrix('Q', Q);.
    printMatrix('\nR', R);  // EN: Execute a statement: printMatrix('\nR', R);.

    // 驗證 QᵀQ = I
    const QT = transpose(Q);  // EN: Execute a statement: const QT = transpose(Q);.
    const QTQ = matrixMultiply(QT, Q);  // EN: Execute a statement: const QTQ = matrixMultiply(QT, Q);.
    console.log('\n驗證 QᵀQ = I：');  // EN: Execute a statement: console.log('\n驗證 QᵀQ = I：');.
    printMatrix('QᵀQ', QTQ);  // EN: Execute a statement: printMatrix('QᵀQ', QTQ);.

    // 驗證 A = QR
    const QR_result = matrixMultiply(Q, R);  // EN: Execute a statement: const QR_result = matrixMultiply(Q, R);.
    console.log('\n驗證 A = QR：');  // EN: Execute a statement: console.log('\n驗證 A = QR：');.
    printMatrix('QR', QR_result);  // EN: Execute a statement: printMatrix('QR', QR_result);.

    // ========================================
    // 2. 用 QR 解最小平方
    // ========================================
    printSeparator('2. 用 QR 解最小平方');  // EN: Execute a statement: printSeparator('2. 用 QR 解最小平方');.

    // 數據
    const t = [0, 1, 2];  // EN: Execute a statement: const t = [0, 1, 2];.
    const b = [1, 3, 4];  // EN: Execute a statement: const b = [1, 3, 4];.

    console.log('數據點：');  // EN: Execute a statement: console.log('數據點：');.
    for (let i = 0; i < t.length; i++) {  // EN: Loop control flow: for (let i = 0; i < t.length; i++) {.
        console.log(`  (${t[i]}, ${b[i]})`);  // EN: Execute a statement: console.log(` (${t[i]}, ${b[i]})`);.
    }  // EN: Structure delimiter for a block or scope.

    // 設計矩陣
    const A_ls = t.map(ti => [1, ti]);  // EN: Execute a statement: const A_ls = t.map(ti => [1, ti]);.

    console.log('\n設計矩陣 A：');  // EN: Execute a statement: console.log('\n設計矩陣 A：');.
    printMatrix('A', A_ls);  // EN: Execute a statement: printMatrix('A', A_ls);.
    printVector('觀測值 b', b);  // EN: Execute a statement: printVector('觀測值 b', b);.

    // QR 分解
    const [Q_ls, R_ls] = qrDecomposition(A_ls);  // EN: Execute a statement: const [Q_ls, R_ls] = qrDecomposition(A_ls);.
    printMatrix('\nQ', Q_ls);  // EN: Execute a statement: printMatrix('\nQ', Q_ls);.
    printMatrix('R', R_ls);  // EN: Execute a statement: printMatrix('R', R_ls);.

    // 解最小平方
    const x = qrLeastSquares(A_ls, b);  // EN: Execute a statement: const x = qrLeastSquares(A_ls, b);.
    printVector('\n解 x', x);  // EN: Execute a statement: printVector('\n解 x', x);.

    console.log(`\n最佳直線：y = ${x[0].toFixed(4)} + ${x[1].toFixed(4)}t`);  // EN: Execute a statement: console.log(`\n最佳直線：y = ${x[0].toFixed(4)} + ${x[1].toFixed(4)}t`);.

    // ========================================
    // 3. 3×3 矩陣的 QR 分解
    // ========================================
    printSeparator('3. 3×3 矩陣的 QR 分解');  // EN: Execute a statement: printSeparator('3. 3×3 矩陣的 QR 分解');.

    const A2 = [  // EN: Execute line: const A2 = [.
        [1, 1, 0],  // EN: Execute line: [1, 1, 0],.
        [1, 0, 1],  // EN: Execute line: [1, 0, 1],.
        [0, 1, 1]  // EN: Execute line: [0, 1, 1].
    ];  // EN: Execute a statement: ];.

    console.log('輸入矩陣 A：');  // EN: Execute a statement: console.log('輸入矩陣 A：');.
    printMatrix('A', A2);  // EN: Execute a statement: printMatrix('A', A2);.

    const [Q2, R2] = qrDecomposition(A2);  // EN: Execute a statement: const [Q2, R2] = qrDecomposition(A2);.

    console.log('\nQR 分解結果：');  // EN: Execute a statement: console.log('\nQR 分解結果：');.
    printMatrix('Q', Q2);  // EN: Execute a statement: printMatrix('Q', Q2);.
    printMatrix('\nR', R2);  // EN: Execute a statement: printMatrix('\nR', R2);.

    // 總結
    printSeparator('總結');  // EN: Execute a statement: printSeparator('總結');.
    console.log(`  // EN: Execute line: console.log(`.
QR 分解核心：  // EN: Execute line: QR 分解核心：.

1. A = QR  // EN: Execute line: 1. A = QR.
   - Q: 標準正交矩陣 (QᵀQ = I)  // EN: Execute line: - Q: 標準正交矩陣 (QᵀQ = I).
   - R: 上三角矩陣  // EN: Execute line: - R: 上三角矩陣.

2. Gram-Schmidt 演算法：  // EN: Execute line: 2. Gram-Schmidt 演算法：.
   - 對 A 的行向量正交化得到 Q  // EN: Execute line: - 對 A 的行向量正交化得到 Q.
   - R 的元素是投影係數  // EN: Execute line: - R 的元素是投影係數.

3. 用 QR 解最小平方：  // EN: Execute line: 3. 用 QR 解最小平方：.
   min ‖Ax - b‖²  // EN: Execute line: min ‖Ax - b‖².
   → Rx = Qᵀb  // EN: Execute line: → Rx = Qᵀb.

4. 優勢：  // EN: Execute line: 4. 優勢：.
   - 比正規方程更穩定  // EN: Execute line: - 比正規方程更穩定.
   - 避免計算 AᵀA  // EN: Execute line: - 避免計算 AᵀA.
`);  // EN: Execute a statement: `);.

    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log('示範完成！');  // EN: Execute a statement: console.log('示範完成！');.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

main();  // EN: Execute a statement: main();.
