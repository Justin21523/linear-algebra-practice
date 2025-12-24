/**
 * QR 分解 (QR Decomposition)
 *
 * 執行：node qr_decomposition.js
 */

function printSeparator(title) {
    console.log();
    console.log('='.repeat(60));
    console.log(title);
    console.log('='.repeat(60));
}

function printVector(name, v) {
    const formatted = v.map(x => x.toFixed(4)).join(', ');
    console.log(`${name} = [${formatted}]`);
}

function printMatrix(name, M) {
    console.log(`${name} =`);
    for (const row of M) {
        const formatted = row.map(x => x.toFixed(4).padStart(8)).join(', ');
        console.log(`  [${formatted}]`);
    }
}

// 基本向量運算
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

// 取得矩陣的第 j 行（column）
function getColumn(A, j) {
    return A.map(row => row[j]);
}

// Gram-Schmidt QR 分解
function qrDecomposition(A) {
    const m = A.length;
    const n = A[0].length;

    // Q: m×n, R: n×n
    const Q = Array.from({length: m}, () => Array(n).fill(0));
    const R = Array.from({length: n}, () => Array(n).fill(0));

    for (let j = 0; j < n; j++) {
        // 取得 A 的第 j 行
        let v = getColumn(A, j);

        // 減去前面所有 q 向量的投影
        for (let i = 0; i < j; i++) {
            const qi = getColumn(Q, i);
            R[i][j] = dotProduct(qi, getColumn(A, j));
            const proj = scalarMultiply(R[i][j], qi);
            v = vectorSubtract(v, proj);
        }

        // 標準化
        R[j][j] = vectorNorm(v);

        if (R[j][j] > 1e-10) {
            for (let i = 0; i < m; i++) {
                Q[i][j] = v[i] / R[j][j];
            }
        }
    }

    return [Q, R];
}

// 回代法解上三角方程組 Rx = b
function solveUpperTriangular(R, b) {
    const n = b.length;
    const x = Array(n).fill(0);

    for (let i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (let j = i + 1; j < n; j++) {
            x[i] -= R[i][j] * x[j];
        }
        x[i] /= R[i][i];
    }

    return x;
}

// 用 QR 分解解最小平方問題
function qrLeastSquares(A, b) {
    const [Q, R] = qrDecomposition(A);

    // 計算 Qᵀb
    const n = Q[0].length;
    const Qt_b = [];
    for (let j = 0; j < n; j++) {
        const qj = getColumn(Q, j);
        Qt_b.push(dotProduct(qj, b));
    }

    // 解 Rx = Qᵀb
    return solveUpperTriangular(R, Qt_b);
}

// 矩陣乘法
function matrixMultiply(A, B) {
    const m = A.length;
    const k = B.length;
    const n = B[0].length;

    const result = Array.from({length: m}, () => Array(n).fill(0));
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            for (let p = 0; p < k; p++) {
                result[i][j] += A[i][p] * B[p][j];
            }
        }
    }
    return result;
}

// 矩陣轉置
function transpose(A) {
    const m = A.length;
    const n = A[0].length;
    return Array.from({length: n}, (_, j) =>
        Array.from({length: m}, (_, i) => A[i][j])
    );
}

function main() {
    printSeparator('QR 分解示範 (JavaScript)');

    // ========================================
    // 1. 基本 QR 分解
    // ========================================
    printSeparator('1. 基本 QR 分解');

    const A = [
        [1, 1],
        [1, 0],
        [0, 1]
    ];

    console.log('輸入矩陣 A：');
    printMatrix('A', A);

    const [Q, R] = qrDecomposition(A);

    console.log('\nQR 分解結果：');
    printMatrix('Q', Q);
    printMatrix('\nR', R);

    // 驗證 QᵀQ = I
    const QT = transpose(Q);
    const QTQ = matrixMultiply(QT, Q);
    console.log('\n驗證 QᵀQ = I：');
    printMatrix('QᵀQ', QTQ);

    // 驗證 A = QR
    const QR_result = matrixMultiply(Q, R);
    console.log('\n驗證 A = QR：');
    printMatrix('QR', QR_result);

    // ========================================
    // 2. 用 QR 解最小平方
    // ========================================
    printSeparator('2. 用 QR 解最小平方');

    // 數據
    const t = [0, 1, 2];
    const b = [1, 3, 4];

    console.log('數據點：');
    for (let i = 0; i < t.length; i++) {
        console.log(`  (${t[i]}, ${b[i]})`);
    }

    // 設計矩陣
    const A_ls = t.map(ti => [1, ti]);

    console.log('\n設計矩陣 A：');
    printMatrix('A', A_ls);
    printVector('觀測值 b', b);

    // QR 分解
    const [Q_ls, R_ls] = qrDecomposition(A_ls);
    printMatrix('\nQ', Q_ls);
    printMatrix('R', R_ls);

    // 解最小平方
    const x = qrLeastSquares(A_ls, b);
    printVector('\n解 x', x);

    console.log(`\n最佳直線：y = ${x[0].toFixed(4)} + ${x[1].toFixed(4)}t`);

    // ========================================
    // 3. 3×3 矩陣的 QR 分解
    // ========================================
    printSeparator('3. 3×3 矩陣的 QR 分解');

    const A2 = [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ];

    console.log('輸入矩陣 A：');
    printMatrix('A', A2);

    const [Q2, R2] = qrDecomposition(A2);

    console.log('\nQR 分解結果：');
    printMatrix('Q', Q2);
    printMatrix('\nR', R2);

    // 總結
    printSeparator('總結');
    console.log(`
QR 分解核心：

1. A = QR
   - Q: 標準正交矩陣 (QᵀQ = I)
   - R: 上三角矩陣

2. Gram-Schmidt 演算法：
   - 對 A 的行向量正交化得到 Q
   - R 的元素是投影係數

3. 用 QR 解最小平方：
   min ‖Ax - b‖²
   → Rx = Qᵀb

4. 優勢：
   - 比正規方程更穩定
   - 避免計算 AᵀA
`);

    console.log('='.repeat(60));
    console.log('示範完成！');
    console.log('='.repeat(60));
}

main();
