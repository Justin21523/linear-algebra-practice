/**
 * 行列式的性質 (Determinant Properties)
 *
 * 執行：node determinant.js
 */

function printSeparator(title) {
    console.log();
    console.log('='.repeat(60));
    console.log(title);
    console.log('='.repeat(60));
}

function printMatrix(name, M) {
    console.log(`${name} =`);
    for (const row of M) {
        const formatted = row.map(x => x.toFixed(4).padStart(8)).join(', ');
        console.log(`  [${formatted}]`);
    }
}

// 2×2 行列式
function det2x2(A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

// 3×3 行列式
function det3x3(A) {
    const [a, b, c] = A[0];
    const [d, e, f] = A[1];
    const [g, h, i] = A[2];

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

// n×n 行列式（列運算化為上三角）
function detNxN(A) {
    const n = A.length;
    const M = A.map(row => [...row]);

    let sign = 1;

    for (let col = 0; col < n; col++) {
        // 找主元
        let pivotRow = -1;
        for (let row = col; row < n; row++) {
            if (Math.abs(M[row][col]) > 1e-10) {
                pivotRow = row;
                break;
            }
        }

        if (pivotRow === -1) return 0;

        // 列交換
        if (pivotRow !== col) {
            [M[col], M[pivotRow]] = [M[pivotRow], M[col]];
            sign *= -1;
        }

        // 消去
        for (let row = col + 1; row < n; row++) {
            const factor = M[row][col] / M[col][col];
            for (let j = col; j < n; j++) {
                M[row][j] -= factor * M[col][j];
            }
        }
    }

    let det = sign;
    for (let i = 0; i < n; i++) {
        det *= M[i][i];
    }

    return det;
}

// 矩陣乘法
function matrixMultiply(A, B) {
    const m = A.length, k = B.length, n = B[0].length;
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
    const m = A.length, n = A[0].length;
    return Array.from({length: n}, (_, j) =>
        Array.from({length: m}, (_, i) => A[i][j])
    );
}

// 純量乘矩陣
function scalarMultiply(c, A) {
    return A.map(row => row.map(x => c * x));
}

function main() {
    printSeparator('行列式性質示範 (JavaScript)');

    // ========================================
    // 1. 基本計算
    // ========================================
    printSeparator('1. 基本行列式計算');

    const A2 = [[3, 8], [4, 6]];
    printMatrix('A (2×2)', A2);
    console.log(`det(A) = ${det2x2(A2)}`);

    let A3 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ];
    printMatrix('\nA (3×3)', A3);
    console.log(`det(A) = ${det3x3(A3)}`);

    // ========================================
    // 2. 性質 1：det(I) = 1
    // ========================================
    printSeparator('2. 性質 1：det(I) = 1');

    const I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    printMatrix('I₃', I3);
    console.log(`det(I₃) = ${det3x3(I3)}`);

    // ========================================
    // 3. 性質 2：列交換變號
    // ========================================
    printSeparator('3. 性質 2：列交換變號');

    let A = [[1, 2], [3, 4]];
    printMatrix('A', A);
    console.log(`det(A) = ${det2x2(A)}`);

    const A_swap = [[3, 4], [1, 2]];
    printMatrix('\nA（交換列）', A_swap);
    console.log(`det(交換後) = ${det2x2(A_swap)}`);
    console.log('驗證：變號 ✓');

    // ========================================
    // 4. 乘積公式
    // ========================================
    printSeparator('4. 乘積公式：det(AB) = det(A)·det(B)');

    A = [[1, 2], [3, 4]];
    const B = [[5, 6], [7, 8]];
    const AB = matrixMultiply(A, B);

    printMatrix('A', A);
    printMatrix('B', B);
    printMatrix('AB', AB);

    let detA = det2x2(A);
    const detB = det2x2(B);
    const detAB = det2x2(AB);

    console.log(`\ndet(A) = ${detA}`);
    console.log(`det(B) = ${detB}`);
    console.log(`det(A)·det(B) = ${detA * detB}`);
    console.log(`det(AB) = ${detAB}`);

    // ========================================
    // 5. 轉置公式
    // ========================================
    printSeparator('5. 轉置公式：det(Aᵀ) = det(A)');

    A3 = [[1, 2, 3], [4, 5, 6], [7, 8, 10]];
    const AT = transpose(A3);

    printMatrix('A', A3);
    printMatrix('Aᵀ', AT);

    console.log(`\ndet(A) = ${det3x3(A3)}`);
    console.log(`det(Aᵀ) = ${det3x3(AT)}`);

    // ========================================
    // 6. 純量乘法
    // ========================================
    printSeparator('6. 純量乘法：det(cA) = cⁿ·det(A)');

    A = [[1, 2], [3, 4]];
    const c = 2;
    const cA = scalarMultiply(c, A);

    printMatrix('A (2×2)', A);
    console.log(`c = ${c}`);
    printMatrix('cA', cA);

    detA = det2x2(A);
    const detcA = det2x2(cA);
    const n = 2;

    console.log(`\ndet(A) = ${detA}`);
    console.log(`cⁿ·det(A) = ${c}² × ${detA} = ${Math.pow(c, n) * detA}`);
    console.log(`det(cA) = ${detcA}`);

    // ========================================
    // 7. 上三角矩陣
    // ========================================
    printSeparator('7. 上三角矩陣：det = 對角線乘積');

    const U = [[2, 3, 1], [0, 4, 5], [0, 0, 6]];
    printMatrix('U（上三角）', U);
    console.log(`對角線乘積：2 × 4 × 6 = ${2 * 4 * 6}`);
    console.log(`det(U) = ${det3x3(U)}`);

    // ========================================
    // 8. 奇異矩陣
    // ========================================
    printSeparator('8. 奇異矩陣：det(A) = 0');

    const A_singular = [[1, 2], [2, 4]];
    printMatrix('A（列成比例）', A_singular);
    console.log(`det(A) = ${det2x2(A_singular)}`);
    console.log('此矩陣不可逆');

    // 總結
    printSeparator('總結');
    console.log(`
行列式三大性質：
1. det(I) = 1
2. 列交換 → det 變號
3. 對單列線性

重要公式：
- det(AB) = det(A)·det(B)
- det(Aᵀ) = det(A)
- det(A⁻¹) = 1/det(A)
- det(cA) = cⁿ·det(A)
`);

    console.log('='.repeat(60));
    console.log('示範完成！');
    console.log('='.repeat(60));
}

main();
