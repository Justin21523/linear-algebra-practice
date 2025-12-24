/**
 * 行列式的性質 (Determinant Properties)
 *
 * 執行：node determinant.js
 */

function printSeparator(title) {  // EN: Execute line: function printSeparator(title) {.
    console.log();  // EN: Execute a statement: console.log();.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log(title);  // EN: Execute a statement: console.log(title);.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

function printMatrix(name, M) {  // EN: Execute line: function printMatrix(name, M) {.
    console.log(`${name} =`);  // EN: Execute a statement: console.log(`${name} =`);.
    for (const row of M) {  // EN: Loop control flow: for (const row of M) {.
        const formatted = row.map(x => x.toFixed(4).padStart(8)).join(', ');  // EN: Execute a statement: const formatted = row.map(x => x.toFixed(4).padStart(8)).join(', ');.
        console.log(`  [${formatted}]`);  // EN: Execute a statement: console.log(` [${formatted}]`);.
    }  // EN: Structure delimiter for a block or scope.
}  // EN: Structure delimiter for a block or scope.

// 2×2 行列式
function det2x2(A) {  // EN: Execute line: function det2x2(A) {.
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Return from the current function: return A[0][0] * A[1][1] - A[0][1] * A[1][0];.
}  // EN: Structure delimiter for a block or scope.

// 3×3 行列式
function det3x3(A) {  // EN: Execute line: function det3x3(A) {.
    const [a, b, c] = A[0];  // EN: Execute a statement: const [a, b, c] = A[0];.
    const [d, e, f] = A[1];  // EN: Execute a statement: const [d, e, f] = A[1];.
    const [g, h, i] = A[2];  // EN: Execute a statement: const [g, h, i] = A[2];.

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);  // EN: Return from the current function: return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);.
}  // EN: Structure delimiter for a block or scope.

// n×n 行列式（列運算化為上三角）
function detNxN(A) {  // EN: Execute line: function detNxN(A) {.
    const n = A.length;  // EN: Execute a statement: const n = A.length;.
    const M = A.map(row => [...row]);  // EN: Execute a statement: const M = A.map(row => [...row]);.

    let sign = 1;  // EN: Execute a statement: let sign = 1;.

    for (let col = 0; col < n; col++) {  // EN: Loop control flow: for (let col = 0; col < n; col++) {.
        // 找主元
        let pivotRow = -1;  // EN: Execute a statement: let pivotRow = -1;.
        for (let row = col; row < n; row++) {  // EN: Loop control flow: for (let row = col; row < n; row++) {.
            if (Math.abs(M[row][col]) > 1e-10) {  // EN: Conditional control flow: if (Math.abs(M[row][col]) > 1e-10) {.
                pivotRow = row;  // EN: Execute a statement: pivotRow = row;.
                break;  // EN: Execute a statement: break;.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.

        if (pivotRow === -1) return 0;  // EN: Conditional control flow: if (pivotRow === -1) return 0;.

        // 列交換
        if (pivotRow !== col) {  // EN: Conditional control flow: if (pivotRow !== col) {.
            [M[col], M[pivotRow]] = [M[pivotRow], M[col]];  // EN: Execute a statement: [M[col], M[pivotRow]] = [M[pivotRow], M[col]];.
            sign *= -1;  // EN: Execute a statement: sign *= -1;.
        }  // EN: Structure delimiter for a block or scope.

        // 消去
        for (let row = col + 1; row < n; row++) {  // EN: Loop control flow: for (let row = col + 1; row < n; row++) {.
            const factor = M[row][col] / M[col][col];  // EN: Execute a statement: const factor = M[row][col] / M[col][col];.
            for (let j = col; j < n; j++) {  // EN: Loop control flow: for (let j = col; j < n; j++) {.
                M[row][j] -= factor * M[col][j];  // EN: Execute a statement: M[row][j] -= factor * M[col][j];.
            }  // EN: Structure delimiter for a block or scope.
        }  // EN: Structure delimiter for a block or scope.
    }  // EN: Structure delimiter for a block or scope.

    let det = sign;  // EN: Execute a statement: let det = sign;.
    for (let i = 0; i < n; i++) {  // EN: Loop control flow: for (let i = 0; i < n; i++) {.
        det *= M[i][i];  // EN: Execute a statement: det *= M[i][i];.
    }  // EN: Structure delimiter for a block or scope.

    return det;  // EN: Return from the current function: return det;.
}  // EN: Structure delimiter for a block or scope.

// 矩陣乘法
function matrixMultiply(A, B) {  // EN: Execute line: function matrixMultiply(A, B) {.
    const m = A.length, k = B.length, n = B[0].length;  // EN: Execute a statement: const m = A.length, k = B.length, n = B[0].length;.
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
    const m = A.length, n = A[0].length;  // EN: Execute a statement: const m = A.length, n = A[0].length;.
    return Array.from({length: n}, (_, j) =>  // EN: Return from the current function: return Array.from({length: n}, (_, j) =>.
        Array.from({length: m}, (_, i) => A[i][j])  // EN: Execute line: Array.from({length: m}, (_, i) => A[i][j]).
    );  // EN: Execute a statement: );.
}  // EN: Structure delimiter for a block or scope.

// 純量乘矩陣
function scalarMultiply(c, A) {  // EN: Execute line: function scalarMultiply(c, A) {.
    return A.map(row => row.map(x => c * x));  // EN: Return from the current function: return A.map(row => row.map(x => c * x));.
}  // EN: Structure delimiter for a block or scope.

function main() {  // EN: Execute line: function main() {.
    printSeparator('行列式性質示範 (JavaScript)');  // EN: Execute a statement: printSeparator('行列式性質示範 (JavaScript)');.

    // ========================================
    // 1. 基本計算
    // ========================================
    printSeparator('1. 基本行列式計算');  // EN: Execute a statement: printSeparator('1. 基本行列式計算');.

    const A2 = [[3, 8], [4, 6]];  // EN: Execute a statement: const A2 = [[3, 8], [4, 6]];.
    printMatrix('A (2×2)', A2);  // EN: Execute a statement: printMatrix('A (2×2)', A2);.
    console.log(`det(A) = ${det2x2(A2)}`);  // EN: Execute a statement: console.log(`det(A) = ${det2x2(A2)}`);.

    let A3 = [  // EN: Execute line: let A3 = [.
        [1, 2, 3],  // EN: Execute line: [1, 2, 3],.
        [4, 5, 6],  // EN: Execute line: [4, 5, 6],.
        [7, 8, 10]  // EN: Execute line: [7, 8, 10].
    ];  // EN: Execute a statement: ];.
    printMatrix('\nA (3×3)', A3);  // EN: Execute a statement: printMatrix('\nA (3×3)', A3);.
    console.log(`det(A) = ${det3x3(A3)}`);  // EN: Execute a statement: console.log(`det(A) = ${det3x3(A3)}`);.

    // ========================================
    // 2. 性質 1：det(I) = 1
    // ========================================
    printSeparator('2. 性質 1：det(I) = 1');  // EN: Execute a statement: printSeparator('2. 性質 1：det(I) = 1');.

    const I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];  // EN: Execute a statement: const I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];.
    printMatrix('I₃', I3);  // EN: Execute a statement: printMatrix('I₃', I3);.
    console.log(`det(I₃) = ${det3x3(I3)}`);  // EN: Execute a statement: console.log(`det(I₃) = ${det3x3(I3)}`);.

    // ========================================
    // 3. 性質 2：列交換變號
    // ========================================
    printSeparator('3. 性質 2：列交換變號');  // EN: Execute a statement: printSeparator('3. 性質 2：列交換變號');.

    let A = [[1, 2], [3, 4]];  // EN: Execute a statement: let A = [[1, 2], [3, 4]];.
    printMatrix('A', A);  // EN: Execute a statement: printMatrix('A', A);.
    console.log(`det(A) = ${det2x2(A)}`);  // EN: Execute a statement: console.log(`det(A) = ${det2x2(A)}`);.

    const A_swap = [[3, 4], [1, 2]];  // EN: Execute a statement: const A_swap = [[3, 4], [1, 2]];.
    printMatrix('\nA（交換列）', A_swap);  // EN: Execute a statement: printMatrix('\nA（交換列）', A_swap);.
    console.log(`det(交換後) = ${det2x2(A_swap)}`);  // EN: Execute a statement: console.log(`det(交換後) = ${det2x2(A_swap)}`);.
    console.log('驗證：變號 ✓');  // EN: Execute a statement: console.log('驗證：變號 ✓');.

    // ========================================
    // 4. 乘積公式
    // ========================================
    printSeparator('4. 乘積公式：det(AB) = det(A)·det(B)');  // EN: Execute a statement: printSeparator('4. 乘積公式：det(AB) = det(A)·det(B)');.

    A = [[1, 2], [3, 4]];  // EN: Execute a statement: A = [[1, 2], [3, 4]];.
    const B = [[5, 6], [7, 8]];  // EN: Execute a statement: const B = [[5, 6], [7, 8]];.
    const AB = matrixMultiply(A, B);  // EN: Execute a statement: const AB = matrixMultiply(A, B);.

    printMatrix('A', A);  // EN: Execute a statement: printMatrix('A', A);.
    printMatrix('B', B);  // EN: Execute a statement: printMatrix('B', B);.
    printMatrix('AB', AB);  // EN: Execute a statement: printMatrix('AB', AB);.

    let detA = det2x2(A);  // EN: Execute a statement: let detA = det2x2(A);.
    const detB = det2x2(B);  // EN: Execute a statement: const detB = det2x2(B);.
    const detAB = det2x2(AB);  // EN: Execute a statement: const detAB = det2x2(AB);.

    console.log(`\ndet(A) = ${detA}`);  // EN: Execute a statement: console.log(`\ndet(A) = ${detA}`);.
    console.log(`det(B) = ${detB}`);  // EN: Execute a statement: console.log(`det(B) = ${detB}`);.
    console.log(`det(A)·det(B) = ${detA * detB}`);  // EN: Execute a statement: console.log(`det(A)·det(B) = ${detA * detB}`);.
    console.log(`det(AB) = ${detAB}`);  // EN: Execute a statement: console.log(`det(AB) = ${detAB}`);.

    // ========================================
    // 5. 轉置公式
    // ========================================
    printSeparator('5. 轉置公式：det(Aᵀ) = det(A)');  // EN: Execute a statement: printSeparator('5. 轉置公式：det(Aᵀ) = det(A)');.

    A3 = [[1, 2, 3], [4, 5, 6], [7, 8, 10]];  // EN: Execute a statement: A3 = [[1, 2, 3], [4, 5, 6], [7, 8, 10]];.
    const AT = transpose(A3);  // EN: Execute a statement: const AT = transpose(A3);.

    printMatrix('A', A3);  // EN: Execute a statement: printMatrix('A', A3);.
    printMatrix('Aᵀ', AT);  // EN: Execute a statement: printMatrix('Aᵀ', AT);.

    console.log(`\ndet(A) = ${det3x3(A3)}`);  // EN: Execute a statement: console.log(`\ndet(A) = ${det3x3(A3)}`);.
    console.log(`det(Aᵀ) = ${det3x3(AT)}`);  // EN: Execute a statement: console.log(`det(Aᵀ) = ${det3x3(AT)}`);.

    // ========================================
    // 6. 純量乘法
    // ========================================
    printSeparator('6. 純量乘法：det(cA) = cⁿ·det(A)');  // EN: Execute a statement: printSeparator('6. 純量乘法：det(cA) = cⁿ·det(A)');.

    A = [[1, 2], [3, 4]];  // EN: Execute a statement: A = [[1, 2], [3, 4]];.
    const c = 2;  // EN: Execute a statement: const c = 2;.
    const cA = scalarMultiply(c, A);  // EN: Execute a statement: const cA = scalarMultiply(c, A);.

    printMatrix('A (2×2)', A);  // EN: Execute a statement: printMatrix('A (2×2)', A);.
    console.log(`c = ${c}`);  // EN: Execute a statement: console.log(`c = ${c}`);.
    printMatrix('cA', cA);  // EN: Execute a statement: printMatrix('cA', cA);.

    detA = det2x2(A);  // EN: Execute a statement: detA = det2x2(A);.
    const detcA = det2x2(cA);  // EN: Execute a statement: const detcA = det2x2(cA);.
    const n = 2;  // EN: Execute a statement: const n = 2;.

    console.log(`\ndet(A) = ${detA}`);  // EN: Execute a statement: console.log(`\ndet(A) = ${detA}`);.
    console.log(`cⁿ·det(A) = ${c}² × ${detA} = ${Math.pow(c, n) * detA}`);  // EN: Execute a statement: console.log(`cⁿ·det(A) = ${c}² × ${detA} = ${Math.pow(c, n) * detA}`);.
    console.log(`det(cA) = ${detcA}`);  // EN: Execute a statement: console.log(`det(cA) = ${detcA}`);.

    // ========================================
    // 7. 上三角矩陣
    // ========================================
    printSeparator('7. 上三角矩陣：det = 對角線乘積');  // EN: Execute a statement: printSeparator('7. 上三角矩陣：det = 對角線乘積');.

    const U = [[2, 3, 1], [0, 4, 5], [0, 0, 6]];  // EN: Execute a statement: const U = [[2, 3, 1], [0, 4, 5], [0, 0, 6]];.
    printMatrix('U（上三角）', U);  // EN: Execute a statement: printMatrix('U（上三角）', U);.
    console.log(`對角線乘積：2 × 4 × 6 = ${2 * 4 * 6}`);  // EN: Execute a statement: console.log(`對角線乘積：2 × 4 × 6 = ${2 * 4 * 6}`);.
    console.log(`det(U) = ${det3x3(U)}`);  // EN: Execute a statement: console.log(`det(U) = ${det3x3(U)}`);.

    // ========================================
    // 8. 奇異矩陣
    // ========================================
    printSeparator('8. 奇異矩陣：det(A) = 0');  // EN: Execute a statement: printSeparator('8. 奇異矩陣：det(A) = 0');.

    const A_singular = [[1, 2], [2, 4]];  // EN: Execute a statement: const A_singular = [[1, 2], [2, 4]];.
    printMatrix('A（列成比例）', A_singular);  // EN: Execute a statement: printMatrix('A（列成比例）', A_singular);.
    console.log(`det(A) = ${det2x2(A_singular)}`);  // EN: Execute a statement: console.log(`det(A) = ${det2x2(A_singular)}`);.
    console.log('此矩陣不可逆');  // EN: Execute a statement: console.log('此矩陣不可逆');.

    // 總結
    printSeparator('總結');  // EN: Execute a statement: printSeparator('總結');.
    console.log(`  // EN: Execute line: console.log(`.
行列式三大性質：  // EN: Execute line: 行列式三大性質：.
1. det(I) = 1  // EN: Execute line: 1. det(I) = 1.
2. 列交換 → det 變號  // EN: Execute line: 2. 列交換 → det 變號.
3. 對單列線性  // EN: Execute line: 3. 對單列線性.

重要公式：  // EN: Execute line: 重要公式：.
- det(AB) = det(A)·det(B)  // EN: Execute line: - det(AB) = det(A)·det(B).
- det(Aᵀ) = det(A)  // EN: Execute line: - det(Aᵀ) = det(A).
- det(A⁻¹) = 1/det(A)  // EN: Execute line: - det(A⁻¹) = 1/det(A).
- det(cA) = cⁿ·det(A)  // EN: Execute line: - det(cA) = cⁿ·det(A).
`);  // EN: Execute a statement: `);.

    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log('示範完成！');  // EN: Execute a statement: console.log('示範完成！');.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

main();  // EN: Execute a statement: main();.
