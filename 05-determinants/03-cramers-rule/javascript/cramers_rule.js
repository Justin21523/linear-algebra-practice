/**
 * 克萊姆法則 (Cramer's Rule)
 *
 * 執行：node cramers_rule.js
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

function printVector(name, v) {  // EN: Execute line: function printVector(name, v) {.
    const formatted = v.map(x => x.toFixed(4)).join(', ');  // EN: Execute a statement: const formatted = v.map(x => x.toFixed(4)).join(', ');.
    console.log(`${name} = [${formatted}]`);  // EN: Execute a statement: console.log(`${name} = [${formatted}]`);.
}  // EN: Structure delimiter for a block or scope.

// 2×2 行列式
function det2x2(A) {  // EN: Execute line: function det2x2(A) {.
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Return from the current function: return A[0][0] * A[1][1] - A[0][1] * A[1][0];.
}  // EN: Structure delimiter for a block or scope.

// 3×3 行列式
function det3x3(A) {  // EN: Execute line: function det3x3(A) {.
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])  // EN: Return from the current function: return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]).
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])  // EN: Execute line: - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]).
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);  // EN: Execute a statement: + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);.
}  // EN: Structure delimiter for a block or scope.

function determinant(A) {  // EN: Execute line: function determinant(A) {.
    const n = A.length;  // EN: Execute a statement: const n = A.length;.
    if (n === 2) return det2x2(A);  // EN: Conditional control flow: if (n === 2) return det2x2(A);.
    if (n === 3) return det3x3(A);  // EN: Conditional control flow: if (n === 3) return det3x3(A);.
    throw new Error('僅支援 2×2 和 3×3 矩陣');  // EN: Execute a statement: throw new Error('僅支援 2×2 和 3×3 矩陣');.
}  // EN: Structure delimiter for a block or scope.

// 替換第 j 行
function replaceColumn(A, b, j) {  // EN: Execute line: function replaceColumn(A, b, j) {.
    return A.map((row, i) =>  // EN: Return from the current function: return A.map((row, i) =>.
        row.map((val, k) => k === j ? b[i] : val)  // EN: Execute line: row.map((val, k) => k === j ? b[i] : val).
    );  // EN: Execute a statement: );.
}  // EN: Structure delimiter for a block or scope.

// 克萊姆法則
function cramersRule(A, b) {  // EN: Execute line: function cramersRule(A, b) {.
    const n = A.length;  // EN: Execute a statement: const n = A.length;.
    const detA = determinant(A);  // EN: Execute a statement: const detA = determinant(A);.

    if (Math.abs(detA) < 1e-10) {  // EN: Conditional control flow: if (Math.abs(detA) < 1e-10) {.
        throw new Error('矩陣奇異');  // EN: Execute a statement: throw new Error('矩陣奇異');.
    }  // EN: Structure delimiter for a block or scope.

    const x = [];  // EN: Execute a statement: const x = [];.
    for (let j = 0; j < n; j++) {  // EN: Loop control flow: for (let j = 0; j < n; j++) {.
        const Aj = replaceColumn(A, b, j);  // EN: Execute a statement: const Aj = replaceColumn(A, b, j);.
        x.push(determinant(Aj) / detA);  // EN: Execute a statement: x.push(determinant(Aj) / detA);.
    }  // EN: Structure delimiter for a block or scope.
    return x;  // EN: Return from the current function: return x;.
}  // EN: Structure delimiter for a block or scope.

function main() {  // EN: Execute line: function main() {.
    printSeparator('克萊姆法則示範 (JavaScript)');  // EN: Execute a statement: printSeparator('克萊姆法則示範 (JavaScript)');.

    // ========================================
    // 1. 2×2 系統
    // ========================================
    printSeparator('1. 2×2 系統');  // EN: Execute a statement: printSeparator('1. 2×2 系統');.

    const A2 = [[2, 3], [4, 5]];  // EN: Execute a statement: const A2 = [[2, 3], [4, 5]];.
    const b2 = [8, 14];  // EN: Execute a statement: const b2 = [8, 14];.

    console.log('方程組：');  // EN: Execute a statement: console.log('方程組：');.
    console.log('  2x + 3y = 8');  // EN: Execute a statement: console.log(' 2x + 3y = 8');.
    console.log('  4x + 5y = 14');  // EN: Execute a statement: console.log(' 4x + 5y = 14');.

    printMatrix('\nA', A2);  // EN: Execute a statement: printMatrix('\nA', A2);.
    printVector('b', b2);  // EN: Execute a statement: printVector('b', b2);.

    const detA2 = determinant(A2);  // EN: Execute a statement: const detA2 = determinant(A2);.
    console.log(`\ndet(A) = ${detA2.toFixed(4)}`);  // EN: Execute a statement: console.log(`\ndet(A) = ${detA2.toFixed(4)}`);.

    const x2 = cramersRule(A2, b2);  // EN: Execute a statement: const x2 = cramersRule(A2, b2);.

    for (let j = 0; j < 2; j++) {  // EN: Loop control flow: for (let j = 0; j < 2; j++) {.
        const Aj = replaceColumn(A2, b2, j);  // EN: Execute a statement: const Aj = replaceColumn(A2, b2, j);.
        const detAj = determinant(Aj);  // EN: Execute a statement: const detAj = determinant(Aj);.
        console.log(`\nA${j+1}（第 ${j+1} 行換成 b）：`);  // EN: Execute a statement: console.log(`\nA${j+1}（第 ${j+1} 行換成 b）：`);.
        printMatrix('', Aj);  // EN: Execute a statement: printMatrix('', Aj);.
        console.log(`det(A${j+1}) = ${detAj.toFixed(4)}`);  // EN: Execute a statement: console.log(`det(A${j+1}) = ${detAj.toFixed(4)}`);.
        console.log(`x${j+1} = ${x2[j].toFixed(4)}`);  // EN: Execute a statement: console.log(`x${j+1} = ${x2[j].toFixed(4)}`);.
    }  // EN: Structure delimiter for a block or scope.

    console.log(`\n解：x = ${x2[0].toFixed(4)}, y = ${x2[1].toFixed(4)}`);  // EN: Execute a statement: console.log(`\n解：x = ${x2[0].toFixed(4)}, y = ${x2[1].toFixed(4)}`);.

    // ========================================
    // 2. 3×3 系統
    // ========================================
    printSeparator('2. 3×3 系統');  // EN: Execute a statement: printSeparator('2. 3×3 系統');.

    const A3 = [  // EN: Execute line: const A3 = [.
        [2, 1, -1],  // EN: Execute line: [2, 1, -1],.
        [-3, -1, 2],  // EN: Execute line: [-3, -1, 2],.
        [-2, 1, 2]  // EN: Execute line: [-2, 1, 2].
    ];  // EN: Execute a statement: ];.
    const b3 = [8, -11, -3];  // EN: Execute a statement: const b3 = [8, -11, -3];.

    console.log('方程組：');  // EN: Execute a statement: console.log('方程組：');.
    console.log('   2x +  y -  z =  8');  // EN: Execute a statement: console.log(' 2x + y - z = 8');.
    console.log('  -3x -  y + 2z = -11');  // EN: Execute a statement: console.log(' -3x - y + 2z = -11');.
    console.log('  -2x +  y + 2z = -3');  // EN: Execute a statement: console.log(' -2x + y + 2z = -3');.

    printMatrix('\nA', A3);  // EN: Execute a statement: printMatrix('\nA', A3);.
    printVector('b', b3);  // EN: Execute a statement: printVector('b', b3);.

    const x3 = cramersRule(A3, b3);  // EN: Execute a statement: const x3 = cramersRule(A3, b3);.

    console.log(`\n解：x = ${x3[0].toFixed(4)}, y = ${x3[1].toFixed(4)}, z = ${x3[2].toFixed(4)}`);  // EN: Execute a statement: console.log(`\n解：x = ${x3[0].toFixed(4)}, y = ${x3[1].toFixed(4)}, z = ….

    // 驗證
    console.log('\n驗證：');  // EN: Execute a statement: console.log('\n驗證：');.
    console.log(`  2(${x3[0]}) + (${x3[1]}) - (${x3[2]}) = ${(2*x3[0] + x3[1] - x3[2]).toFixed(4)}`);  // EN: Execute a statement: console.log(` 2(${x3[0]}) + (${x3[1]}) - (${x3[2]}) = ${(2*x3[0] + x3[1….
    console.log(`  -3(${x3[0]}) - (${x3[1]}) + 2(${x3[2]}) = ${(-3*x3[0] - x3[1] + 2*x3[2]).toFixed(4)}`);  // EN: Execute a statement: console.log(` -3(${x3[0]}) - (${x3[1]}) + 2(${x3[2]}) = ${(-3*x3[0] - x….
    console.log(`  -2(${x3[0]}) + (${x3[1]}) + 2(${x3[2]}) = ${(-2*x3[0] + x3[1] + 2*x3[2]).toFixed(4)}`);  // EN: Execute a statement: console.log(` -2(${x3[0]}) + (${x3[1]}) + 2(${x3[2]}) = ${(-2*x3[0] + x….

    // 總結
    printSeparator('總結');  // EN: Execute a statement: printSeparator('總結');.
    console.log(`  // EN: Execute line: console.log(`.
克萊姆法則：  // EN: Execute line: 克萊姆法則：.
  xⱼ = det(Aⱼ) / det(A)  // EN: Execute line: xⱼ = det(Aⱼ) / det(A).
  Aⱼ = A 的第 j 行換成 b  // EN: Execute line: Aⱼ = A 的第 j 行換成 b.

適用條件：  // EN: Execute line: 適用條件：.
  - det(A) ≠ 0  // EN: Execute line: - det(A) ≠ 0.
  - 方陣系統  // EN: Execute line: - 方陣系統.

時間複雜度：O(n! × n)  // EN: Execute line: 時間複雜度：O(n! × n).
`);  // EN: Execute a statement: `);.

    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log('示範完成！');  // EN: Execute a statement: console.log('示範完成！');.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

main();  // EN: Execute a statement: main();.
