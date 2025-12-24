/**
 * 克萊姆法則 (Cramer's Rule)
 *
 * 執行：node cramers_rule.js
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

function printVector(name, v) {
    const formatted = v.map(x => x.toFixed(4)).join(', ');
    console.log(`${name} = [${formatted}]`);
}

// 2×2 行列式
function det2x2(A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

// 3×3 行列式
function det3x3(A) {
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

function determinant(A) {
    const n = A.length;
    if (n === 2) return det2x2(A);
    if (n === 3) return det3x3(A);
    throw new Error('僅支援 2×2 和 3×3 矩陣');
}

// 替換第 j 行
function replaceColumn(A, b, j) {
    return A.map((row, i) =>
        row.map((val, k) => k === j ? b[i] : val)
    );
}

// 克萊姆法則
function cramersRule(A, b) {
    const n = A.length;
    const detA = determinant(A);

    if (Math.abs(detA) < 1e-10) {
        throw new Error('矩陣奇異');
    }

    const x = [];
    for (let j = 0; j < n; j++) {
        const Aj = replaceColumn(A, b, j);
        x.push(determinant(Aj) / detA);
    }
    return x;
}

function main() {
    printSeparator('克萊姆法則示範 (JavaScript)');

    // ========================================
    // 1. 2×2 系統
    // ========================================
    printSeparator('1. 2×2 系統');

    const A2 = [[2, 3], [4, 5]];
    const b2 = [8, 14];

    console.log('方程組：');
    console.log('  2x + 3y = 8');
    console.log('  4x + 5y = 14');

    printMatrix('\nA', A2);
    printVector('b', b2);

    const detA2 = determinant(A2);
    console.log(`\ndet(A) = ${detA2.toFixed(4)}`);

    const x2 = cramersRule(A2, b2);

    for (let j = 0; j < 2; j++) {
        const Aj = replaceColumn(A2, b2, j);
        const detAj = determinant(Aj);
        console.log(`\nA${j+1}（第 ${j+1} 行換成 b）：`);
        printMatrix('', Aj);
        console.log(`det(A${j+1}) = ${detAj.toFixed(4)}`);
        console.log(`x${j+1} = ${x2[j].toFixed(4)}`);
    }

    console.log(`\n解：x = ${x2[0].toFixed(4)}, y = ${x2[1].toFixed(4)}`);

    // ========================================
    // 2. 3×3 系統
    // ========================================
    printSeparator('2. 3×3 系統');

    const A3 = [
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ];
    const b3 = [8, -11, -3];

    console.log('方程組：');
    console.log('   2x +  y -  z =  8');
    console.log('  -3x -  y + 2z = -11');
    console.log('  -2x +  y + 2z = -3');

    printMatrix('\nA', A3);
    printVector('b', b3);

    const x3 = cramersRule(A3, b3);

    console.log(`\n解：x = ${x3[0].toFixed(4)}, y = ${x3[1].toFixed(4)}, z = ${x3[2].toFixed(4)}`);

    // 驗證
    console.log('\n驗證：');
    console.log(`  2(${x3[0]}) + (${x3[1]}) - (${x3[2]}) = ${(2*x3[0] + x3[1] - x3[2]).toFixed(4)}`);
    console.log(`  -3(${x3[0]}) - (${x3[1]}) + 2(${x3[2]}) = ${(-3*x3[0] - x3[1] + 2*x3[2]).toFixed(4)}`);
    console.log(`  -2(${x3[0]}) + (${x3[1]}) + 2(${x3[2]}) = ${(-2*x3[0] + x3[1] + 2*x3[2]).toFixed(4)}`);

    // 總結
    printSeparator('總結');
    console.log(`
克萊姆法則：
  xⱼ = det(Aⱼ) / det(A)
  Aⱼ = A 的第 j 行換成 b

適用條件：
  - det(A) ≠ 0
  - 方陣系統

時間複雜度：O(n! × n)
`);

    console.log('='.repeat(60));
    console.log('示範完成！');
    console.log('='.repeat(60));
}

main();
