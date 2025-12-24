/**
 * 餘因子展開 (Cofactor Expansion)
 *
 * 執行：node cofactor.js
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

// 取得子矩陣
function getMinorMatrix(A, row, col) {
    return A.filter((_, i) => i !== row)
            .map(r => r.filter((_, j) => j !== col));
}

// 行列式（遞迴餘因子展開）
function determinant(A) {
    const n = A.length;
    if (n === 1) return A[0][0];
    if (n === 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];

    let det = 0;
    for (let j = 0; j < n; j++) {
        const sub = getMinorMatrix(A, 0, j);
        const sign = (j % 2 === 0) ? 1 : -1;
        det += sign * A[0][j] * determinant(sub);
    }
    return det;
}

// 子行列式
function minor(A, i, j) {
    return determinant(getMinorMatrix(A, i, j));
}

// 餘因子
function cofactor(A, i, j) {
    const sign = ((i + j) % 2 === 0) ? 1 : -1;
    return sign * minor(A, i, j);
}

// 餘因子矩陣
function cofactorMatrix(A) {
    const n = A.length;
    return Array.from({length: n}, (_, i) =>
        Array.from({length: n}, (_, j) => cofactor(A, i, j))
    );
}

// 轉置
function transpose(A) {
    const n = A.length;
    return Array.from({length: n}, (_, j) =>
        Array.from({length: n}, (_, i) => A[i][j])
    );
}

// 伴隨矩陣
function adjugate(A) {
    return transpose(cofactorMatrix(A));
}

// 逆矩陣
function inverse(A) {
    const det = determinant(A);
    const adj = adjugate(A);
    return adj.map(row => row.map(x => x / det));
}

// 矩陣乘法
function multiply(A, B) {
    const n = A.length;
    return Array.from({length: n}, (_, i) =>
        Array.from({length: n}, (_, j) =>
            A[i].reduce((sum, _, k) => sum + A[i][k] * B[k][j], 0)
        )
    );
}

function main() {
    printSeparator('餘因子展開示範 (JavaScript)');

    // ========================================
    // 1. 子行列式與餘因子
    // ========================================
    printSeparator('1. 子行列式與餘因子');

    const A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ];

    printMatrix('A', A);

    console.log('\n所有餘因子 Cᵢⱼ：');
    for (let i = 0; i < 3; i++) {
        let line = '';
        for (let j = 0; j < 3; j++) {
            line += `  C${i+1}${j+1} = ${cofactor(A, i, j).toFixed(4).padStart(8)}`;
        }
        console.log(line);
    }

    // ========================================
    // 2. 餘因子展開
    // ========================================
    printSeparator('2. 餘因子展開計算行列式');

    console.log('沿第一列展開：');
    console.log('det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃');
    console.log(`       = ${A[0][0]}×${cofactor(A, 0, 0)} + ${A[0][1]}×${cofactor(A, 0, 1)} + ${A[0][2]}×${cofactor(A, 0, 2)}`);
    console.log(`       = ${determinant(A)}`);

    // ========================================
    // 3. 餘因子矩陣與伴隨矩陣
    // ========================================
    printSeparator('3. 餘因子矩陣與伴隨矩陣');

    const B = [
        [2, 1, 3],
        [1, 0, 2],
        [4, 1, 5]
    ];

    printMatrix('A', B);
    console.log(`\ndet(A) = ${determinant(B)}`);

    const C = cofactorMatrix(B);
    printMatrix('\n餘因子矩陣 C', C);

    const adj = adjugate(B);
    printMatrix('\n伴隨矩陣 adj(A) = Cᵀ', adj);

    // ========================================
    // 4. 用伴隨矩陣求逆矩陣
    // ========================================
    printSeparator('4. 用伴隨矩陣求逆矩陣');

    console.log('A⁻¹ = adj(A) / det(A)');

    const B_inv = inverse(B);
    printMatrix('\nA⁻¹', B_inv);

    // 驗證
    const I = multiply(B, B_inv);
    printMatrix('\n驗證 A × A⁻¹', I);

    // ========================================
    // 5. 2×2 特例
    // ========================================
    printSeparator('5. 2×2 伴隨矩陣公式');

    const A2 = [[3, 4], [5, 6]];
    printMatrix('A', A2);

    console.log('\n對於 [[a,b],[c,d]]:');
    console.log(`adj(A) = [[d,-b],[-c,a]] = [[${A2[1][1]},${-A2[0][1]}],[${-A2[1][0]},${A2[0][0]}]]`);

    const adj2 = adjugate(A2);
    printMatrix('\n計算得到的 adj(A)', adj2);

    // 總結
    printSeparator('總結');
    console.log(`
餘因子展開公式：
  Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ
  det(A) = Σⱼ aᵢⱼ Cᵢⱼ

伴隨矩陣：
  adj(A) = Cᵀ

逆矩陣：
  A⁻¹ = adj(A) / det(A)

時間複雜度：O(n!)
`);

    console.log('='.repeat(60));
    console.log('示範完成！');
    console.log('='.repeat(60));
}

main();
