/**
 * 餘因子展開 (Cofactor Expansion)
 *
 * 執行：node cofactor.js
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

// 取得子矩陣
function getMinorMatrix(A, row, col) {  // EN: Execute line: function getMinorMatrix(A, row, col) {.
    return A.filter((_, i) => i !== row)  // EN: Return from the current function: return A.filter((_, i) => i !== row).
            .map(r => r.filter((_, j) => j !== col));  // EN: Execute a statement: .map(r => r.filter((_, j) => j !== col));.
}  // EN: Structure delimiter for a block or scope.

// 行列式（遞迴餘因子展開）
function determinant(A) {  // EN: Execute line: function determinant(A) {.
    const n = A.length;  // EN: Execute a statement: const n = A.length;.
    if (n === 1) return A[0][0];  // EN: Conditional control flow: if (n === 1) return A[0][0];.
    if (n === 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Conditional control flow: if (n === 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];.

    let det = 0;  // EN: Execute a statement: let det = 0;.
    for (let j = 0; j < n; j++) {  // EN: Loop control flow: for (let j = 0; j < n; j++) {.
        const sub = getMinorMatrix(A, 0, j);  // EN: Execute a statement: const sub = getMinorMatrix(A, 0, j);.
        const sign = (j % 2 === 0) ? 1 : -1;  // EN: Execute a statement: const sign = (j % 2 === 0) ? 1 : -1;.
        det += sign * A[0][j] * determinant(sub);  // EN: Execute a statement: det += sign * A[0][j] * determinant(sub);.
    }  // EN: Structure delimiter for a block or scope.
    return det;  // EN: Return from the current function: return det;.
}  // EN: Structure delimiter for a block or scope.

// 子行列式
function minor(A, i, j) {  // EN: Execute line: function minor(A, i, j) {.
    return determinant(getMinorMatrix(A, i, j));  // EN: Return from the current function: return determinant(getMinorMatrix(A, i, j));.
}  // EN: Structure delimiter for a block or scope.

// 餘因子
function cofactor(A, i, j) {  // EN: Execute line: function cofactor(A, i, j) {.
    const sign = ((i + j) % 2 === 0) ? 1 : -1;  // EN: Execute a statement: const sign = ((i + j) % 2 === 0) ? 1 : -1;.
    return sign * minor(A, i, j);  // EN: Return from the current function: return sign * minor(A, i, j);.
}  // EN: Structure delimiter for a block or scope.

// 餘因子矩陣
function cofactorMatrix(A) {  // EN: Execute line: function cofactorMatrix(A) {.
    const n = A.length;  // EN: Execute a statement: const n = A.length;.
    return Array.from({length: n}, (_, i) =>  // EN: Return from the current function: return Array.from({length: n}, (_, i) =>.
        Array.from({length: n}, (_, j) => cofactor(A, i, j))  // EN: Execute line: Array.from({length: n}, (_, j) => cofactor(A, i, j)).
    );  // EN: Execute a statement: );.
}  // EN: Structure delimiter for a block or scope.

// 轉置
function transpose(A) {  // EN: Execute line: function transpose(A) {.
    const n = A.length;  // EN: Execute a statement: const n = A.length;.
    return Array.from({length: n}, (_, j) =>  // EN: Return from the current function: return Array.from({length: n}, (_, j) =>.
        Array.from({length: n}, (_, i) => A[i][j])  // EN: Execute line: Array.from({length: n}, (_, i) => A[i][j]).
    );  // EN: Execute a statement: );.
}  // EN: Structure delimiter for a block or scope.

// 伴隨矩陣
function adjugate(A) {  // EN: Execute line: function adjugate(A) {.
    return transpose(cofactorMatrix(A));  // EN: Return from the current function: return transpose(cofactorMatrix(A));.
}  // EN: Structure delimiter for a block or scope.

// 逆矩陣
function inverse(A) {  // EN: Execute line: function inverse(A) {.
    const det = determinant(A);  // EN: Execute a statement: const det = determinant(A);.
    const adj = adjugate(A);  // EN: Execute a statement: const adj = adjugate(A);.
    return adj.map(row => row.map(x => x / det));  // EN: Return from the current function: return adj.map(row => row.map(x => x / det));.
}  // EN: Structure delimiter for a block or scope.

// 矩陣乘法
function multiply(A, B) {  // EN: Execute line: function multiply(A, B) {.
    const n = A.length;  // EN: Execute a statement: const n = A.length;.
    return Array.from({length: n}, (_, i) =>  // EN: Return from the current function: return Array.from({length: n}, (_, i) =>.
        Array.from({length: n}, (_, j) =>  // EN: Execute line: Array.from({length: n}, (_, j) =>.
            A[i].reduce((sum, _, k) => sum + A[i][k] * B[k][j], 0)  // EN: Execute line: A[i].reduce((sum, _, k) => sum + A[i][k] * B[k][j], 0).
        )  // EN: Execute line: ).
    );  // EN: Execute a statement: );.
}  // EN: Structure delimiter for a block or scope.

function main() {  // EN: Execute line: function main() {.
    printSeparator('餘因子展開示範 (JavaScript)');  // EN: Execute a statement: printSeparator('餘因子展開示範 (JavaScript)');.

    // ========================================
    // 1. 子行列式與餘因子
    // ========================================
    printSeparator('1. 子行列式與餘因子');  // EN: Execute a statement: printSeparator('1. 子行列式與餘因子');.

    const A = [  // EN: Execute line: const A = [.
        [1, 2, 3],  // EN: Execute line: [1, 2, 3],.
        [4, 5, 6],  // EN: Execute line: [4, 5, 6],.
        [7, 8, 9]  // EN: Execute line: [7, 8, 9].
    ];  // EN: Execute a statement: ];.

    printMatrix('A', A);  // EN: Execute a statement: printMatrix('A', A);.

    console.log('\n所有餘因子 Cᵢⱼ：');  // EN: Execute a statement: console.log('\n所有餘因子 Cᵢⱼ：');.
    for (let i = 0; i < 3; i++) {  // EN: Loop control flow: for (let i = 0; i < 3; i++) {.
        let line = '';  // EN: Execute a statement: let line = '';.
        for (let j = 0; j < 3; j++) {  // EN: Loop control flow: for (let j = 0; j < 3; j++) {.
            line += `  C${i+1}${j+1} = ${cofactor(A, i, j).toFixed(4).padStart(8)}`;  // EN: Execute a statement: line += ` C${i+1}${j+1} = ${cofactor(A, i, j).toFixed(4).padStart(8)}`;.
        }  // EN: Structure delimiter for a block or scope.
        console.log(line);  // EN: Execute a statement: console.log(line);.
    }  // EN: Structure delimiter for a block or scope.

    // ========================================
    // 2. 餘因子展開
    // ========================================
    printSeparator('2. 餘因子展開計算行列式');  // EN: Execute a statement: printSeparator('2. 餘因子展開計算行列式');.

    console.log('沿第一列展開：');  // EN: Execute a statement: console.log('沿第一列展開：');.
    console.log('det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃');  // EN: Execute a statement: console.log('det(A) = a₁₁C₁₁ + a₁₂C₁₂ + a₁₃C₁₃');.
    console.log(`       = ${A[0][0]}×${cofactor(A, 0, 0)} + ${A[0][1]}×${cofactor(A, 0, 1)} + ${A[0][2]}×${cofactor(A, 0, 2)}`);  // EN: Execute a statement: console.log(` = ${A[0][0]}×${cofactor(A, 0, 0)} + ${A[0][1]}×${cofactor….
    console.log(`       = ${determinant(A)}`);  // EN: Execute a statement: console.log(` = ${determinant(A)}`);.

    // ========================================
    // 3. 餘因子矩陣與伴隨矩陣
    // ========================================
    printSeparator('3. 餘因子矩陣與伴隨矩陣');  // EN: Execute a statement: printSeparator('3. 餘因子矩陣與伴隨矩陣');.

    const B = [  // EN: Execute line: const B = [.
        [2, 1, 3],  // EN: Execute line: [2, 1, 3],.
        [1, 0, 2],  // EN: Execute line: [1, 0, 2],.
        [4, 1, 5]  // EN: Execute line: [4, 1, 5].
    ];  // EN: Execute a statement: ];.

    printMatrix('A', B);  // EN: Execute a statement: printMatrix('A', B);.
    console.log(`\ndet(A) = ${determinant(B)}`);  // EN: Execute a statement: console.log(`\ndet(A) = ${determinant(B)}`);.

    const C = cofactorMatrix(B);  // EN: Execute a statement: const C = cofactorMatrix(B);.
    printMatrix('\n餘因子矩陣 C', C);  // EN: Execute a statement: printMatrix('\n餘因子矩陣 C', C);.

    const adj = adjugate(B);  // EN: Execute a statement: const adj = adjugate(B);.
    printMatrix('\n伴隨矩陣 adj(A) = Cᵀ', adj);  // EN: Execute a statement: printMatrix('\n伴隨矩陣 adj(A) = Cᵀ', adj);.

    // ========================================
    // 4. 用伴隨矩陣求逆矩陣
    // ========================================
    printSeparator('4. 用伴隨矩陣求逆矩陣');  // EN: Execute a statement: printSeparator('4. 用伴隨矩陣求逆矩陣');.

    console.log('A⁻¹ = adj(A) / det(A)');  // EN: Execute a statement: console.log('A⁻¹ = adj(A) / det(A)');.

    const B_inv = inverse(B);  // EN: Execute a statement: const B_inv = inverse(B);.
    printMatrix('\nA⁻¹', B_inv);  // EN: Execute a statement: printMatrix('\nA⁻¹', B_inv);.

    // 驗證
    const I = multiply(B, B_inv);  // EN: Execute a statement: const I = multiply(B, B_inv);.
    printMatrix('\n驗證 A × A⁻¹', I);  // EN: Execute a statement: printMatrix('\n驗證 A × A⁻¹', I);.

    // ========================================
    // 5. 2×2 特例
    // ========================================
    printSeparator('5. 2×2 伴隨矩陣公式');  // EN: Execute a statement: printSeparator('5. 2×2 伴隨矩陣公式');.

    const A2 = [[3, 4], [5, 6]];  // EN: Execute a statement: const A2 = [[3, 4], [5, 6]];.
    printMatrix('A', A2);  // EN: Execute a statement: printMatrix('A', A2);.

    console.log('\n對於 [[a,b],[c,d]]:');  // EN: Execute a statement: console.log('\n對於 [[a,b],[c,d]]:');.
    console.log(`adj(A) = [[d,-b],[-c,a]] = [[${A2[1][1]},${-A2[0][1]}],[${-A2[1][0]},${A2[0][0]}]]`);  // EN: Execute a statement: console.log(`adj(A) = [[d,-b],[-c,a]] = [[${A2[1][1]},${-A2[0][1]}],[${….

    const adj2 = adjugate(A2);  // EN: Execute a statement: const adj2 = adjugate(A2);.
    printMatrix('\n計算得到的 adj(A)', adj2);  // EN: Execute a statement: printMatrix('\n計算得到的 adj(A)', adj2);.

    // 總結
    printSeparator('總結');  // EN: Execute a statement: printSeparator('總結');.
    console.log(`  // EN: Execute line: console.log(`.
餘因子展開公式：  // EN: Execute line: 餘因子展開公式：.
  Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ  // EN: Execute line: Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ.
  det(A) = Σⱼ aᵢⱼ Cᵢⱼ  // EN: Execute line: det(A) = Σⱼ aᵢⱼ Cᵢⱼ.

伴隨矩陣：  // EN: Execute line: 伴隨矩陣：.
  adj(A) = Cᵀ  // EN: Execute line: adj(A) = Cᵀ.

逆矩陣：  // EN: Execute line: 逆矩陣：.
  A⁻¹ = adj(A) / det(A)  // EN: Execute line: A⁻¹ = adj(A) / det(A).

時間複雜度：O(n!)  // EN: Execute line: 時間複雜度：O(n!).
`);  // EN: Execute a statement: `);.

    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log('示範完成！');  // EN: Execute a statement: console.log('示範完成！');.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

main();  // EN: Execute a statement: main();.
