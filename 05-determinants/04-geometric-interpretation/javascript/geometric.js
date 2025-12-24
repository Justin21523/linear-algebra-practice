/**
 * 行列式的幾何解釋 (Geometric Interpretation)
 *
 * 執行：node geometric.js
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

// 2D 叉積（純量）
function cross2D(a, b) {  // EN: Execute line: function cross2D(a, b) {.
    return a[0] * b[1] - a[1] * b[0];  // EN: Return from the current function: return a[0] * b[1] - a[1] * b[0];.
}  // EN: Structure delimiter for a block or scope.

// 3D 叉積
function cross3D(a, b) {  // EN: Execute line: function cross3D(a, b) {.
    return [  // EN: Return from the current function: return [.
        a[1] * b[2] - a[2] * b[1],  // EN: Execute line: a[1] * b[2] - a[2] * b[1],.
        a[2] * b[0] - a[0] * b[2],  // EN: Execute line: a[2] * b[0] - a[0] * b[2],.
        a[0] * b[1] - a[1] * b[0]  // EN: Execute line: a[0] * b[1] - a[1] * b[0].
    ];  // EN: Execute a statement: ];.
}  // EN: Structure delimiter for a block or scope.

// 內積
function dot(a, b) {  // EN: Execute line: function dot(a, b) {.
    return a.reduce((sum, ai, i) => sum + ai * b[i], 0);  // EN: Return from the current function: return a.reduce((sum, ai, i) => sum + ai * b[i], 0);.
}  // EN: Structure delimiter for a block or scope.

// 2×2 行列式
function det2x2(A) {  // EN: Execute line: function det2x2(A) {.
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];  // EN: Return from the current function: return A[0][0] * A[1][1] - A[0][1] * A[1][0];.
}  // EN: Structure delimiter for a block or scope.

// 平行四邊形面積
function parallelogramArea(a, b) {  // EN: Execute line: function parallelogramArea(a, b) {.
    return Math.abs(cross2D(a, b));  // EN: Return from the current function: return Math.abs(cross2D(a, b));.
}  // EN: Structure delimiter for a block or scope.

// 平行六面體體積
function parallelepipedVolume(a, b, c) {  // EN: Execute line: function parallelepipedVolume(a, b, c) {.
    const bxc = cross3D(b, c);  // EN: Execute a statement: const bxc = cross3D(b, c);.
    return Math.abs(dot(a, bxc));  // EN: Return from the current function: return Math.abs(dot(a, bxc));.
}  // EN: Structure delimiter for a block or scope.

// 三角形面積
function triangleArea(x1, y1, x2, y2, x3, y3) {  // EN: Execute line: function triangleArea(x1, y1, x2, y2, x3, y3) {.
    return Math.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;  // EN: Return from the current function: return Math.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;.
}  // EN: Structure delimiter for a block or scope.

function main() {  // EN: Execute line: function main() {.
    printSeparator('行列式幾何解釋示範 (JavaScript)');  // EN: Execute a statement: printSeparator('行列式幾何解釋示範 (JavaScript)');.

    // ========================================
    // 1. 平行四邊形面積
    // ========================================
    printSeparator('1. 平行四邊形面積');  // EN: Execute a statement: printSeparator('1. 平行四邊形面積');.

    let a = [3, 0];  // EN: Execute a statement: let a = [3, 0];.
    let b = [1, 2];  // EN: Execute a statement: let b = [1, 2];.

    printVector('a', a);  // EN: Execute a statement: printVector('a', a);.
    printVector('b', b);  // EN: Execute a statement: printVector('b', b);.

    const area = parallelogramArea(a, b);  // EN: Execute a statement: const area = parallelogramArea(a, b);.
    const signedArea = cross2D(a, b);  // EN: Execute a statement: const signedArea = cross2D(a, b);.

    console.log('\n平行四邊形：');  // EN: Execute a statement: console.log('\n平行四邊形：');.
    console.log(`  有號面積 = a × b = ${signedArea.toFixed(4)}`);  // EN: Execute a statement: console.log(` 有號面積 = a × b = ${signedArea.toFixed(4)}`);.
    console.log(`  面積 = |a × b| = ${area.toFixed(4)}`);  // EN: Execute a statement: console.log(` 面積 = |a × b| = ${area.toFixed(4)}`);.

    // ========================================
    // 2. 定向判斷
    // ========================================
    printSeparator('2. 定向判斷');  // EN: Execute a statement: printSeparator('2. 定向判斷');.

    a = [1, 0];  // EN: Execute a statement: a = [1, 0];.
    b = [0, 1];  // EN: Execute a statement: b = [0, 1];.
    let signedVal = cross2D(a, b);  // EN: Execute a statement: let signedVal = cross2D(a, b);.

    printVector('a', a);  // EN: Execute a statement: printVector('a', a);.
    printVector('b', b);  // EN: Execute a statement: printVector('b', b);.
    console.log(`有號面積 = ${signedVal.toFixed(4)}`);  // EN: Execute a statement: console.log(`有號面積 = ${signedVal.toFixed(4)}`);.
    console.log(`定向：${signedVal > 0 ? '逆時針（正向）' : '順時針（負向）'}`);  // EN: Execute a statement: console.log(`定向：${signedVal > 0 ? '逆時針（正向）' : '順時針（負向）'}`);.

    console.log('\n交換 a, b 順序：');  // EN: Execute a statement: console.log('\n交換 a, b 順序：');.
    signedVal = cross2D(b, a);  // EN: Execute a statement: signedVal = cross2D(b, a);.
    console.log(`有號面積 = ${signedVal.toFixed(4)}`);  // EN: Execute a statement: console.log(`有號面積 = ${signedVal.toFixed(4)}`);.
    console.log(`定向：${signedVal > 0 ? '逆時針（正向）' : '順時針（負向）'}`);  // EN: Execute a statement: console.log(`定向：${signedVal > 0 ? '逆時針（正向）' : '順時針（負向）'}`);.

    // ========================================
    // 3. 平行六面體體積
    // ========================================
    printSeparator('3. 平行六面體體積');  // EN: Execute a statement: printSeparator('3. 平行六面體體積');.

    const v1 = [1, 0, 0];  // EN: Execute a statement: const v1 = [1, 0, 0];.
    const v2 = [0, 2, 0];  // EN: Execute a statement: const v2 = [0, 2, 0];.
    const v3 = [0, 0, 3];  // EN: Execute a statement: const v3 = [0, 0, 3];.

    printVector('a', v1);  // EN: Execute a statement: printVector('a', v1);.
    printVector('b', v2);  // EN: Execute a statement: printVector('b', v2);.
    printVector('c', v3);  // EN: Execute a statement: printVector('c', v3);.

    const vol = parallelepipedVolume(v1, v2, v3);  // EN: Execute a statement: const vol = parallelepipedVolume(v1, v2, v3);.
    console.log(`\n體積 = |a · (b × c)| = ${vol.toFixed(4)}`);  // EN: Execute a statement: console.log(`\n體積 = |a · (b × c)| = ${vol.toFixed(4)}`);.

    // ========================================
    // 4. 三角形面積
    // ========================================
    printSeparator('4. 三角形面積');  // EN: Execute a statement: printSeparator('4. 三角形面積');.

    const x1 = 0, y1 = 0;  // EN: Execute a statement: const x1 = 0, y1 = 0;.
    const x2 = 4, y2 = 0;  // EN: Execute a statement: const x2 = 4, y2 = 0;.
    const x3 = 0, y3 = 3;  // EN: Execute a statement: const x3 = 0, y3 = 3;.

    console.log('三角形頂點：');  // EN: Execute a statement: console.log('三角形頂點：');.
    console.log(`  P1 = (${x1}, ${y1})`);  // EN: Execute a statement: console.log(` P1 = (${x1}, ${y1})`);.
    console.log(`  P2 = (${x2}, ${y2})`);  // EN: Execute a statement: console.log(` P2 = (${x2}, ${y2})`);.
    console.log(`  P3 = (${x3}, ${y3})`);  // EN: Execute a statement: console.log(` P3 = (${x3}, ${y3})`);.

    const triArea = triangleArea(x1, y1, x2, y2, x3, y3);  // EN: Execute a statement: const triArea = triangleArea(x1, y1, x2, y2, x3, y3);.
    console.log(`\n面積 = ${triArea.toFixed(4)}`);  // EN: Execute a statement: console.log(`\n面積 = ${triArea.toFixed(4)}`);.

    // ========================================
    // 5. 線性變換的體積縮放
    // ========================================
    printSeparator('5. 線性變換的體積縮放');  // EN: Execute a statement: printSeparator('5. 線性變換的體積縮放');.

    const A = [[2, 0], [0, 3]];  // EN: Execute a statement: const A = [[2, 0], [0, 3]];.
    printMatrix('縮放矩陣 A', A);  // EN: Execute a statement: printMatrix('縮放矩陣 A', A);.
    console.log(`det(A) = ${det2x2(A).toFixed(4)}`);  // EN: Execute a statement: console.log(`det(A) = ${det2x2(A).toFixed(4)}`);.
    console.log('\n單位正方形 → 2×3 長方形');  // EN: Execute a statement: console.log('\n單位正方形 → 2×3 長方形');.
    console.log(`面積從 1 變成 ${Math.abs(det2x2(A)).toFixed(4)}`);  // EN: Execute a statement: console.log(`面積從 1 變成 ${Math.abs(det2x2(A)).toFixed(4)}`);.

    const theta = Math.PI / 4;  // EN: Execute a statement: const theta = Math.PI / 4;.
    const R = [  // EN: Execute line: const R = [.
        [Math.cos(theta), -Math.sin(theta)],  // EN: Execute line: [Math.cos(theta), -Math.sin(theta)],.
        [Math.sin(theta), Math.cos(theta)]  // EN: Execute line: [Math.sin(theta), Math.cos(theta)].
    ];  // EN: Execute a statement: ];.
    console.log(`\n旋轉矩陣：det(R) = ${det2x2(R).toFixed(4)}（面積不變）`);  // EN: Execute a statement: console.log(`\n旋轉矩陣：det(R) = ${det2x2(R).toFixed(4)}（面積不變）`);.

    const H = [[1, 0], [0, -1]];  // EN: Execute a statement: const H = [[1, 0], [0, -1]];.
    console.log(`反射矩陣：det(H) = ${det2x2(H).toFixed(4)}（面積不變，定向反轉）`);  // EN: Execute a statement: console.log(`反射矩陣：det(H) = ${det2x2(H).toFixed(4)}（面積不變，定向反轉）`);.

    const S = [[1, 2], [0, 1]];  // EN: Execute a statement: const S = [[1, 2], [0, 1]];.
    console.log(`剪切矩陣：det(S) = ${det2x2(S).toFixed(4)}（面積不變）`);  // EN: Execute a statement: console.log(`剪切矩陣：det(S) = ${det2x2(S).toFixed(4)}（面積不變）`);.

    // 總結
    printSeparator('總結');  // EN: Execute a statement: printSeparator('總結');.
    console.log(`  // EN: Execute line: console.log(`.
行列式的幾何意義：  // EN: Execute line: 行列式的幾何意義：.

1. |det| = 體積/面積的縮放因子  // EN: Execute line: 1. |det| = 體積/面積的縮放因子.
2. sign(det) = 定向保持/反轉  // EN: Execute line: 2. sign(det) = 定向保持/反轉.
3. det = 0 → 降維  // EN: Execute line: 3. det = 0 → 降維.

特殊矩陣：  // EN: Execute line: 特殊矩陣：.
   - 旋轉：det = 1  // EN: Execute line: - 旋轉：det = 1.
   - 反射：det = -1  // EN: Execute line: - 反射：det = -1.
   - 剪切：det = 1  // EN: Execute line: - 剪切：det = 1.
`);  // EN: Execute a statement: `);.

    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
    console.log('示範完成！');  // EN: Execute a statement: console.log('示範完成！');.
    console.log('='.repeat(60));  // EN: Execute a statement: console.log('='.repeat(60));.
}  // EN: Structure delimiter for a block or scope.

main();  // EN: Execute a statement: main();.
