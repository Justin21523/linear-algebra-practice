/**
 * 行列式的幾何解釋 (Geometric Interpretation)
 *
 * 執行：node geometric.js
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

// 2D 叉積（純量）
function cross2D(a, b) {
    return a[0] * b[1] - a[1] * b[0];
}

// 3D 叉積
function cross3D(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

// 內積
function dot(a, b) {
    return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
}

// 2×2 行列式
function det2x2(A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

// 平行四邊形面積
function parallelogramArea(a, b) {
    return Math.abs(cross2D(a, b));
}

// 平行六面體體積
function parallelepipedVolume(a, b, c) {
    const bxc = cross3D(b, c);
    return Math.abs(dot(a, bxc));
}

// 三角形面積
function triangleArea(x1, y1, x2, y2, x3, y3) {
    return Math.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;
}

function main() {
    printSeparator('行列式幾何解釋示範 (JavaScript)');

    // ========================================
    // 1. 平行四邊形面積
    // ========================================
    printSeparator('1. 平行四邊形面積');

    let a = [3, 0];
    let b = [1, 2];

    printVector('a', a);
    printVector('b', b);

    const area = parallelogramArea(a, b);
    const signedArea = cross2D(a, b);

    console.log('\n平行四邊形：');
    console.log(`  有號面積 = a × b = ${signedArea.toFixed(4)}`);
    console.log(`  面積 = |a × b| = ${area.toFixed(4)}`);

    // ========================================
    // 2. 定向判斷
    // ========================================
    printSeparator('2. 定向判斷');

    a = [1, 0];
    b = [0, 1];
    let signedVal = cross2D(a, b);

    printVector('a', a);
    printVector('b', b);
    console.log(`有號面積 = ${signedVal.toFixed(4)}`);
    console.log(`定向：${signedVal > 0 ? '逆時針（正向）' : '順時針（負向）'}`);

    console.log('\n交換 a, b 順序：');
    signedVal = cross2D(b, a);
    console.log(`有號面積 = ${signedVal.toFixed(4)}`);
    console.log(`定向：${signedVal > 0 ? '逆時針（正向）' : '順時針（負向）'}`);

    // ========================================
    // 3. 平行六面體體積
    // ========================================
    printSeparator('3. 平行六面體體積');

    const v1 = [1, 0, 0];
    const v2 = [0, 2, 0];
    const v3 = [0, 0, 3];

    printVector('a', v1);
    printVector('b', v2);
    printVector('c', v3);

    const vol = parallelepipedVolume(v1, v2, v3);
    console.log(`\n體積 = |a · (b × c)| = ${vol.toFixed(4)}`);

    // ========================================
    // 4. 三角形面積
    // ========================================
    printSeparator('4. 三角形面積');

    const x1 = 0, y1 = 0;
    const x2 = 4, y2 = 0;
    const x3 = 0, y3 = 3;

    console.log('三角形頂點：');
    console.log(`  P1 = (${x1}, ${y1})`);
    console.log(`  P2 = (${x2}, ${y2})`);
    console.log(`  P3 = (${x3}, ${y3})`);

    const triArea = triangleArea(x1, y1, x2, y2, x3, y3);
    console.log(`\n面積 = ${triArea.toFixed(4)}`);

    // ========================================
    // 5. 線性變換的體積縮放
    // ========================================
    printSeparator('5. 線性變換的體積縮放');

    const A = [[2, 0], [0, 3]];
    printMatrix('縮放矩陣 A', A);
    console.log(`det(A) = ${det2x2(A).toFixed(4)}`);
    console.log('\n單位正方形 → 2×3 長方形');
    console.log(`面積從 1 變成 ${Math.abs(det2x2(A)).toFixed(4)}`);

    const theta = Math.PI / 4;
    const R = [
        [Math.cos(theta), -Math.sin(theta)],
        [Math.sin(theta), Math.cos(theta)]
    ];
    console.log(`\n旋轉矩陣：det(R) = ${det2x2(R).toFixed(4)}（面積不變）`);

    const H = [[1, 0], [0, -1]];
    console.log(`反射矩陣：det(H) = ${det2x2(H).toFixed(4)}（面積不變，定向反轉）`);

    const S = [[1, 2], [0, 1]];
    console.log(`剪切矩陣：det(S) = ${det2x2(S).toFixed(4)}（面積不變）`);

    // 總結
    printSeparator('總結');
    console.log(`
行列式的幾何意義：

1. |det| = 體積/面積的縮放因子
2. sign(det) = 定向保持/反轉
3. det = 0 → 降維

特殊矩陣：
   - 旋轉：det = 1
   - 反射：det = -1
   - 剪切：det = 1
`);

    console.log('='.repeat(60));
    console.log('示範完成！');
    console.log('='.repeat(60));
}

main();
