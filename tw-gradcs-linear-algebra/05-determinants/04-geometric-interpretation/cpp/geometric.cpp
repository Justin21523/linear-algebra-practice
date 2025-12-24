/**
 * 行列式的幾何解釋 (Geometric Interpretation)
 *
 * 編譯：g++ -std=c++17 -O2 geometric.cpp -o geometric
 * 執行：./geometric
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

void printSeparator(const string& title) {
    cout << endl;
    cout << string(60, '=') << endl;
    cout << title << endl;
    cout << string(60, '=') << endl;
}

void printVector(const string& name, const vector<double>& v) {
    cout << name << " = [";
    for (size_t i = 0; i < v.size(); i++) {
        cout << fixed << setprecision(4) << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}

void printMatrix(const string& name, const vector<vector<double>>& M) {
    cout << name << " =" << endl;
    for (const auto& row : M) {
        cout << "  [";
        for (size_t i = 0; i < row.size(); i++) {
            cout << fixed << setprecision(4) << setw(8) << row[i];
            if (i < row.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
    }
}

// 2D 叉積（純量）
double cross2D(const vector<double>& a, const vector<double>& b) {
    return a[0] * b[1] - a[1] * b[0];
}

// 3D 叉積
vector<double> cross3D(const vector<double>& a, const vector<double>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

// 內積
double dot(const vector<double>& a, const vector<double>& b) {
    double result = 0;
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

// 2×2 行列式
double det2x2(const vector<vector<double>>& A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

// 3×3 行列式
double det3x3(const vector<vector<double>>& A) {
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

// 平行四邊形面積
double parallelogramArea(const vector<double>& a, const vector<double>& b) {
    return abs(cross2D(a, b));
}

// 平行六面體體積
double parallelepipedVolume(const vector<double>& a,
                             const vector<double>& b,
                             const vector<double>& c) {
    auto bxc = cross3D(b, c);
    return abs(dot(a, bxc));
}

// 三角形面積
double triangleArea(double x1, double y1,
                    double x2, double y2,
                    double x3, double y3) {
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;
}

int main() {
    printSeparator("行列式幾何解釋示範 (C++)");

    // ========================================
    // 1. 平行四邊形面積
    // ========================================
    printSeparator("1. 平行四邊形面積");

    vector<double> a = {3, 0};
    vector<double> b = {1, 2};

    printVector("a", a);
    printVector("b", b);

    double area = parallelogramArea(a, b);
    double signedArea = cross2D(a, b);

    cout << "\n平行四邊形：" << endl;
    cout << "  有號面積 = a × b = " << signedArea << endl;
    cout << "  面積 = |a × b| = " << area << endl;

    // ========================================
    // 2. 定向判斷
    // ========================================
    printSeparator("2. 定向判斷");

    a = {1, 0};
    b = {0, 1};
    double signed_val = cross2D(a, b);

    printVector("a", a);
    printVector("b", b);
    cout << "有號面積 = " << signed_val << endl;
    cout << "定向：" << (signed_val > 0 ? "逆時針（正向）" : "順時針（負向）") << endl;

    cout << "\n交換 a, b 順序：" << endl;
    signed_val = cross2D(b, a);
    cout << "有號面積 = " << signed_val << endl;
    cout << "定向：" << (signed_val > 0 ? "逆時針（正向）" : "順時針（負向）") << endl;

    // ========================================
    // 3. 平行六面體體積
    // ========================================
    printSeparator("3. 平行六面體體積");

    vector<double> v1 = {1, 0, 0};
    vector<double> v2 = {0, 2, 0};
    vector<double> v3 = {0, 0, 3};

    printVector("a", v1);
    printVector("b", v2);
    printVector("c", v3);

    double vol = parallelepipedVolume(v1, v2, v3);
    cout << "\n體積 = |a · (b × c)| = " << vol << endl;

    // ========================================
    // 4. 三角形面積
    // ========================================
    printSeparator("4. 三角形面積");

    double x1 = 0, y1 = 0;
    double x2 = 4, y2 = 0;
    double x3 = 0, y3 = 3;

    cout << "三角形頂點：" << endl;
    cout << "  P1 = (" << x1 << ", " << y1 << ")" << endl;
    cout << "  P2 = (" << x2 << ", " << y2 << ")" << endl;
    cout << "  P3 = (" << x3 << ", " << y3 << ")" << endl;

    double triArea = triangleArea(x1, y1, x2, y2, x3, y3);
    cout << "\n面積 = " << triArea << endl;

    // ========================================
    // 5. 線性變換的體積縮放
    // ========================================
    printSeparator("5. 線性變換的體積縮放");

    // 縮放矩陣
    vector<vector<double>> A = {{2, 0}, {0, 3}};
    printMatrix("縮放矩陣 A", A);
    cout << "det(A) = " << det2x2(A) << endl;
    cout << "\n單位正方形 → 2×3 長方形" << endl;
    cout << "面積從 1 變成 " << abs(det2x2(A)) << endl;

    // 旋轉矩陣
    double theta = M_PI / 4;
    vector<vector<double>> R = {
        {cos(theta), -sin(theta)},
        {sin(theta), cos(theta)}
    };
    cout << "\n旋轉矩陣：det(R) = " << det2x2(R) << "（面積不變）" << endl;

    // 反射矩陣
    vector<vector<double>> H = {{1, 0}, {0, -1}};
    cout << "反射矩陣：det(H) = " << det2x2(H) << "（面積不變，定向反轉）" << endl;

    // 剪切矩陣
    vector<vector<double>> S = {{1, 2}, {0, 1}};
    cout << "剪切矩陣：det(S) = " << det2x2(S) << "（面積不變）" << endl;

    // 總結
    printSeparator("總結");
    cout << R"(
行列式的幾何意義：

1. |det| = 體積/面積的縮放因子
2. sign(det) = 定向保持/反轉
3. det = 0 → 降維

特殊矩陣：
   - 旋轉：det = 1
   - 反射：det = -1
   - 剪切：det = 1
)" << endl;

    cout << string(60, '=') << endl;
    cout << "示範完成！" << endl;
    cout << string(60, '=') << endl;

    return 0;
}
