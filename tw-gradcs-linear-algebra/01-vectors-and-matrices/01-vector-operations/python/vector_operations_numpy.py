"""
向量運算：NumPy 版本 (Vector Operations: NumPy Implementation)

本程式示範：
1. 使用 NumPy 進行向量運算
2. 比較手刻版本與 NumPy 的語法差異
3. NumPy 的效能優勢（向量化運算）

This program demonstrates vector operations using NumPy,
comparing with manual implementation and showcasing vectorized operations.
"""

import numpy as np

# 設定輸出格式 (Set output format)
np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線 (Print separator)"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():
    """主程式 (Main program)"""

    print_separator("向量運算示範 - NumPy 版本\nVector Operations Demo - NumPy Implementation")

    # ========================================
    # 建立向量 (Creating Vectors)
    # ========================================
    print_separator("建立向量 (Creating Vectors)")

    # 從 list 建立
    u = np.array([3.0, 4.0])
    v = np.array([1.0, 2.0])

    print(f"u = np.array([3.0, 4.0]) → {u}")
    print(f"v = np.array([1.0, 2.0]) → {v}")

    # 其他建立方式
    zeros = np.zeros(3)
    ones = np.ones(3)
    range_vec = np.arange(1, 6)  # [1, 2, 3, 4, 5]
    linspace_vec = np.linspace(0, 1, 5)  # 0到1之間均勻分布5個點

    print(f"\nnp.zeros(3)      → {zeros}")
    print(f"np.ones(3)       → {ones}")
    print(f"np.arange(1, 6)  → {range_vec}")
    print(f"np.linspace(0,1,5) → {linspace_vec}")

    # ========================================
    # 1. 向量加法與減法 (Addition and Subtraction)
    # ========================================
    print_separator("1. 向量加法與減法 (Addition & Subtraction)")

    # NumPy 使用運算子重載，語法非常直覺
    print(f"u + v = {u} + {v} = {u + v}")
    print(f"u - v = {u} - {v} = {u - v}")

    # ========================================
    # 2. 純量乘法 (Scalar Multiplication)
    # ========================================
    print_separator("2. 純量乘法 (Scalar Multiplication)")

    c = 2.5
    print(f"{c} * u = {c} * {u} = {c * u}")
    print(f"-u = {-u}")
    print(f"u / 2 = {u / 2}")

    # ========================================
    # 3. 元素級運算 (Element-wise Operations)
    # ========================================
    print_separator("3. 元素級運算 (Element-wise Operations)")

    # NumPy 的強大之處：向量化運算
    print(f"u * v (元素相乘) = {u * v}")
    print(f"u ** 2 (元素平方) = {u ** 2}")
    print(f"np.sqrt(u) = {np.sqrt(u)}")
    print(f"np.exp(v) = {np.exp(v)}")

    # ========================================
    # 4. 向量長度 (Vector Norm)
    # ========================================
    print_separator("4. 向量長度 (Vector Norm)")

    # 方法一：使用 np.linalg.norm
    norm_u = np.linalg.norm(u)
    print(f"np.linalg.norm(u) = {norm_u}")

    # 方法二：手動計算（但仍使用 NumPy）
    norm_manual = np.sqrt(np.sum(u ** 2))
    print(f"np.sqrt(np.sum(u**2)) = {norm_manual}")

    # 方法三：使用內積
    norm_dot = np.sqrt(np.dot(u, u))
    print(f"np.sqrt(np.dot(u, u)) = {norm_dot}")

    # 不同的範數
    print(f"\n不同類型的範數 (Different norms):")
    x = np.array([3.0, -4.0])
    print(f"x = {x}")
    print(f"L2 norm (歐幾里得): np.linalg.norm(x) = {np.linalg.norm(x)}")
    print(f"L1 norm (曼哈頓):   np.linalg.norm(x, 1) = {np.linalg.norm(x, 1)}")
    print(f"L∞ norm (最大值):   np.linalg.norm(x, np.inf) = {np.linalg.norm(x, np.inf)}")

    # ========================================
    # 5. 正規化 (Normalization)
    # ========================================
    print_separator("5. 正規化 (Normalization)")

    u_normalized = u / np.linalg.norm(u)
    print(f"û = u / ‖u‖ = {u} / {np.linalg.norm(u):.4f}")
    print(f"û = {u_normalized}")
    print(f"‖û‖ = {np.linalg.norm(u_normalized):.10f} (應該是 1)")

    # ========================================
    # 6. 內積 (Dot Product)
    # ========================================
    print_separator("6. 內積 (Dot Product)")

    # 方法一：np.dot
    dot1 = np.dot(u, v)
    print(f"np.dot(u, v) = {dot1}")

    # 方法二：@ 運算子（Python 3.5+）
    dot2 = u @ v
    print(f"u @ v = {dot2}")

    # 方法三：元素相乘後求和
    dot3 = np.sum(u * v)
    print(f"np.sum(u * v) = {dot3}")

    # ========================================
    # 7. 夾角計算 (Angle Calculation)
    # ========================================
    print_separator("7. 夾角計算 (Angle Between Vectors)")

    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)

    print(f"cos(θ) = (u·v) / (‖u‖×‖v‖) = {cos_theta:.4f}")
    print(f"θ = {theta_rad:.4f} 弧度 (radians)")
    print(f"θ = {theta_deg:.2f}° 度 (degrees)")

    # ========================================
    # 8. 正交檢驗 (Orthogonality Check)
    # ========================================
    print_separator("8. 正交檢驗 (Orthogonality Check)")

    # 正交向量
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])

    print(f"a = {a}, b = {b}")
    print(f"a · b = {np.dot(a, b)}")
    print(f"a 和 b 正交？ {np.isclose(np.dot(a, b), 0)}")

    # 另一組正交向量
    p = np.array([1.0, 1.0])
    q = np.array([1.0, -1.0])

    print(f"\np = {p}, q = {q}")
    print(f"p · q = {np.dot(p, q)}")
    print(f"p 和 q 正交？ {np.isclose(np.dot(p, q), 0)}")

    # ========================================
    # 9. 向量投影 (Vector Projection)
    # ========================================
    print_separator("9. 向量投影 (Vector Projection)")

    # proj_v(u) = ((u·v) / ‖v‖²) * v
    proj = (np.dot(u, v) / np.dot(v, v)) * v
    scalar_proj = np.dot(u, v) / np.linalg.norm(v)

    print(f"將 u={u} 投影到 v={v}")
    print(f"proj_v(u) = {proj}")
    print(f"comp_v(u) = {scalar_proj:.4f} (純量投影)")

    # 驗證
    perp = u - proj
    print(f"\nperp = u - proj = {perp}")
    print(f"perp · v = {np.dot(perp, v):.10f} (應該是 0)")

    # ========================================
    # 10. 3D 向量與外積 (3D Vectors and Cross Product)
    # ========================================
    print_separator("10. 3D 向量與外積 (3D Vectors & Cross Product)")

    p3 = np.array([1.0, 0.0, 0.0])
    q3 = np.array([0.0, 1.0, 0.0])

    print(f"p = {p3}")
    print(f"q = {q3}")

    # 外積只定義在 3D
    cross = np.cross(p3, q3)
    print(f"p × q = {cross}")
    print("（外積結果垂直於 p 和 q 形成的平面）")

    # 驗證外積垂直於兩個原向量
    print(f"\n驗證外積垂直於原向量：")
    print(f"(p×q) · p = {np.dot(cross, p3)}")
    print(f"(p×q) · q = {np.dot(cross, q3)}")

    # ========================================
    # 11. 廣播機制 (Broadcasting)
    # ========================================
    print_separator("11. 廣播機制 (Broadcasting)")

    # NumPy 的廣播機制讓不同形狀的陣列可以運算
    vectors = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    print("多個向量的矩陣：")
    print(vectors)

    # 同時計算所有向量的長度
    norms = np.linalg.norm(vectors, axis=1)
    print(f"\n每個向量的長度: {norms}")

    # 同時正規化所有向量
    normalized = vectors / norms[:, np.newaxis]
    print(f"\n正規化後：")
    print(normalized)

    # ========================================
    # 12. 效能比較提示 (Performance Note)
    # ========================================
    print_separator("12. NumPy 效能優勢")

    print("""
NumPy 的向量化運算比 Python 迴圈快很多：

❌ 慢的寫法（Python 迴圈）：
    result = []
    for i in range(len(u)):
        result.append(u[i] + v[i])

✅ 快的寫法（NumPy 向量化）：
    result = u + v

原因：
1. NumPy 底層用 C 語言實作
2. 向量化運算避免 Python 直譯器的 overhead
3. 可利用 CPU 的 SIMD 指令

經驗法則：盡量避免在 NumPy 陣列上使用 Python 迴圈！
""")

    print("=" * 60)
    print("所有 NumPy 向量運算示範完成！")
    print("All NumPy vector operations demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
