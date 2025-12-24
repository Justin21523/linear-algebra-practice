"""
投影 - NumPy 版本 (Projections - NumPy Implementation)

本程式示範：
1. 投影到直線
2. 投影矩陣及其性質
3. 投影到子空間
4. 視覺化投影

使用 NumPy 提供高效的向量化運算。
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def print_separator(title: str) -> None:
    """印出分隔線"""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ========================================
# 投影函數
# ========================================

def project_onto_line(b: np.ndarray, a: np.ndarray) -> dict:
    """
    計算向量 b 到直線（方向為 a）的投影

    公式：p = (aᵀb / aᵀa) * a
    """
    # 投影係數
    x_hat = np.dot(a, b) / np.dot(a, a)

    # 投影向量
    p = x_hat * a

    # 誤差向量
    e = b - p

    return {
        'x_hat': x_hat,
        'projection': p,
        'error': e,
        'error_norm': np.linalg.norm(e)
    }


def projection_matrix_line(a: np.ndarray) -> np.ndarray:
    """
    計算投影到直線的投影矩陣

    公式：P = aaᵀ / (aᵀa)
    """
    a = a.reshape(-1, 1)  # 確保是列向量
    return (a @ a.T) / (a.T @ a)


def project_onto_subspace(b: np.ndarray, A: np.ndarray) -> dict:
    """
    計算向量 b 到子空間 C(A) 的投影

    公式：
        x̂ = (AᵀA)⁻¹ Aᵀb
        p = Ax̂
    """
    # 正規方程
    ATA = A.T @ A
    ATb = A.T @ b

    # 解 x̂
    x_hat = np.linalg.solve(ATA, ATb)

    # 投影
    p = A @ x_hat

    # 誤差
    e = b - p

    return {
        'x_hat': x_hat,
        'projection': p,
        'error': e,
        'error_norm': np.linalg.norm(e)
    }


def projection_matrix_subspace(A: np.ndarray) -> np.ndarray:
    """
    計算投影到子空間 C(A) 的投影矩陣

    公式：P = A(AᵀA)⁻¹Aᵀ
    """
    ATA_inv = np.linalg.inv(A.T @ A)
    return A @ ATA_inv @ A.T


# ========================================
# 驗證函數
# ========================================

def verify_projection_matrix(P: np.ndarray, name: str = "P") -> None:
    """驗證投影矩陣的性質"""
    print(f"\n驗證 {name} 的性質：")

    # 對稱性
    is_symmetric = np.allclose(P, P.T)
    print(f"  對稱性 ({name}ᵀ = {name})：{is_symmetric}")

    # 冪等性
    is_idempotent = np.allclose(P @ P, P)
    print(f"  冪等性 ({name}² = {name})：{is_idempotent}")

    # 秩
    rank = np.linalg.matrix_rank(P)
    print(f"  秩 rank({name}) = {rank}")


def main():
    """主程式"""

    print_separator("投影示範（NumPy 版）\nProjection Demo (NumPy)")

    # ========================================
    # 1. 投影到直線
    # ========================================
    print_separator("1. 投影到直線")

    a = np.array([1, 1], dtype=float)
    b = np.array([2, 0], dtype=float)

    print(f"方向 a = {a}")
    print(f"向量 b = {b}")

    result = project_onto_line(b, a)

    print(f"\n投影係數 x̂ = (aᵀb)/(aᵀa) = {result['x_hat']:.4f}")
    print(f"投影 p = x̂a = {result['projection']}")
    print(f"誤差 e = b - p = {result['error']}")

    # 驗證正交性
    print(f"\n驗證 e ⊥ a：e · a = {np.dot(result['error'], a):.6f}")

    # ========================================
    # 2. 投影矩陣（到直線）
    # ========================================
    print_separator("2. 投影矩陣（到直線）")

    a = np.array([1, 2], dtype=float)
    print(f"方向 a = {a}")

    P = projection_matrix_line(a)
    print(f"\n投影矩陣 P = aaᵀ/(aᵀa)：\n{P}")

    verify_projection_matrix(P)

    # 用投影矩陣計算
    b = np.array([3, 4], dtype=float)
    print(f"\n向量 b = {b}")
    print(f"投影 p = Pb = {P @ b}")

    # ========================================
    # 3. 投影到子空間
    # ========================================
    print_separator("3. 投影到平面（子空間）")

    A = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ], dtype=float)

    b = np.array([1, 2, 3], dtype=float)

    print(f"矩陣 A（平面的基底）：\n{A}")
    print(f"\n向量 b = {b}")

    result = project_onto_subspace(b, A)

    print(f"\n係數 x̂ = (AᵀA)⁻¹Aᵀb = {result['x_hat']}")
    print(f"投影 p = Ax̂ = {result['projection']}")
    print(f"誤差 e = b - p = {result['error']}")
    print(f"誤差長度 ‖e‖ = {result['error_norm']:.4f}")

    # 驗證 e ⊥ C(A)
    print(f"\n驗證 Aᵀe = {A.T @ result['error']}")

    # ========================================
    # 4. 投影矩陣（子空間）
    # ========================================
    print_separator("4. 投影矩陣（子空間）")

    P = projection_matrix_subspace(A)
    print(f"P = A(AᵀA)⁻¹Aᵀ：\n{P}")

    verify_projection_matrix(P)

    # 用投影矩陣計算
    print(f"\n用 P 計算投影：")
    print(f"Pb = {P @ b}")

    # ========================================
    # 5. 另一個子空間例子
    # ========================================
    print_separator("5. 投影到斜平面")

    A = np.array([
        [1, 1],
        [1, 0],
        [0, 1]
    ], dtype=float)

    b = np.array([2, 3, 4], dtype=float)

    print(f"矩陣 A：\n{A}")
    print(f"\n向量 b = {b}")

    result = project_onto_subspace(b, A)

    print(f"\n係數 x̂ = {result['x_hat']}")
    print(f"投影 p = {result['projection']}")
    print(f"誤差 e = {result['error']}")
    print(f"‖e‖ = {result['error_norm']:.4f}")

    P = projection_matrix_subspace(A)
    verify_projection_matrix(P)

    # ========================================
    # 6. I - P 投影到正交補
    # ========================================
    print_separator("6. 補空間投影矩陣 I - P")

    a = np.array([1, 0], dtype=float)
    P = projection_matrix_line(a)

    print(f"方向 a = {a}")
    print(f"P = aaᵀ/(aᵀa)：\n{P}")

    I = np.eye(2)
    I_minus_P = I - P

    print(f"\nI - P：\n{I_minus_P}")

    verify_projection_matrix(I_minus_P, "I-P")

    # 正交分解
    b = np.array([3, 4], dtype=float)
    print(f"\n向量 b = {b}")

    p = P @ b
    e = I_minus_P @ b

    print(f"Pb = {p}（投影到 a）")
    print(f"(I-P)b = {e}（投影到 a⊥）")
    print(f"Pb + (I-P)b = {p + e}")

    # ========================================
    # 7. 正交基底的簡化
    # ========================================
    print_separator("7. 正交基底的簡化")

    # 正交基底
    Q = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ], dtype=float)

    print(f"正交矩陣 Q：\n{Q}")
    print(f"QᵀQ = \n{Q.T @ Q}")

    # 當 Q 有標準正交行向量時，P = QQᵀ
    P = Q @ Q.T
    print(f"\n投影矩陣 P = QQᵀ：\n{P}")

    # ========================================
    # 8. 批次投影
    # ========================================
    print_separator("8. 批次投影多個向量")

    a = np.array([1, 1], dtype=float)
    P = projection_matrix_line(a)

    # 多個向量同時投影
    vectors = np.array([
        [1, 0],
        [0, 1],
        [2, 2],
        [3, -1]
    ], dtype=float)

    print(f"方向 a = {a}")
    print(f"\n原始向量：\n{vectors}")

    projections = (P @ vectors.T).T
    print(f"\n投影結果：\n{projections}")

    # 總結
    print_separator("NumPy 投影函數總結")
    print("""
投影到直線：
  x_hat = np.dot(a, b) / np.dot(a, a)
  p = x_hat * a
  P = np.outer(a, a) / np.dot(a, a)

投影到子空間：
  x_hat = np.linalg.solve(A.T @ A, A.T @ b)
  p = A @ x_hat
  P = A @ np.linalg.inv(A.T @ A) @ A.T

驗證：
  np.allclose(P, P.T)      # 對稱
  np.allclose(P @ P, P)    # 冪等
""")

    print("=" * 60)
    print("示範完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
