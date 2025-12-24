"""
投影 - NumPy 版本 (Projections - NumPy Implementation)

本程式示範：
1. 投影到直線
2. 投影矩陣及其性質
3. 投影到子空間
4. 視覺化投影

使用 NumPy 提供高效的向量化運算。
"""  # EN: Execute statement: """.

import numpy as np  # EN: Import module(s): import numpy as np.

np.set_printoptions(precision=4, suppress=True)  # EN: Execute statement: np.set_printoptions(precision=4, suppress=True).


def print_separator(title: str) -> None:  # EN: Define print_separator and its behavior.
    """印出分隔線"""  # EN: Execute statement: """印出分隔線""".
    print()  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.
    print(title)  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


# ========================================
# 投影函數
# ========================================

def project_onto_line(b: np.ndarray, a: np.ndarray) -> dict:  # EN: Define project_onto_line and its behavior.
    """
    計算向量 b 到直線（方向為 a）的投影

    公式：p = (aᵀb / aᵀa) * a
    """  # EN: Execute statement: """.
    # 投影係數
    x_hat = np.dot(a, b) / np.dot(a, a)  # EN: Assign x_hat from expression: np.dot(a, b) / np.dot(a, a).

    # 投影向量
    p = x_hat * a  # EN: Assign p from expression: x_hat * a.

    # 誤差向量
    e = b - p  # EN: Assign e from expression: b - p.

    return {  # EN: Return a value: return {.
        'x_hat': x_hat,  # EN: Execute statement: 'x_hat': x_hat,.
        'projection': p,  # EN: Execute statement: 'projection': p,.
        'error': e,  # EN: Execute statement: 'error': e,.
        'error_norm': np.linalg.norm(e)  # EN: Execute statement: 'error_norm': np.linalg.norm(e).
    }  # EN: Execute statement: }.


def projection_matrix_line(a: np.ndarray) -> np.ndarray:  # EN: Define projection_matrix_line and its behavior.
    """
    計算投影到直線的投影矩陣

    公式：P = aaᵀ / (aᵀa)
    """  # EN: Execute statement: """.
    a = a.reshape(-1, 1)  # 確保是列向量  # EN: Assign a from expression: a.reshape(-1, 1) # 確保是列向量.
    return (a @ a.T) / (a.T @ a)  # EN: Return a value: return (a @ a.T) / (a.T @ a).


def project_onto_subspace(b: np.ndarray, A: np.ndarray) -> dict:  # EN: Define project_onto_subspace and its behavior.
    """
    計算向量 b 到子空間 C(A) 的投影

    公式：
        x̂ = (AᵀA)⁻¹ Aᵀb
        p = Ax̂
    """  # EN: Execute statement: """.
    # 正規方程
    ATA = A.T @ A  # EN: Assign ATA from expression: A.T @ A.
    ATb = A.T @ b  # EN: Assign ATb from expression: A.T @ b.

    # 解 x̂
    x_hat = np.linalg.solve(ATA, ATb)  # EN: Assign x_hat from expression: np.linalg.solve(ATA, ATb).

    # 投影
    p = A @ x_hat  # EN: Assign p from expression: A @ x_hat.

    # 誤差
    e = b - p  # EN: Assign e from expression: b - p.

    return {  # EN: Return a value: return {.
        'x_hat': x_hat,  # EN: Execute statement: 'x_hat': x_hat,.
        'projection': p,  # EN: Execute statement: 'projection': p,.
        'error': e,  # EN: Execute statement: 'error': e,.
        'error_norm': np.linalg.norm(e)  # EN: Execute statement: 'error_norm': np.linalg.norm(e).
    }  # EN: Execute statement: }.


def projection_matrix_subspace(A: np.ndarray) -> np.ndarray:  # EN: Define projection_matrix_subspace and its behavior.
    """
    計算投影到子空間 C(A) 的投影矩陣

    公式：P = A(AᵀA)⁻¹Aᵀ
    """  # EN: Execute statement: """.
    ATA_inv = np.linalg.inv(A.T @ A)  # EN: Assign ATA_inv from expression: np.linalg.inv(A.T @ A).
    return A @ ATA_inv @ A.T  # EN: Return a value: return A @ ATA_inv @ A.T.


# ========================================
# 驗證函數
# ========================================

def verify_projection_matrix(P: np.ndarray, name: str = "P") -> None:  # EN: Define verify_projection_matrix and its behavior.
    """驗證投影矩陣的性質"""  # EN: Execute statement: """驗證投影矩陣的性質""".
    print(f"\n驗證 {name} 的性質：")  # EN: Print formatted output to the console.

    # 對稱性
    is_symmetric = np.allclose(P, P.T)  # EN: Assign is_symmetric from expression: np.allclose(P, P.T).
    print(f"  對稱性 ({name}ᵀ = {name})：{is_symmetric}")  # EN: Print formatted output to the console.

    # 冪等性
    is_idempotent = np.allclose(P @ P, P)  # EN: Assign is_idempotent from expression: np.allclose(P @ P, P).
    print(f"  冪等性 ({name}² = {name})：{is_idempotent}")  # EN: Print formatted output to the console.

    # 秩
    rank = np.linalg.matrix_rank(P)  # EN: Assign rank from expression: np.linalg.matrix_rank(P).
    print(f"  秩 rank({name}) = {rank}")  # EN: Print formatted output to the console.


def main():  # EN: Define main and its behavior.
    """主程式"""  # EN: Execute statement: """主程式""".

    print_separator("投影示範（NumPy 版）\nProjection Demo (NumPy)")  # EN: Call print_separator(...) to perform an operation.

    # ========================================
    # 1. 投影到直線
    # ========================================
    print_separator("1. 投影到直線")  # EN: Call print_separator(...) to perform an operation.

    a = np.array([1, 1], dtype=float)  # EN: Assign a from expression: np.array([1, 1], dtype=float).
    b = np.array([2, 0], dtype=float)  # EN: Assign b from expression: np.array([2, 0], dtype=float).

    print(f"方向 a = {a}")  # EN: Print formatted output to the console.
    print(f"向量 b = {b}")  # EN: Print formatted output to the console.

    result = project_onto_line(b, a)  # EN: Assign result from expression: project_onto_line(b, a).

    print(f"\n投影係數 x̂ = (aᵀb)/(aᵀa) = {result['x_hat']:.4f}")  # EN: Print formatted output to the console.
    print(f"投影 p = x̂a = {result['projection']}")  # EN: Print formatted output to the console.
    print(f"誤差 e = b - p = {result['error']}")  # EN: Print formatted output to the console.

    # 驗證正交性
    print(f"\n驗證 e ⊥ a：e · a = {np.dot(result['error'], a):.6f}")  # EN: Print formatted output to the console.

    # ========================================
    # 2. 投影矩陣（到直線）
    # ========================================
    print_separator("2. 投影矩陣（到直線）")  # EN: Call print_separator(...) to perform an operation.

    a = np.array([1, 2], dtype=float)  # EN: Assign a from expression: np.array([1, 2], dtype=float).
    print(f"方向 a = {a}")  # EN: Print formatted output to the console.

    P = projection_matrix_line(a)  # EN: Assign P from expression: projection_matrix_line(a).
    print(f"\n投影矩陣 P = aaᵀ/(aᵀa)：\n{P}")  # EN: Print formatted output to the console.

    verify_projection_matrix(P)  # EN: Call verify_projection_matrix(...) to perform an operation.

    # 用投影矩陣計算
    b = np.array([3, 4], dtype=float)  # EN: Assign b from expression: np.array([3, 4], dtype=float).
    print(f"\n向量 b = {b}")  # EN: Print formatted output to the console.
    print(f"投影 p = Pb = {P @ b}")  # EN: Print formatted output to the console.

    # ========================================
    # 3. 投影到子空間
    # ========================================
    print_separator("3. 投影到平面（子空間）")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 0],  # EN: Execute statement: [1, 0],.
        [0, 1],  # EN: Execute statement: [0, 1],.
        [0, 0]  # EN: Execute statement: [0, 0].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    b = np.array([1, 2, 3], dtype=float)  # EN: Assign b from expression: np.array([1, 2, 3], dtype=float).

    print(f"矩陣 A（平面的基底）：\n{A}")  # EN: Print formatted output to the console.
    print(f"\n向量 b = {b}")  # EN: Print formatted output to the console.

    result = project_onto_subspace(b, A)  # EN: Assign result from expression: project_onto_subspace(b, A).

    print(f"\n係數 x̂ = (AᵀA)⁻¹Aᵀb = {result['x_hat']}")  # EN: Print formatted output to the console.
    print(f"投影 p = Ax̂ = {result['projection']}")  # EN: Print formatted output to the console.
    print(f"誤差 e = b - p = {result['error']}")  # EN: Print formatted output to the console.
    print(f"誤差長度 ‖e‖ = {result['error_norm']:.4f}")  # EN: Print formatted output to the console.

    # 驗證 e ⊥ C(A)
    print(f"\n驗證 Aᵀe = {A.T @ result['error']}")  # EN: Print formatted output to the console.

    # ========================================
    # 4. 投影矩陣（子空間）
    # ========================================
    print_separator("4. 投影矩陣（子空間）")  # EN: Call print_separator(...) to perform an operation.

    P = projection_matrix_subspace(A)  # EN: Assign P from expression: projection_matrix_subspace(A).
    print(f"P = A(AᵀA)⁻¹Aᵀ：\n{P}")  # EN: Print formatted output to the console.

    verify_projection_matrix(P)  # EN: Call verify_projection_matrix(...) to perform an operation.

    # 用投影矩陣計算
    print(f"\n用 P 計算投影：")  # EN: Print formatted output to the console.
    print(f"Pb = {P @ b}")  # EN: Print formatted output to the console.

    # ========================================
    # 5. 另一個子空間例子
    # ========================================
    print_separator("5. 投影到斜平面")  # EN: Call print_separator(...) to perform an operation.

    A = np.array([  # EN: Assign A from expression: np.array([.
        [1, 1],  # EN: Execute statement: [1, 1],.
        [1, 0],  # EN: Execute statement: [1, 0],.
        [0, 1]  # EN: Execute statement: [0, 1].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    b = np.array([2, 3, 4], dtype=float)  # EN: Assign b from expression: np.array([2, 3, 4], dtype=float).

    print(f"矩陣 A：\n{A}")  # EN: Print formatted output to the console.
    print(f"\n向量 b = {b}")  # EN: Print formatted output to the console.

    result = project_onto_subspace(b, A)  # EN: Assign result from expression: project_onto_subspace(b, A).

    print(f"\n係數 x̂ = {result['x_hat']}")  # EN: Print formatted output to the console.
    print(f"投影 p = {result['projection']}")  # EN: Print formatted output to the console.
    print(f"誤差 e = {result['error']}")  # EN: Print formatted output to the console.
    print(f"‖e‖ = {result['error_norm']:.4f}")  # EN: Print formatted output to the console.

    P = projection_matrix_subspace(A)  # EN: Assign P from expression: projection_matrix_subspace(A).
    verify_projection_matrix(P)  # EN: Call verify_projection_matrix(...) to perform an operation.

    # ========================================
    # 6. I - P 投影到正交補
    # ========================================
    print_separator("6. 補空間投影矩陣 I - P")  # EN: Call print_separator(...) to perform an operation.

    a = np.array([1, 0], dtype=float)  # EN: Assign a from expression: np.array([1, 0], dtype=float).
    P = projection_matrix_line(a)  # EN: Assign P from expression: projection_matrix_line(a).

    print(f"方向 a = {a}")  # EN: Print formatted output to the console.
    print(f"P = aaᵀ/(aᵀa)：\n{P}")  # EN: Print formatted output to the console.

    I = np.eye(2)  # EN: Assign I from expression: np.eye(2).
    I_minus_P = I - P  # EN: Assign I_minus_P from expression: I - P.

    print(f"\nI - P：\n{I_minus_P}")  # EN: Print formatted output to the console.

    verify_projection_matrix(I_minus_P, "I-P")  # EN: Call verify_projection_matrix(...) to perform an operation.

    # 正交分解
    b = np.array([3, 4], dtype=float)  # EN: Assign b from expression: np.array([3, 4], dtype=float).
    print(f"\n向量 b = {b}")  # EN: Print formatted output to the console.

    p = P @ b  # EN: Assign p from expression: P @ b.
    e = I_minus_P @ b  # EN: Assign e from expression: I_minus_P @ b.

    print(f"Pb = {p}（投影到 a）")  # EN: Print formatted output to the console.
    print(f"(I-P)b = {e}（投影到 a⊥）")  # EN: Print formatted output to the console.
    print(f"Pb + (I-P)b = {p + e}")  # EN: Print formatted output to the console.

    # ========================================
    # 7. 正交基底的簡化
    # ========================================
    print_separator("7. 正交基底的簡化")  # EN: Call print_separator(...) to perform an operation.

    # 正交基底
    Q = np.array([  # EN: Assign Q from expression: np.array([.
        [1, 0],  # EN: Execute statement: [1, 0],.
        [0, 1],  # EN: Execute statement: [0, 1],.
        [0, 0]  # EN: Execute statement: [0, 0].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"正交矩陣 Q：\n{Q}")  # EN: Print formatted output to the console.
    print(f"QᵀQ = \n{Q.T @ Q}")  # EN: Print formatted output to the console.

    # 當 Q 有標準正交行向量時，P = QQᵀ
    P = Q @ Q.T  # EN: Assign P from expression: Q @ Q.T.
    print(f"\n投影矩陣 P = QQᵀ：\n{P}")  # EN: Print formatted output to the console.

    # ========================================
    # 8. 批次投影
    # ========================================
    print_separator("8. 批次投影多個向量")  # EN: Call print_separator(...) to perform an operation.

    a = np.array([1, 1], dtype=float)  # EN: Assign a from expression: np.array([1, 1], dtype=float).
    P = projection_matrix_line(a)  # EN: Assign P from expression: projection_matrix_line(a).

    # 多個向量同時投影
    vectors = np.array([  # EN: Assign vectors from expression: np.array([.
        [1, 0],  # EN: Execute statement: [1, 0],.
        [0, 1],  # EN: Execute statement: [0, 1],.
        [2, 2],  # EN: Execute statement: [2, 2],.
        [3, -1]  # EN: Execute statement: [3, -1].
    ], dtype=float)  # EN: Execute statement: ], dtype=float).

    print(f"方向 a = {a}")  # EN: Print formatted output to the console.
    print(f"\n原始向量：\n{vectors}")  # EN: Print formatted output to the console.

    projections = (P @ vectors.T).T  # EN: Assign projections from expression: (P @ vectors.T).T.
    print(f"\n投影結果：\n{projections}")  # EN: Print formatted output to the console.

    # 總結
    print_separator("NumPy 投影函數總結")  # EN: Call print_separator(...) to perform an operation.
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
""")  # EN: Execute statement: """).

    print("=" * 60)  # EN: Print formatted output to the console.
    print("示範完成！")  # EN: Print formatted output to the console.
    print("=" * 60)  # EN: Print formatted output to the console.


if __name__ == "__main__":  # EN: Branch on a condition: if __name__ == "__main__":.
    main()  # EN: Call main(...) to perform an operation.
