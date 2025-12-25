# 第四章：正交性與最小平方 (Orthogonality and Least Squares)

> 當 Ax = b 無解時怎麼辦？——找最接近的解

## 本章概述

正交性是線性代數中最美的概念之一。本章探討向量的正交關係，並應用於解決「無解」方程組的最佳近似問題——這就是機器學習中無處不在的**最小平方法**。

## 為什麼重要？

| 應用 | 說明 |
|------|------|
| **線性迴歸** | 機器學習最基本的模型 |
| **曲線擬合** | 從數據點找出最佳曲線 |
| **信號處理** | 濾波、去噪 |
| **電腦視覺** | 相機校正、3D 重建 |
| **數值計算** | QR 分解求解更穩定 |

## 本章單元

```
01-inner-product-and-orthogonality/  內積與正交性
02-projections/                      投影（核心概念）
03-least-squares-regression/         最小平方回歸（ML 必學！）
04-gram-schmidt-process/             Gram-Schmidt 正交化
05-qr-decomposition/                 QR 分解
06-least-squares-normal-qr-svd/      解法比較：Normal vs QR vs SVD（病態/共線性）
07-regularization-ridge-and-truncated-svd/ 正則化：Ridge（Tikhonov）與 TSVD
08-regularization-model-selection/   參數選擇：用 hold-out 選 λ / k（bias–variance）
09-regularization-kfold-cross-validation/ 參數選擇（更穩）：k-fold CV + ASCII 曲線
10-iterative-solvers-for-ridge/      迭代法解 Ridge：GD vs CG（條件數與收斂）
11-preconditioning-for-ridge/       預條件化：CG vs PCG（Jacobi）
12-early-stopping-implicit-regularization/ Early Stopping（隱式正則化）：用迭代次數選模型
13-lsqr-iterative-least-squares/    LSQR：不形成 AᵀA 的最小平方迭代法
14-lsqr-damped-and-stopping-criteria/ Damped LSQR（Ridge）+ stopping criteria（更實務）
15-lsqr-lsmr-preconditioning/      LSQR/LSMR 類方法的預條件化：column scaling（右預條件）
16-damped-lsqr-damp-selection-cv/  用 k-fold CV 選 damp（Ridge λ=damp²）：更貼近 ML 超參數選擇
17-lsmr-iterative-least-squares/   LSMR：normal equations 上的 MINRES（教學版）+ 與 LSQR 對照
18-lsmr-advanced-preconditioning-randomized-qr/ 進階預條件化：Randomized QR（Blendenpik/LSRN 風格）加速 LSMR/LSQR
19-lsmr-damped-stopping-and-cv/   Damped LSMR（Ridge）+ stopping criteria + k-fold CV 選 damp
```

## 學習目標

完成本章後，你應該能夠：

1. **內積與正交**
   - 計算向量內積、長度、夾角
   - 判斷向量是否正交
   - 理解正交補空間

2. **投影**
   - 計算向量到直線的投影
   - 計算向量到平面的投影
   - 推導投影矩陣公式

3. **最小平方法**
   - 理解「無解」方程組的最佳近似
   - 推導正規方程 AᵀAx̂ = Aᵀb
   - 應用於線性迴歸

4. **Gram-Schmidt**
   - 將任意基底正交化
   - 理解正交基底的優勢

5. **QR 分解**
   - A = QR（Q 正交、R 上三角）
   - 用 QR 解最小平方問題

## 核心公式

```
投影公式：
  投影到向量 a：proj_a(b) = (aᵀb/aᵀa) a
  投影到子空間：proj = A(AᵀA)⁻¹Aᵀ b
  投影矩陣：P = A(AᵀA)⁻¹Aᵀ

最小平方：
  正規方程：AᵀA x̂ = Aᵀb
  最佳解：x̂ = (AᵀA)⁻¹Aᵀb

QR 分解：
  A = QR
  最小平方解：Rx̂ = Qᵀb

正則化（Ridge / TSVD）：
  Ridge：min ‖Ax-b‖² + λ‖x‖²  ⇔  (AᵀA + λI)x = Aᵀb
  TSVD：x̂_k = V_k Σ_k⁻¹ U_kᵀ b（只保留前 k 個奇異值/方向）
```

## 考試重點

### 常見題型

1. **投影計算**：求向量到直線/平面的投影
2. **投影矩陣**：求投影矩陣並驗證性質
3. **最小平方**：給定超定系統求最佳解
4. **Gram-Schmidt**：正交化向量組
5. **QR 分解**：分解矩陣並應用

### 關鍵性質

```
投影矩陣 P：
  P² = P（冪等）
  Pᵀ = P（對稱）

正交矩陣 Q：
  QᵀQ = I
  Q⁻¹ = Qᵀ
  保持長度：‖Qx‖ = ‖x‖
```

## 先修知識

- 第一章：內積、向量長度
- 第三章：行空間、零空間、四大子空間

## 延伸閱讀

- Strang, Chapter 4: Orthogonality
- MIT 18.06 Lecture 15-17
- 3Blue1Brown: Dot products and duality
