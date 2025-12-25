# LSMR：MINRES on Normal Equations 的最小平方迭代法

> 目標：用 matvec-only 的方式解 `min‖Ax-b‖`，並以「正常方程殘差」`‖Aᵀr‖` 為核心收斂指標

## 學習目標

1. 理解 LSMR 的定位：等價於在 normal equations `AᵀA x = Aᵀb` 上做 MINRES（但不形成 `AᵀA`）
2. 對照 LSQR：兩者都用 Golub–Kahan bidiagonalization，但收斂行為/殘差指標有差異
3. 用可驗算的量檢查：`‖Ax-b‖`、`‖Aᵀ(Ax-b)‖`、迭代數

## 本單元實作（Python）

```
python/
└── lsmr_manual.py   # 教學版 LSMR：用 bidiagonalization 建 tridiagonal，再解小型 least-squares（MINRES 定義）
```

