# LSQR / LSMR 的預條件化（Preconditioning）

> 目標：在不形成 `AᵀA` 的情況下，加速大型最小平方迭代法的收斂（特別是共線性/病態矩陣）

## 學習目標

1. 理解「右預條件化」的改寫：令 `x = D⁻¹y`，改解 `min‖A D⁻¹ y - b‖`
2. 用最直觀的預條件器：**column scaling**（把 `A` 的每一欄縮放到相近尺度）
3. 比較 LSQR（matvec-only）在預條件化前後的迭代數、`‖Ax-b‖` 與 `‖Aᵀr‖`

## 本單元實作（Python）

```
python/
└── lsqr_lsmr_preconditioning_numpy.py   # LSQR + column scaling（並加 normal-eq baseline 供對照）
```

