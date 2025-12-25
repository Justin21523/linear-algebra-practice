# Randomized SVD vs Oja：同一份資料的品質 / 成本對照

> 目標：用同一份中心化資料 `Xc`，比較兩條「大規模 PCA」路線：
>
> - **Randomized SVD**：一次（或少數次）做隨機投影 + 小矩陣 SVD，直接拿到 top-k 主成分子空間
> - **Oja’s online PCA**：用串流/小批次 SGD 逐步學出 top-k 主成分子空間

## 學習目標

1. 用相同資料與相同指標，公平比較兩種方法的品質：principal angles、EVR
2. 觀察 Randomized SVD 的 `p/q` 與 Oja 的 learning rate 對品質/成本的影響
3. 建立你在 ML/數值線代常用的評估習慣：先看「子空間是否對」，再看「成本是否划算」

## 本單元實作（Python）

```
python/
└── benchmark_randomized_svd_vs_oja_numpy.py   # 同資料 sweep (k,p,q) 與 learning-rate，印出品質/成本表格
```

