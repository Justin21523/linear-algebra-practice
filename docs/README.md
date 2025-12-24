# 文件說明（docs/）

本資料夾用來存放「實作說明文件」與其他輔助文件。從現在起，**每一個新增或修改的程式實作**都需要在 `docs/` 補上一份對應的說明（繁體中文），並包含必要的程式碼節錄與執行方式。

## 實作說明文件放哪裡？

採用「對應單元資料夾」的方式，讓文件可直接追溯到該單元的所有語言實作：

- 單元路徑：`04-orthogonality-and-least-squares/02-projections/`
- 說明文件：`docs/implementations/04-orthogonality-and-least-squares/02-projections/README.md`

> 規則：`docs/implementations/<章節>/<單元>/README.md`

同一個單元若有多語言（或 `*_manual.py` / `*_numpy.py` 兩版本），請在同一份 `README.md` 中以小節分開說明。

## 文件內容最低要求

每份實作說明文件請至少包含：

1. **目標與背景**：這個檔案要示範/解決什麼線性代數概念？
2. **如何執行**：清楚列出指令與工作目錄（例如 `python ...` / `gcc ...` / `node ...`）。
3. **核心做法**：用條列說明主要步驟/公式/演算法。
4. **程式碼區段**：節錄關鍵函數或邏輯（小段即可），並解釋其意義。
5. **驗證方式**：如何手動檢查輸出是否合理（含浮點誤差容忍度的說明，如適用）。

## 範本

可直接複製 `docs/IMPLEMENTATION_TEMPLATE.md` 來撰寫。
