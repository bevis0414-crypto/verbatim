# 逐字 Verbatim

錄音轉逐字稿網站。轉錄在 Google Colab 的免費 GPU 上執行（faster-whisper），網站本身放在 GitHub Pages。

- 網站：https://bevis0414-crypto.github.io/verbatim/
- 使用：打開網站 → 按「啟動 Colab」→ 在 Colab 按 Ctrl+F9 → 點「回到逐字網站」

檔案
- `index.html`：網站介面
- `server.py`：Colab 上執行的轉錄伺服器（faster-whisper、講者分辨）
- `verbatim_colab.ipynb`：一鍵啟動筆記本
