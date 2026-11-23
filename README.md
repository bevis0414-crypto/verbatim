# 逐字 Verbatim

錄音轉逐字稿網站：https://bevis0414-crypto.github.io/verbatim/

轉錄在 Google Colab 的免費 GPU 上執行，網站本身放在 GitHub Pages。

**兩種引擎：**
- **Google Colab**（主要）：網站按「啟動 Colab」→ 在 Colab 按 Ctrl+F9 → 點「回到逐字網站」。支援所有模型、講者分辨、AI 校對。
- **這台電腦的 GPU**（Colab 分不到 GPU 時的備案）：不用安裝，在瀏覽器裡用 WebGPU 跑 Whisper（transformers.js）。可用 Large v3 Turbo、Small、Base；沒有講者分辨和 AI 校對。

然後把錄音檔拖進網站即可。

## 穩定性設計

- **固定版本的執行環境**：引擎在獨立的 Python 3.11 虛擬環境中執行，所有套件由 `engine/requirements.lock` 鎖定（含 CUDA 12 函式庫、ffmpeg），不使用 Colab 內建套件，Colab 升級不會影響。
- **自動測試**：每次更新及每週一，GitHub Actions 會用真的 Whisper 模型和三人對話錄音，完整跑一遍上傳 → 轉錄 → 講者分辨 → AI 校對。
- **stable 分支**：通過測試才會發佈到 `stable`，Colab 筆記本只使用 `stable`。
- **自動恢復**：Colab 有看守程式，引擎當掉會自動重啟；網站斷線時說明原因並在恢復後自動接續。
- **錯誤回報**：網站上的「複製錯誤資訊」會收集版本、GPU、錯誤原因。

## 檔案

| 路徑 | 用途 |
|---|---|
| `index.html` | 網站介面 |
| `verbatim_colab.ipynb` | 一鍵啟動筆記本（由 `tools/make_notebook.py` 產生） |
| `engine/server.py` | 轉錄引擎（faster-whisper、講者分辨、API） |
| `engine/diarize.py` | 講者分辨（3D-Speaker CAM++ 聲紋 + 分群，sherpa-onnx） |
| `engine/requirements.in` / `.lock` | 依賴與完整鎖定版本 |
| `tests/` | 測試錄音產生器與端對端測試 |

更新依賴：改 `engine/requirements.in` 後執行
`uv pip compile engine/requirements.in --python-version 3.11 --python-platform x86_64-manylinux_2_28 -o engine/requirements.lock`
