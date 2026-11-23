# 逐字 Verbatim

錄音轉逐字稿網站：https://bevis0414-crypto.github.io/verbatim/

轉錄在 Google Colab 的免費 GPU 上執行，網站本身放在 GitHub Pages。

**兩種引擎：**
- **這台電腦**：網站上下載安裝程式（Windows 點兩下；Mac／Linux 在終端機貼一行指令），安裝一次後開機自動在背景待命，之後打開網站就自動連上。有 NVIDIA 顯示卡用 GPU，沒有用 CPU。移除：`launch.py --uninstall`。
- **Google Colab**：網站按「啟動 Colab」→ 在 Colab 按 Ctrl+F9 → 點「回到逐字網站」。

然後把錄音檔拖進網站即可。

## 穩定性設計

- **固定版本的執行環境**：引擎在獨立的 Python 3.11 虛擬環境中執行，所有套件由 `engine/requirements.lock` 鎖定（含 CUDA 12 函式庫、ffmpeg），不使用 Colab 內建套件，Colab 升級不會影響。
- **自動測試**：每次更新及每週一，GitHub Actions 會用真的 Whisper 模型和三人對話錄音，完整跑一遍上傳 → 轉錄 → 講者分辨 → AI 校對；Windows 和 Mac 另外實際執行安裝程式、測試後再移除。
- **stable 分支**：三個平台都通過才會發佈到 `stable`，Colab 筆記本和本機安裝程式只使用 `stable`。
- **自動恢復**：Colab 與本機都有看守程式，引擎當掉會自動重啟；網站斷線時說明原因並在恢復後自動接續。
- **錯誤回報**：網站上的「複製錯誤資訊」會收集版本、GPU、錯誤原因。

## 檔案

| 路徑 | 用途 |
|---|---|
| `index.html` | 網站介面 |
| `verbatim_colab.ipynb` | 一鍵啟動筆記本（由 `tools/make_notebook.py` 產生） |
| `engine/server.py` | 轉錄引擎（faster-whisper、講者分辨、API） |
| `engine/diarize.py` | 講者分辨（3D-Speaker CAM++ 聲紋 + 分群，sherpa-onnx） |
| `engine/requirements.in` / `*.lock` | 依賴與各平台（Linux、Windows、Mac）完整鎖定版本 |
| `local/` | 本機安裝程式（`install.bat`、`install.sh`）與啟動器 `launch.py` |
| `tests/` | 測試錄音產生器與端對端測試 |

更新依賴：改 `engine/requirements.in` 後，對三個平台各執行一次
`uv pip compile engine/requirements.in --python-version 3.11 --python-platform <x86_64-manylinux_2_28 | x86_64-pc-windows-msvc | aarch64-apple-darwin> -o engine/<requirements.lock | requirements-win.lock | requirements-mac.lock>`
