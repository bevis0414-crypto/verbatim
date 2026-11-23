#!/bin/sh
# 逐字 Verbatim 本機引擎安裝（Mac／Linux）：curl -fsSL https://bevis0414-crypto.github.io/verbatim/local/install.sh | sh
set -e
case "$(uname)" in
  Darwin) VB="$HOME/Library/Application Support/Verbatim" ;;
  *)      VB="$HOME/.local/share/verbatim" ;;
esac
mkdir -p "$VB"
if [ ! -x "$VB/uv/uv" ] && [ ! -x "$VB/uv/bin/uv" ]; then
  echo "[準備] 下載安裝工具 uv ..."
  curl -LsSf https://astral.sh/uv/0.11.32/install.sh | env UV_INSTALL_DIR="$VB/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null
fi
UV="$VB/uv/uv"; [ -x "$UV" ] || UV="$VB/uv/bin/uv"
[ -x "$UV" ] || { echo "下載 uv 失敗，請確認網路連線後再執行一次。"; exit 1; }
if [ -n "$VERBATIM_LAUNCH_SRC" ]; then cp "$VERBATIM_LAUNCH_SRC" "$VB/launch.py"
else curl -fsSL -o "$VB/launch.py" "https://raw.githubusercontent.com/bevis0414-crypto/verbatim/stable/local/launch.py"; fi
VERBATIM_UV="$UV" "$UV" run --no-project --python 3.11 "$VB/launch.py" --install "$@"
