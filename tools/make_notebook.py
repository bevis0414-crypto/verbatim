"""產生 verbatim_colab.ipynb（一格、程式碼收起來）。改筆記本請改這支，再執行：python tools/make_notebook.py"""
import json, os

OWNER, REPO = "bevis0414-crypto", "verbatim"
CODE = r'''#@title ▶ 啟動逐字 Verbatim　　（按 Ctrl+F9，第一次約 3 分鐘）
OWNER, REPO = "__OWNER__", "__REPO__"
SITE = f"https://{OWNER}.github.io/{REPO}/"
UV_VERSION = "0.11.32"
CLOUDFLARED = "https://github.com/cloudflare/cloudflared/releases/download/2026.10.0/cloudflared-linux-amd64"
ROOT = "/content/verbatim"

import hashlib, io, json, os, re, shutil, subprocess, sys, tarfile, threading, time, urllib.request
from IPython.display import HTML, display, clear_output

def say(msg, color="#8b9098"):
    display(HTML(f'<div style="font:15px sans-serif;color:{color};margin:4px 0">{msg}</div>'))

def card(title, body, color="#eceae4", button=None):
    btn = (f'<a href="{button[1]}" target="_blank" style="display:inline-block;background:#ff5a1f;color:#fff;padding:14px 28px;'
           f'border-radius:12px;font-size:17px;font-weight:600;text-decoration:none;margin-top:18px">{button[0]}</a>') if button else ""
    display(HTML(f'<div style="font-family:sans-serif;background:#15171b;color:#eceae4;border-radius:16px;padding:28px;max-width:680px">'
                 f'<div style="font-size:22px;font-weight:700;margin-bottom:8px;color:{color}">{title}</div>'
                 f'<div style="color:#b9bcc2;line-height:1.7">{body}</div>{btn}</div>'))

def fail(title, detail):
    clear_output()
    card("❌ " + title, detail + "<br><br>請把這個畫面截圖給網站管理者。", "#ff5a1f")
    raise SystemExit

def alive(url):
    try:
        return json.load(urllib.request.urlopen(url + "/api/system", timeout=5))
    except Exception:
        return None

# ① GPU
if subprocess.run("nvidia-smi", shell=True, capture_output=True).returncode != 0:
    say("⚠️ 這次沒有分到 GPU，轉錄會非常慢。請到上方選單「執行階段 → 變更執行階段類型」選 T4 GPU 後再按一次 Ctrl+F9；"
        "如果選不到，可能是今天的免費額度用完了，可以晚一點再試。", "#ff5a1f")

# ② 下載引擎（只用通過自動測試的 stable 版本）
say("① 下載轉錄引擎…")
src = None
for ref in ("stable", "main"):
    try:
        raw = urllib.request.urlopen(f"https://codeload.github.com/{OWNER}/{REPO}/tar.gz/refs/heads/{ref}", timeout=60).read()
        src = (ref, raw); break
    except Exception:
        continue
if not src:
    fail("下載引擎失敗", "連不到 GitHub，請確認網路後再按一次 Ctrl+F9。")
shutil.rmtree(f"{ROOT}/engine", ignore_errors=True)
with tarfile.open(fileobj=io.BytesIO(src[1])) as t:
    for m in t.getmembers():
        parts = m.name.split("/", 1)
        if len(parts) == 2 and parts[1].startswith("engine/") and m.isfile():
            m.name = parts[1]
            t.extract(m, ROOT)

# ③ 固定版本的獨立環境：Python 3.11 + requirements.lock，不使用 Colab 內建套件
say("② 準備執行環境（第一次約 1–2 分鐘）…")
lock = open(f"{ROOT}/engine/requirements.lock", "rb").read()
stamp = f"{ROOT}/venv/.lock-sha"
want = hashlib.sha256(lock).hexdigest()
if not (os.path.exists(stamp) and open(stamp).read() == want):
    r = subprocess.run(f"{sys.executable} -m pip install -q uv=={UV_VERSION}", shell=True, capture_output=True, text=True)
    uv = shutil.which("uv") or os.path.join(os.path.dirname(sys.executable), "uv")
    shutil.rmtree(f"{ROOT}/venv", ignore_errors=True)
    for cmd in ([uv, "venv", "-q", "-p", "3.11", f"{ROOT}/venv"],
                [uv, "pip", "install", "-q", "--python", f"{ROOT}/venv/bin/python", "-r", f"{ROOT}/engine/requirements.lock"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail("執行環境安裝失敗", "<pre style='white-space:pre-wrap;font-size:12px'>" + (r.stderr or r.stdout)[-1500:] + "</pre>")
    open(stamp, "w").write(want)

# ④ 啟動引擎（重新執行時先關掉舊的）
say("③ 啟動引擎並載入 Large v2 模型（第一次約 1–2 分鐘）…")
subprocess.run("pkill -f verbatim/engine/server.py", shell=True)
time.sleep(1)
env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "LD_LIBRARY_PATH"))}
env.update(PORT="8000", VERBATIM_PRELOAD="large-v2", VERBATIM_DATA=ROOT)
log = open(f"{ROOT}/engine.log", "w")
subprocess.Popen([f"{ROOT}/venv/bin/python", f"{ROOT}/engine/server.py"], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
st = None
for _ in range(480):
    st = alive("http://localhost:8000")
    if st and st.get("ready"):
        break
    time.sleep(1)
if not st:
    fail("引擎啟動失敗", "<pre style='white-space:pre-wrap;font-size:12px'>" + open(f"{ROOT}/engine.log").read()[-1500:] + "</pre>")

# ⑤ 對外連線網址（Cloudflare 免費通道，固定版本）
say("④ 建立連線網址…")
url = globals().get("_verbatim_url")
if not (url and alive(url)):
    subprocess.run("pkill -f cloudflared", shell=True)
    if not os.path.exists(f"{ROOT}/cloudflared"):
        urllib.request.urlretrieve(CLOUDFLARED, f"{ROOT}/cloudflared")
        os.chmod(f"{ROOT}/cloudflared", 0o755)
    proc = subprocess.Popen([f"{ROOT}/cloudflared", "tunnel", "--no-autoupdate", "--url", "http://localhost:8000"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url, t0 = None, time.time()
    for line in proc.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0); break
        if time.time() - t0 > 90:
            break
    if not url:
        fail("建立連線網址失敗", "Cloudflare 暫時沒有回應，請再按一次 Ctrl+F9。")
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
    for _ in range(30):
        if alive(url): break
        time.sleep(2)
    _verbatim_url = url

clear_output()
dev = f"GPU {st.get('gpu') or ''}" if st.get("device") == "cuda" else "CPU（會很慢）"
warn = f'<div style="color:#ff5a1f;margin-top:12px">⚠️ {st["note"]}</div>' if st.get("note") else ""
card("✅ 轉錄引擎已啟動",
     f"點下面的按鈕回到逐字網站。<b style='color:#eceae4'>這個 Colab 分頁請保持開著</b>，關掉後網站就會斷線。{warn}"
     f"<div style='color:#8b9098;font-size:13px;margin-top:14px'>引擎版本 {st.get('version')}（{src[0]}）· {dev} · {url}</div>",
     button=("回到逐字網站 ↗", SITE + "#server=" + url))
'''

code = CODE.replace("__OWNER__", OWNER).replace("__REPO__", REPO)
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "name": "逐字 Verbatim"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [
          {"cell_type": "markdown", "metadata": {}, "source": [
              "# 逐字 Verbatim — 雲端轉錄引擎\n",
              "按 **Ctrl+F9**（Mac：⌘+F9）全部執行。跳出「這個筆記本並非由 Google 撰寫」時選 **仍要執行**。\n",
              "完成後點橘色按鈕回到網站，這個分頁保持開著就好。"]},
          {"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None, "outputs": [],
           "source": code.splitlines(True)}]}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verbatim_colab.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
compile(code.replace("#@title", "#"), "nb", "exec")
print("written", os.path.abspath(out))
