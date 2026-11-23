"""逐字 Verbatim — 本機引擎啟動器（只用 Python 標準函式庫）

用法（一般使用者不需要手動執行，安裝程式會處理）：
  launch.py --install     第一次安裝：下載引擎、建立固定版本環境、設定開機自動啟動、啟動並開啟網站
  launch.py --background  開機時由系統自動執行：更新引擎後在背景待命，當掉會自動重啟
  launch.py --uninstall   移除：關掉引擎、取消開機啟動、刪除所有檔案（含模型）
  launch.py --prepare     只下載引擎並建立環境（自動測試用）
"""
import hashlib, io, json, os, platform, shutil, subprocess, sys, tarfile, time, urllib.request, webbrowser

OWNER, REPO = "bevis0414-crypto", "verbatim"
SITE = f"https://{OWNER}.github.io/{REPO}/"
PORT = 8765
URL = f"http://localhost:{PORT}"
WIN, MAC = sys.platform == "win32", sys.platform == "darwin"

if WIN:
    ROOT = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Verbatim")
elif MAC:
    ROOT = os.path.expanduser("~/Library/Application Support/Verbatim")
else:
    ROOT = os.path.expanduser("~/.local/share/verbatim")
ROOT = os.environ.get("VERBATIM_ROOT", ROOT)
VENV_PY = os.path.join(ROOT, "venv", "Scripts" if WIN else "bin", "python.exe" if WIN else "python")
PIDFILE = os.path.join(ROOT, "launcher.pid")
STOPFILE = os.path.join(ROOT, "stop.flag")
LOCK = {"win32": "requirements-win.lock", "darwin": "requirements-mac.lock"}.get(sys.platform, "requirements.lock")


def say(msg):
    try:
        print(msg, flush=True)
    except Exception:
        pass
    try:
        with open(os.path.join(ROOT, "launcher.log"), "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except Exception:
        pass


def uv_exe():
    exe = "uv.exe" if WIN else "uv"
    for c in (os.environ.get("VERBATIM_UV"), os.path.join(ROOT, "uv", exe), os.path.join(ROOT, "uv", "bin", exe), shutil.which("uv")):
        if c and os.path.exists(c):
            return c
    raise SystemExit("找不到 uv，請重新執行安裝程式")


def engine_status():
    try:
        s = json.load(urllib.request.urlopen(URL + "/api/system", timeout=3))
        return s if s.get("app") == "verbatim" else None
    except Exception:
        return None


# ---------- 下載引擎（只用通過自動測試的 stable 版） ----------
def update_engine():
    src_dir = os.environ.get("VERBATIM_ENGINE_SRC")  # 自動測試：用目前這次提交的引擎
    new = os.path.join(ROOT, "engine.new")
    shutil.rmtree(new, ignore_errors=True)
    if src_dir:
        shutil.copytree(src_dir, new)
        ref = "local"
    else:
        raw, ref = None, None
        for ref in ("stable", "main"):
            try:
                raw = urllib.request.urlopen(f"https://codeload.github.com/{OWNER}/{REPO}/tar.gz/refs/heads/{ref}",
                                             timeout=60).read()
                break
            except Exception:
                raw = None
        if raw is None:
            if os.path.exists(os.path.join(ROOT, "engine", "server.py")):
                say("連不到 GitHub，先用上次下載的引擎")
                return "cached"
            raise SystemExit("下載引擎失敗：連不到 GitHub，請確認網路後再試一次")
        os.makedirs(new)
        with tarfile.open(fileobj=io.BytesIO(raw)) as t:
            for m in t.getmembers():
                parts = m.name.split("/", 1)
                if len(parts) == 2 and parts[1].startswith("engine/") and m.isfile():
                    m.name = parts[1][len("engine/"):]
                    kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                    t.extract(m, new, **kw)
    old = os.path.join(ROOT, "engine")
    shutil.rmtree(old, ignore_errors=True)
    os.replace(new, old)
    return ref


def ensure_env():
    lock_path = os.path.join(ROOT, "engine", LOCK)
    want = hashlib.sha256(open(lock_path, "rb").read()).hexdigest()
    stamp = os.path.join(ROOT, "venv", ".lock-sha")
    if os.path.exists(stamp) and open(stamp, encoding="utf-8").read() == want and os.path.exists(VENV_PY):
        return False
    say("準備執行環境（第一次約 3–5 分鐘，之後只有更新時才會再裝）…")
    uv = uv_exe()
    shutil.rmtree(os.path.join(ROOT, "venv"), ignore_errors=True)
    for cmd in ([uv, "venv", "-q", "-p", "3.11", os.path.join(ROOT, "venv")],
                [uv, "pip", "install", "-q", "--python", VENV_PY, "-r", lock_path]):
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise SystemExit("安裝執行環境失敗：\n" + (r.stderr or r.stdout)[-1500:])
    open(stamp, "w", encoding="utf-8").write(want)
    return True


# ---------- 啟動與看守 ----------
def has_nvidia():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def start_engine(hidden):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(PORT=str(PORT), VERBATIM_LOCAL="1", VERBATIM_DATA=ROOT, VERBATIM_UV=uv_exe(), PYTHONIOENCODING="utf-8")
    env.pop("VERBATIM_PRELOAD", None)  # 本機不預先載入模型：平常待命幾乎不佔記憶體，第一次轉錄時才載入
    kw = {}
    if WIN and hidden:
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    log = open(os.path.join(ROOT, "engine.log"), "w", encoding="utf-8")
    return subprocess.Popen([VENV_PY, os.path.join(ROOT, "engine", "server.py")], env=env, cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, **kw)


def wait_ready(proc, limit=120):
    for _ in range(limit):
        if proc.poll() is not None:
            return None
        s = engine_status()
        if s and s.get("pid") == proc.pid:
            return s
        time.sleep(1)
    return None


def supervise(hidden=True):
    """在前景執行：啟動引擎，當掉自動重啟；10 分鐘內當掉 3 次就停止"""
    crashes = []
    open(PIDFILE, "w", encoding="utf-8").write(str(os.getpid()))
    while True:
        if os.path.exists(STOPFILE):  # 安裝程式要求停止（例如正在更新環境）
            return
        proc = start_engine(hidden)
        if not wait_ready(proc):
            say("引擎啟動失敗，紀錄：" + open(os.path.join(ROOT, "engine.log"), encoding="utf-8", errors="ignore").read()[-800:])
        else:
            say(f"引擎待命中：{URL}")
        while proc.poll() is None:
            if os.path.exists(STOPFILE):
                proc.terminate()
                return
            time.sleep(2)
        crashes = [t for t in crashes if time.time() - t < 600] + [time.time()]
        say(f"引擎停止（結束代碼 {proc.returncode}）")
        if len(crashes) >= 3:
            say("引擎在 10 分鐘內反覆停止，已暫停自動重啟。請重新開機或重新執行安裝程式。")
            return
        time.sleep(3)


def launcher_python():
    """用 uv 管理的 Python 3.11 跑啟動器（不放在引擎環境裡，更新引擎環境時才不會被鎖住）"""
    exe = sys.executable
    if WIN:
        w = os.path.join(os.path.dirname(exe), "pythonw.exe")
        if os.path.exists(w):
            return w
    return exe


def copy_self():
    dst = os.path.join(ROOT, "launch.py")
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copyfile(__file__, dst)
    return dst


def background_cmd():
    """開機啟動／背景執行的指令。Windows 優先用 uvw.exe（不跳出視窗），由 uv 找 Python 3.11，不寫死 Python 路徑"""
    launch = copy_self()
    if WIN:
        uvw = os.path.join(os.path.dirname(uv_exe()), "uvw.exe")
        if os.path.exists(uvw):
            return [uvw, "run", "--no-project", "--python", "3.11", launch, "--background"]
    return [launcher_python(), launch, "--background"]


def register_autostart():
    cmd = background_cmd()
    if WIN:
        import winreg
        line = " ".join(f'"{c}"' for c in cmd)
        # CreateKey：全新的 Windows 帳號可能還沒有這個機碼，沒有就建立
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as k:
            winreg.SetValueEx(k, "VerbatimEngine", 0, winreg.REG_SZ, line)
    elif MAC:
        plist = os.path.expanduser("~/Library/LaunchAgents/io.github.verbatim.engine.plist")
        os.makedirs(os.path.dirname(plist), exist_ok=True)
        args = "".join(f"<string>{c}</string>" for c in cmd)
        open(plist, "w", encoding="utf-8").write(
            '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd"><plist version="1.0"><dict>'
            f'<key>Label</key><string>io.github.verbatim.engine</string><key>ProgramArguments</key><array>{args}</array>'
            '<key>RunAtLoad</key><true/></dict></plist>')
    else:
        d = os.path.expanduser("~/.config/autostart")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "verbatim-engine.desktop"), "w", encoding="utf-8").write(
            "[Desktop Entry]\nType=Application\nName=逐字 Verbatim 引擎\nExec=" + " ".join(f'"{c}"' for c in cmd) +
            "\nX-GNOME-Autostart-enabled=true\nNoDisplay=true\n")


def unregister_autostart():
    if WIN:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                                winreg.KEY_SET_VALUE) as k:
                winreg.DeleteValue(k, "VerbatimEngine")
        except OSError:
            pass
    elif MAC:
        p = os.path.expanduser("~/Library/LaunchAgents/io.github.verbatim.engine.plist")
        subprocess.run(["launchctl", "unload", p], capture_output=True)
        if os.path.exists(p):
            os.remove(p)
    else:
        p = os.path.expanduser("~/.config/autostart/verbatim-engine.desktop")
        if os.path.exists(p):
            os.remove(p)


def kill_pid(pid):
    try:
        if WIN:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(int(pid), 15)
    except Exception:
        pass


def stop_engine():
    """先請背景看守員停下（避免它把引擎重新開起來），再關引擎"""
    open(STOPFILE, "w", encoding="utf-8").close()
    try:
        kill_pid(int(open(PIDFILE, encoding="utf-8").read()))
    except Exception:
        pass
    s = engine_status()
    if s and s.get("pid"):
        kill_pid(s["pid"])
    for _ in range(10):
        if not engine_status():
            break
        time.sleep(1)


def start_background_detached():
    if os.path.exists(STOPFILE):
        os.remove(STOPFILE)
    cmd = background_cmd()
    if WIN:
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW，再加上 CREATE_BREAKAWAY_FROM_JOB：
        # 安裝視窗所在的「工作群組」關閉時（例如某些終端機程式），背景引擎才不會被一起關掉。
        # 有些環境不允許脫離群組，那就退回不脫離的方式。
        base = 0x00000008 | 0x00000200 | 0x08000000
        try:
            subprocess.Popen(cmd, creationflags=base | 0x01000000, close_fds=True)
        except OSError:
            subprocess.Popen(cmd, creationflags=base, close_fds=True)
    else:
        subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    os.makedirs(ROOT, exist_ok=True)
    mode = (sys.argv[1] if len(sys.argv) > 1 else "--install").lower()

    if mode == "--uninstall":
        unregister_autostart()
        stop_engine()
        time.sleep(2)
        shutil.rmtree(ROOT, ignore_errors=True)
        say("已移除逐字 Verbatim 本機引擎（含下載的模型）。")
        return

    if mode == "--prepare":
        say(f"引擎來源：{update_engine()}")
        ensure_env()
        say(f"環境就緒：{VENV_PY}")
        return

    if mode == "--background":
        if engine_status():  # 已經有引擎在跑（例如安裝時剛啟動的）
            return
        try:
            update_engine()
            ensure_env()
        except SystemExit as e:  # 更新失敗時用舊的繼續跑
            say(f"更新略過：{e}")
            if not os.path.exists(VENV_PY):
                return
        supervise(hidden=True)
        return

    # --install（安裝程式呼叫，也可以重複執行來修復或更新）
    say("── 逐字 Verbatim 本機引擎 ──")
    say(f"安裝位置：{ROOT}")
    say("① 下載轉錄引擎…")
    stop_engine()
    update_engine()
    say("② 準備執行環境…")
    ensure_env()
    say("③ 設定開機自動啟動…")
    try:
        register_autostart()
    except Exception as e:  # 設定失敗不影響這次使用，只是下次開機要再執行一次安裝程式
        say(f"⚠️ 無法設定開機自動啟動（{e}）。這次可以正常使用；之後開機若網站連不上，再執行一次安裝程式即可。")
    say("④ 啟動引擎…")
    start_background_detached()
    s = None
    for _ in range(90):
        s = engine_status()
        if s:
            break
        time.sleep(1)
    if not s:
        raise SystemExit("引擎沒有啟動，紀錄：" +
                         open(os.path.join(ROOT, "engine.log"), encoding="utf-8", errors="ignore").read()[-800:])
    gpu = "NVIDIA 顯示卡" if has_nvidia() else ("Apple 晶片" if MAC and platform.machine() == "arm64" else "CPU")
    say(f"✅ 安裝完成！引擎版本 {s.get('version')}，使用 {gpu}。")
    say("   之後每次開機都會自動在背景待命，直接打開逐字網站就能用。")
    say("   第一次連線時 Chrome 會詢問是否允許存取這台裝置，請按「允許」。")
    if "--no-browser" not in sys.argv:
        webbrowser.open(SITE + "#server=" + URL)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            say(f"❌ {e.code}")
        raise
