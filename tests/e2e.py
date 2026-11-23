"""端對端測試：啟動引擎 → 分段上傳 → 轉錄（真的 Whisper）→ 講者分辨 → 檢查結果。
用法：python tests/e2e.py <dialog.mp3> <dialog_truth.json> [model]
自動測試（GitHub Actions）在 CPU 上執行；在 Colab 上也可以拿來自我檢查。
"""
import json, os, subprocess, sys, time, urllib.request, uuid

MP3, TRUTH = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "small"
PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"
ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "server.py")


def call(path, fields=None, files=None, js=None):
    if js is not None:
        req = urllib.request.Request(BASE + path, data=json.dumps(js).encode(), headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60))
    if fields is None and files is None:
        return json.load(urllib.request.urlopen(BASE + path, timeout=30))
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in (fields or {}).items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (name, data) in (files or {}).items():
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                 f'Content-Type: application/octet-stream\r\n\r\n').encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def main():
    env = dict(os.environ, PORT=str(PORT), VERBATIM_CPU=os.environ.get("VERBATIM_CPU", "1"))
    proc = subprocess.Popen([sys.executable, ENGINE], env=env)
    try:
        for _ in range(60):
            try:
                print("system:", call("/api/system")); break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("引擎沒有啟動")
        data = open(MP3, "rb").read()
        uid = uuid.uuid4().hex[:16]
        half = len(data) // 2  # 故意分兩段上傳，測試分段合併
        call(f"/api/upload/{uid}", {"part": 0}, {"chunk": ("p", data[:half])})
        call(f"/api/upload/{uid}", {"part": 1}, {"chunk": ("p", data[half:])})
        jid = call("/api/transcribe", {"uid": uid, "filename": "dialog.mp3", "language": "zh", "model": MODEL,
                                       "trad": "1", "hint": "", "speakers": "auto"})["id"]
        t0 = time.time()
        while True:
            j = call(f"/api/jobs/{jid}?since=0&rev=-1")
            if j["status"] in ("done", "error", "cancelled"):
                break
            if time.time() - t0 > 1200:
                raise SystemExit("逾時")
            time.sleep(2)
        print(json.dumps(j, ensure_ascii=False, indent=1))
        print("diag:", json.dumps(call("/api/diag"), ensure_ascii=False))
        truth = json.load(open(TRUTH))
        segs = j["segments"]
        text = "".join(s["text"] for s in segs)
        n_spk = len({s.get("speaker") for s in segs})
        checks = {
            "完成": j["status"] == "done",
            "沒有警告": not j["warning"],
            "有逐字稿": len(segs) >= 4,
            # 看字而不是看詞：要有繁體專用字、不能有簡體字（不受小模型聽錯詞影響）
            "繁體輸出": sum(ch in text for ch in "們這說個題點還見論開聽議對讓會時裡為後來") >= 3
                        and not any(ch in text for ch in "们这说个题点还见论开听议对让会时里为后"),
            "內容大致正確": sum(k in text for k in ("工作坊", "公園", "照明", "座椅", "長輩", "意見", "討論", "車道")) >= 3,
            "沒有「壹」": "壹" not in text,
            "講者人數": n_spk == len({t["speaker"] for t in truth}),
        }
        if os.environ.get("VERBATIM_TEST_PROOF") == "1":  # AI 校對
            pid = call("/api/proofread", js={"segments": [{"text": x["text"]} for x in segs], "hint": "工作坊、公園", "trad": True})["id"]
            t0 = time.time()
            while True:
                p = call(f"/api/proofread/{pid}")
                if p["status"] in ("done", "error", "cancelled") or time.time() - t0 > 1500:
                    break
                time.sleep(3)
            print("proofread:", json.dumps(p, ensure_ascii=False, indent=1)[:3000])
            res = p.get("result") or []
            checks["校對完成"] = p["status"] == "done"
            checks["校對沒有掉句"] = len(res) == len(segs) and all(r["text"] for r in res)
            checks["校對後仍是繁體"] = not any(ch in "".join(r["text"] for r in res) for ch in "这们说长园议题")
        for k, ok in checks.items():
            print(("✅" if ok else "❌"), k)
        if not all(checks.values()):
            sys.exit(1)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
