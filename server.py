"""逐字 Verbatim — 在 Colab GPU 上跑的錄音轉逐字稿伺服器（faster-whisper + 講者分辨）"""
import os, uuid, time, queue, threading, traceback, gc

try:  # 先載入 torch，讓 Colab 內建的 cuDNN 進到行程裡
    import torch
except Exception:
    torch = None

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
MOCK = os.environ.get("VERBATIM_MOCK") == "1"

MODELS = {"large-v2": "large-v2", "large-v3": "large-v3", "turbo": "large-v3-turbo",
          "medium": "medium", "small": "small"}
ZH_PROMPT = "以下是一段繁體中文的逐字稿，包含適當的標點符號。"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs: dict = {}
work_q: "queue.Queue[str]" = queue.Queue()
_model = {"name": None, "obj": None}
_spk = {"enc": None}
_state = {"device": None, "gpu": None, "note": ""}
_cc = None


def device_info():
    if _state["device"] is None:
        try:
            import ctranslate2
            n = ctranslate2.get_cuda_device_count()
        except Exception:
            n = 0
        _state["device"] = "cuda" if n > 0 else "cpu"
        if n > 0:
            try:
                _state["gpu"] = torch.cuda.get_device_name(0)
            except Exception:
                _state["gpu"] = "NVIDIA GPU"
    return _state


def get_model(key):
    name = MODELS.get(key, "large-v2")
    if _model["name"] == name:
        return _model["obj"]
    _model["obj"] = _model["name"] = None  # 一次只放一個模型在 GPU，避免記憶體不足
    gc.collect()
    from faster_whisper import WhisperModel
    dev = device_info()["device"]
    try:
        m = WhisperModel(name, device=dev, compute_type="float16" if dev == "cuda" else "int8")
        if dev == "cuda":  # 先試跑一秒空白音訊，確認 GPU 函式庫（cuBLAS／cuDNN）真的載得到
            import numpy as np
            list(m.transcribe(np.zeros(16000, np.float32), language="en")[0])
    except Exception as e:
        if dev != "cuda":
            raise
        traceback.print_exc()
        _state["device"], _state["note"] = "cpu", f"GPU 無法使用，暫時改用 CPU（會很慢）：{e}"
        m = WhisperModel(name, device="cpu", compute_type="int8")
    _model.update(name=name, obj=m)
    return m


def load_audio(path, sr=16000):
    """用 ffmpeg 解碼成 16 kHz 單聲道，不依賴 PyAV（新版 PyAV 與 faster-whisper 不相容）"""
    import subprocess, numpy as np
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1",
                        "-acodec", "pcm_s16le", "-ar", str(sr), "-"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("無法讀取這個檔案的音訊：" + (r.stderr.decode(errors="ignore").strip()[-200:] or "格式不支援"))
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0


def to_trad(text):
    global _cc
    if _cc is None:
        try:
            from opencc import OpenCC
            _cc = OpenCC("s2twp")
        except Exception:
            _cc = False
    return _cc.convert(text) if _cc else text


# ---------- 講者分辨：每段取聲紋（ECAPA），再分群 ----------
def speaker_encoder():
    if _spk["enc"] is None:
        try:  # 新版 torchaudio 拿掉了這個函式，舊版 speechbrain 還會呼叫
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]
        except Exception:
            pass
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        dev = "cuda" if device_info()["device"] == "cuda" else "cpu"
        _spk["enc"] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=os.path.join(APP_DIR, "ecapa"),
            run_opts={"device": dev})
    return _spk["enc"]


def diarize(audio, segs, n_speakers):
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    if len(segs) < 2:
        return [0] * len(segs)
    enc, sr = speaker_encoder(), 16000
    embs, idx = [], []
    for i, s in enumerate(segs):
        a, b = int(s["start"] * sr), int(s["end"] * sr)
        if b - a < sr * 0.8:  # 太短的段落聲紋不可靠，之後跟著鄰近段落
            continue
        if b - a > sr * 12:  # 長段落取中間 12 秒
            c = (a + b) // 2
            a, b = c - 6 * sr, c + 6 * sr
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(audio[a:b].copy())[None]).squeeze().cpu().numpy()
        embs.append(e / (np.linalg.norm(e) + 1e-9))
        idx.append(i)
    if len(embs) < 2:
        return [0] * len(segs)
    X = np.stack(embs)
    if n_speakers and n_speakers > 1:
        cl = AgglomerativeClustering(n_clusters=min(n_speakers, len(X)), metric="cosine", linkage="average")
    else:
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=0.62, metric="cosine", linkage="average")
    raw = cl.fit_predict(X)
    order, labels = {}, [None] * len(segs)
    for i, r in zip(idx, raw):  # 依出場順序重新編號：講者 1、講者 2…
        labels[i] = order.setdefault(int(r), len(order))
    for i in range(len(labels)):  # 短段落：找時間上最近、有標記的段落
        if labels[i] is None:
            near = min(idx, key=lambda j: abs(j - i))
            labels[i] = labels[near]
    return labels


def run_job(jid):
    job = jobs[jid]
    try:
        if MOCK:
            return mock_run(job)
        job["status"] = "loading"
        audio = load_audio(job["path"])
        model = get_model(job["model"])
        job["status"] = "transcribing"
        lang = job["language"] or None
        prompt = " ".join(p for p in [ZH_PROMPT if lang in (None, "zh") else "", job["hint"]] if p) or None
        segs, info = model.transcribe(audio, language=lang, beam_size=5, vad_filter=True,
                                      initial_prompt=prompt, condition_on_previous_text=False)
        job["duration"], job["language"] = round(info.duration, 2), info.language
        trad = job["trad"] and info.language == "zh"
        for s in segs:
            if job.get("cancel"):
                job["status"] = "cancelled"
                return
            text = s.text.strip()
            if trad:
                text = to_trad(text)
            if text:
                job["segments"].append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
            job["progress"] = min(s.end / info.duration, 1.0) if info.duration else 0
        job["progress"] = 1.0
        if job["speakers"] != "off" and job["segments"]:
            job["status"] = "diarizing"
            n = int(job["speakers"]) if job["speakers"].isdigit() else 0
            try:
                for s, lab in zip(job["segments"], diarize(audio, job["segments"], n)):
                    s["speaker"] = lab
                job["rev"] += 1
            except Exception as e:
                traceback.print_exc()
                job["warning"] = f"講者分辨失敗，只輸出逐字稿：{e}"
        job["status"] = "done"
    except Exception as e:
        job["status"], job["error"] = "error", str(e)
        traceback.print_exc()
    finally:
        try:
            os.remove(job["path"])
        except Exception:
            pass


def mock_run(job):  # 本機測試用
    lines = ["大家好，今天的工作坊主要想聽聽各位對公園改善的想法。", "第一個議題是高架橋下的照明，晚上走起來會不會怕？",
             "我覺得燈太暗了，而且自行車騎很快。", "那座椅呢？很多長輩說想要有遮蔭的休息區。", "好，我們把這些意見記下來，下一輪再討論配置。"]
    job["status"], job["duration"] = "transcribing", 44.0
    t = 0.0
    for i in range(10):
        time.sleep(0.3)
        d = 3.4 + (i % 3)
        job["segments"].append({"start": round(t, 2), "end": round(t + d, 2), "text": lines[i % 5]})
        t += d
        job["progress"] = min(t / 44.0, 1)
    if job["speakers"] != "off":
        job["status"] = "diarizing"
        time.sleep(1)
        for i, s in enumerate(job["segments"]):
            s["speaker"] = [0, 0, 1, 2, 0][i % 5]
        job["rev"] += 1
    job["progress"], job["status"] = 1.0, "done"


def worker():
    while True:
        jid = work_q.get()
        if jid in jobs and not jobs[jid].get("cancel"):
            run_job(jid)


threading.Thread(target=worker, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index():
    return "<meta charset=utf-8><body style='background:#0d0e11;color:#eee;font:16px sans-serif;padding:40px'>逐字 Verbatim 伺服器運作中，請回到逐字網站使用。"


@app.get("/api/system")
def system():
    s = device_info()
    return {"app": "verbatim", "device": s["device"], "gpu": s["gpu"], "note": s["note"], "mock": MOCK,
            "loaded": next((k for k, v in MODELS.items() if v == _model["name"]), None)}


# 分段上傳：Cloudflare 免費通道單次請求上限 100 MB，所以前端把檔案切成 40 MB 一段送
@app.post("/api/upload/{uid}")
async def upload_part(uid: str, part: int = Form(...), chunk: UploadFile = File(...)):
    if not uid.isalnum() or len(uid) > 32:
        raise HTTPException(400, "bad id")
    path = os.path.join(UPLOAD_DIR, uid + ".part")
    with open(path, "ab" if part > 0 else "wb") as out:
        while data := await chunk.read(1 << 20):
            out.write(data)
    return {"ok": True, "size": os.path.getsize(path)}


@app.post("/api/transcribe")
def transcribe(uid: str = Form(...), filename: str = Form(""), language: str = Form(""), model: str = Form("large-v2"),
               trad: str = Form("1"), hint: str = Form(""), speakers: str = Form("off")):
    src = os.path.join(UPLOAD_DIR, uid + ".part")
    if not uid.isalnum() or not os.path.exists(src):
        raise HTTPException(400, "找不到上傳的檔案")
    jid = uuid.uuid4().hex[:10]
    path = os.path.join(UPLOAD_DIR, jid + os.path.splitext(filename)[1][:8])
    os.replace(src, path)
    jobs[jid] = {"id": jid, "filename": filename, "path": path, "language": language.strip(), "model": model,
                 "trad": trad == "1", "hint": hint.strip()[:400], "speakers": speakers, "status": "queued",
                 "progress": 0.0, "duration": None, "segments": [], "rev": 0, "error": None, "warning": None,
                 "created": time.time()}
    work_q.put(jid)
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str, since: int = 0, rev: int = 0):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "找不到這個工作（Colab 可能重新啟動過）")
    ahead = sum(1 for j in jobs.values() if j["status"] == "queued" and j["created"] < job["created"])
    full = job["rev"] != rev  # 講者標好後整份重送
    return {k: job[k] for k in ("id", "status", "progress", "duration", "language", "error", "warning", "rev")} | {
        "segments": job["segments"] if full else job["segments"][since:], "full": full,
        "model_loaded": _model["name"] == MODELS.get(job["model"]), "ahead": ahead}


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    if jid in jobs:
        jobs[jid]["cancel"] = True
        if jobs[jid]["status"] == "queued":
            jobs[jid]["status"] = "cancelled"
    return {"ok": True}


def preload(key="large-v2"):
    """啟動後就在背景先載入常用模型，第一次轉錄不用等"""
    if MOCK:
        return
    def _go():
        try:
            device_info(); get_model(key)
        except Exception:
            traceback.print_exc()
    threading.Thread(target=_go, daemon=True).start()


def start(port=8000):
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(80):
        if server.started:
            break
        time.sleep(0.1)
    return server


if __name__ == "__main__":
    import uvicorn
    if os.environ.get("VERBATIM_PRELOAD"):
        preload(os.environ["VERBATIM_PRELOAD"])
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
