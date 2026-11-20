"""逐字 Verbatim — 在 Colab GPU 上執行的錄音轉逐字稿引擎

執行環境由 engine/requirements.lock 完整鎖定（Python 3.11 獨立虛擬環境），不使用 Colab 內建套件。
- 轉錄：faster-whisper（GPU，CUDA 12 函式庫來自 pip 的 nvidia-*-cu12）
- 講者分辨：3D-Speaker CAM++ 聲紋（sherpa-onnx，CPU），在轉錄進行中逐句計算
- 解碼：imageio-ffmpeg 內附的 ffmpeg
"""
import os, sys, uuid, time, queue, threading, traceback, gc, platform, collections

VERSION = "2026.11.20-5"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("VERBATIM_DATA", APP_DIR)
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(UPLOAD_DIR, exist_ok=True)
MOCK = os.environ.get("VERBATIM_MOCK") == "1"
FORCE_CPU = os.environ.get("VERBATIM_CPU") == "1"
sys.path.insert(0, APP_DIR)


def _preload_cuda12():
    """CTranslate2 以檔名尋找 libcublas.so.12／libcudnn*.so.9。先把虛擬環境裡 nvidia-*-cu12 的函式庫載進行程，
    就不必依賴 Colab 系統的 CUDA 版本或 LD_LIBRARY_PATH。"""
    import ctypes, glob, site
    files = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for pat in ("nvidia/cuda_runtime/lib/*.so.12", "nvidia/cuda_nvrtc/lib/*.so.12",
                    "nvidia/cublas/lib/libcublas*.so.12", "nvidia/cudnn/lib/libcudnn*.so.9"):
            files += glob.glob(os.path.join(sp, pat))
    left = sorted(set(files), key=lambda f: ("Lt" not in f, "cudnn.so" in f, f))
    for _ in range(5):  # 有相依順序，載不進的下一輪再試
        left = [f for f in left if not _try_load(ctypes, f)]
        if not left:
            break


def _try_load(ctypes, f):
    try:
        ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


if not MOCK:
    _preload_cuda12()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

MODELS = {"large-v2": "large-v2", "large-v3": "large-v3", "turbo": "large-v3-turbo",
          "medium": "medium", "small": "small"}
ZH_PROMPT = "以下是一段繁體中文的逐字稿，包含適當的標點符號。"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs: dict = {}
work_q: "queue.Queue[str]" = queue.Queue()
_model = {"name": None, "obj": None}
_state = {"device": None, "gpu": None, "note": "", "ready": False}
_errors = collections.deque(maxlen=5)
_cc = None
_emb = {"obj": None, "err": None}


# ---------- 環境 ----------
def device_info():
    if _state["device"] is None:
        n = 0
        if not FORCE_CPU:
            try:
                import ctranslate2
                n = ctranslate2.get_cuda_device_count()
            except Exception:
                n = 0
        _state["device"] = "cuda" if n > 0 else "cpu"
        if n > 0:
            try:
                import subprocess
                _state["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                               capture_output=True, text=True).stdout.strip().split("\n")[0] or "GPU"
            except Exception:
                _state["gpu"] = "GPU"
    return _state


_model_lock = threading.Lock()


def get_model(key):
    with _model_lock:  # 背景預先載入和轉錄工作可能同時要模型，只載一次
        return _get_model(key)


def _get_model(key):
    name = MODELS.get(key, "large-v2")
    if _model["name"] == name:
        return _model["obj"]
    _model["obj"] = _model["name"] = None  # 一次只放一個模型，避免 GPU 記憶體不足
    gc.collect()
    from faster_whisper import WhisperModel
    import numpy as np
    dev = device_info()["device"]
    try:
        m = WhisperModel(name, device=dev, compute_type="float16" if dev == "cuda" else "int8",
                         download_root=os.path.join(MODEL_DIR, "whisper"))
        if dev == "cuda":  # 試跑一秒空白音訊，確認 GPU 函式庫真的可用
            list(m.transcribe(np.zeros(16000, np.float32), language="en")[0])
    except Exception as e:
        if dev != "cuda":
            raise
        log_error("GPU 試跑失敗", e)
        _state["device"], _state["note"] = "cpu", f"GPU 無法使用，暫時改用 CPU（會很慢）。原因：{short(e)}"
        m = WhisperModel(name, device="cpu", compute_type="int8", download_root=os.path.join(MODEL_DIR, "whisper"))
    _model.update(name=name, obj=m)
    return m


_emb_lock = threading.Lock()


def get_embedder():
    with _emb_lock:
        return _get_embedder()


def _get_embedder():
    if _emb["obj"] is None and _emb["err"] is None:
        try:
            import diarize
            _emb["obj"] = diarize.Embedder(os.path.join(MODEL_DIR, "speaker"))
        except Exception as e:
            _emb["err"] = short(e)
            log_error("講者模型載入失敗", e)
    return _emb["obj"]


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def load_audio(path, sr=16000):
    import subprocess, numpy as np
    r = subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1",
                        "-acodec", "pcm_s16le", "-ar", str(sr), "-"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise UserError("讀不到這個檔案的聲音，可能是格式不支援或檔案損毀。可以先轉成 mp3 或 m4a 再試。")
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0


_YI = None


def fix_text(text):
    """Whisper 繁體輸出偶爾把「一」寫成「壹」；金額寫法（壹佰、壹萬元…）保留"""
    global _YI
    if _YI is None:
        import re
        _YI = re.compile(r"壹(?![拾佰仟萬億圓元角])")
    return _YI.sub("一", text)


def to_trad(text):
    global _cc
    if _cc is None:
        try:
            from opencc import OpenCC
            _cc = OpenCC("s2twp")
        except Exception:
            _cc = False
    return _cc.convert(text) if _cc else text


class UserError(Exception):
    """可以直接顯示給使用者看的錯誤"""


def short(e):
    return (str(e) or e.__class__.__name__).strip().split("\n")[0][:300]


def log_error(where, e):
    traceback.print_exc()
    _errors.append({"time": time.strftime("%H:%M:%S"), "where": where, "error": short(e)})


def friendly(e):
    s = str(e)
    if isinstance(e, UserError):
        return s
    if "out of memory" in s.lower():
        return "GPU 記憶體不足。請改用 Large v3 Turbo 或 Medium 模型再試。"
    if any(k in s for k in ("huggingface", "HTTPSConnection", "Connection", "timed out", "LocalEntryNotFound")):
        return "模型下載失敗（網路問題），請稍等一下再試一次。"
    return "轉錄時發生錯誤：" + short(e)


# ---------- 轉錄工作 ----------
def run_job(jid):
    job = jobs[jid]
    try:
        if MOCK:
            return mock_run(job)
        job["status"] = "loading"
        audio = load_audio(job["path"])
        model = get_model(job["model"])
        n_spk = 0 if job["speakers"] in ("off", "auto") else int(job["speakers"])
        want_spk = job["speakers"] != "off"
        embs, emb_q = [], queue.Queue()
        if want_spk:  # 聲紋在另一條執行緒逐句計算，和 GPU 轉錄同時進行
            def emb_worker():
                emb = get_embedder()
                while True:
                    s = emb_q.get()
                    if s is None:
                        break
                    try:
                        embs.append(emb.embed(audio, s["start"], s["end"]) if emb else None)
                    except Exception as e:
                        log_error("聲紋計算", e)
                        embs.append(None)
            emb_t = threading.Thread(target=emb_worker, daemon=True)
            emb_t.start()

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
                break
            text = s.text.strip()
            if trad:
                text = to_trad(text)
            if info.language == "zh":
                text = fix_text(text)
            if text:
                seg = {"start": round(s.start, 2), "end": round(s.end, 2), "text": text}
                job["segments"].append(seg)
                if want_spk:
                    emb_q.put(seg)
            job["progress"] = min(s.end / info.duration, 1.0) if info.duration else 0

        if want_spk:
            emb_q.put(None)
            if job["status"] != "cancelled":
                job["status"] = "diarizing"
            emb_t.join()
            if job["status"] == "diarizing":
                if _emb["obj"] is None:
                    job["warning"] = f"講者分辨無法使用，只輸出逐字稿。（{_emb['err'] or '模型未載入'}）"
                else:
                    import diarize
                    for s, lab in zip(job["segments"], diarize.cluster(embs, n_spk)):
                        s["speaker"] = lab
                    job["rev"] += 1
        if job["status"] != "cancelled":
            job["progress"], job["status"] = 1.0, "done"
    except Exception as e:
        log_error("轉錄", e)
        job["status"], job["error"] = "error", friendly(e)
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


def preload(key="large-v2"):
    """啟動後在背景先載入常用模型（含 GPU 試跑）和講者模型"""
    def _go():
        try:
            if not MOCK:
                device_info()
                get_model(key)
                get_embedder()
        except Exception as e:
            log_error("預先載入", e)
            _state["note"] = _state["note"] or f"模型預先載入失敗：{short(e)}"
        finally:
            _state["ready"] = True
    threading.Thread(target=_go, daemon=True).start()


# ---------- API ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return ("<meta charset=utf-8><body style='background:#0d0e11;color:#eee;font:16px sans-serif;padding:40px'>"
            "逐字 Verbatim 引擎運作中，請回到逐字網站使用。")


@app.get("/api/system")
def system():
    s = device_info()
    return {"app": "verbatim", "version": VERSION, "device": s["device"], "gpu": s["gpu"], "note": s["note"],
            "ready": s["ready"] or MOCK, "mock": MOCK,
            "loaded": next((k for k, v in MODELS.items() if v == _model["name"]), None)}


@app.get("/api/diag")
def diag():
    """給「複製錯誤資訊」用：版本、環境、最近的錯誤"""
    from importlib import metadata
    pk = {}
    for p in ("faster-whisper", "ctranslate2", "sherpa-onnx", "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "numpy"):
        try:
            pk[p] = metadata.version(p)
        except Exception:
            pk[p] = None
    s = device_info()
    return {"version": VERSION, "python": platform.python_version(), "device": s["device"], "gpu": s["gpu"],
            "note": s["note"], "model": _model["name"], "speaker_model": "ok" if _emb["obj"] else _emb["err"],
            "packages": pk, "errors": list(_errors)}


# 分段上傳：Cloudflare 免費通道單次請求上限 100 MB，前端把檔案切成 40 MB 一段送
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
        raise HTTPException(400, "找不到上傳的檔案，請重新上傳")
    if speakers not in ("off", "auto") and not speakers.isdigit():
        speakers = "auto"
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
        raise HTTPException(404, "找不到這個工作（Colab 可能重新啟動過），請重新上傳")
    ahead = sum(1 for j in jobs.values() if j["status"] == "queued" and j["created"] < job["created"])
    full = job["rev"] != rev
    return {k: job[k] for k in ("id", "status", "progress", "duration", "language", "error", "warning", "rev")} | {
        "segments": job["segments"] if full else job["segments"][since:], "full": full, "ahead": ahead}


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    if jid in jobs:
        jobs[jid]["cancel"] = True
        if jobs[jid]["status"] == "queued":
            jobs[jid]["status"] = "cancelled"
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    if os.environ.get("VERBATIM_PRELOAD"):
        preload(os.environ["VERBATIM_PRELOAD"])
    else:
        _state["ready"] = True
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="warning")
