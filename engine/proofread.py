"""AI 校對：用開源語言模型（Qwen3，GGUF Q4_K_M，llama.cpp）修正錯字、標點、分段。
Colab／Apple 晶片 Mac 用 GPU 跑 8B；Windows 本機用 CPU 跑 4B。

設計原則：
- 只校對，不改寫：逐句編號送進模型、逐句取回，時間戳與講者不會跑掉。
- 改太多就不採用：校對後和原句差異過大（可能是模型自己發揮）時保留原句。
- 第一次使用時才安裝 llama-cpp-python（CUDA 12 預先編譯版）和下載模型，平常不佔啟動時間。
"""
import os, re, subprocess, sys, difflib, shutil

import platform as _pf

_REL = "https://github.com/abetlen/llama-cpp-python/releases/download"
# 各平台的 llama-cpp-python 0.3.19 預先編譯版（同一版 llama.cpp）
WHEELS = {
    "linux": f"{_REL}/v0.3.19-cu124/llama_cpp_python-0.3.19-cp311-cp311-linux_x86_64.whl",   # Colab：CUDA 12
    "win32": f"{_REL}/v0.3.19/llama_cpp_python-0.3.19-cp311-cp311-win_amd64.whl",           # Windows：CPU
    "darwin": f"{_REL}/v0.3.19-metal/llama_cpp_python-0.3.19-cp311-cp311-macosx_11_0_arm64.whl",  # Apple 晶片：Metal
}
LLM_WHEEL = WHEELS["linux"]
COMMON_EXTRAS = ["diskcache==5.6.3", "jinja2==3.1.6", "MarkupSafe==3.0.3"]
# 有 GPU 加速的平台用 8B；只能用 CPU 的平台改用 4B，速度約快一倍
_GPU_MODEL = ("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf")
_CPU_MODEL = ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf")


def use_gpu(whisper_device):
    if sys.platform == "darwin":
        return _pf.machine() == "arm64"
    return sys.platform == "linux" and whisper_device == "cuda"


def model_for(gpu):
    repo, fname = _GPU_MODEL if gpu else _CPU_MODEL
    return os.environ.get("VERBATIM_LLM_REPO", repo), os.environ.get("VERBATIM_LLM_FILE", fname)


def wheel_and_extras():
    if sys.platform == "linux":
        return WHEELS["linux"], COMMON_EXTRAS + ["nvidia-cuda-runtime-cu12==12.9.79"]
    if sys.platform == "darwin" and _pf.machine() != "arm64":
        raise LLMUnavailable("Intel 晶片的 Mac 目前不支援 AI 校對")
    return WHEELS.get(sys.platform), COMMON_EXTRAS

SYSTEM = (
    "你是專業的繁體中文（臺灣）逐字稿校對員。輸入是語音辨識產生的逐字稿，每行開頭有編號。\n"
    "請只做以下修正：\n"
    "1. 語音辨識造成的錯字、同音字、近音字（依上下文判斷正確的詞）。\n"
    "2. 標點符號：補上或修正為全形標點（，。？！、：「」）。\n"
    "3. 分段：如果某行開始了新的話題或段落，在那一行的編號前加上 ¶。\n"
    "嚴格禁止：改寫句子、刪減內容、合併或拆開行、摘要、加入原文沒有的資訊、把口語改成書面語。"
    "口語詞（例如「然後」「就是」「對」）要保留。不確定的地方保持原樣。\n"
    "輸出格式：每行「[編號] 校對後的句子」，行數和編號必須和輸入完全相同，不要輸出其他文字。"
)


class LLMUnavailable(Exception):
    pass


def ensure_installed(status=lambda s: None):
    try:
        import llama_cpp  # noqa: F401
        return
    except Exception:
        pass
    if os.environ.get("VERBATIM_LLM_NO_INSTALL") == "1":
        raise LLMUnavailable("校對模組未安裝")
    status("installing")
    wheel, extras = wheel_and_extras()
    uv = os.environ.get("VERBATIM_UV") or shutil.which("uv")
    cmd = ([uv, "pip", "install", "-q", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install", "-q"])
    r = subprocess.run(cmd + [wheel] + extras, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise LLMUnavailable("校對模組安裝失敗：" + (r.stderr or r.stdout).strip()[-300:])


def load(model_dir, use_gpu, status=lambda s: None, preload_cuda=None):
    ensure_installed(status)
    if preload_cuda:
        preload_cuda()  # 讓新裝的 libcudart.so.12 也先載進行程
    status("downloading")
    from huggingface_hub import hf_hub_download
    repo, fname = model_for(use_gpu)
    path = hf_hub_download(repo, fname, cache_dir=os.path.join(model_dir, "llm"))
    status("loading")
    from llama_cpp import Llama
    return Llama(model_path=path, n_gpu_layers=-1 if use_gpu else 0, n_ctx=8192, n_batch=512,
                 verbose=False, seed=0)


def chunks(texts, max_chars=900, max_lines=30):
    out, cur, n = [], [], 0
    for i, t in enumerate(texts):
        if cur and (n + len(t) > max_chars or len(cur) >= max_lines):
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += len(t)
    if cur:
        out.append(cur)
    return out


LINE = re.compile(r"^\s*(¶)?\s*[\[【(（]?\s*(\d+)\s*[\]】)）]\s*(¶)?\s*(.*)$")


def parse(raw, ids):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    got = {}
    for line in raw.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        k = int(m.group(2))
        if k in ids and k not in got:
            got[k] = (m.group(4).strip(), bool(m.group(1) or m.group(3)))
    return got


def accept(old, new):
    """校對結果是否可信：不能空、不能長太多、和原句要夠像"""
    if not new:
        return False
    if len(new) > len(old) * 1.6 + 8 or len(new) < len(old) * 0.6 - 4:
        return False
    strip = lambda s: re.sub(r"[\s，。？！、：；「」『』,.?!:;\"'（）()…—-]", "", s)
    a, b = strip(old), strip(new)
    if not a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.55


def proofread(llm, texts, hint="", progress=lambda done: None, cancelled=lambda: False, post=lambda s: s,
              pause_wait=lambda: None):
    """回傳 [(校對後文字, 是否新段落, 是否有修改)]，長度與 texts 相同"""
    result = [(t, False, False) for t in texts]
    groups = chunks(texts)
    done = 0
    for gi, g in enumerate(groups):
        pause_wait()
        if cancelled():
            break
        ctx = ""
        if gi:  # 前文兩行當參考，幫助判斷上下文
            prev = groups[gi - 1][-2:]
            ctx = "前文（僅供參考，不要輸出）：\n" + "\n".join(result[i][0] for i in prev) + "\n\n"
        ids = list(range(1, len(g) + 1))
        body = "\n".join(f"[{k}] {texts[i]}" for k, i in zip(ids, g))
        user = (f"專有名詞參考：{hint}\n\n" if hint else "") + ctx + "請校對以下逐字稿：\n" + body + "\n/no_think"
        got = {}
        for attempt in range(2):
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                temperature=0.0, repeat_penalty=1.05, max_tokens=int(sum(len(texts[i]) for i in g) * 2.2) + 300)
            got = parse(out["choices"][0]["message"]["content"], set(ids))
            if len(got) >= len(ids) * 0.8:
                break
        for k, i in zip(ids, g):
            if k not in got:
                continue
            new, para = got[k]
            new = post(new)
            if accept(texts[i], new):
                result[i] = (new, para, new != texts[i])
            else:
                result[i] = (texts[i], para, False)
        done += len(g)
        progress(done)
    return result
