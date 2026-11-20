# 用 sherpa-onnx 中文 TTS 做一段三人輪流說話的工作坊錄音，寫出 mp3 和正確答案（自動測試用）
import sherpa_onnx, numpy as np, wave, json, sys, os, subprocess, tarfile, urllib.request
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
os.makedirs(OUT, exist_ok=True)
d = os.path.join(OUT, "vits-zh-aishell3")
if not os.path.exists(d):
    tgz = os.path.join(OUT, "tts.tar.bz2")
    urllib.request.urlretrieve("https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-aishell3.tar.bz2", tgz)
    with tarfile.open(tgz) as t:
        t.extractall(OUT)
tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
    vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=f"{d}/vits-aishell3.int8.onnx", lexicon=f"{d}/lexicon.txt", tokens=f"{d}/tokens.txt"),
    num_threads=2), rule_fsts=f"{d}/phone.fst,{d}/date.fst,{d}/number.fst"))
lines = [(10, "大家好，今天的工作坊主要想聽聽各位對公園改善的想法。"),
         (66, "我覺得高架橋下的燈太暗了，晚上走路會怕。"),
         (10, "了解，那照明是第一個議題。還有其他意見嗎？"),
         (120, "很多長輩希望多一點有遮蔭的座椅，可以坐下來休息。"),
         (66, "另外自行車騎得很快，最好把步道和車道分開。"),
         (10, "好，我們把這些意見記下來，下一輪再討論配置。")]
out, truth, t, sr = [], [], 0.0, None
for sid, text in lines:
    a = tts.generate(text, sid=sid, speed=1.0); sr = a.sample_rate
    x = np.array(a.samples, dtype=np.float32)
    truth.append({"speaker": sid, "start": round(t, 2), "end": round(t + len(x)/sr, 2), "text": text})
    out += [x, np.zeros(int(sr*0.6), np.float32)]; t += len(x)/sr + 0.6
y = np.concatenate(out)
pcm = (np.clip(y, -1, 1)*32767).astype(np.int16)
with wave.open(os.path.join(OUT, "dialog_raw.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
subprocess.run([FF,"-v","error","-y","-i",os.path.join(OUT, "dialog_raw.wav"),"-ar","16000","-ac","1",os.path.join(OUT, "dialog.wav")], check=True)
subprocess.run([FF,"-v","error","-y","-i",os.path.join(OUT, "dialog_raw.wav"),"-b:a","64k",os.path.join(OUT, "dialog.mp3")], check=True)
json.dump(truth, open(os.path.join(OUT, "dialog_truth.json"),"w"), ensure_ascii=False, indent=1)
print("sr", sr, "dur", round(t,1)); [print(x) for x in truth]
