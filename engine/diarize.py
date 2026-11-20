"""講者分辨：每一句逐字稿取一次聲紋（3D-Speaker CAM++，中英文模型，透過 sherpa-onnx 執行，不需要 PyTorch），
最後依聲紋相似度分群。聲紋在轉錄進行中逐句計算，轉錄結束時只剩分群，幾乎不用等。
"""
import os, urllib.request
import numpy as np

EMB_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
           "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
SR = 16000
AUTO_THRESHOLD = 0.40   # 餘弦距離；小於這個值的兩群視為同一人（自動判斷人數時使用）。
# 寧可多分：多分出來的講者在網站上改成同一個名字就能合併；兩人被合成一人則無法事後拆開。
MIN_SEC, MAX_SEC = 0.6, 10.0


def ensure_model(root):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "campplus_zh_en.onnx")
    if not os.path.exists(path):
        tmp = path + ".part"
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(EMB_URL, tmp)
                os.replace(tmp, path)
                break
            except Exception:
                if attempt == 2:
                    raise
    return path


class Embedder:
    def __init__(self, model_root, threads=2):
        import sherpa_onnx
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=ensure_model(model_root), num_threads=threads))

    def embed(self, audio, start, end):
        """回傳正規化後的聲紋；太短（聲紋不可靠）回傳 None。長句只取中間 10 秒。"""
        if end - start < MIN_SEC:
            return None
        if end - start > MAX_SEC:
            mid = (start + end) / 2
            start, end = mid - MAX_SEC / 2, mid + MAX_SEC / 2
        clip = audio[int(start * SR):int(end * SR)]
        if len(clip) < MIN_SEC * SR:
            return None
        s = self.ex.create_stream()
        s.accept_waveform(SR, np.ascontiguousarray(clip, dtype=np.float32))
        s.input_finished()
        e = np.asarray(self.ex.compute(s), dtype=np.float32)
        n = np.linalg.norm(e)
        return e / n if n > 0 else None


def cluster(embs, num_speakers=0):
    """embs：每句一個聲紋或 None。回傳每句的講者編號（依出場順序 0、1、2…）。"""
    from sklearn.cluster import AgglomerativeClustering
    idx = [i for i, e in enumerate(embs) if e is not None]
    if not embs:
        return []
    if len(idx) < 2:
        return [0] * len(embs)
    X = np.stack([embs[i] for i in idx])
    if num_speakers and num_speakers > 1:
        cl = AgglomerativeClustering(n_clusters=min(num_speakers, len(idx)), metric="cosine", linkage="average")
    else:
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=AUTO_THRESHOLD, metric="cosine",
                                     linkage="average")
    raw = cl.fit_predict(X)
    order, labels = {}, [None] * len(embs)
    for i, r in zip(idx, raw):
        labels[i] = order.setdefault(int(r), len(order))
    for i in range(len(labels)):  # 太短的句子：跟時間上最近的句子
        if labels[i] is None:
            labels[i] = labels[min(idx, key=lambda j: abs(j - i))]
    return labels
