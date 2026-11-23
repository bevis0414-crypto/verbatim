import onnx, os, sys, traceback
from huggingface_hub import hf_hub_download, list_repo_files
repo = "markno5/Breeze-ASR-25-ONNX"
print("FILES:", list_repo_files(repo), flush=True)
def ext_locs(g, acc):
    for t in g.initializer:
        if t.data_location == onnx.TensorProto.EXTERNAL:
            d = {e.key: e.value for e in t.external_data}; acc[d.get("location")] = acc.get(d.get("location"), 0) + 1
    for node in g.node:
        for a in node.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                ext_locs(a.g, acc)
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_location == onnx.TensorProto.EXTERNAL:
                d = {e.key: e.value for e in a.t.external_data}; k = "(Constant)" + str(d.get("location")); acc[k] = acc.get(k, 0) + 1
for f in ["onnx/encoder_model_q4f16.onnx", "onnx/decoder_model_merged_q4f16.onnx", "onnx/encoder_model_q4.onnx", "onnx/decoder_model_merged_q4.onnx"]:
    try:
        p = hf_hub_download(repo, f)
        m = onnx.load(p, load_external_data=False)
        acc = {}; ext_locs(m.graph, acc)
        print(f"\n== {f} size={os.path.getsize(p)} opset={[(o.domain, o.version) for o in m.opset_import]}", flush=True)
        print("   external data locations (tensor count):", acc, flush=True)
        names = [n.name for n in m.graph.node if "Concat_2" in n.name][:5]
        print("   sample Concat_2 nodes:", names, flush=True)
    except Exception:
        traceback.print_exc()
