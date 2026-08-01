#!/usr/bin/env python3
"""fetch_glm_tensors.py — download individual tensors from a HF safetensors
repo using HTTP range requests, without pulling whole shards.

Usage:
    python3 fetch_glm_tensors.py --repo amd/GLM-5.2-MXFP4 --out /tmp/glm-probe \
        --tensors model.layers.3.mlp.experts.0.gate_proj.weight,\
model.layers.3.mlp.experts.0.gate_proj.weight_scale
"""
import argparse, json, os, struct, requests
from huggingface_hub import hf_hub_download, hf_hub_url

def shard_header(url, token=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    n = struct.unpack("<Q", requests.get(url, headers={**h, "Range": "bytes=0-7"},
                                         timeout=60).content)[0]
    js = requests.get(url, headers={**h, "Range": f"bytes=8-{8 + n - 1}"},
                      timeout=120).content
    return json.loads(js.decode()), 8 + n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="amd/GLM-5.2-MXFP4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensors", required=True)
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    os.makedirs(args.out, exist_ok=True)

    idx = json.load(open(hf_hub_download(args.repo, "model.safetensors.index.json")))
    wmap = idx["weight_map"]

    headers_cache = {}
    for name in args.tensors.split(","):
        name = name.strip()
        if name not in wmap:
            raise SystemExit(f"tensor not in index: {name}")
        shard = wmap[name]
        url = hf_hub_url(args.repo, shard)
        if shard not in headers_cache:
            headers_cache[shard] = shard_header(url, token)
        hdr, base = headers_cache[shard]
        t = hdr[name]
        beg, end = t["data_offsets"]
        lo, hi = base + beg, base + end - 1
        h = {"Range": f"bytes={lo}-{hi}"}
        if token: h["Authorization"] = f"Bearer {token}"
        data = requests.get(url, headers=h, timeout=600).content
        assert len(data) == end - beg, f"{name}: got {len(data)} want {end-beg}"
        open(os.path.join(args.out, name + ".bin"), "wb").write(data)
        json.dump({"dtype": t["dtype"], "shape": t["shape"]},
                  open(os.path.join(args.out, name + ".json"), "w"))
        print(f"{name}: {t['dtype']} {t['shape']} {len(data)} B")

if __name__ == "__main__":
    main()
