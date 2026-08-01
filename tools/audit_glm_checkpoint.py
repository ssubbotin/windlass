#!/usr/bin/env python3
"""audit_glm_checkpoint.py — classify GLM-5.2 checkpoint tensors into routed
experts (streamed from NVMe), the excluded MTP head (layer 78), and resident
(VRAM-resident) weight, and total the bytes.

Sizes are MEASURED, not inferred: the safetensors index has no shapes, so
this fetches only the safetensors shard HEADERS via HTTP Range requests (a
few KB per shard) and reads dtype+shape from there. No weight data is ever
downloaded — these repos are hundreds of GB. Also downloads config.json and
model.safetensors.index.json (small JSON files).

Usage:
    python3 audit_glm_checkpoint.py --repo amd/GLM-5.2-MXFP4
"""
import argparse, hashlib, json, os, re, struct, collections, sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download, hf_hub_url

# safetensors dtype -> bytes per stored element
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8,
               "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1, "BOOL": 1}

EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")

HEADER_FETCH_WORKERS = 16
CACHE_DIR = "/tmp/audit_glm_checkpoint_cache"


def fetch_shard_header(session, url, timeout=300):
    """Range-fetch only the safetensors header: 8-byte LE length prefix
    followed by that many bytes of header JSON. Never touches tensor data."""
    n = struct.unpack("<Q", session.get(
        url, headers={"Range": "bytes=0-7"}, timeout=timeout).content)[0]
    body = session.get(
        url, headers={"Range": f"bytes=8-{8 + n - 1}"}, timeout=timeout).content
    return json.loads(body.decode())


def fetch_shard_header_cached(shard, url):
    """Same as fetch_shard_header, but memoized to a local scratch file so a
    re-run (e.g. while iterating on classification/reporting) is instant.
    The cache is disposable scratch, not part of the repo."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.sha1(shard.encode()).hexdigest()
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    session = requests.Session()
    header = fetch_shard_header(session, url)
    with open(path, "w") as f:
        json.dump(header, f)
    return header


def tensor_bytes(entry):
    n = DTYPE_BYTES.get(entry["dtype"], 0)
    for d in entry["shape"]:
        n *= d
    return n


def component_of(name):
    if ".self_attn.indexer." in name:
        return "indexer"
    if ".self_attn." in name:
        return "attention"
    if ".shared_experts." in name:
        return "shared_expert"
    if ".mlp.gate." in name:
        return "router"
    if ".mlp." in name:
        return "dense_mlp"
    if "embed_tokens" in name:
        return "embeddings"
    if "lm_head" in name:
        return "lm_head"
    return "norms/other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="amd/GLM-5.2-MXFP4")
    ap.add_argument("--mtp-layer", type=int, default=78,
                     help="layer index to exclude entirely (MTP head)")
    args = ap.parse_args()

    cfg = json.load(open(hf_hub_download(args.repo, "config.json")))
    idx = json.load(open(hf_hub_download(args.repo, "model.safetensors.index.json")))
    weight_map = idx["weight_map"]

    print(f"config: layers={cfg['num_hidden_layers']} hidden={cfg['hidden_size']} "
          f"routed_experts={cfg['n_routed_experts']} per_tok={cfg['num_experts_per_tok']} "
          f"moe_inter={cfg['moe_intermediate_size']} dense_first={cfg['first_k_dense_replace']}")

    mtp_prefix = f"model.layers.{args.mtp_layer}."
    expert_tensor_count = 0
    resident_tensor_count = 0
    mtp_tensor_count = 0
    for name in weight_map:
        if name.startswith(mtp_prefix):
            mtp_tensor_count += 1
        elif EXPERT_RE.search(name):
            expert_tensor_count += 1
        else:
            resident_tensor_count += 1

    print(f"\nrouted-expert tensors total: {expert_tensor_count}")
    print(f"MTP-layer ({args.mtp_layer}) tensors: {mtp_tensor_count}")
    print(f"resident (non-expert, non-MTP) tensors: {resident_tensor_count}")

    # --- Config-derived arithmetic, kept as an independent cross-check ---
    H = cfg["hidden_size"]
    MI = cfg["moe_intermediate_size"]
    NE = cfg["n_routed_experts"]
    L = cfg["num_hidden_layers"]
    DENSE = cfg["first_k_dense_replace"]
    moe_layers = L - DENSE

    # MXFP4: 32 values -> 16 payload bytes + 1 e8m0 scale byte
    params_per_expert = 3 * H * MI
    bytes_per_expert_computed = params_per_expert // 32 * 17
    computed_expert_bytes = bytes_per_expert_computed * NE * moe_layers

    print(f"\nparams per routed expert (computed) : {params_per_expert:,}")
    print(f"bytes  per routed expert (computed) : {bytes_per_expert_computed:,} "
          f"({bytes_per_expert_computed/1e6:.1f} MB)")
    print(f"routed experts                      : {NE * moe_layers:,} "
          f"({moe_layers} MoE layers x {NE})")
    print(f"total expert bytes (computed)       : {computed_expert_bytes/1e9:.1f} GB")

    # --- Measured pass: range-fetch every shard header, classify every tensor ---
    # Shard headers on this checkpoint carry tens of thousands of tensor
    # entries each and there are ~90 shards; fetching them serially blew a
    # 900s budget with zero output. Fetch concurrently (network-bound, so
    # threads parallelize cleanly) and cache each parsed header to local
    # scratch so a re-run while iterating on the classification/reporting
    # logic below is instant.
    shards = sorted(set(weight_map.values()))
    print(f"\nfetching {len(shards)} shard headers (metadata only, no weight data), "
          f"{HEADER_FETCH_WORKERS} workers...", file=sys.stderr)

    expert_bytes = 0
    mtp_bytes = 0
    resident_by_dtype = collections.Counter()
    resident_by_component = collections.Counter()
    resident_total = 0

    done = 0
    with ThreadPoolExecutor(max_workers=HEADER_FETCH_WORKERS) as pool:
        futures = {
            pool.submit(fetch_shard_header_cached, shard, hf_hub_url(args.repo, shard)): shard
            for shard in shards
        }
        for fut in as_completed(futures):
            shard = futures[fut]
            header = fut.result()
            for name, entry in header.items():
                if name == "__metadata__" or "dtype" not in entry:
                    continue
                b = tensor_bytes(entry)
                if name.startswith(mtp_prefix):
                    mtp_bytes += b
                    continue
                if EXPERT_RE.search(name):
                    expert_bytes += b
                    continue
                resident_total += b
                resident_by_dtype[entry["dtype"]] += b
                resident_by_component[component_of(name)] += b
            done += 1
            if done % 10 == 0 or done == len(shards):
                print(f"  {done}/{len(shards)} shards", file=sys.stderr)

    print(f"\nMEASURED routed-expert bytes        : {expert_bytes:,} ({expert_bytes/1e9:.1f} GB)")
    print(f"MEASURED MTP-layer ({args.mtp_layer}) bytes    : {mtp_bytes:,} ({mtp_bytes/1e9:.1f} GB)")
    print(f"MEASURED RESIDENT bytes (excl. MTP) : {resident_total:,} ({resident_total/1e9:.1f} GB)")

    divergence = expert_bytes - computed_expert_bytes
    print(f"\nmeasured vs computed expert bytes divergence: {divergence:,} "
          f"({divergence/1e9:+.3f} GB, {100*divergence/computed_expert_bytes:+.2f}%)")

    print("\nMEASURED resident bytes by dtype:")
    for d, b in resident_by_dtype.most_common():
        print(f"  {d:10s} {b:>16,} ({b/1e9:6.1f} GB)")

    print("\nMEASURED resident bytes by component:")
    for g, b in resident_by_component.most_common():
        print(f"  {g:14s} {b:>16,} ({b/1e9:6.1f} GB)")

    bf16_f16 = resident_by_dtype.get("BF16", 0) + resident_by_dtype.get("F16", 0)
    fp8_savings = bf16_f16 // 2
    resident_after_fp8 = resident_total - fp8_savings
    print(f"\nBF16/F16 resident bytes             : {bf16_f16:,} ({bf16_f16/1e9:.1f} GB)")
    print(f"savings if BF16/F16 resident -> FP8  : {fp8_savings:,} ({fp8_savings/1e9:.1f} GB)")
    print(f"resident total AFTER FP8 conversion  : {resident_after_fp8:,} "
          f"({resident_after_fp8/1e9:.1f} GB)")

    gate_gb = 45.0
    resident_gb = resident_total / 1e9
    verdict = "PASS" if resident_gb <= gate_gb else "FAIL"
    print(f"\ngate check: measured resident {resident_gb:.1f} GB vs {gate_gb:.0f} GB budget "
          f"-> {verdict}")


if __name__ == "__main__":
    main()
