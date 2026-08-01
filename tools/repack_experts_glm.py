#!/usr/bin/env python3
"""repack_experts_glm.py — repack GLM-5.2 MXFP4 routed experts into one
contiguous binary per layer for fast pread streaming.

Per-layer file (packed_experts_glm/layer_{N}.bin):
    expert_0 | expert_1 | ... | expert_255
each expert block:
    gate_proj.weight        U8 [2048, 3072] = 6,291,456 B
    gate_proj.weight_scale  U8 [2048,  192] =   393,216 B
    up_proj.weight          U8 [2048, 3072] = 6,291,456 B
    up_proj.weight_scale    U8 [2048,  192] =   393,216 B
    down_proj.weight        U8 [6144, 1024] = 6,291,456 B
    down_proj.weight_scale  U8 [6144,   64] =   393,216 B
Total per expert: 20,054,016 B (~20.1 MB); per layer ~5.1 GB; 75 layers ~385 GB.

Usage:
    python3 repack_experts_glm.py --model ./glm52-mxfp4 \
        --out ./packed_experts --layers 3-77 [--delete-source] [--verify]

    # Re-check an already-packed layer without re-copying it from source.
    # Works even after --delete-source has removed the shards it doesn't need;
    # degrades to INCOMPLETE (not a failure) for sampled tensors whose shard
    # is gone, rather than reporting the packed data as wrong.
    python3 repack_experts_glm.py --model ./glm52-mxfp4 \
        --out ./packed_experts --layers 3 --verify-only

--verify cannot be combined with --delete-source: within a single run an
earlier layer's deletions can remove a shard a later layer's sampled experts
still need, producing a spurious mismatch/incomplete report for otherwise-
correct data. Verification runs as a separate invocation instead.

--verify-only exit codes (also see --help epilog):
    0 = verified   - all sampled experts matched the checkpoint byte-for-byte
    1 = MISMATCH   - packed bytes differ from the checkpoint (data is wrong)
    3 = INCOMPLETE - could not verify: a sampled expert's source shard is
                     absent under --model (e.g. removed by --delete-source
                     on an earlier run), not a data defect
Exit code 2 is reserved for argparse usage errors (e.g. passing
--verify together with --delete-source) and is never returned for any
verification outcome, so a caller can tell "bad command line" apart from
"could not verify the data" by exit code alone.
"""
import argparse, os, struct, json, sys, shutil
from pathlib import Path

SUBOFFS = [
    ("mlp.experts.{e}.gate_proj.weight",       2048 * 3072, "U8"),
    ("mlp.experts.{e}.gate_proj.weight_scale", 2048 *  192, "U8"),
    ("mlp.experts.{e}.up_proj.weight",         2048 * 3072, "U8"),
    ("mlp.experts.{e}.up_proj.weight_scale",   2048 *  192, "U8"),
    ("mlp.experts.{e}.down_proj.weight",       6144 * 1024, "U8"),
    ("mlp.experts.{e}.down_proj.weight_scale", 6144 *   64, "U8"),
]
NUM_EXPERTS = 256
EXPERT_BYTES = sum(n for _, n, _ in SUBOFFS)     # 20,054,016


def read_shard_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode()), 8 + n


def find_tensor(headers, full_name):
    for path, (h, base) in headers.items():
        if full_name in h:
            t = h[full_name]
            beg, end = t["data_offsets"]
            return path, base + beg, end - beg, t["dtype"]
    return None


def verify_layer_content(layer_idx, out_path, headers, sample_experts=(0, 7, 128, 255),
                          index_weight_map=None):
    """Read each of the six sub-tensors for the sampled experts back out of the
    packed file and compare bytes against the checkpoint via the shard headers
    already loaded, exactly as the standalone Step 4 check does.

    Returns one of three status strings, distinguished on purpose:
      "verified"   - every sampled sub-tensor matched byte-for-byte.
      "mismatch"   - at least one sampled sub-tensor's bytes differ: the
                     packed data is actually wrong.
      "incomplete" - at least one sampled sub-tensor's source shard is not
                     among the loaded headers (e.g. removed by an earlier
                     --delete-source run) and no mismatch was found: the
                     packed data was not, and could not be, checked. This is
                     never reported as success and never as a data defect.
    `index_weight_map` (the checkpoint's weight_map, if available) is used
    only to name the missing shard in the "incomplete" case; it does not
    affect the verdict.
    """
    prefix = f"model.layers.{layer_idx}"
    status = "verified"
    with open(out_path, "rb") as pf:
        for e in sample_experts:
            off = 0
            for tmpl, n, _ in SUBOFFS:
                full = prefix + "." + tmpl.format(e=e)
                info = find_tensor(headers, full)
                if info is None:
                    hint = index_weight_map.get(full) if index_weight_map else None
                    if hint:
                        print(f"  CANNOT VERIFY expert {e} {tmpl.format(e=e)}: "
                              f"source shard {hint} not present under --model", file=sys.stderr)
                    else:
                        print(f"  CANNOT VERIFY expert {e} {tmpl.format(e=e)}: "
                              f"tensor not found in any loaded shard", file=sys.stderr)
                    if status != "mismatch":
                        status = "incomplete"
                    off += n
                    continue
                shard_path, soff, sn, dtype = info
                with open(shard_path, "rb") as sf:
                    sf.seek(soff)
                    ref = sf.read(sn)
                pf.seek(e * EXPERT_BYTES + off)
                got = pf.read(n)
                if ref != got:
                    print(f"  MISMATCH expert {e} {tmpl.format(e=e)} at +{off} "
                          f"(packed offset {e * EXPERT_BYTES + off})", file=sys.stderr)
                    status = "mismatch"
                off += n
            assert off == EXPERT_BYTES
    return status


def verify_only(layer_idx, out_dir, headers, index_weight_map):
    """Check an already-packed layer file in place, without re-copying it
    from source. Never writes anything. Distinguishes three outcomes with
    three different exit-relevant statuses (see verify_layer_content)."""
    out_path = Path(out_dir) / f"layer_{layer_idx}.bin"
    if not out_path.exists():
        print(f"  layer {layer_idx}: CANNOT VERIFY - {out_path} does not exist", file=sys.stderr)
        return "incomplete"

    expected = NUM_EXPERTS * EXPERT_BYTES
    actual = out_path.stat().st_size
    if actual != expected:
        print(f"  layer {layer_idx}: MISMATCH - size {actual} != expected {expected}", file=sys.stderr)
        return "mismatch"

    status = verify_layer_content(layer_idx, out_path, headers, index_weight_map=index_weight_map)
    if status == "verified":
        print(f"  layer {layer_idx}: verified")
    elif status == "mismatch":
        print(f"  layer {layer_idx}: VERIFICATION FAILED (content mismatch)", file=sys.stderr)
    else:
        print(f"  layer {layer_idx}: VERIFICATION INCOMPLETE (source shard(s) unavailable)", file=sys.stderr)
    return status


def repack_layer(layer_idx, model_dir, out_dir, headers, verify,
                  delete_source=False, shard_remaining=None):
    out_path = Path(out_dir) / f"layer_{layer_idx}.bin"
    if out_path.exists() and not verify:
        # idempotent: skip if already exists and is correctly sized
        if out_path.stat().st_size == NUM_EXPERTS * EXPERT_BYTES:
            print(f"  layer {layer_idx}: already packed, skipping")
            return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(out_dir).free
    if free < 3 * NUM_EXPERTS * EXPERT_BYTES:
        print(f"  ERROR: only {free/1e9:.0f} GB free, need >{3*NUM_EXPERTS*EXPERT_BYTES/1e9:.0f} GB "
              f"headroom; run with --delete-source or free space", file=sys.stderr)
        sys.exit(1)

    prefix = f"model.layers.{layer_idx}"
    print(f"  layer {layer_idx}: writing {out_path}", flush=True)

    with open(out_path, "wb") as out:
        for e in range(NUM_EXPERTS):
            for tmpl, expected_size, expected_dtype in SUBOFFS:
                full = prefix + "." + tmpl.format(e=e)
                info = find_tensor(headers, full)
                if info is None:
                    print(f"  ERROR: tensor {full} not found in any shard", file=sys.stderr)
                    sys.exit(1)
                shard_path, off, n, dtype = info
                if n != expected_size or dtype != expected_dtype:
                    print(f"  ERROR: {full} size={n} dtype={dtype} expected={expected_size} {expected_dtype}", file=sys.stderr)
                    sys.exit(1)
                with open(shard_path, "rb") as f:
                    f.seek(off)
                    data = f.read(n)
                out.write(data)

                if delete_source and shard_remaining is not None:
                    shard_remaining[shard_path] -= 1
                    if shard_remaining[shard_path] == 0:
                        print(f"    deleting fully-consumed shard {shard_path}", flush=True)
                        os.remove(shard_path)
                        del headers[shard_path]

            if e % 64 == 0:
                print(f"    expert {e}/{NUM_EXPERTS}", flush=True)

    actual = out_path.stat().st_size
    expected = NUM_EXPERTS * EXPERT_BYTES
    if actual != expected:
        print(f"  ERROR: {out_path} size {actual} != expected {expected}", file=sys.stderr)
        sys.exit(1)

    if verify:
        # Real content check: read the six sub-tensors for experts 0, 7, 128
        # and 255 back out of the packed file and diff them against the
        # checkpoint via the shard headers, the same comparison Step 4 does
        # standalone. A pure size/offset-arithmetic check would pass on a
        # corrupted file, so this must touch actual bytes on both sides.
        # Every tensor packed above was already confirmed present in
        # `headers`, so "incomplete" cannot occur on this path.
        status = verify_layer_content(layer_idx, out_path, headers)
        if status == "verified":
            print(f"  layer {layer_idx}: verified")
        else:
            print(f"  layer {layer_idx}: VERIFICATION FAILED", file=sys.stderr)
            sys.exit(1)


def parse_layers(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


EXIT_CODE_HELP = """\
--verify-only exit codes:
  0 = verified   - all sampled experts (0, 7, 128, 255) matched the checkpoint
  1 = MISMATCH   - packed bytes differ from the checkpoint (data is wrong)
  3 = INCOMPLETE - could not verify: a sampled expert's source shard is not
                   present under --model (not a data defect)
  2 is reserved for argparse usage errors (e.g. --verify --delete-source)
  and is never returned for a verification outcome.
"""


def main():
    ap = argparse.ArgumentParser(
        epilog=EXIT_CODE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model",  default="./glm52-mxfp4")
    ap.add_argument("--out",    default="./packed_experts")
    ap.add_argument("--layers", default="3-77")
    ap.add_argument("--verify", action="store_true",
                     help="after packing each layer, content-check it against the "
                          "checkpoint (experts 0/7/128/255, all six sub-tensors). "
                          "Cannot be combined with --delete-source.")
    ap.add_argument("--verify-only", action="store_true",
                     help="skip the write/pack loop entirely and content-check an "
                          "already-packed layer file in place, using whatever shard "
                          "headers are still on disk under --model. Reports one of "
                          "three outcomes (verified / mismatch / incomplete) and "
                          "distinguishes them in the exit code: 0 verified, "
                          "1 mismatch, 3 incomplete (see epilog below). Use this "
                          "to re-check a layer after --delete-source has removed the "
                          "shards it no longer needs.")
    ap.add_argument("--delete-source", action="store_true",
                     help="remove a shard once every tensor it holds has been written "
                          "into a packed layer file. Only deletes shards that become "
                          "fully consumed by tensors this run actually packs.")
    args = ap.parse_args()

    if args.verify and args.delete_source:
        ap.error("--verify cannot be combined with --delete-source: within a single "
                  "run, an earlier layer's deletions can remove a shard a later "
                  "layer's sampled experts (0, 7, 128, 255) still need, producing a "
                  "spurious mismatch/incomplete report for otherwise-correct data. "
                  "Run verification as a separate invocation (--verify-only) instead.")

    print(f"reading shard headers from {args.model}...", flush=True)
    headers = {}
    for fn in sorted(os.listdir(args.model)):
        if fn.endswith(".safetensors"):
            p = os.path.join(args.model, fn)
            h, base = read_shard_header(p)
            headers[p] = (h, base)
    print(f"  {len(headers)} shards indexed")

    layers = parse_layers(args.layers)

    if args.verify_only:
        index_weight_map = None
        idx_path = os.path.join(args.model, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                index_weight_map = json.load(f)["weight_map"]

        # mismatch outranks incomplete for picking the worst status across
        # multiple layers; the exit-code mapping below is intentionally not
        # 0/1/2 in rank order -- 2 is reserved for argparse usage errors.
        rank = {"verified": 0, "incomplete": 1, "mismatch": 2}
        worst = "verified"
        for li in layers:
            status = verify_only(li, args.out, headers, index_weight_map)
            if rank[status] > rank[worst]:
                worst = status

        print("done.")
        sys.exit({"verified": 0, "mismatch": 1, "incomplete": 3}[worst])

    shard_remaining = None
    if args.delete_source:
        # Count how many tensors each shard holds; a shard is deleted once all
        # of its tensors have been consumed by tensors this run packs. Shards
        # that also hold tensors outside the requested --layers range (e.g. a
        # later layer not yet processed) simply never reach zero and are
        # never deleted, which is the safe direction to fail in.
        shard_remaining = {p: sum(1 for k in h if k != "__metadata__")
                           for p, (h, base) in headers.items()}

    for li in layers:
        repack_layer(li, args.model, args.out, headers, args.verify,
                     delete_source=args.delete_source, shard_remaining=shard_remaining)

    print("done.")


if __name__ == "__main__":
    main()
