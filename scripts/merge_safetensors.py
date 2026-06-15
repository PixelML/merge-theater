#!/usr/bin/env python3
"""The entire "model merge", in plain safetensors. No mergekit, no magic.

A linear merge is a weighted element-wise average of the weights: `out = a*W1 + b*W2`,
tensor by tensor. That only makes sense when the checkpoints share a skeleton, which is
why `preflight.py` exists. This script does exactly that and nothing else, so the thing
people call "training an original model" fits on one screen.

Usage:
    python scripts/merge_safetensors.py \\
        --model nex-agi/Nex-N2-mini:0.5 \\
        --model Qwen/Qwen3.6-35B-A3B:0.5 \\
        --out ./merged

Each --model is `hf_id_or_local_path:weight`. Weights are normalized to sum to 1.
HF ids are resolved from your local cache (download them first).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


def weight_map(d: str) -> dict:
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    wm = {}
    with safe_open(os.path.join(d, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            wm[k] = "model.safetensors"
    return wm


def average(model_dirs: list[str], weights: list[float], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ref = model_dirs[0]
    wms = [weight_map(d) for d in model_dirs]
    handles = [
        (wm, {fn: safe_open(os.path.join(d, fn), framework="pt", device="cpu")
              for fn in sorted(set(wm.values()))})
        for d, wm in zip(model_dirs, wms)
    ]

    def get(i: int, k: str):
        wm, h = handles[i]
        return h[wm[k]].get_tensor(k)

    shard_to_keys: dict[str, list[str]] = {}
    for k, fn in wms[0].items():
        shard_to_keys.setdefault(fn, []).append(k)

    new_map, total = {}, 0
    for fn, keys in shard_to_keys.items():
        out_t = {}
        for k in keys:
            ref_t = get(0, k)
            present = all(k in handles[i][0] for i in range(len(weights)))
            if not present or not ref_t.is_floating_point():
                out_t[k] = ref_t
            else:
                acc = None
                for i, w in enumerate(weights):
                    t = get(i, k).to(torch.float32) * w
                    acc = t if acc is None else acc + t
                out_t[k] = acc.to(ref_t.dtype)
            new_map[k] = fn
            total += out_t[k].numel() * out_t[k].element_size()
        save_file(out_t, os.path.join(out_dir, fn), metadata={"format": "pt"})
        del out_t

    if os.path.exists(os.path.join(ref, "model.safetensors.index.json")):
        json.dump(
            {"metadata": {"total_size": total}, "weight_map": new_map},
            open(os.path.join(out_dir, "model.safetensors.index.json"), "w"),
            indent=2,
        )

    for fn in os.listdir(ref):  # config + tokenizer, everything that isn't a shard
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = os.path.join(ref, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, fn))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="ID_OR_PATH:WEIGHT")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids, weights = [], []
    for spec in args.model:
        ref, _, w = spec.rpartition(":")
        ids.append(ref)
        weights.append(float(w))
    s = sum(weights) or 1.0
    weights = [w / s for w in weights]

    dirs = [p if os.path.isdir(p) else snapshot_download(p) for p in ids]
    print(f"merging {list(zip(ids, weights))} -> {args.out}")
    average(dirs, weights, args.out)
    print("done")


if __name__ == "__main__":
    main()
