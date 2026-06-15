"""Run merge-theater on Modal: download -> merge -> eval -> publish.

Everything is cached on a Modal Volume, so reruns and ratio sweeps are cheap.

    pip install modal
    modal token set --token-id ak-... --token-secret as-...   # auth
    modal secret create huggingface HF_TOKEN=hf_...            # write access to your HF org

Entry points:
    modal run modal_app.py                          # single mini merge (cheap dress rehearsal)
    modal run modal_app.py::sweep                    # mini ratio sweep -> pick the winner
    modal run modal_app.py::rematch                  # 397B: Rio baseline vs DARE-TIES challenger ($$$)
    modal run modal_app.py::publish_cmd --merged <name> --repo-id PixelML/<name>

Pull results to tabulate locally:
    modal volume get merge-theater results ./results
    python scripts/compare.py results

NOTE: the 35B path runs on 1xH100. The 397B path (`rematch`) uses merge_big +
evaluate_big (8xH100, fp8) and a ~2.4TB volume. Validate the whole pipeline on the
cheap mini path FIRST, then spend the big credits.
"""
from __future__ import annotations

import pathlib

import modal

APP = modal.App("merge-theater")
VOL = modal.Volume.from_name("merge-theater", create_if_missing=True)
ROOT = "/vol"

# HF token (write scope) lives in this Modal secret; needed for gated pulls + publish.
SECRETS = [modal.Secret.from_name("huggingface")]

BASE_ENV = {"HF_HOME": f"{ROOT}/hf"}

# Tiny image for pulling/pushing weights.
dl_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub>=0.25")
    .env(BASE_ENV)
)

# Our own merge needs nothing exotic: torch + safetensors to average tensors,
# huggingface_hub to resolve the cached snapshots. mergekit is skipped entirely --
# its arch catalog doesn't know Qwen3_5MoeForConditionalGeneration, and a linear
# merge is just a*W1 + b*W2 per tensor.
merge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy", "safetensors", "huggingface_hub>=0.25", "pyyaml")
    .env(BASE_ENV)
)

# Eval image on a CUDA *devel* base: vLLM JIT-compiles Qwen3.5's GDN linear-attention
# kernel at load, which needs nvcc + CUDA headers (absent from debian_slim, which made
# engine init fatal). CUDA_HOME points the compiler at the toolkit.
eval_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "vllm", "lm-eval>=0.4.4", "transformers>=4.46", "accelerate",
        "huggingface_hub>=0.25", "sentencepiece", "pyyaml",
    )
    .env({**BASE_ENV, "CUDA_HOME": "/usr/local/cuda", "HF_ALLOW_CODE_EVAL": "1"})
)


# ---------------------------------------------------------------- helpers (local)
def _models_from_yaml(text: str) -> list[str]:
    import yaml

    doc = yaml.safe_load(text)
    ids: list[str] = []

    def add(mid):
        if mid and mid not in ids:
            ids.append(mid)

    add(doc.get("base_model"))
    for m in doc.get("models", []) or []:
        add(m.get("model") if isinstance(m, dict) else m)
    return ids


def _linear_yaml(a: str, wa: float, b: str, wb: float) -> str:
    return (
        "merge_method: linear\n"
        "dtype: bfloat16\n"
        "models:\n"
        f"  - model: {a}\n    parameters:\n      weight: {wa}\n"
        f"  - model: {b}\n    parameters:\n      weight: {wb}\n"
    )


def _model_card(repo_id: str, merged: str, parents: str, results: str) -> str:
    import sys

    sys.path.insert(0, "scripts")
    from compare import tabulate_markdown  # noqa: E402

    table = tabulate_markdown([results]) if pathlib.Path(results).exists() else "_(eval pending)_"
    parts = [p for p in parents.split(",") if p]
    a = parts[0] if parts else "?"
    b = parts[1] if len(parts) > 1 else "?"
    name = repo_id.split("/")[-1]
    return f"""---
license: apache-2.0
tags:
- merge
- not-an-original-model
---

# {name}

This is a model **merge**, not an original training run. It's a plain element-wise
weighted average of two existing open-weight checkpoints (a ~40-line safetensors script;
no mergekit, since its catalog doesn't know this architecture). No pretraining, no
post-training of our own. The exact recipe ships as `merge.yaml` next to the weights.

## Ingredients
- `{a}`
- `{b}`

Merge variant: `{merged}`. Full pipeline and commands:
https://github.com/PixelML/merge-theater

## Benchmarks (same lm-eval tasks, parents vs merge)

{table}

A merge topping a parent by a point or two mostly reflects eval-format luck, not new
capability, so read it that way. The numbers reproduce with the command below.

## Reproduce
```bash
modal run modal_app.py::sweep --weights-b 0.2,0.3,0.4
```

## License
Both parents (`{a}` and `{b}`) are Apache-2.0, so this merge is released under Apache-2.0.
The merge used a small original script, not mergekit, so no LGPL applies. This card must
travel with the weights.

## Credit
Built by Claude Code (Anthropic's agent) running autonomously: the merge, the eval
pipeline, the ratio sweep, and this card.
"""


# ------------------------------------------------------------------- compute jobs
@APP.function(image=dl_image, volumes={ROOT: VOL}, timeout=6 * 60 * 60, memory=8192, secrets=SECRETS)
def download(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id, ignore_patterns=["*.pth", "original/*"])
    VOL.commit()
    print(f"[download] {model_id} -> {path}")
    return path


@APP.function(image=merge_image, volumes={ROOT: VOL}, timeout=2 * 60 * 60, memory=65536, secrets=SECRETS)
def merge(config_text: str, out_name: str) -> str:
    return _run_merge(config_text, out_name)


@APP.function(image=merge_image, volumes={ROOT: VOL}, timeout=8 * 60 * 60, memory=131072, secrets=SECRETS)
def merge_big(config_text: str, out_name: str) -> str:
    return _run_merge(config_text, out_name)


def _average_safetensors(model_dirs: list, weights: list, out_dir: str) -> None:
    """Weighted element-wise average of same-shape checkpoints. This is the 'merge'."""
    import json
    import os
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    ref = model_dirs[0]

    def weight_map(d):
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            return json.load(open(idx))["weight_map"]
        wm = {}
        with safe_open(os.path.join(d, "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                wm[k] = "model.safetensors"
        return wm

    wms = [weight_map(d) for d in model_dirs]
    handles = []
    for d, wm in zip(model_dirs, wms):
        files = sorted(set(wm.values()))
        handles.append((wm, {fn: safe_open(os.path.join(d, fn), framework="pt", device="cpu")
                             for fn in files}))

    def get(i, k):
        wm, h = handles[i]
        return h[wm[k]].get_tensor(k)

    # Mirror the reference model's shard layout in the output.
    shard_to_keys = {}
    for k, fn in wms[0].items():
        shard_to_keys.setdefault(fn, []).append(k)

    new_map, total = {}, 0
    for fn, keys in shard_to_keys.items():
        out_t = {}
        for k in keys:
            ref_t = get(0, k)
            present = all(k in handles[i][0] for i in range(len(weights)))
            if not present or not ref_t.is_floating_point():
                out_t[k] = ref_t  # buffers / int tensors / key mismatch: keep reference
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
        json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
                  open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=2)

    # Copy config + tokenizer (everything that isn't a weight shard) from the reference.
    for fn in os.listdir(ref):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = os.path.join(ref, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, fn))


def _run_merge(config_text: str, out_name: str) -> str:
    import yaml
    from huggingface_hub import snapshot_download

    cfg = yaml.safe_load(config_text)
    method = cfg.get("merge_method", "linear")
    if method != "linear":
        raise ValueError(f"custom averager supports linear only, got {method!r}")
    entries = cfg["models"]
    ids = [m["model"] for m in entries]
    weights = [float((m.get("parameters") or {}).get("weight", 1.0)) for m in entries]
    s = sum(weights) or 1.0
    weights = [w / s for w in weights]  # normalize blend to sum 1

    out = f"{ROOT}/merged/{out_name}"
    if pathlib.Path(f"{out}/model.safetensors.index.json").exists():
        print(f"[merge] cached, skipping -> {out}")
        return out
    dirs = [snapshot_download(i) for i in ids]  # cache hit on the volume
    print(f"[merge] {list(zip(ids, weights))} -> {out}")
    _average_safetensors(dirs, weights, out)
    pathlib.Path(f"{out}/merge.yaml").write_text(config_text)
    VOL.commit()
    print(f"[merge] done -> {out}")
    return out


@APP.function(image=eval_image, gpu="H200", volumes={ROOT: VOL}, timeout=4 * 60 * 60, secrets=SECRETS)
def evaluate(model_ref: str, tasks: str, label: str) -> None:
    # Single H200 (141GB): a 35B fits with headroom, so TP=1 -- this avoids both the
    # one-80GB-card OOM and the flaky tensor-parallel inter-worker comms timeout that
    # killed the TP=2 run mid-inference.
    _run_eval(model_ref, tasks, label, tp=1, fp8=False)


@APP.function(image=eval_image, gpu="H100:8", volumes={ROOT: VOL}, timeout=6 * 60 * 60, secrets=SECRETS)
def evaluate_big(model_ref: str, tasks: str, label: str) -> None:
    # 397B-A17B served fp8 across 8xH100. fp8 risk noted in README; fall back to
    # bf16 on H100:16 (drop fp8, tensor_parallel_size=16) if vLLM rejects the arch.
    _run_eval(model_ref, tasks, label, tp=8, fp8=True)


def _run_eval(model_ref: str, tasks: str, label: str, tp: int, fp8: bool) -> None:
    import subprocess

    args = (
        f"pretrained={model_ref},dtype=bfloat16,gpu_memory_utilization=0.85,"
        f"trust_remote_code=True,enforce_eager=True,max_model_len=4096,tensor_parallel_size={tp}"
    )
    if fp8:
        args += ",quantization=fp8"
    subprocess.run(
        ["lm_eval", "--model", "vllm", "--model_args", args,
         "--tasks", tasks, "--batch_size", "auto",
         "--confirm_run_unsafe_code", "--output_path", f"{ROOT}/results"],
        check=True,
    )
    VOL.commit()
    print(f"[eval] {label} done")


@APP.function(image=dl_image, volumes={ROOT: VOL}, timeout=12 * 60 * 60, secrets=SECRETS)
def publish(merged_subdir: str, repo_id: str, private: bool = True, card_text: str = "") -> None:
    from huggingface_hub import HfApi

    path = f"{ROOT}/merged/{merged_subdir}"
    if card_text:
        pathlib.Path(f"{path}/README.md").write_text(card_text)
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, folder_path=path, repo_type="model")
    print(f"[publish] https://huggingface.co/{repo_id}  (private={private})")


@APP.function(image=dl_image, volumes={ROOT: VOL}, timeout=600, secrets=SECRETS)
def update_card(repo_id: str, card_text: str) -> None:
    """Replace just the model card (README.md) without re-uploading the weights."""
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=card_text.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[card] updated https://huggingface.co/{repo_id}")


# --------------------------------------------------------------------- entrypoints
@APP.local_entrypoint()
def main(
    config: str = "configs/nex-mini-qwen36.yaml",
    tasks: str = "arc_challenge,hellaswag,winogrande,gsm8k,humaneval,mbpp",
    skip_parents: bool = False,
):
    """One config, one merge, eval parents + child. The cheap dress rehearsal."""
    text = pathlib.Path(config).read_text()
    ids = _models_from_yaml(text)
    name = pathlib.Path(config).stem
    print(f"config={config}  models={ids}  tasks={tasks}")

    list(download.map(ids))
    merged_path = merge.remote(text, name)
    handles = [evaluate.spawn(merged_path, tasks, name)]
    if not skip_parents:
        handles += [evaluate.spawn(m, tasks, m.replace("/", "_")) for m in ids]
    for h in handles:
        h.get()
    _bye()


@APP.local_entrypoint()
def sweep(
    model_a: str = "nex-agi/Nex-N2-mini",
    model_b: str = "Qwen/Qwen3.6-35B-A3B",
    weights_b: str = "0.4,0.5,0.6,0.7",
    tasks: str = "arc_challenge,hellaswag,winogrande,gsm8k,humaneval,mbpp",
    skip_parents: bool = False,
):
    """Merge the same pair at several ratios, eval all, then pick the winner."""
    list(download.map([model_a, model_b]))
    jobs = {}
    for wb in (float(x) for x in weights_b.split(",")):
        wa = round(1 - wb, 3)
        name = f"mini-b{wb}"
        jobs[name] = merge.spawn(_linear_yaml(model_a, wa, model_b, wb), name)
    merged = {k: h.get() for k, h in jobs.items()}

    handles = [evaluate.spawn(p, tasks, k) for k, p in merged.items()]
    if not skip_parents:
        handles += [evaluate.spawn(m, tasks, m.replace("/", "_")) for m in (model_a, model_b)]
    for h in handles:
        h.get()
    _bye()


@APP.local_entrypoint()
def rematch(
    tasks: str = "humaneval,mbpp,gsm8k",
    skip_parents: bool = False,
):
    """397B: reproduce Rio's alleged 0.6/0.4 linear, run a DARE-TIES challenger,
    eval both + parents. This is the expensive, big-credits build."""
    A, B = "nex-agi/Nex-N2-Pro", "Qwen/Qwen3.5-397B-A17B"
    list(download.map([A, B]))

    jobs = {
        "rio-baseline-0.6-0.4": merge_big.spawn(_linear_yaml(A, 0.6, B, 0.4), "rio-baseline-0.6-0.4"),
        "challenger-dareties": merge_big.spawn(
            pathlib.Path("configs/rio-rematch-397b.yaml").read_text(), "challenger-dareties"
        ),
    }
    merged = {k: h.get() for k, h in jobs.items()}

    handles = [evaluate_big.spawn(p, tasks, k) for k, p in merged.items()]
    if not skip_parents:
        handles += [evaluate_big.spawn(m, tasks, m.replace("/", "_")) for m in (A, B)]
    for h in handles:
        h.get()
    _bye()


@APP.local_entrypoint()
def publish_cmd(
    merged: str,
    repo_id: str,
    results: str = "results",
    private: bool = True,
    parents: str = "nex-agi/Nex-N2-mini,Qwen/Qwen3.6-35B-A3B",
):
    """Upload one merged variant to HF with an honest, auto-generated model card.
    Defaults to PRIVATE -- review the card + numbers, then flip --private False."""
    card = _model_card(repo_id, merged, parents, results)
    publish.remote(merged, repo_id, private, card)


@APP.local_entrypoint()
def update_card_cmd(
    repo_id: str,
    merged: str,
    results: str = "results",
    parents: str = "nex-agi/Nex-N2-mini,Qwen/Qwen3.6-35B-A3B",
):
    """Regenerate + push only the model card (no weight re-upload)."""
    update_card.remote(repo_id, _model_card(repo_id, merged, parents, results))


def _bye():
    print("\nDone. Pull + tabulate:")
    print("  modal volume get merge-theater results ./results")
    print("  python scripts/compare.py results")
