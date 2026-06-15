#!/usr/bin/env python3
"""Preflight compatibility check for element-wise model merging.

Two checkpoints can only be linearly / SLERP / TIES merged if they share the same
base architecture: identical layer counts, hidden sizes, attention head config,
MoE expert layout, and tokenizer/vocab. This is the exact property that makes a
merge *detectable* after the fact (see nex-agi/Nex-N2#4) -- and the exact property
a "merge two different families" plan violates.

This script fetches only each model's config.json (no weights) and reports whether
a merge is even defined. Multimodal checkpoints (`...ForConditionalGeneration`)
nest the language-model dims under `text_config`, so we look there too.

Usage:
    python scripts/preflight.py nex-agi/Nex-N2-mini Qwen/Qwen3.6-35B-A3B
    python scripts/preflight.py --config configs/nex-mini-qwen36.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# Multimodal configs nest the LM dims under one of these sub-objects.
NEST_KEYS = ("text_config", "thinker_config", "llm_config", "language_config")

# Keys that must match exactly for an element-wise merge to be valid.
# (Absent-from-both keys are skipped, so dense vs MoE differences still surface.)
SHAPE_KEYS = [
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "moe_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "vocab_size",
]


@dataclass
class ModelSpec:
    repo: str
    config: dict

    @property
    def arch(self) -> str:
        archs = self.config.get("architectures") or []
        return archs[0] if archs else self.config.get("model_type", "?")

    def get(self, key):
        """Resolve a config key at top level, falling back to nested LM config."""
        if self.config.get(key) is not None:
            return self.config[key]
        for nest in NEST_KEYS:
            sub = self.config.get(nest)
            if isinstance(sub, dict) and sub.get(key) is not None:
                return sub[key]
        return None


def load_config(repo: str, revision: str | None = None) -> ModelSpec:
    path = hf_hub_download(repo, "config.json", revision=revision)
    with open(path) as f:
        return ModelSpec(repo, json.load(f))


def models_from_yaml(path: str) -> list[str]:
    if yaml is None:
        sys.exit("pyyaml not installed: pip install pyyaml")
    with open(path) as f:
        doc = yaml.safe_load(f)

    ids: list[str] = []

    def add(mid):
        if mid and mid not in ids:
            ids.append(mid)

    add(doc.get("base_model"))
    for m in doc.get("models", []) or []:
        add(m.get("model") if isinstance(m, dict) else m)
    for sl in doc.get("slices", []) or []:
        for src in sl.get("sources", []) or []:
            add(src.get("model"))
    return ids


def check(ids: list[str]) -> int:
    specs: list[ModelSpec] = []
    for mid in ids:
        try:
            specs.append(load_config(mid))
        except Exception as e:  # network / gated / missing config
            print(f"FAILED to fetch config for {mid}: {e}", file=sys.stderr)
            return 2

    ref = specs[0]
    print(f"reference: {ref.repo}  [{ref.arch}]\n")
    total_mismatch = 0
    for spec in specs[1:]:
        print(f"vs {spec.repo}  [{spec.arch}]")
        local = 0
        if spec.arch != ref.arch:
            local += 1
            print(f"  MISMATCH  architecture: {ref.arch} != {spec.arch}")
        for key in SHAPE_KEYS:
            va, vb = ref.get(key), spec.get(key)
            if va is None and vb is None:
                continue
            if va != vb:
                local += 1
                print(f"  MISMATCH  {key}: {va!r} != {vb!r}")
        if local == 0:
            print("  ok -- architecture + every shape key matches\n")
        else:
            print(f"  {local} mismatch(es)\n")
            total_mismatch += local
    return 0 if total_mismatch == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("models", nargs="*", help="two or more HF model ids")
    p.add_argument("--config", help="a mergekit yaml to read model ids from")
    args = p.parse_args()

    ids = models_from_yaml(args.config) if args.config else args.models
    if len(ids) < 2:
        p.error("need at least two models (positional) or --config <yaml>")

    rc = check(ids)
    print()
    if rc == 0:
        print("VERDICT: MERGEABLE -- same base architecture across all checkpoints.")
        print("A linear / TIES / SLERP merge is well-defined.")
        print("(It is also trivially detectable as a merge afterward -- that's the point.)")
    elif rc == 1:
        print("VERDICT: NOT MERGEABLE -- these are not finetunes of the same base.")
        print("Element-wise blending is undefined here; mergekit will error (or need")
        print("--allow-crimes, which lives up to its name). Crossing families for real")
        print("means distillation / adapter fusion -- i.e. actual training, the very")
        print("work that gets faked when a merge is shipped as an 'original' model.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
