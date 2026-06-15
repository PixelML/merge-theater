#!/usr/bin/env python3
"""Tabulate lm-eval results across base, parents, and merged model.

The merged row usually edges out its parents by a point or two on aggregate.
That proves almost nothing -- benchmark contamination and eval-format overfitting,
not new capability. The table is the joke; this script just renders it.

Usage:
    python scripts/compare.py results               # a directory
    python scripts/compare.py a.json b.json c.json  # explicit files
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

# Metric to read per task, in order of preference.
PREFERRED = [
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "pass@1,create_test",
    "pass_at_1,none",
    "pass@1,none",
    "mc2,none",
]

# Keys that are counts/metadata, never scores.
_SKIP = ("sample_len", "stderr", "alias")


def find_jsons(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            out += glob(os.path.join(p, "**", "*.json"), recursive=True)
        else:
            out.append(p)
    return sorted(set(out))


def primary(task_scores: dict):
    for k in PREFERRED:
        if k in task_scores:
            return task_scores[k]
    # coding metrics with task-specific suffixes (e.g. pass@1,create_test)
    for k, v in task_scores.items():
        if isinstance(v, (int, float)) and (k.startswith("pass@1") or k.startswith("pass_at_1")):
            return v
    # last resort: any real metric float, skipping counts/metadata
    for k, v in task_scores.items():
        if isinstance(v, (int, float)) and not any(s in k for s in _SKIP):
            return v
    return None


def load(path: str):
    with open(path) as f:
        data = json.load(f)
    if "results" not in data:
        return None
    label = (data.get("model_name") or path).rstrip("/").split("/")[-1]
    scores = {}
    for task, s in data["results"].items():
        v = primary(s)
        if v is not None:
            scores[task] = v
    return (label, scores) if scores else None


def aggregate(scores: dict) -> float:
    vs = list(scores.values())
    return sum(vs) / len(vs) if vs else 0.0


def tabulate_markdown(paths: list[str]) -> str:
    """Render a markdown table + aggregate ranking for the given result paths."""
    models = [m for m in (load(f) for f in find_jsons(paths)) if m]
    if not models:
        return "_no lm-eval result JSONs found_"

    tasks = sorted({t for _, s in models for t in s})
    lines: list[str] = []
    header = ["task"] + [lbl for lbl, _ in models]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for task in tasks:
        vals = [s.get(task) for _, s in models]
        present = [v for v in vals if v is not None]
        best = max(present) if present else None
        cells = [task]
        for v in vals:
            if v is None:
                cells.append("-")
                continue
            txt = f"{v * 100:.1f}" if v <= 1 else f"{v:.2f}"
            cells.append(f"**{txt}**" if v == best else txt)
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Aggregate (mean over shared tasks):")
    for lbl, s in sorted(models, key=lambda m: -aggregate(m[1])):
        vs = list(s.values())
        lines.append(f"- {lbl}: {aggregate(s) * 100:.2f}" if vs else f"- {lbl}: n/a")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="result JSON files or a results dir")
    args = ap.parse_args()
    print(tabulate_markdown(args.results))
    print("\n(Top of the table != smarter model. See README, 'The honest part'.)")


if __name__ == "__main__":
    main()
