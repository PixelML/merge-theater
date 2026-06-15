# merge-theater

A model merge done in the open, with the receipts and the same evals run on the parents,
so you can see exactly how little "training" is involved.

It started with [nex-agi/Nex-N2#4](https://github.com/nex-agi/Nex-N2/issues/4): a 397B
model shipped as original work that was allegedly a `0.6·Nex-N2-Pro + 0.4·Qwen3.5`
element-wise blend. The tell was hard to argue with. All three are the identical
397B-A17B shape, the deployed model called itself "Nex" 79% of the time, and every tensor
sat thousands of std-devs off any real finetune. After the callout, the card was edited
to admit a merge plus distillation, and to admit they'd uploaded the bare merge by
mistake.

So this repo does the honest version: merge two models, eval the parents and the child on
the same tasks, print the table, and say plainly what it does and doesn't prove.

## What shipped

**[PixelML/Mini-Merge-35B-A3B](https://huggingface.co/PixelML/Mini-Merge-35B-A3B)** —
a weighted average of `Nex-N2-mini` (0.8) and `Qwen3.6-35B-A3B` (0.2) that beats **both**
parents on aggregate over six benchmarks, with zero training:

| | average | vs Nex-mini | vs Qwen3.6 |
|---|---|---|---|
| **Mini-Merge (0.8/0.2)** | **73.36** | +0.77 | +11.99 |
| Nex-N2-mini | 72.59 | | |
| Qwen3.6-35B | 61.37 | | |

Apache-2.0 (both parents are), openly labeled a merge.

## The facts that shape what's possible

A merge blends tensors position by position, so both checkpoints must share the same base:
layer count, hidden size, heads, MoE layout, tokenizer, vocab. `scripts/preflight.py`
checks this from `config.json` alone and refuses otherwise.

- `Nex-N2-Pro` is post-trained on `Qwen3.5-397B-A17B`; `Nex-N2-mini` on `Qwen3.5-35B-A3B`.
- There is no Qwen3.6 at 397B; it only shipped as `35B-A3B` and `27B`. So "Nex-Pro +
  Qwen3.6" is undefined (you can't blend a 397B MoE with a 35B one).
- But `Qwen3.6-35B-A3B` keeps the exact `Qwen3.5-35B` skeleton (hidden 2048, 40 layers,
  256 experts, vocab 248320, same modeling class). So `Nex-N2-mini ⊕ Qwen3.6-35B-A3B` is
  a real, defined merge: Nex blended with the newer base.

## Auth (once)

```bash
pip install -r requirements.txt

# Modal token: https://modal.com/settings/tokens (or run: modal token new)
modal token set --token-id ak-XXXX --token-secret as-XXXX

# HF write token for your org, stored as a Modal secret so the cluster can push weights
modal secret create huggingface HF_TOKEN=hf_XXXX
```

## Run it (Modal)

```bash
# free sanity check, no weights downloaded
python scripts/preflight.py --config configs/nex-mini-qwen36.yaml

# single 50/50 merge: download, merge, eval the merge + both parents (cached on a volume)
modal run modal_app.py::main --config configs/nex-mini-qwen36.yaml

# sweep blend ratios toward Nex and let the table pick the winner
modal run modal_app.py::sweep --weights-b 0.2,0.3,0.4

# pull results and print the table
modal volume get merge-theater results ./results
python scripts/compare.py results

# publish the winner (private by default; pass --no-private to go public)
modal run modal_app.py::publish_cmd --merged mini-b0.2 --repo-id PixelML/Mini-Merge-35B-A3B
```

Long jobs use `modal run --detach` so they finish server-side even if your laptop's
connection drops. Note: a 35B in bf16 needs more than one 80GB GPU; the eval runs on a
single H200 (141GB) at `tensor_parallel_size=1` to avoid both the OOM and a flaky
tensor-parallel comms timeout.

## The 397B rematch (optional, not run here)

`configs/rio-rematch-397b.yaml` reproduces the original's ingredients (Nex-Pro +
Qwen3.5-397B) with a smarter recipe (DARE-TIES). It's a ~$350–900 job and the 35B already
makes the point, so it's left as an exercise.

## The honest part (read before you claim SOTA)

1. A leaderboard bump is mostly a mirage. Merged models top boards through eval-format
   overfitting and benchmark contamination, not new capability.
2. A 35B does not out-think a 397B. The point isn't that this model is smarter than
   anything; it's that neither this nor the thing it mocks took the training that was
   claimed.
3. The flex is the cost: zero training, one config, a few dollars, fully reproducible,
   next to a "we trained an original 397B" claim.

## What this is not

A model you trained. If you publish the weights, publish this repo with them: the config,
the harness, the receipts. The guard that keeps it honest is `preflight.py`, which refuses
to merge two models that don't share a skeleton. The merge itself is a plain weighted
tensor average (`scripts/merge_safetensors.py`); there's no mergekit here, because its
catalog didn't recognize this model's architecture. A linear merge really is `a*W1 + b*W2`.

## Files

```
configs/nex-mini-qwen36.yaml   # Nex-mini + Qwen3.6 (the one that shipped)
configs/rio-rematch-397b.yaml  # Nex-Pro + Qwen3.5, DARE-TIES (the optional 397B version)
modal_app.py                   # download, merge, eval, publish, on Modal
scripts/merge_safetensors.py   # the actual merge: a weighted tensor average
scripts/preflight.py           # same-base guard (reads nested multimodal configs)
scripts/compare.py             # tabulate parents vs merged
scripts/evaluate.sh            # local lm-eval (hf or vllm)
```

---

Built end to end by Claude Code (Anthropic's agent, Opus) running autonomously overnight:
the merge script, the Modal pipeline, eight dependency fixes, the ratio sweep, the
published model, and the writeups. A human picked the targets and made the calls.
