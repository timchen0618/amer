---
name: amer-retrieval-pipeline
description: Use when generating corpus embeddings, running retrieval, or evaluating MRecall/Recall for a checkpoint in this repo's training/inf_retriever/ pipeline (gen_embed_new.py -> retrieval_inf.py -> eval.py) — for finetuned AMER checkpoints (frozen or joint doc encoder) or non-finetuned baselines (stella/inf-retriever/Qwen/NV-Embed/iterative_retrieval).
---

# AMER embed -> retrieve -> eval pipeline

## Overview

Three scripts, always in this order: `gen_embed_new.py` (or `gen_embed.sbatch`
array job) embeds the corpus -> `retrieval_inf.py` searches it per query ->
`eval.py` scores the result against gold. All three must run via
`sbatch`/`srun`, never directly on the login node (container startup +
loading multi-GB files/checkpoints is memory-heavy enough to hang it).

## Which branch applies?

```
Is the checkpoint's doc encoder frozen (EmbeddingModelFrozenDocEnc*,
--freeze_doc_encoder was set at training time)?
  YES -> reuse existing embeddings (below). Never re-embed.
  NO (joint-trained, EmbeddingModelDocEncNoProj*) -> must re-embed for
       THIS checkpoint specifically (its doc encoder differs from base).
  UNSURE -> check the checkpoint's saved opt.freeze_doc_encoder (load the
       .pth, inspect opt) or ask the user. Don't guess — embedding with
       the wrong corpus silently produces a real but meaningless score.
```

**Fast heuristic before loading a multi-GB checkpoint:** this repo's own
naming convention already tells you, most of the time — `enc_trained_*`
checkpoint dirs are joint-trained (pre-dates this session, one set of
weights doubles as both query and doc encoder); `frozenDocEnc_*` dirs are
frozen (this session's classes, separate frozen doc-encoder copy). Treat
this as a strong prior, not proof — confirm with the `.pth`'s `opt` if the
name is ambiguous or you need certainty before an expensive step.

**Before reusing existing embeddings, confirm they're actually
checkpoint-specific and current**, not stale from some other checkpoint
that happened to reuse the same directory name. Cheap check: the
embeddings directory's file mtimes should be *at or after* the
checkpoint's own `best_model/checkpoint.pth` save time (`ls -la` both). If
the embeddings predate the checkpoint, they're stale — don't reuse them.
Is it a non-finetuned baseline (stella/inf-retriever/Qwen/NV-Embed/
iterative_retrieval passed directly as --model_name_or_path)?
  -> same as frozen: embed once with the base weights, reuse forever.
```

**Frozen / baseline (reuse):** point `--passages_embeddings` at the shared
set, e.g. `/scratch/hc3337/embeddings/inf/qampari_embeddings/*` for QAMPARI.
Check what exists for other datasets before assuming — don't regenerate.

**Joint (fresh embed required):** use the 32-way **parallel array job**
pattern (`gen_embed.sbatch`'s own default: `--array=0-31`, `NUM_SHARDS=32`
matching one shard per array task, ~808K passages/shard for the
~25.86M-passage corpus). Never hand-roll a small `NUM_SHARDS` (e.g. 2) to
"simplify" — this was tried once and produced two ~40GB files that (a) each
took so long to generate the job hit its own time limit before finishing,
and (b) later OOM'd retrieval's FAISS GPU index (`cudaMalloc` fail
allocating ~72GB for one shard's flat index). 32 small shards avoids both
failure modes and is what every other embeddings directory in this repo
uses — matching convention isn't just style here, it's load-bearing.

Full corpus embeddings are ~75GB per checkpoint (32 shards); a joint-trained
checkpoint itself is ~8.7GB. **Check `myquota` before starting** — `df -h`
shows cluster-wide free space, not this account's actual quota, and this
account's `/scratch` has been seen at 98%+ full. If quota is tight, delete
a checkpoint's embeddings (and, if needed, its weights) once you have its
retrieval results — the results JSONL (~150MB) and wandb logs are what the
analysis actually needs, not the raw 75GB embedding set.

## Retrieval: `--max_new_tokens` (k) is required for multi-query checkpoints

`retrieval_inf.py` dispatches on the checkpoint's saved
`opt.training_mode`: `!= 'standard_org_q'` means multi-query, and it calls
`model.generate()` to produce `k = args.max_new_tokens` embeddings per
query (merged across k via `--agg_func`, default `round_robin` — correct
for this project unless the user asks otherwise). **There is no safe
default for k** — it's an empirical, per-dataset choice, and the script
raises `ValueError` if you omit `--max_new_tokens` for a multi-query
checkpoint rather than silently guessing. Known values as of this writing:

| Dataset | k (`--max_new_tokens`) |
|---|---|
| QAMPARI | 5 |
| AmbigQA | 2 |

For any other dataset, don't guess — ask the user. This value was
discovered wrong once already: every script in this repo omitted the flag
for months, silently using an old default of 5 for AmbigQA too (should
have been 2), invalidating several previously-reported AmbigQA
multi-query retrieval numbers until caught and fixed.

Single-query checkpoints (`training_mode == 'standard_org_q'`) never need
`--max_new_tokens` — passing it is harmless but meaningless.

## Eval

`eval.py --data_path <dataset>.jsonl --topk 100 10 --input-file
<retrieval_output>.jsonl` — CPU-bound, fast (well under a minute), but has
historically been launched with an unnecessary GPU allocation in this
repo's scripts (`srun --gres=gpu:h200:1 ...`), which only adds queue wait.
Consider `cpu_short`/`cpu_prem` partitions instead if queue time matters.

**AmbigQA / AmbigQA-2docs need `--no-gold-id`; QAMPARI must not have it.**
AmbigQA's gold passages are grouped per answer without an `id` field, so
without the flag `eval.py` crashes with `KeyError: 'id'` in
`src/eval_utils.py` (`gold_ids_per_cluster = [doc['id'] ...]`).
`scripts/eval/eval_ambignq.sh` sets it via `has_gold_id=false`;
`scripts/eval/eval_qampari.sh` doesn't pass it.

Scores are only printed to stdout — nothing is saved to disk — so record
them (e.g. in `experiment_plan.md`) or capture the job log.

## Common mistakes

- **Confusing generation-time vs. retrieval-time `--num_shards`.** They are
  different parameters with different meanings: at generation
  (`gen_embed_new.py`) it's how many pieces to split the corpus into when
  writing embedding files (use 32, matching convention). At retrieval
  (`retrieval_inf.py`) it's how many *groups* of already-written files to
  load into memory at once (throttles peak RAM/VRAM; e.g. 16 with 32 files
  on disk loads 2 files per group) — it does not need to match the file
  count and is not a bug if it doesn't.
- **Placing throwaway inspection scripts under the harness's `/tmp`
  scratchpad and then running them inside Singularity** — that path isn't
  bound into the container (`python: can't open file
  '/tmp/claude.../scratchpad/...'`). Put temp scripts under the project's
  own `/scratch` tree instead (e.g. a `.tmp_*.py` file, deleted after use).
- **Trusting a Monitor's error-grep right after resubmitting to the same
  `#SBATCH --output=` path.** A stale error from the *previous* failed run
  can still be sitting in that file when the new job is still `PENDING`
  (state check, not just log content, disambiguates this).
- **Assuming a full-precision on-disk size from a truncated `ls`.** Always
  `ls`/`du` the *actual* directory before trusting a file count derived
  from a partial listing.
