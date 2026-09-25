# Experimental Plan: Frozen-Encoder / Causal-Mask Ablations

Follow-up to `debugging_plan.md`. That plan diagnosed why the multi-query
model (`EmbeddingModelDocEncNoProj`, `training/inf_retriever/`) was getting
0% mRecall and proposed fixes for the scheduled-sampling schedule and
possible mode collapse. This plan investigates two further hypotheses on
top of those fixes:

1. Jointly training the document encoder (instead of freezing it, as the
   paper does) may let the model "game" the loss by moving target document
   embeddings instead of learning genuinely diverse query embeddings.
2. The shared backbone (`infly/inf-retriever-v1-1.5b`) runs with
   **bidirectional attention by default** (`config.json` sets
   `"is_causal": false`), not the causal/decoder-like attention the paper's
   "autoregressive" framing and scheduled-sampling rationale assume. The
   main branch's `src/model.py` pipeline (Llama/Qwen decoder LLMs) is
   causal by default, with no override — this is an architectural
   difference between the two pipelines.

Note on the loss function: `loss_fn=hungarian` (the unmasked
`HungarianContrastiveLoss`) was previously suspected to be a bug relative to
`hungarian_masked`, but on closer inspection it matches the paper's Eq. 1-2
exactly (denominator = full batch of `b*m` documents, including a query's
own other gold docs, with no masking described) and matches `main:src/model.py`'s
implementation verbatim. **`loss_fn=hungarian` is paper-faithful and is not
part of this plan's hypotheses.**

## Experiment matrix

| Run | Doc encoder | Query-encoder attention | Purpose |
|---|---|---|---|
| **A** (Phase 1) | Frozen | Causal | Minimal paper-faithful test: does AMER's mechanism work at all on infly? |
| **B** (Phase 2) | Joint (trained) | Causal | Isolates the doc-encoder-freeze variable; adds drift diagnostics |
| **C** (Phase 3) | Frozen | Bidirectional (current default) | Isolates the causal-mask variable, cleanly (no doc-drift confound) |
| **D** (Phase 3, optional) | Joint | Bidirectional | ~ original runs — closes the 2x2, if worth the compute |

Also needed as reference points for Run A: an **untrained infly** zero-shot
eval, and a **single-query fine-tuned** baseline (frozen doc encoder,
`training_mode=standard_org_q`), both under the same frozen/causal setting.

---

## Phase 1 — Run A: minimal viable test

**Implementation changes (`training/inf_retriever/`):**

1. **Freeze the document encoder.** Load a **second, separate**
   `AutoModel.from_pretrained("infly/inf-retriever-v1-1.5b")` instance in
   `EmbeddingModelDocEncNoProj.__init__` (`inbatch.py:378-395`), `.eval()`,
   all params `requires_grad_(False)`, stored as `self.doc_encoder`. Route
   `encode_documents` (`inbatch.py:419-454`) through `self.doc_encoder`
   inside `torch.no_grad()` instead of `self.encoder`. Reusing the *same*
   module under `no_grad()` is **not** sufficient — its weights would still
   drift because they're shared with the trainable query encoder; a
   genuinely separate, never-updated copy is required to match the paper.
   Gate behind a new `--freeze_doc_encoder` flag (`options.py`), default
   `False`, so the current joint-training path is untouched.

2. **Force causal masking for the query encoder only** (leave
   `self.doc_encoder` bidirectional — correct/desired for passage
   encoding). Add `--force_causal` flag. Mechanism not yet fully confirmed:
   `config.json` sets `is_causal: false`, threaded as a default through
   `Qwen2Model.forward(..., is_causal: Optional[bool] = False, ...)` in the
   cached `modeling_qwen.py` — need to verify whether the effective switch
   is (a) `retriever.config.is_causal = True` post-load, or (b) passing
   `is_causal=True` explicitly on every `self.encoder(...)` call in
   `forward()`/`generate()` (`inbatch.py:498-502`, `559-563`). Confirm
   before implementing so the change isn't a silent no-op.

3. **Zero-shot ("untrained") baseline eval:** run `evaluate()`/
   `evaluate_recall()` once against the frozen+causal-configured model
   *before* any optimizer step (step 0). Check whether this already
   happens naturally at `step==0` or needs an explicit call added in
   `finetuning_multi.py` before the training loop.

4. **Single-query reference:** re-run (or reuse an existing checkpoint
   under matching settings, if available) `training_mode=standard_org_q`
   with the same frozen-doc-encoder change, for the 3-way comparison below.

**Run config:** same hyperparameters as current `finetune_ambigqa.sh`
(temp 0.05, `loss_fn=hungarian`, `full_sampling=1`,
`negative_hard_ratio=0.0`) plus `--freeze_doc_encoder --force_causal`.

**Comparison / success criterion:** multi-query (Run A) should beat both
the untrained zero-shot baseline and the single-query fine-tuned baseline
on mRecall@k, *and* show low `eval_repeat_pct` / low
`eval_pairwise_cos_sim` in the `generate()`-based diagnostics (genuine
diversity, not collapse). If Run A fails this, the problem is upstream of
doc-encoder-freezing and causality — likely training dynamics (LR,
negatives, steps) rather than architecture.

### Results (2026-09-17) — training-time diagnostics

Both Run A and the single-query reference (jobs `17927233`, `17927234`)
completed successfully — all Step 1 infra (bf16 loss cast, gradient
isolation, `is_causal` threading, FSDP-bypass) held up over a full 800-step
run. But the diversity criterion **failed**:

| Metric | Step 0 (untrained) | Step 800 (final) |
|---|---|---|
| `pairwise_cos_sim` | 0.827 | 0.929 (↑ = worse) |
| `repeat_pct` | 87.9% | 94.5% (↑ = worse) |
| `same_top1_pct` | 79.7% | 91.3% (↑ = worse) |
| `step_0/1_acc` | — | ~97-98% |
| `step_2/3/4_acc` | — | 37% / 12% / 4.3% (flat all run) |

Training pushed the k query embeddings *closer together*, not apart — mode
collapse (debugging_plan.md issue #3), and it happened **even with the doc
encoder frozen and causal masking forced**, so it isn't explained by
joint-doc-encoder gaming or the bidirectional-attention deviation. Single-
query baseline (job `17927234`) final: `eval_acc: 94.0%`, `mrr: 0.960`.
Note `eval_acc`/`mrr` above are the single-pass teacher-forced metric
already flagged as potentially misleading — see the retrieval evaluation
below for the metric that actually matters.

### Phase 1 — Retrieval Evaluation (mRecall@k)

The training-time diagnostics above are in-batch/teacher-forced proxies.
This step gets actual corpus-level retrieval numbers for three comparison
points: (1) base/untrained `infly/inf-retriever-v1-1.5b`, (2) trained Run A
(multi-query), (3) trained single-query baseline.

**Key simplification:** since both checkpoints' doc encoder is frozen and
therefore bit-identical to the base model, corpus embeddings only need to
exist once and can be reused across all three — no need to re-embed the
corpus per checkpoint (unlike the old joint-doc-encoder-trained checkpoints
under `wikipedia_embeddings/ambigqa/<mode>/`, which needed their own).

| # | Step | Status |
|---|---|---|
| 1 | Verify `--num_shards` against actual corpus shard count | Resolved — not a bug; `--num_shards 16` intentionally loads 2 of the 32 physical shard files at a time |
| 2 | New launch script running `retrieval_inf.py` per checkpoint, reusing existing corpus embeddings (`/scratch/hc3337/embeddings/inf/qampari_embeddings/*`) | Done — `retrieve_phase1.sbatch` |
| 3 | Extend `scripts/eval/eval_ambignq.sh` to cover base model + both new checkpoints | Done — `suffix_list`: `base_model`, `frozenDocEnc_singlequery`, `frozenDocEnc_causal` |
| 4 | Compare mRecall@10/@100 across all three | **Done** — see Results below |

Base model's own retrieval result was already computed by
`scripts/retrieve_base_model.sh` at
`results/base_retrievers/inf/amer_data/ambigqa.jsonl` — reused as-is, not
regenerated.

**Two environment issues hit along the way, both fixed, worth knowing about
for future runs in this env:** (1) `pytrec_eval` was missing from the `nli`
conda env — installed via a workaround (this cluster's Apptainer fakeroot
is degraded — no `/etc/subuid`/`/etc/subgid` entries for this account, so
`mkdir` inside a `:rw` overlay fails; the package's single pure-Python file
was placed as a flat module next to its already-buildable C extension).
Any future package needing a real install into this overlay will likely hit
the same wall — a permanent fix needs an HPC ticket to add subuid/subgid
entries. (2) `src/eval_utils.py` had a module-level `from beir import
LoggingHandler` used only for cosmetic log formatting, unrelated to the
actual eval logic — removed in favor of `logging.StreamHandler()`, avoiding
installing the much larger `beir` package for zero functional need.

### Results (2026-09-18) — retrieval evaluation, AmbigQA test set (827 examples)

| System | MRecall@100 | Recall@100 | Precision@100 | MRecall@10 | Recall@10 | Precision@10 |
|---|---|---|---|---|---|---|
| **Base model** (untrained infly) | **72.55** | **85.97** | **17.57** | **45.95** | **69.23** | **35.66** |
| Trained single-query baseline | 70.25 | 84.47 | 17.18 | 44.01 | 66.05 | 33.40 |
| Trained multi-query (Run A) | 69.65 | 84.21 | 16.04 | 42.20 | 64.74 | 30.68 |

**Every metric, at both k=10 and k=100, ranks in the same order:
base model > single-query fine-tune > multi-query fine-tune (Run A).**
Fine-tuning made retrieval *worse* than doing nothing, and the multi-query
objective made it worse still — confirming, at the actual corpus-retrieval
level (not just the in-batch training diagnostics), that this run did not
meet its success criterion. This is consistent with the training-time
`pairwise_cos_sim`/`repeat_pct` collapse signal above, and also suggests a
second, compounding factor: this pipeline does **full fine-tuning of the
already-strong pretrained `infly` backbone, with no LoRA** — unlike the
paper's own real-data recipe, which uses LoRA specifically because "full
fine-tuning seems to drift the model too much from its base form." Full FT
on a small (5044-example) dataset degrading an already-strong pretrained
embedding space is a plausible independent contributor on top of the
mode-collapse mechanism, and would affect the single-query baseline too
(which also degraded, just less than multi-query) — worth testing directly
(e.g. add LoRA to this pipeline, or a much lower LR / fewer steps) as a
follow-up before concluding the multi-query objective itself is at fault.

### LoRA ablation

Follow-up to the retrieval results above: isolate whether full-FT-without-
LoRA (vs. the multi-query objective itself) is degrading quality.

**Implementation:** `training/inf_retriever/`  had no LoRA support at all
before this (confirmed: no `peft` import anywhere). Added:
- `_maybe_apply_lora(encoder, opt)` (`inbatch.py`, module-level helper) —
  wraps only the trainable `self.encoder` in a `peft` LoRA adapter when
  `opt.use_lora` is set (rank 64, alpha 16, dropout 0.1, on
  q/k/v/o_proj + gate/up/down_proj — matching the paper's Appendix A.6
  recipe). The frozen `self.doc_encoder` never gets LoRA — it isn't trained
  at all either way, same as the full-FT runs.
- Called from `EmbeddingModelFrozenDocEnc.__init__` and
  `EmbeddingModelFrozenDocEncSingleQuery.__init__` right after
  `super().__init__()` sets up `self.encoder`. These are classes added this
  session (not the original `EmbeddingModelDocEncNoProj`), so modifying them
  directly doesn't conflict with the earlier instruction not to touch that
  class.
- New flags in `options.py`: `--use_lora`, `--lora_r` (64), `--lora_alpha`
  (16), `--lora_dropout` (0.1).
- Verified with a real forward+backward dry run on GPU before submitting:
  392 LoRA params, all received gradients; base encoder weights and frozen
  `doc_encoder` correctly received zero gradients; `generate()` also
  confirmed working through the LoRA-wrapped + gradient-checkpointed model.

**Run config:** identical to Run A / the full-FT single-query baseline in
every other respect (same lr=1e-5, 800 steps, batch 150, single GPU,
frozen doc encoder, causal for the multi-query variant) — only `--use_lora`
added, to isolate exactly one variable.
**Caveat:** kept the same LR as the full-FT runs for the cleanest isolation,
even though LoRA conventionally benefits from a higher LR than full
fine-tuning — this run may understate what LoRA could achieve with its own
tuned hyperparameters.

**New scripts:** `finetune_ambigqa_frozen_causal_lora.sh` (job `17945709`),
`finetune_ambigqa_frozen_singlequery_lora.sh` (job `17945710`).

**Retrieval note:** `retrieval_inf.py` needs to run under `nli` (the only
environment with a working `faiss-gpu` build for this repo's retrieval
code), but `nli` doesn't have `peft`, and installing it there hit a genuine
cluster-level limitation: this account has no `/etc/subuid`/`/etc/subgid`
entries anywhere on the cluster, so every available Apptainer/SingularityCE
install's `--fakeroot` mode is degraded and can't `mkdir` inside a `:rw`
overlay (confirmed directly: `getent subuid hc3337` fails, `/etc/subuid`
has no entry for this account, and the apptainer binary isn't setuid-root
despite `allow setuid = yes` in its own config). Worked around this by
**merging the LoRA adapters into the base weights** (`peft`'s
`merge_and_unload()`, done once in `div`, which already has `peft`) —
mathematically exact for inference, and the resulting checkpoints are
plain (no LoRA structure), so `nli` never needs `peft` at all. Merged
checkpoints: append `_merged` to each LoRA checkpoint directory name.

### Results (2026-09-18) — retrieval evaluation, all five configurations

| System | MRecall@100 | Recall@100 | MRecall@10 | Recall@10 |
|---|---|---|---|---|
| **Base model** (untrained) | **72.55** | **85.97** | **45.95** | **69.23** |
| Full-FT single-query | 70.25 | 84.47 | 44.01 | 66.05 |
| LoRA single-query | 70.98 | 84.71 | 44.74 | 67.42 |
| Full-FT multi-query (Run A) | 69.65 | 84.21 | 42.20 | 64.74 |
| LoRA multi-query | 70.13 | 84.38 | 41.72 | 64.15 |

Training-time diagnostics (LoRA multi-query final: `pairwise_cos_sim: 0.915,
repeat_pct: 95.7%, step_2/3/4_acc: 37%/12%/4.3%`) are essentially identical
to full-FT Run A's (`0.929`/`94.5%`/`37%`/`12%`/`4.3%`) — LoRA did not
meaningfully change the collapse dynamics during training either.

**Conclusion: full-FT-without-LoRA is a real but minor contributor, not the
dominant cause.** LoRA single-query beats full-FT single-query on every
metric; LoRA multi-query beats full-FT multi-query at k=100 and is roughly
a wash at k=10 — so the LoRA hypothesis is confirmed as *a* factor. But
every configuration, LoRA included, still underperforms the untrained base
model on every metric, and multi-query is still worse than single-query in
both the full-FT and LoRA settings. The dominant driver is more likely the
training recipe itself (AmbigQA's low target diversity, `full_sampling=1`'s
exposure-bias tradeoff, or the Hungarian loss dynamics at this data scale)
rather than the LoRA/full-FT parameterization choice.

---

### Next-steps priorities (2026-09-18)

Two of the priorities from the "suggested next steps" discussion were run:

**Priority 1 — was checkpoint selection hiding a better answer? No.** Checked
the actual `step` field saved in `best_model/checkpoint.pth` for all four
AmbigQA runs (full-FT single/multi, LoRA single/multi) and both QAMPARI LoRA
runs: **all six saved at step 780/800** — the proxy `mrr` metric kept
improving (or never regressed) essentially to the end of training in every
case. There is no earlier, less-degraded checkpoint being missed by
selection; the retrieval degradation is a property of the whole trained
trajectory, not a checkpoint-picking artifact.

**Priority 2 — does QAMPARI (more diverse targets) show a different
pattern? Partially.** Same recipe (frozen doc encoder, causal, LoRA,
lr=1e-5, 800 steps, batch 150) rerun on QAMPARI
(`finetune_qampari_frozen_causal_lora.sh` job `17965429`,
`finetune_qampari_frozen_singlequery_lora.sh` job `17965430`), retrieval via
`retrieve_qampari_lora.sbatch`, eval via extended `scripts/eval/eval_qampari.sh`:

| System (QAMPARI) | MRecall@100 | Recall@100 | MRecall@10 | Recall@10 |
|---|---|---|---|---|
| Base model (untrained) | 12.24 | 46.51 | 1.69 | 18.91 |
| LoRA single-query | **12.81** | **48.40** | **2.07** | **20.34** |
| LoRA multi-query | 11.30 | 45.02 | 1.32 | 18.71 |

- **Single-query fine-tuning now beats the base model on every metric** —
  the opposite of AmbigQA, where every fine-tuned configuration (including
  LoRA) underperformed the untrained base model. Supports AmbigQA's low
  target diversity as a real contributing factor to *that* degradation.
- **Multi-query still underperforms both base model and single-query**, same
  ordering as AmbigQA. Training diagnostics show collapse is if anything
  *worse* on QAMPARI (`eval_pairwise_cos_sim` reached 0.966, vs AmbigQA's
  0.915-0.929; `repeat_pct` 95.15%, comparable to AmbigQA's ~95%) despite
  QAMPARI's targets being more diverse per the paper's own analysis.
- **Conclusion: the multi-query degradation is not primarily explained by
  AmbigQA's low target diversity.** It persists, and if anything worsens, on
  a dataset with genuinely more diverse targets — pointing at the multi-query
  training mechanism itself (Hungarian loss dynamics, `full_sampling=1`'s
  exposure-bias tradeoff, or the lack of hard negatives) as the dominant
  driver, independent of dataset.

(Older joint-doc-encoder-trained checkpoints also appear in the QAMPARI eval
output for reference — `standard` scores much higher, `multi_hungarian`/
`multi_hungarian_masked` score near zero. These predate this session's fixes
and used a different, much longer training recipe, so they are not a
fair comparison point against the runs above.)

Retrieval note: same `nli`-lacks-`peft` issue as the AmbigQA LoRA runs;
same fix (merge LoRA into base weights via `peft`'s `merge_and_unload()` in
`div`, then retrieve the merged/plain checkpoint under `nli`).

### QAMPARI full-FT Run A (2026-09-21) — filling the missing cell

QAMPARI previously only had a **LoRA** variant of Run A; full-FT Run A
(frozen doc encoder, forced causal, no LoRA) had never been trained on
QAMPARI. New script `training/inf_retriever/finetune_qampari_frozen_causal.sh`
(job `18211613`), identical hyperparameters to the AmbigQA full-FT Run A
(lr=1e-5, 800 steps, warmup 30, batch 150). Retrieval reused the shared
frozen-doc-encoder corpus embeddings (`retrieve_qampari_fullft_causal.sbatch`,
job `18216931`) — no re-embedding needed.

| System (QAMPARI) | MRecall@100 | Recall@100 | MRecall@10 | Recall@10 |
|---|---|---|---|---|
| Base model (untrained) | 12.24 | 46.51 | 1.69 | 18.91 |
| LoRA single-query | **12.81** | **48.40** | **2.07** | **20.34** |
| **Full-FT multi-query (Run A)** | 11.86 | 46.57 | 1.69 | 19.87 |
| LoRA multi-query | 11.30 | 45.02 | 1.32 | 18.71 |

Training-time diagnostics matched the same collapse signature seen
everywhere else this session: final `eval_pairwise_cos_sim: 0.965`,
`eval_repeat_pct: 93.3%`.

Full-FT multi-query sits *between* LoRA multi-query and the base model —
slightly better than LoRA multi-query on most metrics (and actually beats
base model on Recall@10), but still below both the base model on MRecall@100
and below LoRA single-query on every metric. This closes the QAMPARI
full-FT-vs-LoRA comparison: as on AmbigQA, LoRA vs. full-FT is a second-order
effect, not the driver — multi-query underperforms single-query/base under
both parameterizations.

### `multi_hungarian_with_detach` (2026-09-21) — multi-query genuinely wins

Follow-up to the `multi_hungarian_failed` investigation above. The old
joint-doc-encoder-trained checkpoint family for QAMPARI also includes a
`..._multi_hungarian_with_detach` variant — its name implies an earlier
attempt (predating this session) to fix the collapse via a gradient-detach
somewhere in the Hungarian matching. Never evaluated before now.

**Corpus embeddings:** since the doc encoder is jointly trained (not
frozen), this needed its own fresh embeddings — done correctly this time as
a 32-way **parallel array job** (`gen_embed_qampari_multi_hungarian_with_detach.sbatch`,
job `18211616`, one ~808K-passage shard per task), learning from the
`multi_hungarian_failed` episode where a hand-rolled 2-shard sequential job
produced two ~40GB files that both timed out during generation and then
OOM'd retrieval (host RAM on the first attempt, then GPU device memory via
FAISS on the second). All 32 shards completed cleanly, no errors.

**Retrieval:** `retrieve_qampari_multi_hungarian_with_detach.sbatch`
(job `18222722`), same pattern as the other QAMPARI retrieval scripts.

**Results:**

| System (QAMPARI) | MRecall@100 | Recall@100 | MRecall@10 | Recall@10 |
|---|---|---|---|---|
| **multi_hungarian_with_detach** (joint, multi-query, detach fix) | **25.42** | **56.90** | 1.69 | **21.79** |
| standard (joint, single-query, old recipe) | 22.41 | 51.90 | 3.01 | 19.87 |
| LoRA single-query (frozen doc enc) | 12.81 | 48.40 | 2.07 | 20.34 |
| Full-FT multi-query Run A (frozen doc enc) | 11.86 | 46.57 | 1.69 | 19.87 |
| Base model (untrained) | 12.24 | 46.51 | 1.69 | 18.91 |
| LoRA multi-query (frozen doc enc) | 11.30 | 45.02 | 1.32 | 18.71 |
| multi_hungarian_failed (joint, multi-query, no detach) | 0.00 | 1.78 | 0.00 | 0.78 |

**This is the first result where multi-query genuinely and substantially
beats single-query** — +3.0 MRecall@100 / +5.0 Recall@100 over the
joint-trained single-query baseline (`standard`), not just parity. It also
confirms that `multi_hungarian_failed`'s catastrophic ~0 MRecall was a real,
severe training-time collapse (not a retrieval artifact — see the earlier
section), and that whatever the detach mechanism does, it is a genuine,
targeted fix for that specific collapse mode, not a general side effect of
joint doc-encoder training on its own (which alone still leaves multi-query
worse than single-query in the frozen-doc-encoder Run A family above).

### AmbigQA `max_new_tokens` bug fix (2026-09-22)

Discovered that every retrieval script in this repo silently used
`retrieval_inf.py`'s old `--max_new_tokens` default (5) for multi-query
checkpoints, including on AmbigQA — but the correct value for AmbigQA is
**2**, not 5 (empirical per-dataset choice: QAMPARI=5, AmbigQA=2). Fixed:
`retrieval_inf.py` now raises `ValueError` if `--max_new_tokens` is omitted
for a multi-query checkpoint rather than silently defaulting (no more
silent wrong-k bugs); all existing retrieval scripts updated to pass it
explicitly. See the new `amer-retrieval-pipeline` skill
(`.claude/skills/amer-retrieval-pipeline/SKILL.md`) for the per-dataset
table and pipeline conventions generally.

Re-ran the three affected AmbigQA multi-query retrievals with the corrected
k=2:

| System (AmbigQA) | MRecall@100 | Recall@100 | MRecall@10 | Recall@10 |
|---|---|---|---|---|
| Base model (untrained) | 72.55 | 85.97 | 45.95 | 69.23 |
| Full-FT multi-query Run A — old k=5 | 69.65 | 84.21 | 42.20 | 64.74 |
| Full-FT multi-query Run A — **corrected k=2** | 70.50 | 84.57 | 43.17 | 65.61 |
| Old joint `multi_hungarian` — old k=5 | 71.70 | 85.77 | 37.48 | 62.02 |
| Old joint `multi_hungarian` — **corrected k=2** | **74.12** | **87.27** | 44.62 | 66.94 |

**The old joint-trained `multi_hungarian` checkpoint now beats the base
model at k=100** (74.12 vs 72.55) with the corrected k — a real, previously
hidden result, not just noise from the bug. Run A (frozen doc encoder,
full-FT) improved slightly but still trails base model. LoRA multi-query
barely moved (70.13 -> 70.01, essentially unchanged) and still trails base
— so the k bug specifically mattered for Run A and the old joint checkpoint,
not for LoRA. Full corrected comparison:

| System (AmbigQA) | MRecall@100 old k=5 -> new k=2 | Recall@100 | MRecall@10 old -> new | Recall@10 |
|---|---|---|---|---|
| Base model | 72.55 | 85.97 | 45.95 | 69.23 |
| Full-FT Run A | 69.65 -> 70.50 | 84.57 | 42.20 -> 43.17 | 65.61 |
| LoRA multi-query | 70.13 -> 70.01 | 84.41 | 41.72 -> 42.93 | 65.53 |
| Old joint `multi_hungarian` | 71.70 -> **74.12** | **87.27** | 37.48 -> 44.62 | 66.94 |

### QAMPARI reproduction sweep (2026-09-22/23) — concerning early result

Retrained QAMPARI from scratch (joint doc encoder, same hyperparameters as
`with_detach`, 2 GPUs, `accelerate_config_2gpu_fsdp_nooffload.yaml` to avoid
the utilization-cancellation issue) with `--save_every_eval`/`--save_at_steps`
to capture the trajectory. Genuine (non-stale, re-verified) retrieval result
at **step 500: MRecall@100=0.00, Recall@100=0.08** -- catastrophic, not a
mild degradation.

Notable: this run's **step-0 (untrained, zero-shot) eval already shows
`pairwise_cos_sim: 0.8344`** -- reproduced identically across two
independent fresh training runs, confirming it's deterministic, not noise.
**Correction:** initially suspected the `.detach()` fix (commit `0f69a4d`)
as the cause, but `generate()` (the code path used for this zero-shot eval)
has no teacher-forcing and is untouched by that fix -- checked directly.
There's also no real April baseline to compare against (zero-shot logging
was added this session), so the "0.70 vs 0.83" framing was comparing a
several-hundred-step-in number to a true step-0 number -- not apples to
apples. Retracted as the explanation.

**Confirmed genuine result (step-250, properly regenerated after an earlier
stale-file mistake): MRecall@100=0.00, Recall@100=0.45** -- catastrophic,
matching step-500. Both are real, freshly-verified evaluations of the
correct checkpoint, not stale artifacts.

**Cross-check against AmbigQA (same pipeline, same code, same detach fix):**
AmbigQA's step-30 reproduction shows a healthy MRecall@100=70.62 (see next
section) -- ruling out a broken pipeline as the explanation for QAMPARI's
collapse. This points to a genuine, dataset-specific finding: **`with_detach`'s
original positive result (MRecall@100=25.42) looks like a fragile,
non-reproducible artifact**, most likely a lucky early snapshot caught right
before catastrophic collapse -- consistent with the original run's own
restart history showing `pairwise_cos_sim` rising 0.695->0.778 within just
250 steps in *each* independent restart (see the `multi_hungarian_with_detach`
section above). This fresh run, run to completion without an early
interruption saving it, appears to follow that same collapse through to
catastrophic failure rather than being caught early.

Remaining targets (1000/2000/3500/5000) will show whether this pattern
holds throughout, or whether it partially recovers.

**Step-2000 result (2026-09-23): MRecall@100=0.00, Recall@100=0.00, MRecall@10=0.00, Recall@10=0.00, mAP=0.0000, nDCG=0.0000, MRR=0.0000.**
Complete zero across every metric, not just MRecall -- collapse is total,
not partial, and does not recover with more steps. Combined with step-250
and step-500 also at MRecall@100=0.00, the pattern holds from step-250
through step-2000: no sign of the model climbing back out. (Training
itself crashed with no traceback before reaching step-3500/5000 in this
run -- per user instruction, step-5000 is not being pursued further since
the outcome is already clear.)

| System (QAMPARI repro) | MRecall@100 | Recall@100 | MRecall@10 | Recall@10 |
|---|---|---|---|---|
| step-250 | 0.00 | 0.45 | 0.00 | 0.00 |
| step-500 | 0.00 | 0.08 | 0.00 | 0.00 |
| step-2000 | 0.00 | 0.00 | 0.00 | 0.00 |

**Control re-confirmation (2026-09-23):** regenerated corpus embeddings for
the original `multi_hungarian_with_detach` checkpoint from scratch under
today's exact pipeline (`gen_embed_qampari_with_detach_control.sbatch`, job
`18344634`) and re-ran retrieval+eval fresh (`retrieve_qampari_with_detach_control.sbatch`,
job `18357665`, output to a separate `with_detach_control/` dir so as not
to overwrite the original). Result: **MRecall@100=25.42, Recall@100=56.90,
MRecall@10=1.69, Recall@10=21.79** -- an exact match (to 2 decimals) of the
original 2026-09-21 result. This rules out both a stale/corrupted checkpoint
and a changed eval pipeline as explanations -- the checkpoint's strong result
is genuine and stable. Combined with the AmbigQA reproduction being healthy
(rules out a broken pipeline) and the checkpoint reproducing exactly (rules
out checkpoint staleness), the mystery is now narrowed specifically to
*why retraining QAMPARI from scratch with the same hyperparameters produces
a different outcome than the original training run did* -- something about
the original run's actual training dynamics (not the eval side) differed,
despite matching hyperparameters/flags. The one remaining untested candidate
is `fsdp_offload_params` (`true` in the original run's accelerate config,
`false` in the reproduction's, changed to fix the GPU-utilization
cancellation issue).

### AmbigQA reproduction, first result (2026-09-23) — pipeline confirmed healthy

Retrained AmbigQA from scratch the same way (joint doc encoder, same
hyperparameters as the old checkpoint, same no-offload fix, same
`--save_at_steps`-style checkpointing). First target checkpoint, step 30:

| System (AmbigQA) | MRecall@100 | Recall@100 | MRecall@10 | Recall@10 |
|---|---|---|---|---|
| repro step-30 | 70.62 | 84.49 | 39.66 | 61.60 |
| repro step-90 | 72.43 | 85.60 | 41.96 | 64.19 |
| repro step-150 | 72.67 | 85.44 | 41.35 | 63.13 |
| repro step-330 | 70.01 | 84.34 | 38.81 | 61.07 |
| repro step-570 | 70.01 | 84.54 | 38.45 | 60.78 |
| repro step-750 | 70.01 | 84.54 | 38.45 | 60.78 |

**Full sweep complete (2026-09-24), all 6 targets evaluated.** Clean
trajectory: rises 30->90->150 (70.62->72.43->72.67, peak at 150), then
settles down to a stable 70.01 plateau by 330 and stays exactly there
through 570/750 (the lr=0-from-step-400 scheduler artifact explained
above). No further AmbigQA reproduction targets remain.

Step-150 is slightly better than step-30, i.e. training is still improving
steadily rather than collapsing -- further confirms this reproduction is
behaving normally, in contrast to QAMPARI's catastrophic step-250/500.

**Root cause found for step-750/step-570 (2026-09-23): NOT a training dip --
an LR scheduler bug that halves the effective schedule length.** Step-570
and step-750's results are not just similar, they are byte-identical
(`results/finetuned/ambigqa/repro_step570/ambigqa.jsonl` and `.../repro_step750/ambigqa.jsonl`
are the same size, 289787549 bytes, and md5-match). Checked the training
log (`sbatch_outputs/ambigqa_joint_repro.out`) directly: `lr` decays linearly
and hits **exactly 0 at step 400 -- half of `--total_steps 800`** (last
nonzero at step 395: `1.3e-07`; step 400 onward: `0`). Once lr=0, no more
weight updates happen, so every checkpoint from step-400 onward is
functionally the same model -- step-570 and step-750 are two different
saves of the identical frozen weights. Step-330 (lr=1.82e-06, still
nonzero) is the last point with genuine training signal; anything at or
past step-400 tells us nothing new.

**This directly corroborates the DDP-vs-1GPU precision-hypothesis test
running in parallel (jobs 18367406/18367407):** that test's own script
comments already flag "the LR scheduler steps once per step" for the 1-GPU
config vs. "April's 2-process run stepped it twice per step" for DDP/FSDP --
i.e. a known suspicion that multi-GPU runs advance the scheduler 2x too
fast. This AmbigQA finding is hard confirmation of exactly that: total_steps/2
is precisely where lr hits zero, for a run that used the same 2-GPU
`accelerate_config_2gpu_fsdp_nooffload.yaml` config as QAMPARI's
reproduction. Checked QAMPARI's reproduction log too
(`sbatch_outputs/qampari_joint_repro_with_detach.out`): same pattern, lr
decays smoothly and hits exactly 0 at **step 2500 -- half of
`--total_steps 5000`**. So this is not an AmbigQA-specific quirk; it
reproduces identically in both dataset's 2-GPU reproduction runs.

**Caveat on relevance to the original collapse mystery:** the original
`with_detach` run was itself apparently also 2-GPU (per the note in
`finetune_qampari_joint_repro_1gpu.sh`'s comment), so it likely had the
same halved-schedule behavior -- and its best checkpoint (step-250) is far
from the step-2500 zero-point either way, so this specific bug does not by
itself explain why the *original* run succeeded and the *reproduction*
collapsed at step-250/500. It does, however, explain the AmbigQA
step-570/750 puzzle completely, and is valuable independent evidence
supporting whatever the DDP-vs-1GPU test turns up about scheduler-stepping
differences between distributed configs.

**Step-330 result (2026-09-24): MRecall@100=70.01, Recall@100=84.34,
MRecall@10=38.81, Recall@10=61.07.** Essentially identical to step-570/750
(70.01) despite lr still being nonzero (1.82e-06, not yet fully decayed).
So the model had already converged to its steady-state performance by
step-330 -- the true trajectory is: rises to a peak around step-150
(72.67), settles back down to ~70.01 by step-330, then stays flat
(330 ~= 570 ~= 750) for the rest of training, whether or not lr is
literally zero. Step-90 (queued next) will help pin down whether the
peak-then-settle happens gradually between 150 and 330, or right after 150. -- in the normal range alongside every other
multi-query AmbigQA result this session (Run A 70.50, LoRA 70.01, old joint
`multi_hungarian` 74.12), nowhere near the catastrophic ~0 MRecall seen for
QAMPARI's step-250/500. Since this uses the *exact same* pipeline, code
(including the `.detach()` fix), and embed/retrieve/eval scripts as the
QAMPARI reproduction, this rules out a broken pipeline as the explanation
for QAMPARI's collapse -- it's a genuine, dataset-specific finding about
QAMPARI's training dynamics in this reproduction, not a code bug.

**Open question, not yet investigated:** what exactly the detach does.
`enc_trained_..._multi_hungarian_with_detach`'s training script/diff versus
the plain `multi_hungarian_failed` recipe has not been located/read yet in
this session — needed to turn this from "a checkpoint that happens to work"
into an actionable, reproducible recipe (e.g. to also try it with a strong
base model + frozen doc encoder, closing the loop back to the original
"stronger embedding model" question that started this whole investigation).

### Consolidated results table (2026-09-23) — every checkpoint with results

All numbers below were re-computed in one pass (`eval.py --topk 100 10`,
AmbigQA/AmbigQA-2docs with `--no-gold-id`) over every results JSONL under
`results/finetuned/` plus the untrained base model; they match every
number reported in the sections above.

**Shared settings unless the row says otherwise:**
- Base model `infly/inf-retriever-v1-1.5b`, lr 1e-5, temperature 0.05.
- **frozen** = `--freeze_doc_encoder --force_causal --full_sampling`, 800
  steps, bs 150, 1 GPU, retrieves against the base corpus embeddings.
- **joint** = doc encoder trained too, default bidirectional attention,
  own corpus embeddings under `wikipedia_embeddings/`.
- Multi-query uses `loss_fn=hungarian` (unmasked); single-query uses
  `training_mode=standard_org_q`.
- LoRA runs: r=64, alpha=16.
- **k** = `--max_new_tokens` at retrieval. All AmbigQA multi-query rows use
  the corrected k=2 (09-22/23 re-runs), **except**
  `ambigqa_2docs/multi_hungarian`.

| Dataset | Result dir | Doc enc | FT | Queries (k) | Notes | MR@100 | R@100 | MR@10 | R@10 |
|---|---|---|---|---|---|---|---|---|---|
| QAMPARI | base model | — | — | single | untrained | 12.24 | 46.51 | 1.69 | 18.91 |
| QAMPARI | `standard` | joint | full | single | 5000 steps, bs 256 | 22.41 | 51.90 | 3.01 | 19.87 |
| QAMPARI | `multi_hungarian` | joint | full | multi (5) | 5000 steps, bs 50 | 0.00 | 1.78 | 0.00 | 0.78 |
| QAMPARI | `multi_hungarian_masked` | joint | full | multi (5) | `hungarian_masked` loss | 0.00 | 5.26 | 0.00 | 2.39 |
| QAMPARI | `multi_hungarian_failed` | joint | full | multi (5) | outputs identical to `multi_hungarian` | 0.00 | 1.78 | 0.00 | 0.78 |
| QAMPARI | `multi_hungarian_with_detach` | joint | full | multi (5) | detach fix, best ckpt (step 250) | **25.42** | **56.90** | 1.69 | **21.79** |
| QAMPARI | `repro_step250` | joint | full | multi (5) | repro of with_detach, 2 GPUs | 0.00 | 0.45 | 0.00 | 0.00 |
| QAMPARI | `repro_step500` | joint | full | multi (5) | ″ | 0.00 | 0.08 | 0.00 | 0.00 |
| QAMPARI | `repro_step2000` | joint | full | multi (5) | ″ | 0.00 | 0.00 | 0.00 | 0.00 |
| QAMPARI | `frozenDocEnc_causal` | frozen | full | multi (5) | Run A | 11.86 | 46.57 | 1.69 | 19.87 |
| QAMPARI | `frozenDocEnc_causal_lora` | frozen | LoRA | multi (5) | | 11.30 | 45.02 | 1.32 | 18.71 |
| QAMPARI | `frozenDocEnc_singlequery_lora` | frozen | LoRA | single | | 12.81 | 48.40 | 2.07 | 20.34 |
| AmbigQA | base model | — | — | single | untrained | 72.55 | 85.97 | 45.95 | 69.23 |
| AmbigQA | `standard` | joint | full | single | 800 steps, bs 50 | **75.57** | **87.81** | **47.88** | **69.95** |
| AmbigQA | `multi_hungarian` | joint | full | multi (2) | 800 steps, bs 50 | 74.12 | 87.27 | 44.62 | 66.94 |
| AmbigQA | `repro_step30` | joint | full | multi (2) | repro, 2 GPUs | 70.62 | 84.49 | 39.66 | 61.60 |
| AmbigQA | `repro_step150` | joint | full | multi (2) | ″ | 72.67 | 85.44 | 41.35 | 63.13 |
| AmbigQA | `frozenDocEnc_causal` | frozen | full | multi (2) | Run A | 70.50 | 84.57 | 43.17 | 65.61 |
| AmbigQA | `frozenDocEnc_causal_lora` | frozen | LoRA | multi (2) | | 70.01 | 84.41 | 42.93 | 65.53 |
| AmbigQA | `frozenDocEnc_singlequery` | frozen | full | single | | 70.25 | 84.47 | 44.01 | 66.05 |
| AmbigQA | `frozenDocEnc_singlequery_lora` | frozen | LoRA | single | | 70.98 | 84.71 | 44.74 | 67.42 |
| AmbigQA-2docs | base model | — | — | single | untrained | 77.64 | 86.81 | 55.06 | 72.36 |
| AmbigQA-2docs | `standard` | joint | full | single | 400 steps, bs 50 | **80.17** | **88.61** | **56.54** | 71.84 |
| AmbigQA-2docs | `multi_hungarian` | joint | full | multi (**5, stale**) | 400 steps, bs 50 | 69.41 | 81.43 | 43.04 | 61.29 |

**Takeaways:**
- **AmbigQA: the old joint single-query `standard` checkpoint is the best
  system** (75.57 MR@100), beating both joint `multi_hungarian` at k=2
  (74.12) and the base model (72.55). It was left out of the k=2
  comparison above, so the "`multi_hungarian` beats base" result should be
  read against this stronger single-query baseline — multi-query does not
  win on AmbigQA once `standard` is included.
- Same pattern on AmbigQA-2docs: joint `standard` (80.17) > base (77.64)
  > `multi_hungarian` (69.41). The multi row is still the stale May
  retrieval at k=5 and **has not been re-run with k=2** — its number is
  not trustworthy until it is.
- QAMPARI: `multi_hungarian_with_detach` is still the only multi-query
  win, and no repro checkpoint gets near it (all ~0).
- All frozen-doc-encoder runs (full FT or LoRA, single or multi) sit at or
  below the untrained base model on both datasets.

**Still missing:** `ambigqa_2docs/multi_hungarian` at k=2; AmbigQA
`repro_step750` (retrieval script `retrieve_ambigqa_repro_step750.sbatch`
exists, no results yet).

---

## Phase 2 — Run B + drift diagnostics

Only if Run A succeeds. Flip `--freeze_doc_encoder` off (back to joint
training), keep `--force_causal` on, everything else identical (same
seed/data/hyperparameters) to isolate only the freeze/unfreeze variable.

**Implementation (new diagnostics):**

- `train/target_pairwise_cos_sim`: same pairwise-cosine block as
  `inbatch.py:594-601`, applied to `teacher_embeddings` instead of
  `selected_outputs_embeddings`, added right after line 545. Measures
  whether a query's own m gold document embeddings are being spread apart
  or collapsed together as training progresses.
- `eval/doc_encoder_drift`: cache embeddings of a fixed ~200-500 doc sample
  using the pristine pretrained checkpoint once at `main()` start (before
  training), re-embed the same docs at each `eval_freq` checkpoint with
  current weights, log mean cosine distance to the frozen snapshot.

**Comparison:** Run B vs. Run A on final retrieval metrics, plus whether
`target_pairwise_cos_sim` / `doc_encoder_drift` move substantially over
training. This tells us whether unfreezing the doc encoder is what's
costing performance, and whether it's doing so via target-side gaming
specifically.

---

## Phase 3 — Run C (+ optional Run D): causal-mask ablation

**Run C:** same as Run A but `--force_causal` **off** (bidirectional,
current default), doc encoder still frozen — isolates the causal-mask
effect cleanly against Run A, with no doc-drift confound.

**Run D (optional):** same as Run B but `--force_causal` off — close to the
original setup. Check first whether existing runs/checkpoints already give
the retrieval-metric half of this cell; a short re-run with the Phase 2
logging added would still be needed for the drift diagnostics.

**Comparison:** A vs. C isolates causality's effect under frozen
conditions; B vs. D (if run) checks whether that effect holds/changes once
the doc encoder is also trainable (interaction effect).

---

## Sequencing

Each phase gates the next. Do not implement/run any part without explicit
go-ahead. Phase 1's `is_causal` mechanism verification is the first
concrete step once Phase 1 is greenlit, since Phase 1 and Phase 3 both
depend on getting that toggle right.

### Root cause of the repro failures: FSDP fp32 master weights vs. original pure-bf16 DDP (2026-09-23/24)

**Resolved.** Both the QAMPARI and AmbigQA reproductions differ from the originals because the Sept
repros ran under FSDP, where accelerate upcasts the bf16-loaded model to fp32 master weights. The
original April/May runs were sbatch jobs that never picked up the FSDP default config, so they trained
plain 2-GPU DDP with **pure bf16 weights and AdamW state**. That rounds away most updates: only about
7% (QAMPARI) / 10% (AmbigQA) of weights ever change. `fsdp_offload_params` is irrelevant: the
with-offload FSDP attempt collapses identically. Reruns under `accelerate_config_2gpu_ddp.yaml`
reproduce the originals' training curves and checkpoint format, and their retrieval scores:

| Dataset | DDP rerun | Original | FSDP repro |
|---|---|---|---|
| QAMPARI MRecall@100 (step 250) | 26.37 | 25.42 | 0.00 |
| AmbigQA MRecall@100 (step 750) | 75.09 | 74.12 | 72.67 best / 70.01 final |

The DDP/bf16 setting is now the documented **fallback recipe**: see `RECIPE_FALLBACK_bf16_ddp.md`
(git tag `fallback-bf16-ddp`) for exact commands, hyperparameters, verification fingerprints, the
bugs it depends on (bf16 rounding, the 2× scheduler step already noted above, unseeded per-rank
shuffling) and the artifacts to keep. Full investigation:
https://claude.ai/code/artifact/ee8584b3-851e-4d5a-b7d4-a8f1a0c7e28e
