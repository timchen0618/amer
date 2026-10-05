# Code audit, 2026-09-28

Five read-only audits of the working tree on `fsdp-clean-recipe`, which is the code the 24-run grid is running (the in-scope training files were last modified before the grid started at 2026-09-28 05:40):

1. model and loss (`src/inbatch.py`, `inf_retriever.py`, `dist_utils.py`);
2. training loop, optimizer and data loading (`finetuning_multi.py`, `finetuning_data.py`, `options.py`, `utils.py`);
3. embedding, retrieval and eval (`gen_embed_new.py`, `retrieval_inf.py`, `src/inference_utils.py`, `src/retrieval_utils.py`, `src/eval_utils.py`);
4. training vs inference consistency;
5. data building and experiment tooling (`build_clean_splits.py`, hard-negative miner, phase A/C tools).

Nothing here repeats `code_data_audit_2026-09-27.md` (D1–D6, T1–T10, E1–E5, X1–X4, N1–N3) or `step5_progress_2026-09-27.md`. **Verified** means measured or reproduced; **suspected** means from reading the code. Findings re-checked independently in the main session are marked "re-checked".

Scratch scripts and outputs: `.tmp_audit_model/`, `.tmp_audit_train/`, `.tmp_audit_infer/`, `.tmp_audit_consistency/`, `.tmp_audit_data/`, `.tmp_verify/`. Full examples for G1 and G2: `leak_examples_G1_G2.md`.

**Implementation status (updated 2026-10-02): see section 9** — what was fixed, what was explicitly declined, and what is still open.

## 1. Summary

Three problems change what current numbers mean:

1. **G1. QAMPARI clean dev still leaks into train, through identical gold sets.** 127 of the 500 clean-dev questions have a train example with exactly the same gold passages (paraphrases, or sibling templates asking another attribute of the same entities). The clean build grouped only by question text. Those 127 queries gain far more from training than the other 373, so cheap-dev ranking favors later steps and has already driven irreversible pruning in the grid.
2. **G2. QAMPARI test has the same kind of near-duplicates in train.** 44 of 531 test questions (28 identical gold sets). Single-query gains more from them than multi-query, so the single-vs-multi test comparison is biased toward single-query by roughly 1.5–2 points.
3. **R1 + C1. Inference numerics differ from the numerics training optimized, and the obvious fix makes it worse.** `load_retriever` rounds the fp32 checkpoint to bf16 weights and then runs in fp16 (R1). But training ran under bf16 autocast, which also rounds every matmul weight to bf16, so the current pipeline is actually close to what training saw. Loading fp32 weights with fp32 compute moves embeddings further from training numerics and dropped a low-LR multi checkpoint by 3.4 points (60.6 → 57.2 cheap dev MRecall@100); at LR 1e-5 all variants agree within one query. The current numbers are therefore sound (within 0–0.4 of training numerics); if anything changes, it should be fp32 weights under bf16 autocast, which matches training exactly, never plain fp32.

The model and loss code, the training loop, and the sampler are otherwise sound: no finding makes a run train on data or settings other than what its name says. With both paths in fp32, the inference code computes the same embeddings as the training code (cos ≥ 0.99998 for queries at every step, ≥ 0.99976 for documents), so every remaining train/inference difference is numeric precision or the choice of k (C2).

## 2. HIGH

### G1. HIGH: QAMPARI clean dev leaks into train through identical gold sets (verified, re-checked)

- **Where:** `data_creation/build_clean_splits.py:108-133`. Duplicates are keyed on (question, gold set); the split groups by question text only.
- **Size:** 127 of 500 clean-dev questions (25.4%) have a train example with the identical gold-id set; 159 have gold Jaccard ≥ 0.8. Re-checked: 127 / 500.
- **Two kinds:**
  - Paraphrases of one example, e.g. dev "When did Barry Unsworth books get released?" vs train "What are the publication dates of books that were written by Barry Unsworth?" (same 7 passages, same 7 answers). 26 have identical answer sets.
  - Sibling templates on the same entities, e.g. dev "What are the dates of birth of persons that were born in Baildon?" vs train "What are the occupations of persons that were born in Baildon?" (same 8 passages, different answers). QAMPARI is scored by passage, so these are leaks too.
- **Effect on cheap dev MRecall@100:**

| Checkpoint | 127 leaked | 373 clean |
| --- | --- | --- |
| base retriever | 11.0 | 14.8 |
| `grid_qampari_single_nohn_lr1e-5` s2500 | 90.6 | 72.4 |
| `grid_qampari_multi_nohn_lr1e-5` s2500 | 72.4 | 55.5 |

- The leaked-minus-clean gap grows with step and LR. In `single_nohn_lr1e-5`, about 5.6 of the 7.6-point cheap-dev gain from s250 to s2500 comes from the 127 leaked queries.
- **Effect on the grid:** `KEEP_TOP=3` pruning and full-eval queuing rank on all 500. In 10 of 11 finished QAMPARI runs the kept top 3 differ from a ranking on the 373 clean queries. Example: `multi_nohn_lr1e-5` kept 1250/1500/1750; its clean-subset best (500/750/1000) is deleted, and on the clean subset the run declines after step 750. Clean-subset gaps are 0.5–2 points (2–7 queries), so noisy, but the bias is one-directional (toward later steps and more memorization).
- **Fix:** group the split with union-find over shared normalized question OR identical gold set (or Jaccard ≥ 0.8), then rebuild dev500. Until then, rank on the 373-query subset (`.tmp_audit_data/rerank_clean.py`).

### G2. HIGH: QAMPARI test has train near-duplicates by gold set (verified, re-checked)

- **What:** 44 of 531 test questions (8.3%) have a clean-train example with gold Jaccard ≥ 0.8; 28 are identical. Re-checked: 44 / 531, 28 identical. The 2026-09-27 audit (D2) checked question text only and reported test as clean.
- **Origin:** upstream. QAMPARI's train and dev releases share wikidata templates over the same entities, e.g. test "Who directed a film that had P. Balachandran as a screenwriter?" vs train "What are the publication dates of film that had P. Balachandran as screenwriter?" (same 5 passages).
- **Effect on full-corpus test MRecall@100:**

| Checkpoint | 44 near-dups | 487 others | Gap |
| --- | --- | --- | --- |
| base retriever | 13.6 | 12.1 | +1.5 |
| `qampari_single_fix_lr1e-5_s2500` | 72.7 | 39.8 | +33 |
| `qampari_single_fixhn_lr1e-5_s2500` | 90.9 | 51.5 | +39 |
| multi checkpoints | | | +8 to +18 |

- Single-query test MRecall is inflated by about 2.7–3.3 points and multi-query by about 1–1.5, so the comparison is biased toward single-query. It does not flip the current 42.56 vs 33.71.
- **Fix:** drop train examples whose gold set has Jaccard ≥ 0.8 with any test or dev example; meanwhile also report test on the 487-query subset (`.tmp_audit_data/test_leak.py`).

### R1. HIGH: checkpoints are rounded to bf16 when loaded for inference (verified, re-checked by reading)

- **Where:** `src/inference_utils.py:61-63` (`load_retriever`), used by `gen_embed_new.py` and `retrieval_inf.py`. The in-training `evaluate()` (`finetuning_multi.py:486-499`) has the same pattern.
- **What is wrong:** `model_class(opt, None, None)` loads the backbone with `torch_dtype=torch.bfloat16` (`inbatch.py:402-405`, `816-819`). `load_state_dict(strict=True)` copies the fp32 checkpoint into those bf16 parameters, keeping their dtype. Training upcasts first (`finetuning_multi.py:925`, `model.to(torch.float32)`); inference does not. The base weights are bf16, so any update below half a bf16 step rounds back to the base value. The later `.half()` does not matter; the loss happens at the bf16 cast.
- **Evidence (`.tmp_audit_infer/bf16_loss.py`):**

| Checkpoint | Weights changed from base (fp32) | After bf16 cast | L2 error of the update | Cosine, true vs cast update |
| --- | --- | --- | --- | --- |
| grid qampari single LR 1e-6, s2250 | 91.7% | 41.6% | 54% | 0.857 |
| grid ambigqa multi LR 3e-6, s400 | 100% | 47.7% | 51% | 0.876 |
| grid qampari single LR 1e-5, s2500 | 100% | 76.1% | 14.5% | 0.990 |

- **Functional effect (`.tmp_audit_infer/bf16_functional.py`; `grid_qampari_single` LR 1e-6 step 2250, 200 clean-dev queries, ~20k-passage pool):** current pipeline vs fp32 reference: query cosine 0.9952, doc cosine 0.9914, top-20 overlap 0.938; MRecall@20 49.0 vs 51.0, @50 75.0 vs 73.5. Loading in fp32 and then `.half()`: cosine 0.99999 / 0.99994, overlap 0.994, metrics equal to the reference.
- **Consequences:**
  - Compared with an fp32 reference, fine-tuned scores shift by about ±1–2 points, with no fixed direction.
  - **Revised by C1:** the fp32 reference is not what training optimized. Against training numerics (bf16 autocast), the current pipeline is close (−0.2 to −0.4 on two checkpoints), so the earlier worry that low-LR runs are penalized does not hold for the one low-LR checkpoint measured; plain fp32 is what penalizes them.
  - The 2026-09-27 "fp16 inference vs fp32 training (cosine ≥ 0.990)" check was measuring this bf16 cast; fp16 itself is fine.
  - The drift table in `step5_progress` §4.5 describes the fp32 checkpoints, not the evaluated models.
- **Fix: see C1 before changing anything.** Loading fp32 weights alone (`model.to(torch.float32)` before `load_state_dict`) is *not* the right fix: training's forward pass used bf16-rounded matmul weights under autocast, and plain fp32 inference measurably hurts a low-LR checkpoint. Load fp32 weights and run embedding under `torch.autocast("cuda", dtype=torch.bfloat16)`, in `load_retriever` callers and in `evaluate()`.

## 3. MEDIUM

### G3. MEDIUM: `phaseC_collect_scores.py` mis-reports the grid (verified)

- The cheap-dev table globs `results/phaseC/phaseC_{ds}_*`, so no `grid_*` run appears.
- In the full-corpus table, `tags()` keys on `_fix`, so every grid run is labeled "pre-fix", and hn and nohn are not distinguished.
- Scores on the old dev500/dev300, on the new cleandev500, and pre-audit AmbigQA test scores (contaminated by D1) share one column.
- **Fix:** recognize grid naming (`_hn_` / `_nohn_`, dev-set tag) and separate pre-audit from clean runs.

### G4. MEDIUM: disk headroom is tight, and AmbigQA checkpoints are never cleaned up (verified)

- **Where:** `phaseC_run.sh:199` deletes step-0 only when `KEEP_TOP > 0`; AmbigQA runs use `KEEP_TOP=0`.
- Each AmbigQA run keeps six 6.7 GB checkpoints (~40 GB), though only steps 150 and 400 are queued for full eval: about 320 GB unneeded across 12 runs.
- At audit time `myquota` showed 4.73 of 5.00 TB. Projected further growth is ~230 GB (two full-corpus embedding sets in progress, two pending AmbigQA runs, the QAMPARI `multi_hn_lr1e-5` pre-prune peak, remaining cheap-eval outputs). Hitting the quota makes checkpoint saves or embedding fail.
- **Fix:** delete AmbigQA step-0 after the drift job, and steps 30/100/250 after their cheap eval.

### L1. MEDIUM (description), LOW (effect): hard negatives are all-or-nothing per example (verified)

- **Where:** `src/finetuning_data.py:105-118`, `sample_n_hard_negatives` at 205-217; the grid passes `--negative_ctxs 1 --negative_hard_ratio 0.5`.
- One coin is flipped per example (`negative_ctxs` = 1) and the result multiplied by k, so each example gets k hard or k random negatives, never a mix. Measured on the first 300 rows of `clean_hn`: QAMPARI multi 147 all-hard / 153 all-random / 0 mixed; AmbigQA 152 / 148 / 0. `step5_progress` says "half of each example's negatives are hard".
- The in-batch pool is still ~50% hard, so the training effect is probably small. Affects the description of all 12 hn runs.
- **Fix:** draw per negative slot, or correct the description.

### L5. MEDIUM (latent): unknown flags are silently dropped, and several parsed flags do nothing (verified)

- `src/options.py:240` uses `parse_known_args`, so a mistyped flag in `EXTRA` is ignored without error; options are never logged, so the logs can't show it either.
- Parsed but ignored: `--model_path` and `--retriever_model_id` (overwritten at `finetuning_multi.py:897`), `--continue_training` (no resume code; a run meant to continue silently restarts from the base model), `--num_workers` (hard-coded 0), `--label_smoothing`, `--lower_case`.
- All flags the current grid passes exist, so the grid is not affected.
- **Fix:** `parse_args()`, log `opt` at startup, and fail if `--model_path` / `--continue_training` is set to something the code ignores.

### R2. MEDIUM (latent): fp16 L2 normalization can overflow to all-zero embeddings (verified)

- **Where:** `gen_embed_new.py:120,134,142`; `retrieval_inf.py:115,138` (single) and `281,296-298` (multi); `embed_queries_iterative_retrieval` in `src/inference_utils.py`.
- Pooled vectors are fp16 numpy arrays; `np.linalg.norm` sums squares in fp16 and returns `inf` once the norm reaches ~256, so the vector becomes zero and scores 0 against everything, with only a RuntimeWarning.
- Measured raw norms: documents 125–189; multi-query steps median 116 → 188, max 222. Current checkpoints are ~15% below overflow and no overflow warning appears in `sbatch_outputs/`, so existing results are unaffected. A further-drifted checkpoint or larger k would look like a collapse.
- **Fix:** `.float()` before `.cpu().numpy()`.

## 4. LOW

**Training (model, loss, loop)**

| ID | Finding | Where | Status | Grid |
| --- | --- | --- | --- | --- |
| M1 / L3 | Mutable default `iter_stats={}` carries per-step stats (`step_j_teacher_cos_sim`, `pairwise_cos_sim`) across batches, so logged per-step curves are wrong (e.g. `step_7_teacher_cos_sim: 0.717` repeated at steps 25 and 50). Loss and gradients unaffected. | `inbatch.py:519` (also 296, 868, 1033, 151) | verified | logs of all multi runs |
| L2 | When k exceeds an example's hard-negative pool, `random.choices` repeats hard negatives; the cap counts `negative_ctxs`, not `negative_ctxs·k`. QAMPARI: 114 examples with repeats, 79 with none; AmbigQA: 9 / 12. | `finetuning_data.py:116-117, 209` | verified | <1% of examples |
| L12 | Under FSDP, accelerate's bf16 policy sets `param_dtype=reduce_dtype=bf16`: forward uses bf16 weight copies and gradients are reduce-scattered in bf16, unlike 1-GPU/DDP. Master weights and AdamW state stay fp32; RoPE is unaffected. | accelerate 1.9.0 `state.py:983` | verified from source | all grid runs |
| L7 | AdamW weight decay 0.01 applies to norms, biases and embeddings. Negligible at current LR (~1e-4 relative shrink). | `utils.py:143-146` | verified | negligible |
| L8 | `lr_lambda(0)=0` (first step at LR 0); with `--lr_min_ratio > 0` warmup peaks below LR, then jumps. | `utils.py:118-125` | verified | first step only |
| M3 | ~~In-training `evaluate()` is not under `no_grad` for `model.encoder` / `encode_documents`; builds an unused graph (OOM risk only).~~ **Not a bug (2026-10-02):** `evaluate()` is decorated with `@torch.no_grad()`, so no graph is built. | `finetuning_multi.py:485` | wrong | none |
| L9 | In-training multi-query eval pads one negative to k by copying it, and resamples negatives from global `random` on every call, so its numbers aren't comparable across steps. | `finetuning_data.py:126-141` | suspected for dev | in-training eval only |
| L4 | Rank 1 logs its own unaveraged metrics (`dist.reduce` to rank 0 only, both ranks log). | `dist_utils.py:191-199` | verified | logs |
| M6 | Single-query forward prints shapes and loss every step on every rank (~20k lines/run, forces a GPU sync). | `inbatch.py:905-915` | verified | single runs |
| L6 | Under FSDP, a saved optimizer state is rank 0's shard only (unusable for resume). The grid passes `--no_save_optimizer`. | `finetuning_multi.py:718/831/849` | suspected | no |
| L10 | `checkpoint/latest` symlink stores a path that doesn't resolve. | `utils.py:82` | verified | nothing reads it |
| L11 | `opt.txt` is written only when `output_dir` is new; grid runs share `checkpoints/<ds>/`, so none gets one (`checkpoint.pth` has the correct `opt`). | `finetuning_multi.py:885-888` | verified | bookkeeping |
| L13 | `--negative_ctxs > 1` fails in both modes (collator assert in multi; wrong positive split then crash in single). Fails loudly. | collator, `encode_documents` | verified by reading | no |
| M4 | `InBatch` is broken (`inf_retriever` not imported; multi-positive loss is `-log(0)=inf`). Only the old `finetuning.py` reaches it. | `inbatch.py:270, 329-331` | verified by reading | no |
| M5 | `INFRetriever.forward` pools at `mask.sum(1)-1`, wrong for left padding; `random_init` ignored. Right padding everywhere today. | `inf_retriever.py:76` | verified by reading | no |

**Embedding, retrieval, eval**

| ID | Finding | Where | Status | Current evals |
| --- | --- | --- | --- | --- |
| R3 | `--passage_maxlength` ignored; passages truncated at a hard-coded 1024 tokens vs `chunk_length` 512 in training. Only 0.007% of passages exceed 512 tokens (p99 209). | `gen_embed_new.py:100` | verified | negligible |
| R4 | Shards 1–31 read with `skiprows=start_idx` use a data row as the header, so the `dtype` mapping is not applied. Harmless with ids like `"32375442__0"`; purely numeric ids would become int/float and break id matching. | `gen_embed_new.py:208-212` | verified | no |
| R5 | `full_eval_queue.sh` checks success with `-s` on the output file; on a reprocessed tag an old file passes even if the new retrieval failed, and stale results get scored. | `full_eval_queue.sh:51-53` | verified by reading | no |
| R6 | A missing `--selected-indices-file` only warns and scores all queries. | `eval.py:34-43` | verified | no |

**Data and tooling**

| ID | Finding | Where | Status | Grid |
| --- | --- | --- | --- | --- |
| G5 | `norm_q` does not strip articles: test "Who is elected as the vice president of india?" vs train "…as vice president of india?" (identical answers). 8 AmbigQA dev examples share a train gold set (7 with identical answers, NQ paraphrases). ≤0.25% of test, 1.6% of dev. | `build_clean_splits.py:36` | verified | small |
| G6 | `full_eval_queue.sh` skips lines that aren't ready and exits on `END`, so earlier items can be left unevaluated and a missing checkpoint is skipped forever. No `END` lines yet. | `full_eval_queue.sh:91-94` | suspected | not triggered |
| G7 | Silent failure paths: a step whose cheap eval fails is neither pruned nor considered; a driver that exits with ERROR never logs "RUN DONE", so the follower waits forever and that run is never queued; a QAMPARI run with no metrics gets `.queued` with zero steps. A false-negative `job_active` at 10:09 was observed and recovered by the 10:21 restart. | `phaseC_run.sh:149-151`, `grid_launch.sh:57` | verified | recovered |
| G9 | ~18 passages are still wrapped in CSV quotes (no doubled quotes inside), which `csv_unquote` leaves while a csv parser of the corpus strips them. | loader / data | suspected | negligible |
| G10 | Driver restarts resubmit drift jobs: 43 QAMPARI and 25 AmbigQA duplicated (run, step) records. Harmless (the last line is read). | `phaseC_run.sh:177-185` | verified | no |
| G11 | `full_eval_queue.sh` deletes embeddings before the test eval; a transient eval failure redoes the 2–3 h embedding. | `full_eval_queue.sh:106-108` | verified by reading | waste only |

## 5. Training vs inference consistency

**Method.** Three grid checkpoints (`grid_qampari_multi_nohn_lr1e-6` s1000, k=5; `grid_ambigqa_multi_nohn_lr1e-6` s400, k=2; `grid_qampari_single_nohn_lr1e-5` s2500). Six dev questions per dataset and their gold passages; each passage taken both from the training jsonl through `format_passage` and as the verbatim corpus TSV row read with pandas as `gen_embed_new.py` does. Training path: the real collators → `model.forward` (sampling rate 1, loss swapped for a capture) and `generate()` → `encode_documents`. Inference path: the real `retrieval_inf.load_model_and_tokenizer` + `embed_queries_multi` / `embed_queries_single` and `gen_embed_new.embed_passages_iterative_retrieval`, in both the nli and div environments. Scripts and logs: `.tmp_audit_consistency/` (`embed_compare.py`, `compare*.py`, `prec_wrap.py`, `cheap_prec.sbatch`, `job_18732514.log`, `job_18734695.log`, `cheapprec_18735129.log`).

### C1. HIGH: inference numerics differ from training numerics, and plain fp32 inference is further away (verified)

- **Where:** training `finetuning_multi.py:925` (fp32 master weights) with bf16 autocast (`finetuning_multi.py:655-664`, FSDP bf16 params); inference `src/inference_utils.py:61-63` → `inbatch.py:402-405` (bf16 load), then `.half()` (`retrieval_inf.py:83-84`, `gen_embed_new.py:197-198`).
- **Why:** bf16 autocast casts each fp32 linear weight to bf16 for every matmul, so the function training optimized already used bf16-rounded matmul weights. At LR 1e-6 most updates are below bf16 resolution, so an fp32-weight model is a different function from the one training saw. R1's rounding happens to reproduce the matmul weights; what still differs is norms/embeddings and the activation dtype (fp16 vs bf16).
- **Cosine to training numerics (fp32 weights + bf16 autocast), current pipeline:** QAMPARI multi docs min 0.987 / mean 0.998; queries min 0.9984 at step 0 down to 0.9947 at step 4; AmbigQA multi docs min 0.992. Pure fp32 is further from training numerics: QAMPARI multi docs min 0.943 / mean 0.976, queries down to 0.963 at step 3; AmbigQA docs min 0.932.
- **QAMPARI cheap dev MRecall@100 (clean dev500):**

| Checkpoint | Current pipeline (bf16 weights, fp16) | fp32 weights + bf16 autocast (= training) | fp32 weights, fp32 compute |
| --- | --- | --- | --- |
| multi LR 1e-6, s1000 | 60.60 | 61.00 | 57.20 |
| multi LR 1e-5, s1250 | 67.00 | 66.80 | 67.00 |
| single LR 1e-5, s2500 | 77.00 | 77.20 | 77.20 |

  (One query = 0.2 points. The current-pipeline column is the grid's own cheap-dev result for that step.)

- **The effect is specific to low LR.** At LR 1e-5 all three variants agree within one query, for both multi and single; at LR 1e-6 plain fp32 loses 3.4 points (17 queries). This fits the mechanism: at LR 1e-5 most updates exceed bf16 resolution, so bf16-rounded and fp32 weights are nearly the same function.
- **Affects the grid:** current numbers are within 0–0.4 of training numerics at every LR measured, so the grid's ranking is not distorted. "Fixing" R1 with plain fp32 would lower the LR 1e-6 (and probably 3e-6) multi runs by several points and bias the LR comparison against them.
- **Fix:** fp32 weights and `torch.autocast(bf16)` for both corpus and query embedding. The single-query model is barely affected by any variant (cos ≥ 0.998).

### C2. MEDIUM: the number of generated embeddings differs between training and inference (verified)

- **Where:** in training the number of steps equals the batch's gold count (`finetuning_data.py:102-108`, `GoldLengthGroupedBatchSampler`); at inference k is fixed (`phaseC_run.sh` / `grid_launch.sh` → `retrieval_inf.py --max_new_tokens`).
- **QAMPARI:** clean train has 1–8 golds (55% have 6–8); test has 5–8; inference always generates 5, so steps 5–7 are trained but never used, and 8-answer questions get 5 embeddings.
- **AmbigQA:** clean train has 2–5 golds (34% have 3–5); inference always generates 2.
- Round-robin aggregation of the k lists into the top 100 is never seen in training. The in-training `evaluate()` generates gold-count steps (`finetuning_multi.py:538,555`), unlike the retrieval evals.
- A design choice rather than a code bug, but it applies to every multi-query evaluation. **Suggested:** ablate k (dataset-typical gold count, max trained k) or train with a fixed k.

### C3. LOW: in-training eval acc/MRR for multi runs scores only the first embedding (verified)

- `finetuning_multi.py:523-549` calls `model.encoder` directly and pools, which equals generate step 0 (cos 1.0000), so the metric that would select `best_model` ignores steps 1..k-1. Not used by the grid (`--no_save_best_model`).

### C4. LOW (latent): inference ignores some settings saved in the checkpoint's `opt` (verified from code)

- `opt.eval_normalize_text`, `opt.chunk_length` (same root as R3) and `opt.pooling` are never read at inference. They match today (False, 512, last-token), but a run trained with `--eval_normalize_text` would mismatch silently. `force_causal`, `freeze_doc_encoder` and `use_lora` are honored through the model-class choice.

### Steps compared that match

| Step | Result |
| --- | --- |
| Training vs inference code, both pure fp32 | queries ≥ 0.99998 at every step (all 3 checkpoints); docs ≥ 0.99976 |
| nli (torch 2.8.0, transformers 4.57.0) vs div (torch 2.5.1, transformers 4.56.1) | both SDPA, same custom `modeling_qwen.py`; fp32 cos 1.00000, fp16 ≥ 0.99999. Embedding docs in div and queries in nli is harmless |
| Query text | training `question_text`, inference `question` then `question_text`: identical in every dev/test file; neither strips; double spaces treated the same |
| Query token ids | identical; EOS 151643 last; same custom tokenizer |
| Query max length (512 train vs 1024 inference) | no effect; queries are ~40 tokens |
| Single-query: left pad + explicit position ids (train) vs right pad + default positions (inference) | cos ≥ 0.99998 |
| Multi-query left-pad conversion and position ids | identical to the training collator |
| Document strings: `format_passage` vs pandas `title + ' ' + text` | 48 / 48 identical strings and token ids |
| pandas read of shards > 0 | column order unchanged; shard 1 (808,008 rows): 0 id/title/text mismatches |
| Passage length | 30k sampled per reduced corpus: max 366 / 382 tokens; R3 has no effect on the current corpus |
| Encoder used for documents | joint runs `model.encoder`, frozen-doc runs `doc_encoder`, as in training |
| `embedding` vs `encoder.embed_tokens` | identical (max diff 0) |
| Normalization | loss always L2-normalizes; inference normalizes |
| Checkpoint `opt` | grid: `force_causal=False`, `freeze_doc_encoder=False`, `use_lora=False`, `chunk_length=512` |
| `generate()` vs `forward` at sampling rate 1 vs in-training eval step 0 | cos 1.00000 |
| fp16 compute with fp32-loaded weights | ≥ 0.9996 queries, ≥ 0.9994 docs vs fp32; fp16 itself is not the problem |
| FAISS | `IndexFlatIP`, fp32 on GPU |

## 6. Design notes (not bugs)

- **Bidirectional attention.** The grid's model classes never pass `is_causal`, and the loaded Qwen2 defaults to `is_causal=False`. Each generation step re-encodes the whole prefix bidirectionally, so query tokens also attend to fed-back embeddings. This is identical in training and `generate()`, and matches the base retriever, but it is not the paper's causal-LM formulation.
- **Correlated randomness across ranks.** Python `random` and the CUDA RNG are seeded identically on both ranks, so scheduled-sampling masks and negative draws are correlated at the same step. Negligible.
- **Data observations.** `data/phaseA/ambigqa_cleandev500.jsonl` has flat `positive_ctxs`, so it only works with `--no-gold-id` (all scripts pass it). Some AmbigQA clean-dev questions list the same answer several times, which slightly inflates Recall but not MRecall.

## 7. Checked and correct

- **Model and loss:** Hungarian loss indexing and per-rank offsets with different k per rank; Hungarian matching on log-softmax equals matching on similarities; single-query labels; `VarsizeGather`/`Gather` gradients (mean-loss gradient, no double counting); position ids, mask growth and last-token pooling; example-major ordering of golds and negatives; teacher input detached, predicted input not; k=1 batches; SDPA mask with padding; tied embedding counted once. GPU parity check on the real model: left-padded `generate()` = per-query (cos 1.0000); right-padded docs = unpadded (1.0000); `forward()` at sampling rate 1 = `generate()` (1.0000).
- **Training loop:** `--train_data` override takes effect (hn runs load `clean_hn`, ~50% hard negatives); multi-mode sampler gives both ranks the same order, reshuffled each epoch; single-mode `RandomSampler` synced across ranks; scheduler steps once per step; `zero_grad` every step; loop stops exactly at `total_steps` and saves the last step; saved weights are the full gathered state dict after the optimizer step; precision guards pass; FSDP `use_orig_params` and `sync_module_states` on; freeze flags and L2-SP implemented correctly (not used by the grid).
- **Inference:** model-class selection and `strict=True` loading of FSDP full state dicts; frozen checkpoints embed with `.doc_encoder`, joint with `.encoder`; pooling and padding match training; passage formatting matches `format_passage` (0 differences); ids are strings end to end; per-shard top-k then global sort and truncation (E4 fixed in the working tree); round-robin over k lists of 500 gives the same top-100 as n_docs=100; `IndexFlatIP` exact; MRecall/Recall definitions and AmbigQA alias format; T3 fixed in the working tree (`--query_instruct_task` is read).
- **Data:** the clean build implements the 2026-09-27 decisions: 0 question and 0 id overlap among train/dev/test (build normalization); 0 repeated golds in train, gold counts recomputed, 519 single-gold AmbigQA examples dropped; dev sets exactly 500; `clean_hn` and `clean` identical except `hard_negative_ctxs`; 0 hard negatives equal a gold id or have a gold/answer-entity title; 0 contain an answer in a 5% sample; reduced corpora contain every base top-1,000 candidate and all 2,853 QAMPARI dev gold ids.
- **Grid tooling:** k = 5 (QAMPARI multi), 2 (AmbigQA multi), none (single) in queues and cheap-eval embedding shapes; hn/nohn data files correct; all 28 queued checkpoints exist; follower top-2 and prune top-3 agree, including ties; `RUN DONE` and drift grep patterns are delimited, so prefix run names don't collide.

## 8. Proposed actions

**Now (the grid is running):**

| # | Action | Addresses |
| --- | --- | --- |
| 1 | ~~Rank QAMPARI checkpoints on the 373 clean dev queries; consider `KEEP_TOP=0` for runs still training so nothing else is pruned on the leaky ranking~~ **DON'T DO IT (user, 2026-10-02)** | G1 |
| 2 | **No change (2026-10-02).** Decide inference numerics: fp32 weights under bf16 autocast (matches training; within ±0.4 of the current pipeline on the 3 checkpoints measured, so optional). Do **not** switch to plain fp32. Current numbers stay comparable to each other, since every checkpoint went through the same pipeline | R1, C1 |
| 3 | ~~Report QAMPARI test on the 487 non-near-duplicate queries alongside the full 531~~ **DON'T DO IT (user, 2026-10-02)** | G2 |
| 4 | ~~Free AmbigQA checkpoints not queued for full eval~~ **Done 2026-09-28:** steps 0/30/100/250 deleted from the 10 finished AmbigQA runs (~270 GB); the 2 not-yet-started runs still need it after they finish | G4 |

**Before the next round:**

| # | Action | Addresses |
| --- | --- | --- |
| 5 | **Done 2026-10-02 (QAMPARI v2, section 9).** Rebuild splits grouped by shared question OR shared gold set (Jaccard ≥ 0.8), strip articles in `norm_q`; drop train examples near-duplicating test | G1, G2, G5 |
| 6 | **Done 2026-10-02.** Hard negatives per slot, without replacement, topped up with random negatives | L1, L2 |
| 7 | **Done 2026-10-02** (fp32 only on overflow; 512-token passages to be switched on after the grid queue drains). Normalize in fp32 at inference; honor `--passage_maxlength` | R2, R3 |
| 8 | **Done 2026-10-02.** `parse_args()` and log `opt`; fix `iter_stats` default; make `phaseC_collect_scores.py` understand grid runs | L5, M1, G3 |
| 9 | **Done 2026-10-03** (fp32 reduce). Decide FSDP `reduce_dtype` (fp32) or document bf16 reduction | L12 |
| 10 | **Done 2026-10-03** (see C2 in section 9). Ablate inference k for multi-query (typical gold count, max trained k) | C2 |
| 11 | **Done 2026-10-02** (checks and warnings; see section 9). Read `chunk_length`, `eval_normalize_text` and `pooling` from the checkpoint `opt` at inference | C4, R3 |

**Can wait:** everything else in section 4.

## 9. Implementation status (2026-10-02)

Status per finding. **Done** = fixed in the working tree on `fsdp-clean-recipe` (not yet committed as of 2026-10-02); **DON'T DO IT** = explicitly declined by the user; **No change** = decided against by analysis; **Open** = not implemented yet. Code state before these fixes: git tag `grid-clean-v1`.

### Explicitly declined by the user (DON'T DO IT)

| Finding | Declined action |
| --- | --- |
| G1, G2 | Re-scoring the grid's QAMPARI checkpoints on the 373 clean dev / 487 clean test queries (section 8, actions 1 and 3). The leaks are fixed in the data instead (v2, below). |
| R6 | Making `eval.py` fail when `--selected-indices-file` is missing. |

### Done

| Finding | What was done | Where |
| --- | --- | --- |
| G1, G2, G5 | New QAMPARI splits, **v2**: split grouped by shared normalized question or gold-set Jaccard ≥ 0.8; train examples near-duplicating any test example removed; articles stripped in question normalization; gold-count rule on 1–8 *unique* golds (user's change: examples previously dropped for having more than 8 raw gold slots are now included). Test set unchanged; AmbigQA unchanged (links to v1). Verified by an independent checker. Full reproduction steps: `data_creation/CLEAN_V2.md`. | `data_creation/build_clean_v2.py`, `verify_clean_v2.py`, `build_clean_v2.sh`, `clean_v2_base_retrieval.sbatch`, `clean_v2_downstream.sbatch`; data in `data/training/clean_v2{,_hn}/`, `data/phaseA/*_cleanv2dev500.jsonl` |
| G3 (+ pruning record) | Score collector reports grid runs (and any later run family), labels every dev score with its dev set, and marks each step's checkpoint F / K / P / D. | `tools/phaseC_collect_scores.py` |
| G4 | AmbigQA checkpoints not queued for full eval deleted by hand (2026-09-28, ~270 GB); new `KEEP_STEPS` in the driver deletes them automatically after their cheap eval and drift (the launcher sets `KEEP_STEPS="150 400"` for AmbigQA). | `tools/phaseC_run.sh`, `tools/grid_launch.sh` |
| L1, L2 | Hard vs random drawn independently per negative slot; hard negatives sampled without replacement (capped at the pool size), the rest filled with random negatives. On 300 `clean_hn` examples: QAMPARI multi 286 / 298 examples now mixed (was 0), hard fraction 0.49; AmbigQA hard fraction 0.49; all 193 QAMPARI examples with a hard pool smaller than k get k negatives with no repeats. | `src/finetuning_data.py` (`Dataset`) |
| L5 | `parse_args()` (unknown flags fail the run; tested: a typo is rejected, every grid flag parses); `--continue_training`, a non-base `--model_path` or `--retriever_model_id` raise instead of being ignored; every option is printed at startup. | `src/options.py`, `finetuning_multi.py` |
| L11 | `opt.txt` is written to each run's own directory (`<output_dir><run_name>/opt.txt`) by rank 0. | `finetuning_multi.py`, `src/options.py` |
| M1 / L3 | `iter_stats=None` default with a fresh dict per call, in all six `forward()`s. | `src/inbatch.py` |
| M3 | Not a bug: `evaluate()` already runs under `@torch.no_grad()` (entry corrected in section 4). | — |
| R2 | Overflow guard: if any fp16 norm is non-finite, the array is normalized in fp32. Bit-identical otherwise, so evaluations still running are unaffected. | `gen_embed_new.py`, `retrieval_inf.py` (incl. the multi-query path), `src/inference_utils.py` |
| R3, C4 | `--passage_maxlength` is honored (default 1024 = what every evaluation so far used; it warns when the checkpoint's `chunk_length` differs, 512 for all grid checkpoints). Retrieval fails if the checkpoint's `eval_normalize_text` differs from `--normalize_text`; corpus embedding fails if the checkpoint was trained with normalized text. `pooling` needs no check: training also always uses last-token pooling. | `gen_embed_new.py`, `retrieval_inf.py`, `src/inference_utils.py` |
| R5 | The full-eval queue deletes earlier outputs of a tag before submitting it, so a stale file cannot pass the success check. | `tools/full_eval_queue.sh` |
| G11 | Scoring is retried up to 3 times from the retrieval output on disk, instead of a failed score forcing a new 2–3 h embedding; a final failure is logged as ERROR. | `tools/full_eval_queue.sh` |
| R3 (activation) | Corpus embedding now truncates passages at 512 tokens (training's `chunk_length`) for every checkpoint except the grid's (`grid_*`), which keep the 1024 they were all evaluated at; `PASSAGE_MAXLEN` overrides. Note: re-evaluating a pre-grid `phaseC_*` checkpoint would now use 512, not its original 1024. | `tools/gen_embed_ckpt.sbatch`, `tools/cheap_eval_ckpt.sbatch` |
| G7 | The driver logs `RUN FAILED (exit N)` on any non-zero exit (exit trap); the follower warns once when a run's driver failed (and still queues it if a restarted driver finishes), warns about steps without cheap-dev metrics, and warns instead of silently queuing nothing. | `tools/phaseC_run.sh`, `tools/grid_launch.sh` |
| M6 | The four per-step `print`s in the single-query forward (shapes and the loss, which forced a GPU sync) removed. | `src/inbatch.py` |
| L12 | FSDP gets an explicit `MixedPrecision(param_dtype=bf16, reduce_dtype=fp32, buffer_dtype=bf16)` policy, so gradients are reduce-scattered in fp32 like DDP and 1-GPU; training fails at startup if the policy is not fp32-reduce, and logs the plugin's settings. Verified 2026-10-03 with a 15-minute 2-GPU FSDP smoke run on `clean_v2_hn` (job 19106614): the logged policy is `MixedPrecision(param_dtype=bfloat16, reduce_dtype=float32, buffer_dtype=bfloat16)`, the other FSDP settings still come from the config (version 1, FULL_SHARD, single-unit wrap, SHARDED_STATE_DICT, use_orig_params, sync_module_states), loss falls normally (3.98 → 0.50 over 100 steps), and the run's own `opt.txt` records the hard-negative data and ratio. | `finetuning_multi.py` (`build_accelerator`), `accelerate_config_2gpu_fsdp_nooffload.yaml` (comment) |
| C2 | Studied 2026-10-03 (one checkpoint per dataset, full corpus; tables in `grid_results_2026-10-02.md`): QAMPARI is best at k = 5 with **RRF** (+6.0 test MRecall@100 over round-robin); AmbigQA is best at k = 1–2 with round-robin, and RRF hurts at k ≥ 3. A design choice per dataset, not a code fix; the default aggregation is unchanged. Offline scoring of any k and aggregation from one retrieval: `retrieval_inf.py --save_per_step`, `tools/kagg_*`. | `retrieval_inf.py`, `tools/kagg_study.sh`, `kagg_retrieve.sbatch`, `kagg_eval.py`, `kagg_eval.sbatch` |
| — | Tooling for v2: `grid_launch.sh` takes `DATA_ROOT`, `DEV_TAG`, `PREFIX` (own run names and full-eval queue file); `full_eval_queue.sh` takes `QUEUE`; `phaseA_build_reduced_corpus.py` merges `build_info.json` instead of overwriting it. Defaults reproduce the grid. | `tools/` |

Shell scripts that may be running (`full_eval_queue.sh`) were replaced atomically (new file + rename), so the running workers kept executing the old version.

R3 is now active for new checkpoints (see "R3 (activation)" above); the grid's checkpoints keep 1024 so the grid's evaluations stay consistent.

### No change (by analysis)

| Finding | Reason |
| --- | --- |
| R1, C1 | Current inference numerics are within 0–0.4 points of training's own (bf16 autocast) on the 3 checkpoints measured; plain fp32 loading would lose 3.4 points at LR 1e-6. Optional: fp32 weights under bf16 autocast, which matches training exactly. |

### Open

Not planned, by decision of 2026-10-03 (negligible, unused paths, or failing loudly): L7, L8, L9, C3, L4, L6, L10, L13, M4, M5, R4, G6, G9, G10.

| Finding | Severity | Note |
| --- | --- | --- |
| L7 | LOW | Weight decay on norms, biases and embeddings (negligible at current LR). |
| L8 | LOW | First optimizer step at LR 0; `--lr_min_ratio` warmup quirk. |
| L9 | LOW | In-training multi-query eval pads negatives by copying and resamples them each call. |
| C3 | LOW | In-training eval acc/MRR scores only the first embedding (not used for selection). |
| L4 | LOW | Rank 1 logs unaveraged metrics. |
| L6 | LOW | FSDP optimizer state would be rank 0's shard only (grid passes `--no_save_optimizer`). |
| L10 | LOW | Broken `checkpoint/latest` symlink. |
| L13 | LOW | `--negative_ctxs > 1` fails (loudly). |
| M4, M5 | LOW | Broken legacy `InBatch`; `INFRetriever` pooling assumes right padding (unused paths). |
| R4 | LOW | Header handling for shards > 0 in `gen_embed_new.py` (harmless with current ids). |
| G6 | LOW | `END` in the full-eval queue can stop it before earlier items. |
| G9 | LOW | ~18 passages still CSV-quoted. |
| G10 | LOW | Driver restarts duplicate drift records. |
