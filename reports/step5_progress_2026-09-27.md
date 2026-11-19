# Multi-query retrieval with joint doc-encoder training: reproduction, root cause, clean recipe

*Status as of 2026-09-27. Model: `infly/inf-retriever-v1-1.5b`, jointly fine-tuned as query and document encoder. Datasets: QAMPARI (k = 5 query embeddings) and AmbigQA (k = 2).*

## 1. Summary

1. **Why the September reproduction collapsed.** The April/May "with_detach" results came from plain 2-GPU DDP with pure-bf16 weights and optimizer state. Most Adam updates rounded to zero in bf16, so only about 7% (QAMPARI) / 10% (AmbigQA) of weights ever changed. The September runs used FSDP, which upcasts to fp32 master weights. Every weight then trained, and QAMPARI collapsed to 0.00 MRecall@100.
2. **The real cause of that collapse is a label leak in the QAMPARI training data, not precision.** Every QAMPARI gold passage has a trailing space on its title and a leading space on its text. No negative and no corpus passage has either. After joining title and text, every gold carries an extra whitespace token (`ĠĠ`), so a fully trained model learns to spot positives by whitespace. Retrieval against the real corpus then fails. Pure bf16 barely moved the weights, so it could not exploit the leak much, which is why the April recipe "worked". AmbigQA's data has no leak.
3. **With the leak fixed, clean training works at the original hyperparameters.** Clean single-query QAMPARI (LR 1e-5, step 2500) reaches **42.56** MRecall@100 on test, against **26.37** for the old recipe. Multi-query no longer collapses at any LR.
4. **The clean multi-query recipe does not yet beat clean single-query on QAMPARI.** Best multi-query so far: **33.71** on test, against **42.56**. The two were evaluated at different steps (1000 vs 2500); step-matched evaluations are running.
5. **On AmbigQA, multi-query and single-query are within noise.** Best multi-query: **76.66** on test (hard negatives, step 400), against the single-query bar of **76.54**. That is one seed, and dev does not separate them reliably.

## 2. Goal and success criterion

Find a clean multi-query training recipe (no precision bugs) that, on both datasets:

1. matches or beats the old recipe ("fallback", pure-bf16 DDP): QAMPARI 26.37, AmbigQA 75.09;
2. beats the best single-query baseline trained under the same clean setup, counting new runs: currently QAMPARI **42.56**, AmbigQA **76.54**;
3. is stable over 2 seeds.

Test sets are used only for reporting; dev sets select checkpoints.

## 3. What was done, in order

### 3.1 Investigation of the September reproduction

Compared the April/May checkpoints, code, hyperparameters and launch setup with the September runs.

| Finding | Effect |
| --- | --- |
| April runs: 2-GPU DDP, weights loaded in bf16 and never upcast, AdamW state in bf16 | ~7% / 10% of weights change; layernorms never change |
| September runs: FSDP with mixed precision, which upcasts to fp32 master weights | all weights train; QAMPARI 0.00, AmbigQA 70–72.7 (untrained base: 72.55) |
| accelerate steps the LR scheduler once per process per step | 2-GPU runs decayed the LR twice as fast as intended |
| Unseeded Python `random`; each rank drew its own data order | runs not reproducible, ~1 point run-to-run noise |

### 3.2 Fallback recipe (kept as the documented safety net)

Re-ran the April setting (pure-bf16 2-GPU DDP) and reproduced the published numbers.

| Dataset | Checkpoint | MRecall@100 | Recall@100 |
| --- | --- | --- | --- |
| QAMPARI | fallback DDP rerun, step 250 | **26.37** | 58.62 |
| QAMPARI | original April `with_detach`, step 250 | 25.42 | 56.90 |
| AmbigQA | fallback DDP rerun, step 750 | **75.09** | 87.43 |
| AmbigQA | original May `multi_hungarian`, step 750 | 74.12 | 87.27 |

Documented in `RECIPE_FALLBACK_bf16_ddp.md`; code frozen at git tag `fallback-bf16-ddp`.

### 3.3 Clean training code (branch `fsdp-clean-recipe`, pushed)

- Mixed precision everywhere: fp32 weights and AdamW state, bf16 autocast, on 1 GPU, DDP and FSDP.
- The LR scheduler steps exactly once per training step.
- Seeded data order, consistent across ranks.
- Guards that fail the run if trainable weights or optimizer state are not fp32.
- Tests for scheduler stepping and sampler sharding; both pass on 1 GPU, DDP and FSDP.

### 3.4 Phase B: clean baselines (before the leak was known)

2-GPU FSDP, 50 queries per GPU, LR 1e-5; single- and multi-query with identical settings.

| Dataset | Mode | Test MRecall@100 by step | Weights changed |
| --- | --- | --- | --- |
| QAMPARI | multi | 250: 0.00 · 500: 0.00 · 1000: 0.00 · 2500: 0.00 | ~100% |
| QAMPARI | single | 250: 0.00 · 500: 10.73 · 1000: 0.00 · 2500: 0.00 | ~100% |
| AmbigQA | multi | 30: 69.89 · 150: 73.28 · 400: 71.70 | ~100% |
| AmbigQA | single | 30: 72.07 · 150: **76.54** · 400: 76.06 | ~100% |

A control (the old bf16 single-query checkpoint through the same eval scripts) reproduced its known 22.41, so the collapse was real, not an evaluation bug.

### 3.5 Phase A: a cheap evaluation for screening

Full-corpus evaluation embeds ~25.9M passages (~75 GB, 1.5–2.5 h per checkpoint). As a screen, I built a reduced corpus per query set: the base retriever's top-1,000 per query, plus golds, plus 100k shared random passages (about 0.4–0.7M passages). It was validated on 29 checkpoints with known full test scores.

| Dataset | Spearman (cheap vs full MRecall@100) | Pairs ordered correctly |
| --- | --- | --- |
| QAMPARI | 0.956 | 75 / 77 |
| AmbigQA | 0.683 | 53 / 70 |

Cheap eval was adopted for QAMPARI screening and collapse detection. For AmbigQA it cannot discriminate, so AmbigQA candidates get full-corpus dev and test.

Dev sets: QAMPARI 500 dev queries, AmbigQA 300 dev queries.

### 3.6 Phase C, round 1: LR sweep, which exposed the leak

Lower LRs (3e-6, 1e-6, 3e-7) first looked like a fix: QAMPARI multi-query at LR 1e-6 scored 55.00 on cheap dev at step 250. Longer training showed the collapse was only delayed (55.00 → 6.00 → 0.40 at steps 250 / 500 / 1000). Meanwhile, the in-training eval (in-batch accuracy) stayed at ~98%.

Diagnosis:

1. The collapsed checkpoint still retrieved on-topic passages, but gold articles vanished: gold-article recall fell 84.5 → 33.9 → 6.2.
2. The document formatting of training and corpus embedding matched (`title + " " + text`).
3. Whitespace statistics over the training data:

| Passages (first 3,000 examples) | Title ends with a space | Text starts with a space |
| --- | --- | --- |
| QAMPARI train golds (36,522) | 100% | 100% |
| QAMPARI train negatives (75,000) | 0% | 0% |
| QAMPARI dev golds (36,476) | 100% | 100% |
| AmbigQA train golds (7,714) | 0% | 0% |

4. Tokenization confirmed the shortcut: a training gold gets `['John', 'ĠMcG', 'raw', 'ĠĠ', 'ĠJohn', …]`, while the same passage in the corpus gets no `ĠĠ`.

**Fix** (in `training/inf_retriever/src/finetuning_data.py`; not yet committed):

- a new `format_passage()` strips title and text before joining. All 2,926 dev golds found in the corpus now match the corpus strings byte for byte.
- a second bug fixed on the way: the text normalizer always ran during training, because the code tested the imported module (always true) instead of the `normalize` flag. Corpus and query embedding never normalize, so training now matches inference.

Runs trained before the fix are marked **pre-fix** below.

### 3.7 Phase C, fixed loader and round 2 (hard negatives)

- **Round 1, repeated with the fix:** QAMPARI multi- and single-query at LR 1e-5 and 1e-6; AmbigQA multi-query at LR 1e-6.
- **Round 2, hard negatives:** 30 mined negatives per example, from the base retriever's top-100. Golds, answer-containing passages and the top 3 are excluded. Half of each example's negatives are hard. Run for both modes on both datasets.

Common setup: 2-GPU FSDP, 50 queries per GPU, temperature 0.05, passage length 512. QAMPARI: 2,500 steps with 100 warmup. AmbigQA: 400 steps with 15 warmup. Loss: Hungarian-matched contrastive for multi-query, contrastive for single-query.

## 4. Results

MRecall@100 is the primary metric. QAMPARI is scored by gold passage IDs, AmbigQA by answer strings. Cheap dev = reduced-corpus dev. Full = full-corpus.

### 4.1 QAMPARI, full corpus

| Checkpoint | Mode | Loader | Dev MRecall / Recall | Test MRecall / Recall |
| --- | --- | --- | --- | --- |
| single, LR 1e-5, step 2500 | single | fixed | 69.00 / 87.47 | **42.56** / 69.51 |
| multi, LR 1e-6, step 1000 | multi | fixed | 53.60 / 78.80 | 33.71 / 64.44 |
| multi, LR 1e-5, step 1000 | multi | fixed | 51.80 / 77.24 | 27.12 / 58.43 |
| multi, LR 1e-6, step 250 | multi | pre-fix | 43.40 / 73.99 | 26.55 / 58.16 |
| fallback (pure-bf16 DDP), step 250 | multi | reference | 42.80 / 73.29 | 26.37 / 58.62 |
| base retriever (untrained) | single | reference | – | 12.24 / 46.51 |

### 4.2 AmbigQA, full corpus

| Checkpoint | Mode | Loader | Dev MRecall / Recall | Test MRecall / Recall |
| --- | --- | --- | --- | --- |
| multi + hard neg., LR 1e-6, step 400 | multi | fixed | 79.67 / 90.56 | **76.66** / 88.48 |
| single (phase B), LR 1e-5, step 150 | single | pre-fix | 78.00 / 89.02 | 76.54 / 88.01 |
| multi, LR 1e-6, step 150 | multi | pre-fix | 80.33 / 90.78 | 76.06 / 88.16 |
| fallback (pure-bf16 DDP), step 750 | multi | reference | 80.33 / 90.86 | 75.09 / 87.43 |
| multi + hard neg., LR 1e-6, step 150 | multi | fixed | 79.67 / 90.71 | 74.73 / 87.69 |
| multi, LR 3e-6, step 150 | multi | pre-fix | 79.33 / 90.04 | 74.61 / 87.25 |
| base retriever (untrained) | single | reference | – | 72.55 / 85.97 |

AmbigQA "pre-fix" runs differ from fixed ones only in the text-normalization bug; their data has no whitespace leak.

### 4.3 QAMPARI, cheap dev MRecall@100 at every saved step

Cheap dev favors single-query and hard-negative training, so compare within a mode only.

| Run | Mode | Loader | LR | step 250 | step 500 | step 1000 | step 2500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multi | multi | fixed | 1e-5 | 60.40 | 62.60 | 67.00 | 64.40 |
| multi | multi | fixed | 1e-6 | 56.40 | 62.20 | 65.60 | 61.20 |
| multi + hard neg. | multi | fixed | 1e-5 | 70.60 | 75.20 | 77.60 | 76.80 |
| multi | multi | pre-fix | 1e-6 | 55.00 | 6.00 | 0.40 | – |
| multi | multi | pre-fix | 3e-6 | 33.40 | 0.20 | 0.20 | – |
| multi | multi | pre-fix | 3e-7 | 39.20 | 54.60 | 33.80 | – |
| single | single | fixed | 1e-5 | 68.20 | 69.80 | 75.00 | 80.20 |
| single | single | fixed | 1e-6 | 64.20 | 68.20 | 70.80 | 72.40 |
| single + hard neg. | single | fixed | 1e-5 | 79.20 | 80.20 | 81.80 | 87.80 |
| single | single | pre-fix | 1e-6 | 66.60 | 52.20 | 8.40 | 1.60 |
| single | single | pre-fix | 3e-7 | 61.80 | 65.60 | 64.20 | – |

Blank cells: pre-fix runs stopped after the leak was found. Reference points: fallback 50.80, base 24.60.

### 4.4 AmbigQA, cheap dev MRecall@100 at every saved step

Not discriminative on AmbigQA (all 72–76; one question = 0.33 points). Kept for collapse detection only.

| Run | Mode | Loader | LR | step 30 | step 100 | step 150 | step 250 | step 400 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi | multi | fixed | 1e-6 | 74.00 | 74.67 | 74.33 | 75.00 | 74.67 |
| multi + hard neg. | multi | fixed | 1e-6 | 74.67 | 75.67 | 75.67 | 76.00 | 76.33 |
| multi | multi | pre-fix | 1e-6 | 74.00 | 74.67 | 74.67 | 75.00 | 75.00 |
| multi | multi | pre-fix | 3e-6 | 73.00 | 74.33 | 74.00 | 73.67 | 72.00 |
| multi | multi | pre-fix | 3e-7 | 75.33 | 75.00 | 74.33 | 74.00 | 74.33 |
| single + hard neg. | single | fixed | 1e-5 | 73.00 | 74.67 | 75.33 | 74.33 | 76.00 |
| single | single | pre-fix | 1e-6 | 74.67 | 74.67 | 74.67 | 75.00 | 75.33 |
| single | single | pre-fix | 3e-7 | 75.33 | 75.00 | 76.00 | 75.33 | 75.00 |

### 4.5 Parameter drift (share of weights changed / relative L2 distance from step 0)

| Setting | Changed | Relative distance |
| --- | --- | --- |
| fallback (pure bf16) | ~7% (QAMPARI) / ~10% (AmbigQA) | 1.4–1.7e-3 |
| clean, LR 1e-5 (any mode) | ~100% | 2.4e-3 (step 250) → 7.0e-3 (step 2500) |
| clean, LR 1e-6 | 83–87% | 7.6e-4 (step 250) → 1.5e-3 (step 2500) |
| clean, LR 3e-7 | 82–86% | 1.2e-4 → 8.1e-4 |

## 5. Caveats on the evaluation tools

1. **In-training eval (in-batch accuracy / MRR) is saturated and cannot select checkpoints.** The untrained base already scores 96.8%; every run stays at 98.6–99.4%, while full test ranges 12 → 42. It ranks each gold against ~100 easy random passages.
2. **Cheap eval overrates models that drift far from the base retriever.** Only 40–48% of a fine-tuned model's full-corpus top-100 lies inside the reduced corpus. The lowest coverage (multi, LR 1e-5: 40.4%) is exactly the checkpoint cheap eval overrated (cheap 67.00 vs 65.60 for LR 1e-6, but full test 27.12 vs 33.71). Drift grows with training steps, so cheap eval may favor later checkpoints within a run. On the 5 post-fix QAMPARI checkpoints with full dev, cheap dev orders 9 of 10 pairs the same way.
3. **AmbigQA dev (300 queries) is noisy.** The fallback and the single-query bar rank in opposite orders on dev and test. Two checkpoints with identical full dev (79.67) differ by 1.9 points on test.
4. **Mode comparisons so far are not step-matched.** QAMPARI multi was evaluated at step 1000 and single at step 2500 (each chosen by best cheap dev). Matched-step evals are running.
5. **QAMPARI checkpoints between steps 1000 and 2500 were never saved.** Several multi-query runs peak on cheap dev at step 1000, and single-query runs are still rising at step 2500. A better checkpoint may exist in between or beyond.

## 6. Still running

All training is finished; only full-corpus evaluations remain, run one checkpoint at a time per dataset.

| Dataset | Checkpoint | Purpose |
| --- | --- | --- |
| QAMPARI | single + hard neg., LR 1e-5, step 2500 (running) | pair with multi + hard neg. step 2500; may raise the bar |
| QAMPARI | single, LR 1e-5, step 1000 | pair with multi LR 1e-5 step 1000 |
| QAMPARI | multi, LR 1e-5, step 2500 | pair with single LR 1e-5 step 2500 |
| QAMPARI | multi + hard neg., LR 1e-5, steps 2500 and 1000 | hard-negative pairs |
| QAMPARI | single + hard neg., LR 1e-5, step 1000 | hard-negative pair |
| QAMPARI | single, LR 1e-6, steps 1000 and 2500; multi, LR 1e-6, step 2500 | LR 1e-6 pairs |
| AmbigQA | single + hard neg., LR 1e-5, steps 150 and 400 | pairs with the multi + hard neg. checkpoints |
| AmbigQA | multi (fixed), LR 1e-6, step 150 | effect of the normalization fix |
| AmbigQA | multi LR 1e-6 s400, multi LR 3e-7 s150, single LR 1e-6 / 3e-7 s150 (pre-fix) | completes the earlier LR sweep |

Estimated time: QAMPARI 15–20 h, AmbigQA 10–15 h.

## 7. Code and artifacts

| Item | Where | State |
| --- | --- | --- |
| Clean training loop, tests, phase B tools | branch `fsdp-clean-recipe` (commits `bd97733`–`56f9581`) | committed, pushed |
| Fallback recipe | `RECIPE_FALLBACK_bf16_ddp.md`, tag `fallback-bf16-ddp` | committed |
| Loader fix (`format_passage`, normalize flag) | `training/inf_retriever/src/finetuning_data.py` | **uncommitted** |
| New flags: `--no_save_optimizer`, `--freeze_norms`, `--freeze_embeddings`, `--l2sp_decay` | `finetuning_multi.py`, `src/options.py`, `src/utils.py` | **uncommitted** |
| Phase A/C tools: cheap eval, dev sets, reduced corpora, phase C driver, full-eval queue, hard-negative miner, score collector, coverage check | `training/inf_retriever/tools/` | **uncommitted** |
| All scores (regenerable) | `results/phaseC/all_scores.md` via `tools/phaseC_collect_scores.py` | generated |
| Hard-negative training data | `data/training/filtered_hn/{qampari,ambigqa}/` | generated |
