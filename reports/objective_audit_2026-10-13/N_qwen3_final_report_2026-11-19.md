# Qwen3-Embedding-0.6B in F's frozen QAMPARI setting + the legacy one-label step: final report (session N, 2026-11-19)

Branch `qwen3-retriever` (worktree `.claude/worktrees/n-nemotron`, built on `nemotron-retriever` 0d61451, not merged).
Companion report: `N_nemotron_final_report_2026-11-15.md` (same question, Nemotron-3.5-Embed-1B).

## Question

Session F's frozen-doc-encoder QAMPARI recipe, rerun with `Qwen/Qwen3-Embedding-0.6B` in place of
`infly/inf-retriever-v1-1.5b`, as a second replication of the Nemotron study. The metric we care about most is the
**gap = multi large third - single large third** (inf-retriever: 0.0 / 0.0). Round 1 is F's recipe; round 2 repeats the
Nemotron fixes (lr 5e-6, hard negatives mined by base Qwen3, learned input / output layers).

Part 2 (asked 2026-11-16): the legacy pipeline's one-gold step (`train_distributed.py --mix_one_label_shuffled`) used
*other, randomly drawn* questions for the extra update, while F's `--one_label_step` uses the *current batch's*
questions. New flag `--one_label_shuffled` restores the legacy behaviour; it is tested on the best multi runs and their
single runs, with inf-retriever and Qwen3, including F's causal cap run (large third 27.38).

## Setup shared by every run

- As the Nemotron report: S0 split (`data/training/clean_v3_hn_s0`), `--negative_hard_ratio 0.5`, frozen document
  encoder, 10k steps, temperature 0.05, `--one_label_step`, F's `f_run.sh` pipeline, lr 1e-5, hard negatives mined by
  base inf-retriever, unless a row says otherwise.
- Single = one query embedding, 2 updates per step. Multi = Hungarian loss, k = 5, *linear* (scheduled-sampling rate
  0 -> 1) or *cap* (`--sampling_saturate_frac 0.1 --sampling_cap 0.8`). Multi runs: 2 H200 x per-GPU batch 50.
- **Qwen3 singles ran on 1 H200 x batch 100** (same global batch and gathered negative pool): Torch's
  low-GPU-utilization policy cancelled the 2-GPU Qwen3 singles at ~2h15m (steps 9425 / 8075). Their truncated
  results are rows 14-15. inf single runs are F's (2 GPUs).
- Qwen3: causal backbone, last-token pooling, 1024-d; queries `Instruct: <task>\nQuery:<q>`, passages plain; corpus =
  base Qwen3 embeddings of the 25.9M-passage corpus (`wikipedia_embeddings/qampari/qwen3emb06_frozen_base_512`).
  Trains in `div.ext3` (transformers 4.56), retrieves in `nli` (as inf-retriever).
- Selection: test at the dev-selected step (full S0 dev, 2459 questions); MRecall@100; multi fused with RRF (c = 60)
  over the k = 5 lists, round robin (RR) also reported. Thirds = the paper's large / small distance thirds of test
  (168 / 180 questions; one question = 0.6 / 0.56 points), from `tools/fusion_eval.py`. A difference under 2 points
  is a tie.

## Part 1: Qwen3 results

Gap = multi large third minus the large third of the single run in the same block (RR gap uses the same single).

| # | mode | setting | step | dev | test | large / small (RRF) | gap RRF | gap RR |
|---|---|---|---|---|---|---|---|---|
| 1 | - | base Qwen3, no training | - | 8.13 | 6.40 | 1.19 / 15.00 | | |
| | **Round 1: F recipe** | | | | | | | |
| 2 | single | 1 GPU | 10000 | 39.04 | 18.27 | 14.29 / 29.44 | | |
| 3 | multi | linear | 9000 | 36.27 | 13.37 | 10.71 / 21.11 | **-3.6** | -4.8 |
| 4 | multi | cap | 10000 | 36.23 | 14.88 | 10.71 / 22.22 | **-3.6** | -4.8 |
| | **Round 2a: lr 5e-6** | | | | | | | |
| 5 | single | lr 5e-6, 1 GPU | 9000 | 35.10 | 14.12 | 10.12 / 23.33 | | |
| 6 | multi | linear, lr 5e-6 | 10000 | 32.49 | 9.98 | 5.95 / 16.67 | **-4.2** | -6.0 |
| 7 | multi | cap, lr 5e-6 | 10000 | 32.49 | 11.11 | 5.36 / 20.00 | **-4.8** | -5.4 |
| | **Round 2b: hard negatives mined by base Qwen3** | | | | | | | |
| 8 | single | Qwen3 HN, 1 GPU | 10000 | 38.23 | 16.01 | 13.69 / 25.56 | | |
| 9 | multi | linear, Qwen3 HN | 9000 | 35.58 | 13.75 | 12.50 / 20.56 | **-1.2** | -4.8 |
| 10 | multi | cap, Qwen3 HN | 10000 | 36.80 | 13.94 | 11.31 / 21.67 | **-2.4** | -5.4 |
| | **Round 2c: learned projection layers (`--io_proj`, identity init)** | | | | | | | |
| 11 | single | output layer only, 1 GPU | 10000 | 38.80 | 17.33 | 15.48 / 25.00 | | |
| 12 | multi | linear, input + output layers | 7000 | 35.54 | 14.50 | 11.31 / 24.44 | **-4.2** (-3.0 vs row 2) | -5.4 |
| 13 | multi | cap, input + output layers (rerun `_v2`) | 6000 | 33.63 | 13.37 | 11.31 / 21.67 | **-4.2** (-3.0 vs row 2) | -7.1 |
| | **Truncated 2-GPU singles (cancelled; for reference)** | | | | | | | |
| 14 | single | 2 GPU, stopped at 9425 | 9000 | 39.32 | 17.70 | 14.29 / 27.78 | | |
| 15 | single | 2 GPU, lr 5e-6, stopped at 8075 | 8000 | 33.47 | 13.18 | 7.74 / 23.89 | | |

Row 13's dev peaked at step 6000 and fell after (s7000 12.24, s10000 10.36 test).

### Where the multi loss comes from: test MRecall@100, RRF over the first k embeddings

| run | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| linear (3) | 16.95 | 12.05 | 12.43 | 11.86 | 13.37 |
| cap (4) | 16.38 | 13.18 | 13.18 | 13.94 | 14.88 |
| linear lr 5e-6 (6) | 13.94 | 11.30 | 10.73 | 9.42 | 9.98 |
| cap lr 5e-6 (7) | 14.12 | 11.86 | 10.36 | 11.30 | 11.11 |
| linear Qwen3 HN (9) | 16.20 | 12.62 | 12.05 | 11.11 | 13.75 |
| cap Qwen3 HN (10) | 16.01 | 13.56 | 13.18 | 13.37 | 13.94 |
| linear io_proj (12) | 16.57 | 12.99 | 12.43 | 14.69 | 14.50 |
| cap io_proj (13) | 16.20 | 11.11 | 10.36 | 11.68 | 13.37 |
| (inf linear, for comparison) | 34.09 | 29.94 | 34.09 | 33.52 | 33.90 |

### Geometry (test): pairwise cosines, normalized as (pair - random-pair baseline) / (1 - baseline)

Gold = between the gold clusters of a question (first passage per cluster) in the frozen doc space; predicted =
between the k = 5 generated query embeddings. Output: `results/phaseC/fusion/geom_pairwise_cos.json`.

| retriever | gold all (large / small) | predicted linear / cap | Spearman(predicted, gold) per question |
|---|---|---|---|
| inf | 0.44 (0.31 / 0.59) | 0.77 / 0.70 | 0.00 / 0.13 |
| Nemotron | 0.39 (0.29 / 0.49) | 0.65 / 0.56 | -0.18 / -0.29 |
| Qwen3 | 0.44 (0.33 / 0.56) | 0.58 / 0.55 | -0.42 / -0.38 |

After normalization, the gold spread is the same for the three retrievers. The generated embeddings are spread
out about the same on the large and small thirds for every retriever. inf's are the most collapsed (0.70-0.77) and
inf is the one with no gap. Qwen3's and Nemotron's are more spread out (0.55-0.65), and with them the questions
whose golds are close together get the more spread-out query embeddings (negative Spearman).

### Findings (Part 1)

1. **Qwen3 replicates the Nemotron result.** Every Qwen3 multi run is below its single run on the large third with
   RRF (-1.2 to -4.8) and with RR (-4.8 to -7.1). It is also 2.1-4.9 points below it on the full test set.
   inf-retriever: 0.0 / 0.0.
2. **lr 5e-6 is worse everywhere** (single -4.2 test, multi -3.4 / -3.8 test), as with Nemotron.
3. **Qwen3-mined hard negatives give the smallest gaps** (-1.2 / -2.4 RRF). This is a tie with round 1 (2 questions
   each), and mostly because the single run drops (13.69 vs 14.29). The multi large thirds rise by only 1-3
   questions, and the RR gaps are unchanged (-4.8 / -5.4).
   Qwen3 HN data: 29.6 per question, 98.0% at the cap of 30 (inf-mined: 29.0, 95.4%); 20.9% of the Qwen3 ids are also
   in the inf-mined list of the same question (Nemotron vs inf: 23.4%).
4. **Projection layers do not help** (-4.2 / -4.2 RRF, -5.4 / -7.1 RR). The output-only single gains on the large
   third (15.48), which makes the gap against it wider.
5. **Per-k:** as with Nemotron, k = 1 alone (13.9-17.0) is the best setting in every Qwen3 multi run, 1.5-4.0
   points above k = 5. Adding the second list costs 2.3-5.1 points. With inf-retriever, k = 5 is within 1.5 of k = 1 (linear 0.2, cap 1.5).
6. Base Qwen3 is far below base inf-retriever (6.40 vs 24.48 test). Its trained singles reach only half of inf's (18.27 vs 35.03).
   Our Qwen3 encoding equals the model card's sentence-transformers path (`tools/qwen3_check.py`, cosine ~1), so this
   is the model, not the pipeline. The gap is relative to each retriever's own single run, so it is still comparable.

## Part 2: legacy one-label step (`--one_label_shuffled`)

Same runs as before with the flag added; everything else unchanged (seed 0 unless noted). Gap = vs the single run with
the same one-label variant (current vs legacy).

| # | retriever | mode | one-label step | step | dev | test | large / small (RRF) | gap RRF | large RR | gap RR |
|---|---|---|---|---|---|---|---|---|---|---|
| | **inf, causal (`--force_causal`)** | | | | | | | | | |
| 16 | inf | single causal | current (F) | 10000 | 51.16 | 35.22 | 25.00 / 49.44 | | | |
| 17 | inf | cap causal | current (F) | 10000 | 50.63 | 33.52 | 27.38 / 44.44 | **+2.4** | 24.40 | -0.6 |
| 18 | inf | single causal | legacy | 10000 | 52.91 | 36.72 | 26.79 / 51.67 | | | |
| 19 | inf | cap causal | legacy | 10000 | 50.55 | 31.64 | 23.21 / 43.33 | **-3.6** | 22.62 | -4.2 |
| 20 | inf | cap causal | legacy, seed 1 | 10000 | 49.94 | 31.64 | 23.21 / 43.89 | **-3.6** | 22.62 | -4.2 |
| | **inf, bidirectional** | | | | | | | | | |
| 21 | inf | single | current (F) | 10000 | 51.93 | 35.03 | 25.00 / 50.00 | | | |
| 22 | inf | linear | current | 9000 | 50.83 | 33.90 | 25.00 / 45.56 | **0.0** | 23.21 | -1.8 |
| 23 | inf | cap | current (F) | 10000 | 50.71 | 33.15 | 25.00 / 43.89 | **0.0** | 21.43 | -3.6 |
| 24 | inf | single | legacy | 10000 | 53.11 | 36.35 | 26.79 / 50.00 | | | |
| 25 | inf | linear | legacy | 8000 | 50.39 | 34.09 | 22.62 / 48.33 | **-4.2** | 21.43 | -5.4 |
| 26 | inf | cap | legacy | 6000 | 49.69 | 34.46 | 22.62 / 47.78 | **-4.2** | 23.21 | -3.6 |
| | **Qwen3** | | | | | | | | | |
| 27 | Qwen3 | single, 1 GPU | current | 10000 | 39.04 | 18.27 | 14.29 / 29.44 | | | |
| 28 | Qwen3 | cap | current | 10000 | 36.23 | 14.88 | 10.71 / 22.22 | **-3.6** | 9.52 | -4.8 |
| 29 | Qwen3 | single, 1 GPU | legacy | 10000 | 39.45 | 17.14 | 15.48 / 24.44 | | | |
| 30 | Qwen3 | cap | legacy | 9000 | 35.95 | 13.56 | 8.93 / 21.67 | **-6.5** | 9.52 | -6.0 |

Row 26: dev ties at 6000 and 8000 (49.69); s8000 test 33.15, large / small 22.02 / 45.00.
F's other seeds of row 17 (current one-label step) scored 23.81 (seed 1) and 24.40 / 25.60 (partial seeds) on the
large third.

Per-k (RRF, first k lists, test): row 19 32.77 / 31.83 / 31.26 / 31.45 / 31.64; row 20 34.84 / 26.93 / 29.00 / 32.02 /
31.64; row 25 35.22 / 32.96 / 33.33 / 32.20 / 34.09; row 26 35.22 / 32.77 / 31.64 / 32.39 / 34.46; row 30 16.76 /
13.94 / 13.37 / 13.37 / 13.56.

### Findings (Part 2)

7. **The legacy step helps single runs.** inf: test +1.3 / +1.5, large third +1.8 (3 questions) for both the
   bidirectional and causal singles. Qwen3: the large third rises by 1.2, but test falls by 1.1.
8. **It does not help multi runs, and it widens every gap.** inf multi large thirds fall from 25.00-27.38 to
   22.62-23.21, so the RRF gaps go from 0.0 / 0.0 / +2.4 to -4.2 / -4.2 / -3.6. Qwen3 cap goes from -3.6 to -6.5.
   inf multi test scores are a tie (31.64-34.46 vs 33.15-33.90). Only the large third moves, and it moves down.
9. **F's 27.38 was a lucky seed.** F's reseeds of row 17 gave 23.81-25.60. The legacy causal cap gives 23.21 at both
   seeds, which is 1-2 points below F's seed average. The +2.4 gap of row 17 is therefore not a stable effect.
10. With the legacy step, k = 1 is 0.8-3.2 points above k = 5 in the inf multi runs (current step: 0.2 / 1.5), so they
    move toward the Qwen3 / Nemotron pattern, where fusing the later lists loses fully covered questions.

Bottom line: with the frozen Qwen3 document space, as with Nemotron, no fix brings the multi large third up to the
single run's. The legacy one-gold step on random other questions is not the missing ingredient. It improves single
retrieval and makes the multi-vs-single gap on the large third worse for both retrievers.

## Implementation (branch `qwen3-retriever`)

| commit | what |
|---|---|
| 70593a3 | `--retriever qwen3` (Qwen3-Embedding-0.6B): model id + query format on the inf path (`src/qwen3emb.py`), corpus / query embedding (`tools/qwen3_embed.py`, `qwen3_embed_corpus.sbatch`), HN mining (`qwen3_mine_hn.sbatch`), base eval, `tools/q_queue.sh`; check `tools/qwen3_check.py` |
| 5bc12e5 | Qwen3 div-env jobs on h200 / l40s only (div torch 2.5.1 cu121 has no B200 kernels) |
| 9982c2a, 72a501f | `N_1GPU` (1 GPU x batch 100, 1-process FSDP config); GPU-aware queue limit (8 GPUs) |
| 2142257 | `--one_label_shuffled` (default off; needs `--one_label_step`, `--negative_ctxs 1`) |
| d0b0328, a7d2aed | `_v2` reruns of the four runs killed by the quota-full event; seed-1 legacy causal cap |
| 7b9a0e5 | `f_run.sh` phase 4: slim the run's retrieval outputs after RUN DONE (from the disk-cleanup session) |

- `--one_label_shuffled`: wraps `finetuning_multi.prepare_data`. Each training item is paired with a uniformly drawn
  other training question (per-rank RNG, seed `1000003*seed + 7919*(rank+1)`). Multi: the extra batch keeps one random gold and
  one negative per question (k = 1). Single: the extra batch is an ordinary batch of the other questions. The
  extra batch is used for the second update of `--one_label_step`. Smoke test (30 steps): 0-1 of 50 extra questions
  overlap the main batch. Legacy drew without replacement within an epoch; at ~45 epochs of draws the difference is
  negligible.
- Launch: `bash training/inf_retriever/tools/n_launch.sh <suffix> <inf|nemotron|qwen3> "<flags>"` (env `N_LR`,
  `N_DATA_ROOT`, `N_1GPU`, `SMOKE=1`).
- Fusion outputs: `results/phaseC/fusion/q_partial.json` (round 1 / 2a, singles, base), `q_round2.json` (2b / 2c),
  `olshuf_inf_causal.json`, `olshuf_causal_rseed1.json`, `olshuf_inf_noncausal.json`, `olshuf_round2.json` (Qwen3
  legacy runs + row 13), `geom_pairwise_cos.json`.

## Caveats

- One seed per setting except row 20; thirds move 0.6 / 0.56 per question.
- Qwen3 singles trained on 1 GPU x 100, multi runs on 2 GPUs x 50 (same global batch). The truncated 2-GPU singles
  (rows 14-15) have the same large third as the 1-GPU single at lr 1e-5 (14.29 vs 14.29). At lr 5e-6 theirs is lower
  (7.74 vs 10.12), so the lr 5e-6 gaps would be smaller against them (-1.8 / -2.4), still negative.
- 2026-11-16 17:11 the /scratch quota hit 100%. This killed four N trainings (Qwen3 cap io_proj at step ~4000 and
  three legacy-step runs). There is no resume, so they were rerun from scratch as `_v2` (rows 13, 19, 28-30 use the
  reruns), and the dead runs' partial files were deleted with the user's approval. On 2026-11-17 login node a-0 went
  down. Its drivers were restarted on a-2 and resumed their pending training jobs; nothing was lost.
- `perstep_eval.py` fails on every S0 run ("per-embedding scoring failed"; hardcodes the old 500-question dev); it
  only drops an extra diagnostic.
- N's finished retrieval outputs are slimmed in place (titles / texts dropped, ids / scores kept;
  `tools/slim_retrieval.py restore` gives them back); scores are unaffected.
