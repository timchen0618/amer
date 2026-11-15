# Nemotron-3.5-Embed-1B in F's frozen QAMPARI setting: final report (session N, 2026-11-15)

Branch `nemotron-retriever` (worktree `.claude/worktrees/n-nemotron`, not merged). Handoff:
`F_retriever_swap_handoff_2026-11-12.md`. Round-1 notes: `N_nemotron_results_2026-11-13.md` (main checkout).

## Question

Session F's frozen-doc-encoder QAMPARI recipe, rerun with `nvidia/Nemotron-3.5-Embed-1B-BF16` in place of
`infly/inf-retriever-v1-1.5b`. The metric we care about most is the **gap = multi large third - single large third**
(with inf-retriever it is 0). Round 1 found a gap of -4.8 / -6.0 with Nemotron; round 2 tried three fixes the user
asked for (Nemotron-mined hard negatives, lower learning rate, input / output projection); round 3 tried causal
attention with last-token pooling.

## Setup shared by every run

- Data: S0 split (`data/training/clean_v3_hn_s0`), `--negative_hard_ratio 0.5` (half of the negatives are hard
  negatives), frozen document encoder (`--freeze_doc_encoder`), 10k steps, temperature 0.05, one-gold step
  (`--one_label_step`), 2 GPUs (H200), session F's `f_run.sh` pipeline.
- Unless a row says otherwise: lr 1e-5, hard negatives mined by base inf-retriever, no projection layers,
  bidirectional attention.
- Single = one query embedding, 2 updates per step (F's `S0_single_10k_2step`). Multi = Hungarian loss, k = 5
  embeddings, either *linear* (scheduled-sampling rate 0 -> 1) or *cap* (`--sampling_saturate_frac 0.1
  --sampling_cap 0.8`).
- Nemotron: inputs `query: <q>` / `passage: <title> <text>` without BOS / EOS, bf16, mean pooling, 2048-d, its
  bidirectional text backbone; corpus = base Nemotron embeddings of the 25.9M-passage corpus
  (`wikipedia_embeddings/qampari/nemotron_frozen_base_512`).
- Selection: test at the dev-selected step (full S0 dev, 2459 questions); MRecall@100; multi = RRF over the k = 5
  embeddings. A difference under 2 points is a tie. Thirds = the paper's large / small distance thirds of test
  (168 / 180 questions; one question = 0.6 / 0.56 points), from `tools/fusion_eval.py`.

## All results

Gap = this multi run's large third minus the large third of the single run in the same block (round 3: also vs
the bidirectional single, row 6). Thirds for round 3: `results/phaseC/fusion/n_causal.json`.

| # | retriever | mode | setting | step | dev | test | large / small | gap |
|---|---|---|---|---|---|---|---|---|
| 1 | inf | - | base, no training | - | - | 24.48 | | |
| 2 | Nemotron | - | base, no training | - | 11.79 | 18.64 | | |
| | **Round 1: F recipe** | | | | | | | |
| 3 | inf | single | F `S0_single_10k_2step` | 10000 | 51.93 | 35.03 | 25.00 / 50.00 | |
| 4 | inf | multi | linear | 9000 | 50.83 | 33.90 | 25.00 / 45.56 | **0.0** |
| 5 | inf | multi | cap (F `S0_cap10k_onelabel`) | 10000 | 50.71 | 33.15 | 25.00 / 43.89 | **0.0** |
| 6 | Nemotron | single | | 10000 | 54.09 | 34.84 | 26.19 / 50.00 | |
| 7 | Nemotron | multi | linear | 6000 | 49.45 | 31.07 | 21.43 / 45.56 | **-4.8** |
| 8 | Nemotron | multi | cap | 5000 | 46.28 | 28.63 | 20.24 / 44.44 | **-6.0** |
| | **Round 2a: lr 5e-6** | | | | | | | |
| 9 | Nemotron | single | lr 5e-6 | 9000 | 52.42 | 32.58 | 23.21 / 48.33 | |
| 10 | Nemotron | multi | linear, lr 5e-6 | 7000 | 46.40 | 26.74 | 17.86 / 40.56 | **-5.4** |
| 11 | Nemotron | multi | cap, lr 5e-6 | 9000 | 44.57 | 25.80 | 15.48 / 40.56 | **-7.7** |
| | **Round 2b: hard negatives mined by base Nemotron** | | | | | | | |
| 12 | Nemotron | single | Nemotron HN | 8000 | 54.13 | 35.40 | 25.60 / 49.44 | |
| 13 | Nemotron | multi | linear, Nemotron HN | 7000 | 48.56 | 31.83 | 22.02 / 44.44 | **-3.6** |
| 14 | Nemotron | multi | cap, Nemotron HN | 5000 | 46.12 | 28.44 | 20.24 / 40.56 | **-5.4** |
| | **Round 2c: learned projection layers (`--io_proj`, identity init)** | | | | | | | |
| 15 | Nemotron | single | output layer only (`--io_proj_mode out`) | 8000 | 53.72 | 34.84 | 24.40 / 51.67 | |
| 16 | Nemotron | multi | linear, input + output layers | 7000 | 47.74 | 31.26 | 23.81 / 44.44 | **-0.6** |
| 17 | Nemotron | multi | cap, input + output layers | 3000 | 43.92 | 28.25 | 19.64 / 42.22 | **-4.8** |
| | **Round 3: causal query encoder + last-token pooling (`--force_causal`; docs stay bidirectional + mean)** | | | | | | | |
| 18 | Nemotron | single | causal | 8000 | 52.26 | 33.33 | 27.38 / 44.44 | |
| 19 | Nemotron | multi | linear, causal | 8000 | 47.54 | 27.50 | 22.02 / 37.22 | **-5.4** (-4.2 vs row 6) |
| 20 | Nemotron | multi | cap, causal | 6000 | 46.89 | 27.31 | 23.21 / 35.56 | **-4.2** (-3.0 vs row 6) |
| 21 | Nemotron | multi | linear, causal + input / output layers | 5000 | 46.56 | 26.37 | 19.64 / 37.78 | **-7.7** (-6.5 vs row 6) |

Other tested steps of round 3 (not dev-selected): single s7000 32.39, s10000 32.58, s4000 30.32; linear s7000 26.55,
s10000 27.31; cap s7000 28.25, s9000 28.81 (large / small 23.21 / 38.89), s10000 28.44; causal + io_proj s8000 26.55,
s9000 24.86, s10000 25.24.

## Where the multi loss comes from: test MRecall@100, RRF over the first k embeddings

| run | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| inf linear (4) | 34.09 | 29.94 | 34.09 | 33.52 | 33.90 |
| inf cap (5) | 34.65 | 32.77 | 31.26 | 33.33 | 33.15 |
| Nemotron linear (7) | 33.90 | 25.24 | 29.00 | 28.81 | 31.07 |
| Nemotron cap (8) | 32.20 | 25.05 | 25.80 | 25.05 | 28.63 |
| linear lr 5e-6 (10) | 30.51 | 23.16 | 23.16 | 23.92 | 26.74 |
| cap lr 5e-6 (11) | 29.19 | 25.61 | 23.92 | 24.86 | 25.80 |
| linear Nemotron HN (13) | 32.39 | 26.55 | 28.81 | 29.76 | 31.83 |
| cap Nemotron HN (14) | 31.83 | 24.67 | 24.86 | 26.18 | 28.44 |
| linear io_proj (16) | 34.46 | 28.44 | 28.06 | 29.76 | 31.26 |
| cap io_proj (17) | 32.58 | 25.24 | 26.37 | 26.93 | 28.25 |
| linear causal (19) | 29.38 | 23.54 | 27.50 | 27.68 | 27.50 |
| cap causal (20) | 32.39 | 25.99 | 25.24 | 26.74 | 27.31 |
| linear causal + io_proj (21) | 31.83 | 25.99 | 28.44 | 27.87 | 26.37 |

## Findings

1. **The Nemotron gap does not go away.** With inf-retriever the multi runs match the single run's large third
   (gap 0.0 / 0.0). With Nemotron, 10 of the 11 multi runs are 3.6-7.7 points below (6-13 questions of 168). The one
   exception (row 16, -0.6) is explained in point 4. No Nemotron multi run beats its single run on the large third,
   and every one is 3.6-7.0 points lower than its single run on the full test set.
2. **Lower learning rate (5e-6) is worse everywhere**: single -2.3 test, multi -4.3 / -2.8 test, and the gaps
   widen (-5.4 / -7.7).
3. **Nemotron-mined hard negatives are a tie.** Gaps -3.6 / -5.4 vs -4.8 / -6.0 in round 1 (2 questions each),
   with test 31.83 / 28.44 vs 31.07 / 28.63.
4. **Projection layers** (`--io_proj`, identity init): the linear gap shrinks to -0.6, but from both sides. The single
   run's large third drops (24.40 vs 26.19) and the multi run's rises (23.81 vs 21.43, +4 questions). The multi run is
   still 3.6 points below its single run on test (31.26 vs 34.84), and its small third is 7.2 lower. The cap run is
   a tie with round 1 (-4.8). With the causal encoder (row 21), the projection layers give the largest gap (-7.7, tied with the lr 5e-6 cap run).
5. **Causal attention + last-token pooling does not help.** The causal single ties the bidirectional single on test
   (33.33 vs 34.84). It has the highest large third of any Nemotron run (27.38, +2 questions) but a lower small third
   (44.44 vs 50.00). The causal multi runs are lower than their bidirectional counterparts: linear 27.50 vs 31.07,
   cap 27.31 vs 28.63. Their gaps are -5.4 / -4.2 against the causal single and -4.2 / -3.0 against the bidirectional
   one. All of these are in the same range as round 1.
6. **Where the multi loss comes from (per-k table):** in all 11 Nemotron multi runs, the first embedding alone
   (k = 1, 29.2-34.5) is the best of the five settings and is 0.6-5.5 points above the full RRF over k = 5. Adding
   the second embedding costs 3.6-8.7 points, and the later ones win back only part of that. With inf-retriever,
   k = 5 is within 1.5 of k = 1. So with Nemotron, fusing in the 2nd-5th embeddings' lists loses more fully covered
   questions than it gains.
7. **Geometry (round 1, test):** in Nemotron's frozen document space the gold passages of a question are much less
   similar to each other (mean pairwise cosine 0.41 vs 0.86 for inf). Its generated query embeddings are also more
   spread out (pairwise cosine 0.65 linear / 0.56 cap vs 0.77 / 0.70). This fits point 6: each Nemotron step lands
   in a more distinct region, and the RRF of these lists dilutes the first one. We did not test whether this is the
   cause.
8. **Single runs tie across retrievers** (inf 35.03, Nemotron 34.84 / 35.40 with Nemotron HN), even though base
   Nemotron is 5.8 points lower than base inf-retriever. The retriever swap alone therefore costs nothing for single
   retrieval, but it does cost multi retrieval.

Bottom line: with the frozen Nemotron document space, none of the four fixes (Nemotron hard negatives, lr 5e-6,
projection layers, causal attention) brings the multi large third up to the single run's. The best multi setting is
still round 1 / 2b linear (test 31.07 / 31.83, gap -4.8 / -3.6). Since every fix tried here changed the query side,
the next things to try are on the document side: an unfrozen or jointly trained doc encoder, or a different pooling
for documents.

## Implementation (branch `nemotron-retriever`)

| commit | what |
|---|---|
| c4339cd | `--retriever {inf,nemotron}` (default inf, so F's runs are unchanged) in `finetuning_multi_nemo.py`; `src/nemotron.py` (input format, mean pooling, backbone loading); corpus embedding (`tools/nemo_embed_corpus.py` + sbatch); `tools/n_launch.sh` |
| 28dcb7d, 22757e9 | corpus-embedding timing; offline hub in the nemotron overlay |
| 93f2ccd | two-stage retrieval: queries embedded in the nemotron overlay (`retrieval_inf.py --embed_only`), faiss search in nli (`--load_query_embeddings`), since nemotron.ext3's faiss is CPU-only |
| 1490abb | round 2: `--io_proj` allowed with Nemotron, `N_LR` / `N_DATA_ROOT` in `n_launch.sh`, Nemotron hard-negative mining (`tools/nemo_embed_queries.py`, `tools/nemo_mine_hn.sbatch`), `tools/n_queue.sh` |
| 387c386 | `--force_causal` for Nemotron: causal query encoder through the backbone's `config.is_causal` (transformers 5 builds the mask from it; no per-call `is_causal` kwarg), last-token pooling; the frozen doc encoder stays bidirectional + mean pooling; GPU check `tools/nemo_causal_check.py` |
| c0709ec | causal Nemotron single-query retrieval: bf16 -> float before numpy |

- Environment: `/scratch/hc3337/envs/nemotron.ext3` (clone of div + transformers 5.20, torch 2.9 cu128,
  flash-attn 2.8.3, faiss-cpu; needs `HF_HUB_OFFLINE=1`). `div.ext3` untouched.
- Checks: base Nemotron through our code = the model-card path (cosine 1.0000); the backbone is bidirectional in
  every case, padded or not (prefix hidden states change when tokens are appended), so rounds 1-2 ran as intended;
  with `--force_causal` the query encoder is causal (prefix difference 0) while document embeddings are
  bit-identical to the bidirectional model's.
- Nemotron hard negatives (`data/training/clean_v3_hn_s0_nemo`): same procedure as the inf-mined ones (top-100
  base retrieval, skip top 3, drop golds / gold titles / answer passages, keep up to 30); 29.6 per question, 97.7%
  at 30 (inf: 29.0, 95.4%); only 23.4% of ids shared with inf's.
- Launch: `bash training/inf_retriever/tools/n_launch.sh <suffix> nemotron "<flags>"` (env `N_LR`, `N_DATA_ROOT`,
  `SMOKE=1`).
- Fusion outputs: `results/phaseC/fusion/n_nemotron.json`, `n_round2.json`, `n_round2_ioproj_cap.json`,
  `n_round2_nhn_cap.json`, `n_causal.json`.

## Caveats

- One seed per setting. Thirds have 168 / 180 questions, so a 1-2 question change moves a third by 0.6-1.2.
- `perstep_eval.py` fails on every S0 run ("per-embedding scoring failed"; it hardcodes the old 500-question dev);
  it only drops an extra diagnostic (F's runs hit it too).
- Base Nemotron is 5.8 points below base inf-retriever on test, yet trained single runs tie.
- The 30-step smoke-test checkpoints (`*_smoke*`) were deleted after the smoke results were checked; no reported
  row depends on them.
