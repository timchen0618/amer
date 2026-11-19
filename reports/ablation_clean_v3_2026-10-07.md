# Ablation rounds `clean_v3_abl` and `clean_v3_abl2` (2026-10-05 onward)

One change at a time against new single- and multi-query baselines, on clean v3 data. Round 1 (`clean_v3_abl`) is complete on QAMPARI; AmbigQA is on hold. Round 2 (`clean_v3_abl2`, QAMPARI only) is in progress (section after round 1).

## Setup

- **Data:** clean v3 with hard negatives (`data/training/clean_v3_hn`, per-slot draws, ratio 0.5). Dev = the v2 dev sets (`data/phaseA/<ds>_cleanv2dev500.jsonl`); test = the unchanged AMER test sets. v3 = v2 plus the replacement of answer-containing random negatives (`data_creation/CLEAN_V3.md`).
- **Code:** as of `ab68534` (per-slot hard negatives, fp32 FSDP gradient reduction, corpus embedded at 512 tokens for new checkpoints).
- **LR:** the grid's best per dataset for both modes on hard negatives: QAMPARI 1e-5, AmbigQA 1e-6. One seed per run.
- **Runs** (`clean_v3_abl_<ds>_<variant>`, launcher `training/inf_retriever/tools/abl_launch.sh`):

| variant | change against `multi` |
| --- | --- |
| `single` | single-query baseline (`training_mode=standard_org_q`) |
| `multi` | multi-query baseline: `hungarian` loss, scheduled sampling |
| `masked` | `--loss_fn hungarian_masked`: the example's other golds are not used as negatives |
| `fullsamp` | `--full_sampling`: always feed back the model's own output (no teacher forcing) |
| `hps` | `--loss_fn hungarian_plus_single --single_loss_weight 0.5`: 0.5 × Hungarian + 0.5 × single-query contrastive loss on the first embedding |

- **Selection:** QAMPARI saves every 250 steps (2,500 total), keeps the top 3 by cheap dev (reduced corpus) and full-evaluates the top 2; the third kept checkpoint of each run was also fully evaluated afterwards (step 2500 had been pruned). AmbigQA full-evaluates steps 150, 250 and 400. Each run is reported at the checkpoint with the higher **full dev** score; test is never used for selection.
- **Headline metric:** MRecall@100. Multi-query runs are scored at **k = 5 with RRF** (user decision 2026-10-06); round-robin and every k are reported as secondary numbers. Every multi-query full evaluation saves each embedding's own top-500 list, so all k ≤ 5 and both aggregations are scored from one retrieval (`tools/kagg_eval.py`).

## QAMPARI

### Headline (each run at its full-dev-selected checkpoint)

| run | step | dev | test | test, round-robin k=5 |
| --- | --- | --- | --- | --- |
| single | 2000 | 64.80 | **53.30** | – |
| masked | 750 | **65.80** | 47.65 | 49.34 |
| hps | 750 | 63.00 | 46.14 | 45.20 |
| multi | 1500 | 61.20 | 38.42 | 33.90 |
| fullsamp | 750 | 56.80 | 38.04 | 33.71 |

No multi-query variant reaches single-query on test. `masked` beats single-query on dev by 1.0 and trails it on test by 5.65. Picks decided by less than 2 dev points (about two paired standard errors) are effectively ties: `fullsamp` (750 over 500 by 0.4), `multi` (1500 over 1750 by 0.2, over 750 by 2.8), `hps` (750 over 1000 by 0.4).

### Every evaluated checkpoint

MRecall@100, dev / test. The `RRF k=5` column is the headline score; `best test RRF` is the best over k on test, shown for information only, not for selection.

| run, step | round-robin k=5 | RRF k=5 | k=1 | best test RRF (k) |
| --- | --- | --- | --- | --- |
| single, 500 | 63.20 / 53.48 | – | – | – |
| single, 2000 | 64.80 / 53.30 | – | – | – |
| single, 2250 | 64.20 / 52.17 | – | – | – |
| multi, 750 | 56.00 / 37.10 | 58.4 / 40.30 | 53.6 / 31.26 | 41.81 (3) |
| multi, 1500 | 56.00 / 33.90 | 61.2 / 38.42 | 39.0 / 16.57 | 41.62 (4) |
| multi, 1750 | 55.40 / 33.52 | 61.0 / 39.36 | 29.2 / 10.73 | 39.36 (5) |
| masked, 750 | 65.60 / 49.34 | 65.8 / 47.65 | 67.4 / 49.34 | 49.34 (1) |
| masked, 1000 | 64.60 / 49.34 | 63.8 / 48.59 | 64.0 / 47.46 | 48.96 (2) |
| masked, 1500 | 63.60 / 49.91 | 64.0 / 48.40 | 62.4 / 47.08 | 49.15 (2) |
| fullsamp, 250 | 53.60 / 37.10 | 55.8 / 40.30 | 17.6 / 4.52 | 40.30 (5) |
| fullsamp, 500 | 49.80 / 34.84 | 56.4 / 40.30 | 17.0 / 6.78 | 40.30 (5) |
| fullsamp, 750 | 50.60 / 33.71 | 56.8 / 38.04 | 23.0 / 9.60 | 38.04 (5) |
| hps, 250 | 61.80 / 46.14 | 58.4 / 41.62 | 66.2 / 51.60 | 51.60 (1) |
| hps, 750 | 63.20 / 45.20 | 63.0 / 46.14 | **68.2 / 54.05** | 54.05 (1) |
| hps, 1000 | 61.00 / 43.69 | 62.6 / 47.27 | 66.8 / 51.98 | 51.98 (1) |

### Test MRecall@100 by k (RRF)

| run, step | k=1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| multi, 1500 | 16.57 | 25.05 | 34.27 | 41.62 | 38.42 |
| multi, 750 | 31.26 | 36.72 | 41.81 | 41.43 | 40.30 |
| masked, 750 | 49.34 | 48.78 | 48.40 | 47.65 | 47.65 |
| masked, 1500 | 47.08 | 49.15 | 49.15 | 48.59 | 48.40 |
| fullsamp, 750 | 9.60 | 10.73 | 18.64 | 31.07 | 38.04 |
| hps, 750 | 54.05 | 48.96 | 48.02 | 47.83 | 46.14 |

### Cheap dev (reduced corpus) MRecall@100 over training

| run | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | 2250 | 2500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single | 72.6 | 74.2 | 71.0 | 72.2 | 72.2 | 73.0 | 73.4 | 74.2 | 73.8 | 73.6 |
| multi | 66.0 | 66.6 | 69.6 | 68.0 | 67.2 | 70.0 | 68.6 | 67.8 | 65.8 | 62.2 |
| masked | 71.4 | 73.8 | 74.8 | 74.8 | 73.2 | 75.4 | 71.8 | 69.8 | 69.6 | 69.8 |
| fullsamp | 66.4 | 63.6 | 63.0 | 60.8 | 55.4 | 56.0 | 49.6 | 51.0 | 50.8 | 49.6 |
| hps | 73.4 | 71.2 | 72.2 | 71.4 | 66.8 | 69.6 | 65.6 | 64.6 | 65.0 | 63.0 |

All multi-query variants peak in the first half of training and then decline; single-query stays flat.

### Query/gold geometry (full corpus, test)

`align` = cosine of each query embedding to its Hungarian-matched gold; `margin` = matched cosine minus the best cosine to another gold of the same question; `query pair` = mean cosine between a question's k embeddings; `sibling rank` = median number of corpus passages closer to a gold than its nearest sibling gold (`tools/geometry_eval.py`).

| run, step | align | margin | query pair | sibling rank |
| --- | --- | --- | --- | --- |
| single, 2000 | 0.666 | 0.033 | – | 4 |
| multi, 1500 | 0.567 | 0.027 | 0.674 | 20 |
| multi, 750 | 0.517 | 0.005 | 0.696 | 18 |
| masked, 750 | 0.551 | −0.044 | **0.942** | 3 |
| masked, 1500 | 0.587 | −0.049 | **0.949** | 2 |
| fullsamp, 500 | 0.605 | 0.044 | 0.621 | 11 |
| hps, 750 | 0.476 | −0.007 | 0.641 | 5 |

### What this shows

- **`masked` is effectively a single-query model.** Its 5 embeddings are nearly identical (pairwise cosine 0.94–0.95, against about 0.65–0.70 for the other variants) and its margin is negative, meaning no embedding points at its matched gold more than at another gold. Hence the flat k curve: extra embeddings retrieve the same passages. It gets close to single-query by not doing multi-query retrieval.
- **`hps` trains a strong first embedding and weak later ones.** k = 1 is its best setting (test 54.05, above `single`'s 53.30; dev 68.2 vs 64.80), and every further embedding lowers the score. The single-query term works; the multi-query term adds nothing on top.
- **`multi` relies on its later embeddings.** Its first embedding alone is weak (16.57 test at step 1500), and the score keeps rising with k up to k = 4. It is the only variant whose embeddings split up the gold set, but each embedding is individually poor.
- **`fullsamp` hurts.** The first three embeddings are close to useless (k = 1 test 4.5–6.8); the score comes from embeddings 4 and 5, and cheap dev falls throughout training.
- Across variants there is a trade-off: making the first embedding strong (`hps`) or the embeddings alike (`masked`) gives single-query-like results; letting the embeddings diverge (`multi`, `fullsamp`) weakens each one. No variant gets both.

### Comparison with the grid: the v1 test leak

The grid (`reports/grid_results_2026-10-02.md`) trained on v1 data. **44 of the 531 QAMPARI test questions had a near-duplicate in v1 training data** (gold-set Jaccard ≥ 0.8 with a training example); v2/v3 remove all 44. Test MRecall@100, round-robin k=5, split into those 44 and the other 487:

| checkpoint | all 531 | 44 leaked | 487 clean |
| --- | --- | --- | --- |
| grid single (hn, 1e-5, step 2500) | 52.73 | 84.09 | 49.90 |
| v3 single (step 2000) | 53.30 | 68.18 | 51.95 |
| grid multi (hn, 1e-5, step 1750) | 38.23 | 63.64 | 35.93 |
| v3 multi (step 1500) | 33.90 | 27.27 | 34.50 |
| v3 multi (step 750) | 37.10 | 36.36 | 37.17 |
| v3 masked (step 750) | 49.34 | 61.36 | 48.25 |
| v3 hps (step 750) | 45.20 | 54.55 | 44.35 |

- The grid's best multi-query RRF score drops from 44.26 on all 531 to **41.48 on the clean 487** (from its saved per-step lists). v3 `multi` scores 38.42–40.30 at RRF k = 5 on all 531; its own lists were deleted by the queue before this analysis, so it cannot be split the same way.
- On the 487 clean questions the 12 grid multi-query checkpoints range from 27.10 to 36.55 (round-robin); v3 `multi` (34.50–37.17) is at the top of that range. The grid's reported 38.23 was the best of 12 checkpoints, selected on a dev set where 127 of 500 questions had a training example with the same gold set.
- On the 487 clean questions single-query rises by 2.05 (about 10 questions) and multi-query moves by −1.4 to +1.2 (round-robin) or about −1 to −3 (RRF). With one seed per configuration these differences are within checkpoint-to-checkpoint variation; there is no evidence of a regression. Untested differences between the rounds: seed, per-slot hard negatives, and the corpus embedded at 512 instead of 1024 tokens.
- **Grid QAMPARI test numbers are not comparable with v2/v3 runs.** Compare on the 487 clean questions, or retrain the baseline.

### Extra checkpoints (done)

- `multi` step 1750 (check 1): full dev RRF k=5 61.0, below step 1500's 61.2; the multi-query baseline was not held back by step selection.
- The step-2500 checkpoints had been pruned (top 3 by cheap dev kept), so each run's **third-best** cheap-dev checkpoint was fully evaluated instead (single 2250, multi 1750, masked 1000, fullsamp 750, hps 1000; rows above). Only `fullsamp`'s pick changed (500 to 750, by 0.4 dev points).

## QAMPARI round 2: `clean_v3_abl2` (in progress)

Follow-up runs on QAMPARI only, against the round-1 runs above (which are not rerun). Same data, dev set, 2,500 steps and one seed; launcher `training/inf_retriever/tools/abl2_launch.sh`.

### Runs

| # | wave | run | mode | loss | single-query weight w | diversity λ | rate reaches 1 at | LR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | `sat25` | multi | hungarian | – | – | step 625 (25%) | 1e-5 |
| 2 | 1 | `sat50` | multi | hungarian | – | – | step 1250 (50%) | 1e-5 |
| 3 | 1 | `hps_w025` | multi | hungarian + single | 0.25 | – | step 2500 (end) | 1e-5 |
| 4 | 1 | `hps_w010` | multi | hungarian + single | 0.10 | – | step 2500 (end) | 1e-5 |
| 5 | 1 | `single_lr3e-5` | single | contrastive | – | – | – | 3e-5 |
| 6 | 1 | `multi_lr3e-5` | multi | hungarian | – | – | step 2500 (end) | 3e-5 |
| 7 | 1 | `masked_div05` | multi | masked + diversity | – | 0.5 | step 2500 (end) | 1e-5 |
| 8 | 1 | `masked_div2` | multi | masked + diversity | – | 2.0 | step 2500 (end) | 1e-5 |
| 9 | 2 | `hps_fullsamp` | multi | hungarian + single | best of 0.5 / 0.25 / 0.1 | – | step 0 (always 1) | 1e-5 |
| 10 | 2 | `hps_sat50` | multi | hungarian + single | best of 0.5 / 0.25 / 0.1 | – | step 1250 (50%) | 1e-5 |
| opt | 2 | `single_lr2e-5`, `multi_lr2e-5` | single / multi | as 5–6 | – | – | as 5–6 | 2e-5 |

- **Sampling saturation** (`--sampling_saturate_frac f`): rate = min(1, step / (f × total_steps)). Round 1 had f = 1 (`multi`) and rate 1 from the start (`fullsamp`).
- **Diversity penalty** (`--loss_fn hungarian_masked_div --div_weight λ`): L_masked + λ · mean over i ≠ j of relu(cos(q_i, q_j) − cos(g_a(i), g_a(j))), with a(·) the masked loss's Hungarian assignment and the gold–gold cosines detached. Round 1's `masked` had query pairwise cosine 0.94 against 0.57 for its golds; the penalty removes only similarity in excess of the golds'.
- **Wave 2 weight:** the hps weight with the highest full-dev score at k = 5 RRF among round 1's `hps` (0.5) and rows 3–4.

### Selection protocol (changed from round 1)

- Cheap dev ranks multi-query checkpoints by **RRF at k = 5** (`select_metrics.txt`), the same metric as the headline; round 1 ranked by round-robin. Single-query checkpoints are unchanged.
- The top 3 by cheap dev **and the final checkpoint** are kept; the top 2 by cheap dev **plus step 2500** are fully evaluated (2 or 3 per run). Each run is reported at the one with the best full dev score; differences under about 2 points (paired standard error about 1 point on 500 dev questions) are treated as ties.
- Reasons: `reports/cheap_eval_validity_2026-10-07.md` (cheap dev picks the full-dev-best step within a QAMPARI run about half the time).

### Wave 3 (planned, not started): doc-encoder freezing and causal attention

The 2×2 from `experiment_plan.md` (doc encoder frozen or joint × query-encoder attention causal or bidirectional) was never completed: only run A (frozen, causal) was trained, on 2026-09-17/21, and its results are not comparable with anything since. QAMPARI run A predates the 2026-09-26 fix of the QAMPARI gold-label whitespace bug in the training loader. Both datasets' run A used `full_sampling=1`, 800 steps, no hard negatives, leaked splits, bf16 gradient reduction, and AmbigQA was first scored with the k = 5 bug. Everything since (grid, round 1, round 2) is cell D (joint, bidirectional).

Wave 3 reruns the other cells on the current recipe (v3 + hard negatives, LR 1e-5, rate reaching 1 at the end, k = 5 RRF headline, round-1/2 selection protocol), each as a single- and multi-query pair, in this order:

| order | cell | doc encoder | query attention | isolates (against D) | code needed |
| --- | --- | --- | --- | --- | --- |
| 1 | C | frozen (base weights) | bidirectional | freezing the doc encoder | check the frozen-doc-encoder classes against later changes; corpus embeddings = the base model's, shared by every checkpoint |
| 2 | A | frozen | causal | the paper-faithful combination | none beyond C (`--force_causal` exists for frozen classes) |
| 3 | B | joint | causal | the causal mask in the current setup | `--force_causal` for the joint classes |

With a frozen doc encoder every checkpoint shares one corpus embedding, so a full evaluation is a retrieval only (about 30 minutes) and every saved step can be evaluated in full. Reference points: untrained base retriever QAMPARI test 12.24, D single 53.30.

**Comparability with the clean v3 runs.** The D reference is round 1's `single` and `multi` (`clean_v3_abl_qampari_{single,multi}`); wave-3 runs change only the doc encoder and/or the attention mask. Everything else must match those runs, and is checked before launch:

| item | must equal the clean v3 runs | check before launch |
| --- | --- | --- |
| Data | `clean_v3_hn` QAMPARI train, hard-negative ratio 0.5 (per-slot), cleanv2 dev500, unchanged test | same `--train_data` / `DATA_ROOT` / `DEV_TAG` |
| Optimisation | LR 1e-5, 2,500 steps, warmup 100, 50 queries per GPU on 2 GPUs (FSDP, fp32 gradient reduce), temperature 0.05, seed 0 | the old frozen runs used batch 150 and 800 steps: use the phaseC driver, not the old scripts; FSDP must handle the frozen copy (the September runs bypassed FSDP for it) |
| Multi-query objective | `hungarian` loss, rate ramping to 1 at the end, golds shuffled, k = number of golds | the frozen classes duplicate the generation loop (`src/inbatch.py`); diff it against `EmbeddingModelDocEncNoProj` |
| Logging | same training stats and geometry | the frozen classes must call `add_geometry_stats` |
| Selection | cheap dev (RRF k = 5) top 2 + step 2500, best by full dev; `DENSE=1 KEEP_TOP=3 KEEP_FINAL=1` | full evals of every step may be reported as an extra, not used for selection |
| Corpus embedding | 512-token passages, as for clean v3 checkpoints | the existing base-model corpus embeddings may be at another length: re-embed the corpus once with the base model at 512 and reuse it for all frozen checkpoints |
| Checkpoint loading | corpus embedded with the frozen doc encoder, queries with the trained query encoder | `load_retriever` / `gen_embed_new.py` must pick the doc encoder (not the query encoder) for frozen checkpoints; verify the shared embedding equals a checkpoint's own on a sample |
| Retrieval and scoring | same `retrieval_inf.py`, k = 5, RRF headline, per-embedding lists kept, geometry | via `full_eval_queue.sh` with a pre-computed corpus embedding |

### Results

Wave 1: pending. Wave 2: pending. Wave 3: not started.

## AmbigQA (on hold since 2026-10-07; full evals cancelled to focus on QAMPARI)

Multi-query runs generate k = 2 embeddings, so the QAMPARI headline (k = 5, RRF) does not exist here; reported provisionally at **k = 2, RRF**, with k = 1 alongside.

| run | step 150 | step 250 | step 400 |
| --- | --- | --- | --- |
| single | **82.40 / 76.18** | 82.00 / 76.30 | 81.60 / 76.30 |
| multi (round-robin k=2) | 79.60 / 74.00 | running | running |
| masked | queued | queued | queued |
| fullsamp | queued | queued | queued |
| hps | queued | queued | queued |

`multi` step 150: RRF k = 2 79.4 / 73.28; k = 1 80.6 / 75.33.

## Operations and fixes during the round

- `retrieval_inf.py --save_per_step` wrote only the per-step lists and no merged output, so the queue's first multi-query evaluation failed. Fixed in `ab68534`: the merged output is also written, from the same search. The failed evaluation was rebuilt from its saved lists without re-searching.
- `tools/geometry_eval.py` ran out of memory (96 GB) because gold vectors were views that kept whole corpus shards alive; fixed in `ab68534`. The lost dev geometry was re-run.
- `full_eval_queue.sh` deleted the per-step id lists (about 70 MB per split, only recoverable by re-embedding) and kept the passage-text files (about 600 MB, rebuildable from the corpus). Reversed on 2026-10-07; the QAMPARI lists of this round were already deleted.
- B200 nodes gb006, gb011 and gb012 repeatedly failed corpus-embedding shards with "CUDA-capable device(s) is/are busy or unavailable" within 90 seconds. Failed shards were re-embedded individually (`recover_shards.sh`); the cluster rejects `--exclude`.
- On request, AmbigQA full evaluations were paused until every QAMPARI evaluation had finished.
- **2026-10-08, /scratch quota full (round 2, wave 1).** Pruning happens only after each cheap eval, and the cheap evals waited behind the per-user GPU limit (16 GPUs under the `gpu48` QOS) while the 8 training runs held all 16. All 10 checkpoints per run accumulated (~67 GB each) and the quota reached 100% (`myquota` 5.00 TB while `du` showed 4.5 TB: the quota counts about 500 GB more than `du`). Every queued cheap eval then failed at start-up, and the follower queued step-2500-only full evals for runs without cheap metrics (twice: again when it was restarted and read the old `RUN DONE` lines). Those evals were cancelled; nothing was lost (all 77 checkpoints intact). Fixes: 13 pre-grid `phaseC_*` checkpoint runs and older AmbigQA checkpoints deleted on approval (~600 GB); drivers resumed (`abl2_launch.sh` with `RESUME=1`), the follower now only counts `RUN DONE` lines logged after it starts; cheap evals ask for 40 min instead of 2 h and may run on H200 or B200 (group, public and preemptible partitions). Budget disk for every saved checkpoint, and keep at least 4 GPUs free for cheap evals.
- **2026-10-08, B200 back.** 345 of 352 B200 tasks since 2026-09-29 succeeded; the 7 failures ("CUDA-capable device(s) is/are busy or unavailable" at start-up) were on nodes that also ran 18–40 tasks fine. Cheap and full evaluations use B200 again, and that start-up failure now requeues the task (at most twice). A cheap eval on B200 takes about 15 min (H200 about 20) and gives the same scores.
