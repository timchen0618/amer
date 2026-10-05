# 24-run grid results

*Status: complete (2026-10-04): all 48 full-corpus evaluations of the 24 runs.*

**Bottom line:** multi-query does not beat single-query on either dataset under the grid's settings (`hungarian` loss, scheduled sampling, hard negatives drawn once per example). On QAMPARI single-query leads by 8–16 test points in every matched cell; on AmbigQA by 0.6–5.6, mostly within noise for the best cells.

## Code

| Commit | Contents |
| --- | --- |
| `4298660` | Training and data code the grid ran on: clean splits (`data_creation/build_clean_splits.py`), hard-negative miner, training driver (`phaseC_run.sh`, `phaseC_train.sbatch`, `grid_launch.sh`) |
| `c9e1b9e` | Eval pipeline used for the full-corpus evaluations (`retrieval_inf.py`, `full_eval_queue.sh`, embed / retrieve sbatch scripts) |

Branch `fsdp-clean-recipe`. Grid launched 2026-09-28; launch-time snapshot of the uncommitted diff: `results/phaseC/grid_code_snapshot.patch`.

Known issues that affect how these results read: `code_audit_2026-09-28.md` (in particular G1, G2, L1, C2).

## Grid

dataset {QAMPARI, AmbigQA} × mode {single, multi} × data {no hard neg., hard neg.} × LR {1e-6, 3e-6, 1e-5}.

Common settings: INF-Retriever 1.5B as a joint query/document encoder; 2-GPU FSDP, mixed precision, 50 queries per GPU; temperature 0.05; one random negative per gold (multi) or per example (single); hard negatives = 30 mined per example, half the examples' negatives hard (drawn once per example, see L1); clean v1 splits (`data/training/clean`). QAMPARI 2,500 steps (warmup 100), checkpoint every 250 steps; AmbigQA 400 steps (warmup 15), checkpoints 30/100/150/250/400. Multi-query retrieves with k = 5 (QAMPARI) / 2 (AmbigQA) embeddings merged by round-robin.

**Selection:** cheap dev (reduced corpus) picked two checkpoints per QAMPARI run (its top 2); AmbigQA used steps 150 and 400. Both got a full-corpus dev (clean dev500) and test evaluation. Each run is reported at the checkpoint with the higher full dev score; each mode's headline is its best run by full dev. Test was never used for selection.

## Results

All numbers are full-corpus MRecall@100 (Recall@100 in the headline). One question = 0.2 points on dev (500 queries), 0.19 on QAMPARI test (531), 0.12 on AmbigQA test (827).

### Headline (best run per mode by full dev)

| Dataset | Mode | Run (checkpoint) | Dev | Test | Test Recall@100 |
| --- | --- | --- | --- | --- | --- |
| QAMPARI | single | hard neg., LR 1e-5 (step 2500) | 72.20 | **52.73** | 78.26 |
| QAMPARI | multi | hard neg., LR 1e-5 (step 1750) | 64.40 | **38.23** | 67.54 |
| AmbigQA | single | hard neg., LR 1e-6 (step 150) | 82.40 | **76.30** | 88.57 |
| AmbigQA | multi | hard neg., LR 1e-6 (step 150) | 80.00 | **74.00** | 87.45 |

References: untrained base retriever, test 12.24 (QAMPARI) / 72.55 (AmbigQA).

### QAMPARI (dev / test; selected checkpoint in bold)

| Mode | Data | LR | Checkpoint A | Checkpoint B |
| --- | --- | --- | --- | --- |
| single | hard neg. | 1e-5 | **s2500: 72.20 / 52.73** | s2250: 71.80 / 52.17 |
| single | hard neg. | 3e-6 | **s1750: 71.00 / 50.66** | s2000: 70.80 / 51.41 |
| single | hard neg. | 1e-6 | **s1750: 67.40 / 49.15** | s2250: 66.80 / 48.78 |
| single | no hard neg. | 1e-5 | **s2500: 65.80 / 46.33** | s2250: 65.00 / 45.39 |
| single | no hard neg. | 3e-6 | **s2000: 64.40 / 42.00** | s1750: 63.40 / 44.26 |
| single | no hard neg. | 1e-6 | **s2000: 62.60 / 39.74** | s2250: 61.80 / 39.92 |
| multi | hard neg. | 1e-5 | **s1750: 64.40 / 38.23** | s1250: 62.80 / 38.79 |
| multi | hard neg. | 3e-6 | **s1250: 58.60 / 37.85** | s1750: 58.00 / 32.58 |
| multi | hard neg. | 1e-6 | **s1250: 55.40 / 35.40** | s1500: 54.00 / 33.52 |
| multi | no hard neg. | 1e-5 | **s1750: 52.60 / 30.32** | s1250: 51.60 / 32.77 |
| multi | no hard neg. | 3e-6 | **s1000: 51.60 / 31.45** | s1250: 49.60 / 28.81 |
| multi | no hard neg. | 1e-6 | **s1250: 52.40 / 31.83** | s1000: 50.00 / 31.64 |

Test at the selected checkpoint:

| Mode | Data | LR 1e-6 | LR 3e-6 | LR 1e-5 |
| --- | --- | --- | --- | --- |
| single | no hard neg. | 39.74 | 42.00 | 46.33 |
| single | hard neg. | 49.15 | 50.66 | **52.73** |
| multi | no hard neg. | 31.83 | 31.45 | 30.32 |
| multi | hard neg. | 35.40 | 37.85 | **38.23** |

### AmbigQA (dev / test; selected checkpoint in bold)

| Mode | Data | LR | Step 150 | Step 400 |
| --- | --- | --- | --- | --- |
| single | hard neg. | 1e-6 | **82.40 / 76.30** | 81.20 / 76.30 |
| single | hard neg. | 3e-6 | **80.60 / 74.85** | 79.60 / 75.82 |
| single | hard neg. | 1e-5 | 78.80 / 74.00 | **80.80 / 75.70** |
| single | no hard neg. | 1e-6 | **80.80 / 75.45** | 80.60 / 75.33 |
| single | no hard neg. | 3e-6 | **80.60 / 75.82** | 80.20 / 75.94 |
| single | no hard neg. | 1e-5 | **79.80 / 75.45** | 79.20 / 75.45 |
| multi | hard neg. | 1e-6 | **80.00 / 74.00** | 78.80 / 74.97 |
| multi | hard neg. | 3e-6 | **78.60 / 73.76** | 75.20 / 71.22 |
| multi | hard neg. | 1e-5 | **77.60 / 72.55** | 77.00 / 71.83 |
| multi | no hard neg. | 1e-6 | **79.20 / 74.85** | 77.40 / 73.40 |
| multi | no hard neg. | 3e-6 | **78.20 / 72.55** | 75.40 / 71.83 |
| multi | no hard neg. | 1e-5 | 73.00 / 68.32 | **73.60 / 69.89** |

Test at the selected checkpoint:

| Mode | Data | LR 1e-6 | LR 3e-6 | LR 1e-5 |
| --- | --- | --- | --- | --- |
| single | no hard neg. | 75.45 | 75.82 | 75.45 |
| single | hard neg. | **76.30** | 74.85 | 75.70 |
| multi | no hard neg. | 74.85 | 72.55 | 69.89 |
| multi | hard neg. | 74.00 | 73.76 | 72.55 |

### What the grid shows

- **Single vs multi (test, selected checkpoints).** QAMPARI: single leads in all six matched cells, by 12.8–14.5 with hard negatives and 7.9–16.0 without. AmbigQA: single leads in all six, by 1.1–3.2 with hard negatives and 0.6–5.6 without; the best cells differ by 2.3 (about 19 questions).
- **Hard negatives.** QAMPARI: +6.4 to +9.4 for single, +3.6 to +7.9 for multi. AmbigQA: small and mixed for single (−1.0 to +0.9), +2.7 for multi at LR 1e-5 and −0.9 to +1.2 at the other LRs.
- **Learning rate.** QAMPARI: higher is better in every cell except multi without hard negatives (flat, 30–32). AmbigQA: single is flat (74.9–76.3); multi gets worse as LR rises, most without hard negatives (74.85 → 69.89), where LR 1e-5 falls below the untrained base retriever (72.55).
- **Checkpoint selection within a run.** Full dev picked the checkpoint with the higher test score in 7 of 12 QAMPARI runs and in 7 of 10 AmbigQA runs that differ on test (2 ties). The misses are mostly small: at most 2.5 points except multi hard neg. 3e-6 on QAMPARI, where the pick was right by 5.3.
- **k and aggregation** (section below): RRF instead of round-robin raises QAMPARI's best multi-query checkpoint from 38.23 to 44.26 on test; still 8.5 below single-query. On AmbigQA extra embeddings do not help.

### Caveats

- **Hard negatives drawn once per example (audit L1).** In every multi-query hard-negative run, about half the examples got only hard negatives and half only random ones; fixed since (per-slot draws).
- **QAMPARI v1 split residuals (audit G1, G2).** 127 of the 500 v1 dev questions have a train example with the same gold set, and 44 of the 531 test questions have a near-duplicate train example. This may inflate both modes' QAMPARI scores; fixed in the v2 splits.
- **Passage length.** The corpus was embedded with passages truncated at 1,024 tokens; training used 512. Later checkpoints embed at 512.
- **AmbigQA multi-query scores above 75 elsewhere.** The grid's best multi-query *test* score is 74.97 (hard neg., LR 1e-6, step 400; 75.09 when the k/aggregation study re-embedded the same checkpoint on different GPUs, one question). The headline reports 74.00 because selection uses dev, which prefers step 150 (80.00 vs 78.80). Pre-grid phase C checkpoints reached 76.66 and 76.06, but they trained on the pre-audit AmbigQA data that contained 65 of the 827 test questions (audit D1), so they are not comparable.
- **One seed per cell.** No variance estimate; AmbigQA differences under ~1.5 test points (about 12 questions) should be read as ties.
- **Fixed design choices.** `hungarian` loss (other golds of the same example act as negatives), scheduled sampling with shuffled gold feedback, round-robin aggregation. These are the next ablations.

## Number of query embeddings and aggregation (audit C2), 2026-10-03

One multi-query checkpoint per dataset, full corpus, clean dev500 and test. Each was retrieved once with the largest k, saving every embedding's own top-500 list; every smaller k and both aggregations were then scored offline from those lists (`training/inf_retriever/tools/kagg_study.sh`; results in `results/kagg/<ds>/<checkpoint>/{dev,test}_kagg.json`). The offline path was checked against the standard `retrieval_inf.py` path on the same embeddings: identical 500-document rankings for all 500 dev queries.

- **Round-robin** interleaves the k ranked lists (the default so far); **RRF** sums 1 / (60 + rank) over the lists.
- QAMPARI checkpoint: `grid_qampari_multi_hn_lr1e-5_s1750` (best multi-query by clean dev). AmbigQA checkpoint: `grid_ambigqa_multi_hn_lr1e-6_s400`, chosen on **test** (best multi-query test score); its absolute test numbers are therefore optimistic, while comparisons across k and aggregation are unaffected.
- Round-robin at the default k reproduces the grid's full-corpus scores (QAMPARI dev 64.40 / test 38.23; AmbigQA dev 78.80, test 75.09 vs 74.97 in the grid, one question, from embedding the corpus on different GPU hardware).
- One question = 0.2 points on dev, 0.19 on QAMPARI test, 0.12 on AmbigQA test.

### QAMPARI (MRecall@100 / Recall@100)

| k | Dev, round-robin | Dev, RRF | Test, round-robin | Test, RRF |
| --- | --- | --- | --- | --- |
| 1 | 34.0 / 60.7 | 34.0 / 60.7 | 12.05 / 41.9 | 12.05 / 41.9 |
| 3 | 52.8 / 77.0 | 60.4 / 81.0 | 27.68 / 62.1 | 35.22 / 65.9 |
| **5** (default) | 64.4 / 83.6 | **68.6 / 85.6** | 38.23 / 67.5 | **44.26 / 71.7** |
| 8 | 64.8 / 84.3 | 67.6 / 85.3 | 39.55 / 68.8 | 43.50 / 70.6 |

MRecall@10 / Recall@10 at k = 5: round-robin dev 19.6 / 53.2, test 5.46 / 33.4; RRF dev 13.6 / 48.6, test 7.16 / 35.0.

### AmbigQA (MRecall@100 / Recall@100)

| k | Dev, round-robin | Dev, RRF | Test, round-robin | Test, RRF |
| --- | --- | --- | --- | --- |
| 1 | **80.2** / 90.1 | **80.2** / 90.1 | 74.49 / 87.3 | 74.49 / 87.3 |
| **2** (default) | 78.8 / 89.4 | 79.8 / 89.9 | **75.09** / 87.7 | 74.97 / 87.7 |
| 3 | 77.8 / 88.8 | 74.6 / 87.1 | 73.64 / 86.9 | 69.65 / 84.5 |
| 5 | 76.0 / 88.1 | 63.0 / 79.6 | 71.22 / 85.6 | 58.77 / 75.6 |

MRecall@10 is highest at k = 1 (dev 52.0, test 45.34) and falls with k under both aggregations.

### What it shows

- **QAMPARI: RRF is a large, consistent gain.** At k = 5, +4.2 MRecall@100 on dev and +6.0 on test over round-robin, with Recall@100 up 2–4 points; it helps at every k > 1. The gap to the best single-query checkpoint (`grid_qampari_single_hn_lr1e-5_s2500`: dev 72.20, test 52.73) shrinks from 14.5 to 8.5 points on test. Round-robin is better for MRecall@10, since it puts each embedding's top document first.
- **QAMPARI: k = 5 is about right.** k = 8 adds about a point with round-robin and loses about one with RRF, on both dev and test. The first embedding alone (k = 1) is at the untrained base retriever's level (test 12.05 vs 12.24): the value is in combining several.
- **AmbigQA: extra embeddings do not help.** k = 1 is best on dev; k = 1 and 2 are tied on test (5 questions apart); from k = 3 on, both aggregations decline. RRF collapses at k ≥ 3 (k = 5: dev 63.0, test 58.8), consistent with the later embeddings agreeing on off-target passages, which RRF promotes. The best single-query checkpoint (`grid_ambigqa_single_hn_lr1e-6_s150`: dev 82.40, test 76.30) stays ahead.
- **The two datasets point in opposite directions:** many targets per question (QAMPARI, 5–8) benefit from several embeddings combined by RRF; few targets (AmbigQA, mostly 2–3) do not. One checkpoint per dataset, so this shows the pattern, not its variance across checkpoints.

## Notes
