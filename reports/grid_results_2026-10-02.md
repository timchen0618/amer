# 24-run grid results

*Status: placeholder, results to be filled in.*

## Code

| Commit | Contents |
| --- | --- |
| `4298660` | Training and data code the grid ran on: clean splits (`data_creation/build_clean_splits.py`), hard-negative miner, training driver (`phaseC_run.sh`, `phaseC_train.sbatch`, `grid_launch.sh`) |
| `c9e1b9e` | Eval pipeline used for the full-corpus evaluations (`retrieval_inf.py`, `full_eval_queue.sh`, embed / retrieve sbatch scripts) |

Branch `fsdp-clean-recipe`. Grid launched 2026-09-28; launch-time snapshot of the uncommitted diff: `results/phaseC/grid_code_snapshot.patch`.

Known issues that affect how these results read: `code_audit_2026-09-28.md` (in particular G1, G2, L1, C2).

## Grid

dataset {QAMPARI, AmbigQA} × mode {single, multi} × data {no hard neg., hard neg.} × LR {1e-6, 3e-6, 1e-5}.

## Results

### QAMPARI

### AmbigQA

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
