# Fusion combinations (QAMPARI, 2026-10-16)

Follow-up to `inference_fusion_2026-10-15.md`. The user asked for the promising combinations to be tried. Combinations 1–4 ran; 5 (best rule on J's docstop / E1 checkpoints) waits for those runs.

All numbers: MRecall@100. Dev = full dev (500), test = full test (531), large / small = the paper's distance thirds of test (168 / 180; test only). One parameter per method, chosen on mean dev over the setting's checkpoints. Means are over 10 frozen (dev-selected F runs, incl. C_hps05 and C_seq_used) and 16 joint checkpoints (as in the fusion study, plus hps_seq_detach s1000). The single-query references are C single s500 (frozen, 31.8 / 31.26) and joint single s2000 (64.8 / 53.30).

## Bottom line

- **Frozen:** no combination clears the tie band. The best (single list + all 5 multi queries, RRF, single weight 2) gives +1.3 test on average and 32.8 / 33.15 on its best-by-dev checkpoint, but a second *single* model does as well (control below).
- **Joint:** the hybrids are the first multi-query results above joint single: single list + all 5 queries (RRF, single weight 3) averages 67.3 / 54.39 (+2.5 dev / +1.1 test; large +1.9, small +0.1). **But fusing joint single with another single-query model gains the same** (67.2 / 54.43 with single_lr3e-5 s2000). The gain is ensembling two models, not multiple queries. Still inside the 2-point band.
- **Weighted summed scores** (combination 1) and the **gated hybrid** (3) add nothing beyond their parts. A **rule chosen per family** (4) adds +0.8 test on joint over the best one-for-all rule, still far below single (45.65 vs 53.30).

## Combinations 1–3

### Frozen (10 checkpoints)

| Method | Chosen on dev | Dev | Test | Large | Small | Best checkpoint by dev: dev / test / large / small |
| --- | --- | --- | --- | --- | --- | --- |
| Single model alone (reference) | `single` | 31.80 | 31.26 | 12.50 | 49.44 | |
| Plain RRF over the 5 queries | `rrf_c60` | 31.20 | 31.13 | 12.74 | 48.33 | 32.00 / 32.02 / 13.69 / 49.44 (C_seq_used) |
| 1. Weighted summed scores | `wsum_w2` | 31.58 | 31.81 | 13.39 | 49.78 | 32.20 / 32.02 / 12.50 / 50.56 (C_seq_used) |
| 2. Hybrid: single (w) + queries 2–5, RRF | `hyb_rrf_ws3` | 31.70 | 32.56 | 12.44 | 50.39 | 32.60 / 32.77 / 11.31 / 52.22 (C_seq) |
| 2b. Hybrid: single (w) + all 5 queries, RRF | `hyb5_rrf_ws2` | 31.86 | 32.60 | 12.80 | 50.67 | 32.80 / 33.15 / 12.50 / 52.78 (C_seq) |
| 2c. Hybrid: single (w) + queries 2–5, summed scores | `hyb_sum_ws3` | 31.70 | 32.56 | 12.50 | 50.72 | 32.20 / 31.64 / 11.90 / 49.44 (C_detach) |
| 3. Gated hybrid (single alone when queries agree) | `gate0.93_hyb_rrf_ws3` | 31.64 | 31.86 | 12.86 | 49.39 | 32.40 / 32.20 / 14.29 / 49.44 (C_hps05) |

### Joint (16 checkpoints)

| Method | Chosen on dev | Dev | Test | Large | Small | Best checkpoint by dev: dev / test / large / small |
| --- | --- | --- | --- | --- | --- | --- |
| Single model alone (reference) | `single` | 64.80 | 53.30 | 41.67 | 61.67 | |
| Plain RRF over the 5 queries | `rrf_c60` | 61.59 | 43.97 | 33.78 | 50.21 | 65.20 / 48.40 / 38.69 / 56.67 (hps_seq) |
| 1. Weighted summed scores | `wsum_w1.5` | 62.52 | 45.53 | 34.78 | 52.33 | 67.00 / 50.85 / 38.10 / 61.11 (hps_detach) |
| 2. Hybrid: single (w) + queries 2–5, RRF | `hyb_rrf_ws3` | 66.73 | 53.75 | 42.56 | 60.55 | 68.20 / 54.05 / 42.26 / 62.22 (hps_detach) |
| 2b. Hybrid: single (w) + all 5 queries, RRF | `hyb5_rrf_ws3` | **67.30** | **54.39** | **43.53** | 61.74 | 69.00 / 54.05 / 44.05 / 60.00 (hps_w010) |
| 2c. Hybrid: single (w) + queries 2–5, summed scores | `hyb_sum_ws3` | 67.05 | 54.26 | 43.04 | 61.53 | 69.00 / 55.18 / 45.24 / 63.33 (hps_detach) |
| 3. Gated hybrid | `gate0.97_hyb_rrf_ws3` | 66.71 | 53.75 | 42.56 | 60.55 | (= 2; the gate almost never fires) |

The dev-chosen weights sit at the edge of the grid (ws = 3): the more the single model's list counts, the better.

## Control: single + another single-query model

Same fusion (RRF, c = 60, single weight r : partner 1, r chosen on dev).

| Setting | Partner (alone: dev / test) | Ensemble: dev / test / large / small |
| --- | --- | --- |
| Frozen | A_single s2250 (30.8 / 31.45) | 31.4 / 32.77 / 13.69 / 49.44 |
| Frozen | C_single s2500, same run (31.4 / 34.09) | 31.6 / 33.52 / 14.29 / 51.67 |
| Joint | single_lr3e-5 s2000 (60.4 / 45.57) | **67.2 / 54.43 / 43.45 / 61.11** |
| Joint | B_single s2250 (64.0 / 51.79) | 66.0 / 54.05 / 42.26 / 62.78 |
| Joint | single s2250, same run (64.2 / 52.17) | 65.2 / 53.30 / 42.26 / 61.67 |

A weaker single-query partner (45.57 test alone) lifts joint single as much as the best multi-query hybrid, on both distance thirds. So the hybrid's gain is a two-model ensemble effect. The multi-query lists add no retrieval the second single model does not.

## Combination 4: rule chosen per model family

Families: `hps` (runs with the single-query term) vs the rest; within each family the fusion_eval method with the highest mean dev.

| Setting | Plain RRF | One rule for all | Rule per family |
| --- | --- | --- | --- |
| Frozen (12; hps → `lead1_N100`, rest → `wrrf_w3_c60`) | 30.87 / 30.81 | `wrrf_w3_c60` 31.37 / 31.42 | 31.38 / 31.45 |
| Joint (15; hps → `wrrf_w5_c10`, rest → `sum_score`) | 61.41 / 43.85 | `sum_score` 62.25 / 44.82 | 62.67 / 45.65 |

(dev / test.) Frozen has one hps checkpoint, so its family rule is fitted on one run.

## What this means

- Inference alone does not make multi-query beat single-query, frozen or joint. The joint gap (53.3 vs 48.4 for the best multi-query model) stays a training problem.
- The one multi-query result above single (joint hybrid, +1.1 test) is matched by a plain two-single-model ensemble. Reporting it as a multi-query gain would need it to beat that control.
- Combination 5 (best rule on docstop / E1 checkpoints) can reuse `fusion_combo.py` once J's E3 runs land, but given the control it is unlikely to change the conclusion.

## Reproduce

    python training/inf_retriever/tools/fusion_combo.py --single_dir <single ckpt dir> --dirs <multi ckpt dirs> --out results/phaseC/fusion/combo_<set>.json
    python training/inf_retriever/tools/fusion_combo.py --single_dir <single ckpt dir> --partners <other single dirs> --out results/phaseC/fusion/control_<set>.json
    python training/inf_retriever/tools/fusion_combo_summary.py --combo results/phaseC/fusion/combo_frozen.json results/phaseC/fusion/combo_joint.json \
        --fe frozen=results/phaseC/fusion/frozen.json,results/phaseC/fusion/extra_besttest.json joint=results/phaseC/fusion/joint.json

## Addendum (2026-10-17): new frozen checkpoints

Combinations rerun on the frozen checkpoints finished after the study (`results/phaseC/fusion/combo_frozen_new.json`). dev / test / large / small:

| Checkpoint | Plain RRF | 1. Weighted summed (w 2) | 2b. Hybrid, single (w 2) + all 5 |
| --- | --- | --- | --- |
| C_seq_ioproj s1250 (dev-selected) | 32.6 / 31.45 / 13.10 / 48.89 | 33.0 / 32.02 / 13.69 / 49.44 | 33.2 / 33.33 / 13.69 / 51.67 |
| C_seq_ioproj s2500 (best test) | 30.2 / 33.15 / 17.26 / 50.56 | 30.6 / 33.33 / 17.26 / 50.56 | 31.6 / 34.46 / 17.26 / 52.78 |
| C_hps025 s1750 | 31.8 / 33.33 / 14.88 / 52.22 | 31.6 / 33.90 / 14.29 / 53.33 | 31.8 / 34.09 / 14.29 / 52.22 |
| C_hps010 s1750 | 31.6 / 32.58 / 13.10 / 50.56 | — | 32.6 / 34.84 / 14.88 / 51.67 |

Frozen single alone: 31.8 / 31.26 / 12.50 / 49.44; single + a second single model (control): 31.4–31.6 / 32.77–33.52 / 13.69–14.29 / 49.44–51.67. The hybrids sit about 1 point above the control on test but tie it on dev, so the conclusion is unchanged: no dev-visible gain from multi-query.
