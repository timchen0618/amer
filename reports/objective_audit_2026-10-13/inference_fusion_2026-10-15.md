# Inference fusion study and paper distance subsets (QAMPARI, 2026-10-15)

**Bottom line.** No inference change closes the gap to single-query. The best fusions gain about +1 test point on average (summed scores for joint checkpoints, weighted RRF for frozen ones), within the 2-point tie band. On the paper's large-distance subset (questions whose gold passages are far apart) the joint multi-query models nearly match single-query, and one checkpoint passes it (round-1 hps at step 1000, the nearest saved to its dev-selected 750, with weighted RRF: 42.3 vs 41.7). The whole remaining gap sits on the small-distance subset. Frozen multi-query models gain nothing on either subset: they tie frozen single at 12.5 large-distance.

All numbers: MRecall@100, 5 query embeddings, scored by gold id exactly as `src/eval_utils.eval_retrieve_docs` (the tool reproduces every logged headline number, e.g. C_seq 31.8 / 31.45, C_multi 31.4 / 31.07, C single 31.26).

## Paper distance subsets

- Source: `data/qampari/{large,small}_distance_indices_inf.txt` (thirds, as in the paper; built by `analysis/group_by_distance.py` from the mean pairwise distance of each question's gold embeddings). They index the paper's 1000-question file `diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl`.
- All 531 test questions are in that file (matched by question text; gold ids identical). Mapped indices: `data/qampari/test531_{large,small}_distance_indices_inf.txt` (168 large, 180 small). The test set itself is unchanged.
- No dev question is in the paper file, so the subsets are test-only and never used for selection.

### Frozen document encoder (dev-selected checkpoints, plain RRF)

| Run | Step | Dev | Test | Test, large distance (168) | Test, small distance (180) |
| --- | --- | --- | --- | --- | --- |
| Untrained base | - | 21.40 | 24.48 | 8.93 | 41.11 |
| C single | 500 | 31.8 | 31.26 | 12.50 | 49.44 |
| C_multi | 2250 | 31.4 | 31.07 | 12.50 | 48.33 |
| C_seq (E1) | 1250 | 31.8 | 31.45 | 12.50 | 48.89 |
| C_detach (E2) | 2250 | 31.0 | 29.38 | 11.90 | 45.56 |
| C_seq_detach (E1 + E2) | 2500 | 30.0 | 30.70 | 13.69 | 46.67 |
| C_fednorm | 1250 | 30.6 | 30.13 | 11.31 | 47.22 |
| C_sat50 | 1500 | 31.2 | 31.45 | 11.90 | 48.33 |
| C_sat75 | 2250 | 31.0 | 30.51 | 13.10 | 47.22 |
| C_masked | 1500 | 31.8 | 32.39 | 12.50 | 51.11 |

C_masked's test number comes from this tool; its driver's test evals were still finishing.

### Joint encoder (selected rows, plain RRF)

| Run | Step | Dev | Test | Test, large distance | Test, small distance |
| --- | --- | --- | --- | --- | --- |
| single | 2000 | 64.8 | 53.30 | 41.67 | 61.67 |
| hps_seq | 1000 | 65.2 | 48.40 | 38.69 | 56.67 |
| hps 0.5 | 1000 (nearest with lists to dev-selected 750) | 62.6 | 47.27 | 40.48 | 53.89 |
| masked | 1000 (nearest to dev-selected 750) | 63.8 | 48.59 | 36.31 | 56.11 |
| multi | 1750 (nearest to dev-selected 1500) | 61.0 | 39.36 | 28.57 | 46.11 |

Gap to single: about −1 to −5 on the large subset, −5 to −8 on the small subset. Every joint run is in `results/phaseC/fusion/summary.md`.

## Inference methods 1–5

Tool: `training/inf_retriever/tools/fusion_eval.py` (CPU, saved per-query top-500 lists + query vectors). One setting per method per setting, chosen by mean full-dev MRecall over that setting's checkpoints; then reported on test and both subsets. Frozen: 8 checkpoints. Joint: 15 checkpoints. **Flag:** round-1 hps, masked and multi use the nearest checkpoint with saved lists (hps 1000 and masked 1000 instead of 750, multi 1750 instead of 1500).

| # | Method | Exact rule | Grid |
| --- | --- | --- | --- |
| 1 | First query leads | query 1's top N, then round robin over queries 2–5 | N ∈ {0, 20, 40, 60, 80, 100} |
| 2 | Weighted RRF | Σⱼ wⱼ / (c + rankⱼ); query 1 weight w | w ∈ {1, 1.5, 2, 3, 5}, c ∈ {10, 60} |
| 3a | Best score | passage score = its best cosine over queries | – |
| 3b | Summed scores | sum of cosines; a missing list contributes its 500th score | – |
| 4 | Drop near-duplicates | drop query j if cos > t to an earlier kept query, then RRF | t ∈ {0.90, 0.93, 0.95, 0.97, 0.99} |
| 5 | Agreement gate | mean query–query cos > t → query 1 alone, else RRF | t ∈ {0.85, 0.90, 0.93, 0.95, 0.97} |

### Frozen (8 checkpoints)

| Method | Chosen on dev | Dev | Test | Test, large distance | Test, small distance |
| --- | --- | --- | --- | --- | --- |
| Plain RRF, c = 60 (headline) | `rrf_c60` | 31.10 | 30.88 | 12.43 | 47.92 |
| Round robin | `rr` | 29.83 (-1.27) | 30.02 (-0.87) | 12.80 (+0.37) | 46.81 (-1.11) |
| 1. First query leads (top N of query 1, then round robin) | `lead1_N20` | 30.07 (-1.03) | 29.64 (-1.25) | 12.13 (-0.30) | 45.76 (-2.15) |
| 2. Weighted RRF (query 1 weight w, constant c) | `wrrf_w3_c60` | 31.60 (+0.50) | 31.73 (+0.85) | 13.62 (+1.19) | 49.59 (+1.67) |
| 3a. Score fusion: best score over queries | `max_score` | 28.68 (-2.43) | 28.01 (-2.87) | 10.79 (-1.64) | 45.21 (-2.71) |
| 3b. Score fusion: summed scores | `sum_score` | 31.30 (+0.20) | 31.19 (+0.31) | 12.50 (+0.08) | 48.82 (+0.90) |
| 4. Drop near-duplicate queries (cos > t), then RRF | `dedup0.99_rrf` | 31.32 (+0.22) | 31.26 (+0.38) | 12.87 (+0.45) | 48.54 (+0.63) |
| 5. Agreement gate (mean q-q cos > t: query 1 only) | `gate0.95` | 31.12 (+0.02) | 30.63 (-0.26) | 12.87 (+0.45) | 47.36 (-0.55) |

Means over checkpoints; brackets = change vs plain RRF. Per-checkpoint numbers: `results/phaseC/fusion/summary.md`.

### Joint (15 checkpoints)

| Method | Chosen on dev | Dev | Test | Test, large distance | Test, small distance |
| --- | --- | --- | --- | --- | --- |
| Plain RRF, c = 60 (headline) | `rrf_c60` | 61.41 | 43.85 | 33.69 | 50.04 |
| Round robin | `rr` | 58.32 (-3.09) | 40.13 (-3.73) | 30.20 (-3.49) | 46.52 (-3.52) |
| 1. First query leads (top N of query 1, then round robin) | `lead1_N60` | 59.00 (-2.41) | 41.36 (-2.50) | 31.79 (-1.90) | 47.70 (-2.33) |
| 2. Weighted RRF (query 1 weight w, constant c) | `wrrf_w1.5_c60` | 61.79 (+0.37) | 44.53 (+0.68) | 34.29 (+0.59) | 51.22 (+1.19) |
| 3a. Score fusion: best score over queries | `max_score` | 58.12 (-3.29) | 39.89 (-3.97) | 30.99 (-2.70) | 45.74 (-4.30) |
| 3b. Score fusion: summed scores | `sum_score` | 62.25 (+0.84) | 44.82 (+0.97) | 34.29 (+0.60) | 51.26 (+1.22) |
| 4. Drop near-duplicate queries (cos > t), then RRF | `dedup0.93_rrf` | 61.49 (+0.08) | 44.22 (+0.36) | 34.05 (+0.36) | 50.41 (+0.37) |
| 5. Agreement gate (mean q-q cos > t: query 1 only) | `gate0.85` | 61.49 (+0.08) | 43.93 (+0.08) | 33.89 (+0.20) | 50.07 (+0.04) |

Means over checkpoints; brackets = change vs plain RRF. Per-checkpoint numbers: `results/phaseC/fusion/summary.md`.

**What it shows**

- **No method clears the 2-point tie band on average.** Best: summed scores on joint (+0.84 dev, +0.97 test), weighted RRF with query 1 at 3× on frozen (+0.50 dev, +0.85 test).
- **Giving query 1 more room helps only models whose query 1 is strong** (hps family, masked): first-query-leads N=60 lifts hps_detach 46.3 → 50.1 test and hps_seq 48.4 → 49.3, but sinks multi / sat / fullsamp (weak query 1), so the one shared setting loses on average.
- **Best-score fusion is unstable:** it is the best single result (hps_detach 51.2 test) and the worst on average (−4.0 joint), because one bad query can dominate.
- **Near-duplicate dropping and the gate change little** (joint +0.08 dev each; frozen picks high thresholds, 0.99 and 0.95).
- **The large-distance subset gains as much as the full test set**; no method changes which subset holds the gap.

## Promising combinations (proposed, not run)

1. **Weighted summed scores:** score = w·s₁ + Σⱼ sⱼ (methods 2 + 3b), w tuned on dev. The two best families combined; eval-only.
2. **Single-query list + multi-query extras (hybrid):** the single model's list as the lead list, fused (weighted RRF or summed scores) with a multi model's queries 2–5. Single already wins the small subset and multi nearly matches it on the large one, so the union could beat both; eval-only, needs both runs' saved lists on the same test questions.
3. **Spread-gated hybrid:** per question, use single alone when the multi queries agree (high q–q cos, small-distance-like) and the hybrid otherwise (methods 5 + idea 2). Targets exactly the subset split above.
4. **Method choice per model family:** first-query-leads or weighted RRF for q1-anchored models (hps, masked), summed scores otherwise; selected on dev within each family.
5. **Training + inference:** the E3 docstop + E1 checkpoint (once J's runs land) with the best fusion above.

## Reproduce

    python training/inf_retriever/tools/fusion_eval.py --dirs <ckpt dirs> --single name=<test jsonl> --out results/phaseC/fusion/<set>.json
    python training/inf_retriever/tools/fusion_summary.py results/phaseC/fusion/frozen.json results/phaseC/fusion/joint.json
