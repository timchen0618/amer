# Session F plan: closing the gaps to the paper's training (QAMPARI, frozen document encoder)

User-approved 2026-10-22. Run autonomously, in stages; each stage builds on the previous stage's best setting (selected on full dev, MRecall@100, RRF k = 5; test for reporting only; < 2 points = tie). Push notification after Stage A.

**Base recipe (all runs unless changed):** C_multi = frozen INF-Retriever document encoder, INF-1.5B query model (full finetune, bidirectional), plain Hungarian (the paper's loss), temperature 0.05, lr 1e-5, hard negatives (ratio 0.5, 1 negative per gold), linear 0 → 1 sampling ramp, 2500 steps, no input/output layers. References: C_multi 31.4 / 31.07, frozen single 31.8 / 31.26 (dev / test).

| Stage | What changes | Runs | Question |
| --- | --- | --- | --- |
| A | Sampling schedule {linear 0→1, paper cap `min(10·t/T, 0.8)`, constant 1.0} × length {2500, 10000 steps} | 5 new (linear @ 2500 = C_multi) | Does the paper's schedule or longer training help, alone or together? |
| B | On A's best: paper one-label step × random-init linear input/output layers (2 × 2) | 3 new | Do the paper's extra one-gold step and its projection layers help? |
| C | On B's best, single changes: temperature 0.01 / 0.02 / 0.03; 15 negatives per gold (`--negative_ctxs 15`; memory probe: peak 56 GB of 140 GB, same as 8) | 4 | RepLlama-style contrastive settings: sharper softmax, more hard negatives |
| D | On the best so far, single ablations: no hard negatives; causal attention (query-model ablation) | 2 | Do hard negatives or bidirectional attention matter? |
| E | Only if nothing beats C_multi: data filter (drop 1-gold questions, or keep only 5–8 golds as the paper did) | 1–2 | Does training on the paper's gold-count range help? |
| F | After everything, on the final best: re-test input layers: linear input + output, linear input only, **2-layer MLP input, no output layer** | 3 | Do learned input layers help once the rest of the recipe is right? |

**Stage A re-run on a larger, seeded dev set (user, 2026-10-23).** The 500-question dev set is too small to select reliably. Stage B (both tracks) was cancelled before training; Stage A is re-run (7 runs, `S0_*`, single-query baseline included) on a seeded 90/10 split of the training pool (`tools/build_seed_split.py --seed 0`: 23,732 train / 2,637 dev; no test overlap), as in the legacy code's `train_test_split(test_size=0.1)`. The 2,637 dev questions are both the full-dev selection set and the in-training eval set; a full-dev eval takes ~11 min (≈10 min at 500: corpus loading dominates). Other seeds give other splits for variance runs once the recipe is settled. Note: ~20% of training-split questions are "born in X / died in X" questions that the test set does not have; a test-like dev score without them is reported alongside. GPU use starts after the other session's Llama-8B runs finish (user, 2026-10-23).

**Stage A re-run results (all 7 done, 2026-10-30).** Split as finally built (user 2026-10-27): only questions with 5–8 unique golds are kept before splitting, so dev is 2,459 questions (`data/phaseA/qampari_clean_v3_dev_s0.jsonl`), not the 2,637 above; S0 numbers are comparable with each other, not with the original Stage A runs. Base recipe as above (frozen doc encoder, full finetune, plain Hungarian, HN 0.5, lr 1e-5). Score: multi = MRecall@100 RRF k = 5; single = MRecall@100. Test is evaluated on the top 2 checkpoints by full dev, the final step and the lowest-`eval_loss_sr1` step. Source: `results/phaseC/clean_v3_f_qampari_S0_*.driver.out` (`RESULT full` lines).

| Run | Schedule × length | Dev-selected step: dev / test | Loss-selected step: test | Other tested steps (test) |
| --- | --- | --- | --- | --- |
| S0_single | single query, 2500 | 2500: 39.16 / **32.39** | – | 2250: 32.39 |
| S0_multi | linear 0→1, 2500 | 2250: 38.10 / **30.32** | 2500: 30.13 | 1750: 31.07 |
| S0_cap | cap 0.8, 2500 | 2500: 39.81 / **29.94** | 2500: 29.94 | 2250: 30.32 |
| S0_const | constant 1.0, 2500 | 2250: 41.52 / **29.38** | 2500: 29.57 | – |
| S0_10k | linear 0→1, 10000 | 10000: 46.03 / **31.64** | 9000: 32.02 | – |
| S0_cap_10k | cap 0.8, 10000 | 7000: 46.28 / **31.07** | 10000: 30.89 | 9000: 31.26 |
| S0_const_10k | constant 1.0, 10000 | 4000: 43.76 / **29.00** | 7000: 28.81 | 5000: 29.94, 10000: 28.06 |
| S0_mult10_10k | multiplier 10, no cap (`min(10·t/T, 1)`), 10000 | 6000: 43.31 / **29.76** | 7000: 29.38 | 4000: 30.32, 10000: 28.44 |

- At 2500 steps no multi schedule beats single (−2.1 to −3.0 test). At 10k, linear 0→1 (S0_10k) reaches 31.6–32.0, level with single (32.39, < 2 points = tie); the paper cap and constant schedules do not help at either length.
- Dev and test disagree: 10k runs gain 5–8 dev points over 2500 but ≤ 1.7 test. S0_const_10k peaks on dev at step 4000 and drifts down after (41.64 at 10000).
- Selection rules: the lowest-`eval_loss_sr1` checkpoint tests within 0.4 of the full-dev pick in every run (better in 3 of 6, worse in 2).
- S0_const_10k was cancelled at step ~2,100 on 2026-10-29 when the Slurm account `torch_pr_152_courant` was retired, and was re-run from step 0 on `torch_pr_152_echoi` (job 21396564).
- Paper's large / small test thirds (168 / 180 questions; `tools/fusion_eval.py`, `results/phaseC/fusion_S0_*test.json`), dev-selected step: single 15.48 / 49.44, multi 13.10 / 47.22, cap 13.69 / 45.00, const 13.69 / 45.56, 10k 20.83 / 42.78 (s9000 21.43), cap_10k 19.64 / 43.89 (s9000 22.62), const_10k 13.69 / 41.67. 10k with the linear ramp or the cap beats single on the large third by 4–7 points and loses ~6 on the small third.
- S0_mult10_10k (added 2026-10-31, user): the paper's multiplier 10 without the 0.8 cap. Dev peaks at step 6000 and drifts down (like constant 1.0); test 29.76 at the dev-selected step (large / small 19.64 / 41.67; s4000 30.32, 16.67 / 44.44), below S0_cap_10k (31.07) on every rule, so the cap helps.
- Stage A winner on this split: S0_10k (linear 0→1, 10000 steps). **Stage B base = S0_cap_10k** (user 2026-11-01: the higher-test of S0_cap_10k / S0_mult10_10k). Stage B queued 2026-11-01: S0_cap10k_onelabel, S0_cap10k_ioprojrand, S0_cap10k_onelabel_ioprojrand.

**Stage B results (all 3 done, 2026-11-02).** Base S0_cap_10k (cap 0.8, multiplier 10, 10000 steps) on the S0 split; same scoring and checkpoint rules as Stage A (multi = MRecall@100 RRF k = 5; large / small = the paper's test distance thirds, `results/phaseC/fusion_S0_test.json`). Source: `results/phaseC/clean_v3_f_qampari_S0_cap10k_*.driver.out`.

| Run | Change vs base | Dev-selected step: dev / test | Large / small | Loss-selected step: test | Other tested steps (test) |
| --- | --- | --- | --- | --- | --- |
| S0_cap_10k (base) | – | 7000: 46.28 / 31.07 | 19.64 / 43.89 | 10000: 30.89 | 9000: 31.26 |
| S0_cap10k_onelabel | `--one_label_step` | 10000: **50.71 / 33.15** | **25.00** / 43.89 | 9000: 33.52 (25.60 / 44.44) | – |
| S0_cap10k_ioprojrand | `--io_proj --io_proj_init random` | 7000: 28.26 / 12.62 | 7.14 / 18.33 | 5000: 12.81 | 8000: 12.05, 10000: 11.30 |
| S0_cap10k_onelabel_ioprojrand | both | 10000: 37.01 / 19.96 | 14.29 / 27.22 | 7000: 19.02 | 8000: 19.21 |

- The paper's one-gold step is the first change to beat the base clearly: +4.4 dev, +2.1 test (+2.5 at step 9000), and +5.4 on the large third with the small third unchanged. It is F's best multi run and the first above S0_single (32.39 test; < 2 points, so a tie by the plan's rule), with the large third 10 points above single (15.48).
- Dev was still rising at step 10000 (49.49 / 50.18 / 50.55 / 50.71 at 7k–10k), so the setting is re-run at 20000 steps (S0_cap20k_onelabel, 2026-11-01, user) before Stage C.
- Random-init linear input / output layers hurt badly: dev starts near zero (3.25 at step 1000, vs 37.05 with the one-gold step and no layers) and plateaus at ~28 dev / ~12.5 test. The one-gold step partly offsets it (37.01 / 19.96) but stays 13 test points below the one-gold run without layers. So the paper's projection layers are not part of the best recipe here; Stage F (learned input layers on the final recipe) is the remaining check.
- The selection rules agree within 0.5 test in every run (loss-selected better in 2 of 3).
- The driver's per-embedding (copy / best-single-query) scoring failed on the two io_proj runs; the RRF scores are unaffected.
- Stage B winner: S0_cap10k_onelabel. Stages C–F not started.

**One-gold step at 20,000 steps (S0_cap20k_onelabel, 2026-11-03).** Same flags as S0_cap10k_onelabel, run twice as long (warmup 800, ramp saturating at step 2000 under the 0.8 cap, lr decay over 20k; a fresh run, not a continuation). Full dev every 2000 steps: 39.89 / 45.02 / 47.99 / **49.29** / 48.92 / 48.88 / 48.31 / 48.80 / 49.13 / 49.04 (2k–20k).

| Step | Why tested | Dev | Test | Large / small |
| --- | --- | --- | --- | --- |
| 8000 | dev-selected | 49.29 | **33.15** | 23.21 / 46.11 |
| 12000 | lowest dev loss | 48.88 | 32.77 | 25.00 / 45.56 |
| 18000 | 2nd by dev | 49.13 | 31.83 | 24.40 / 43.89 |
| 20000 | final | 49.04 | 32.39 | 24.40 / 44.44 |

- Twice the steps does not help: dev plateaus at ~49 from step 8000 (below the 10k run's 50.71) and test at the dev-selected step equals the 10k run (33.15 vs 33.15; 10k run's s9000 33.52). Large third 23–25 in both. The 10k run's rising dev at its end came from its own lr decay / schedule, not from under-training.
- So the Stage B best stays S0_cap10k_onelabel (10k steps), the cheaper base for Stage C.

### Step-matched single-query baselines (added 2026-11-04)

The S0 single baseline ran 2,500 steps; the one-gold multi run ran 10,000 steps with two updates per step. Two single runs match that budget (same data, lr, warmup 400, 10k steps): `S0_single_10k` (one update per step) and `S0_single_10k_2step` (`--one_label_step` in single mode: a second update on the same questions with an independently redrawn gold and negatives).

| Run | Step | Dev | Test | Large / small |
|---|---|---|---|---|
| S0_single (2.5k) | 2500 | 39.16 | 32.39 | 15.48 / 49.44 |
| S0_single_10k | 10000 (dev-selected) | 47.46 | **35.59** | 22.62 / 50.00 |
| S0_single_10k | 9000 (lowest dev loss) | 46.60 | 35.78 | 22.62 / 50.56 |
| S0_single_10k_2step | 10000 (dev-selected) | 51.93 | **35.03** | 25.00 / 50.00 |
| S0_single_10k_2step | 8000 (lowest dev loss) | 51.24 | 35.78 | 25.00 / 49.44 |
| S0_single_10k_2step | 9000 (2nd by dev) | 51.65 | 34.84 | 24.40 / 49.44 |
| S0_cap10k_onelabel (multi) | 10000 | 50.71 | 33.15 | 25.00 / 43.89 |

- At matched steps the single baseline beats the best multi run on full test (35.59 vs 33.15). Multi's remaining edge is the large third (25.00 vs 22.62, 2.4 points on ~168 questions), and it loses 6 points on the small third.
- Most of the earlier multi-over-single large-third gap (25.00 vs 15.48) came from training length, not from multiple queries.
- With the second (redrawn-gold) update too, single matches the one-gold multi run on every axis or beats it: dev 51.93 vs 50.71, test 35.03 vs 33.15, large third 25.00 vs 25.00, small third 50.00 vs 43.89. The multi run's large-third edge disappears at matched updates; the second update helps single on dev (+4.5) and the large third (+2.4) but not on full test (35.03 vs 35.59).
- Stage D (no hard negatives, causal attention) therefore runs on both the multi base and S0_single_10k_2step (user 2026-11-04); C / E / F wait.

### Stage D results (all 4 done, 2026-11-05)

Run before Stage C (user 2026-11-04), each on both bases: multi = S0_cap10k_onelabel, single = S0_single_10k_2step. Same scoring and checkpoint rules as Stage B; large / small thirds from `results/phaseC/fusion_S0_test.json` (multi, RRF k = 5) and `fusion_S0_single_test.json`.

| Run | Dev-selected step: dev / test | Large / small | Other tested steps (test) |
|---|---|---|---|
| Multi base (S0_cap10k_onelabel) | 10000: 50.71 / 33.15 | 25.00 / 43.89 | 9000: 33.52 |
| Multi, no hard negatives | 8000: 49.69 / 32.39 | 24.40 / 43.89 | 10000: 32.02, 9000 (loss): 32.39 |
| Multi, causal attention | 10000: 50.63 / 33.52 | 27.38 / 44.44 | 9000 (loss): 34.09 (26.19 / 46.67) |
| Single base (S0_single_10k_2step) | 10000: 51.93 / 35.03 | 25.00 / 50.00 | 8000 (loss): 35.78 |
| Single, no hard negatives | 10000: 50.75 / 34.09 | 24.40 / 48.33 | 8000 (loss): 34.65 |
| Single, causal attention | 10000: 51.16 / 35.22 | 25.00 / 49.44 | 5000 (loss): 35.03 |

- Every change is within the < 2-point tie band on dev and on test, for both bases.
- Hard negatives help slightly and consistently: removing them costs ~1 dev and ~1 test in both multi and single.
- Causal attention is neutral: single 35.22 vs 35.03 test, multi 33.52 vs 33.15 (large third +2.4, within noise on 168 questions).
- Single stays ahead of multi under every D setting (test 34.1–35.2 vs 32.0–34.1). Bidirectional attention with hard negatives stays the recipe.

**Two tracks from Stage B on (user, 2026-10-23).** Track 1 builds on the full-dev best (C_multi). Track 2 builds on the setting best on the paper's large-distance test third, paper cap 0.8 × 10k (its step 10000: 26.0 / 32.77 test, large 23.81 vs single 13.69, query–query cosine 0.68). Every later stage runs on both, doubling the runs. Track 2 reports its **final checkpoint (step 10000)**, a fixed rule: the dev large-distance third (rebuilt with the paper's thresholds; 173 / 165 of 500) sits at 2–5 MRecall for every checkpoint and does not follow test-large, so it cannot select. Track 1 keeps full-dev selection.

**Selection-rule comparison (user, 2026-10-23).** Every run from Stage B on also logs the legacy rule's metric, `eval_loss_sr1` (Hungarian loss on the 3000 held-out dev questions at sampling rate 1.0, every 250 steps); the lowest-loss checkpoint is kept and test-evaluated next to the top 2 by full dev, so the two rules can be compared on test.

Notes:

- The RepLlama-inspired changes (Stage C, added 2026-10-22) run before the deferred ablations (Stage D), so the ablations are done on the final best setting.
- Code: options live in F's own entry script `training/inf_retriever/finetuning_multi_paper.py` (`--sampling_cap`, `--one_label_step`, `--io_proj_init random`) and `src/inbatch_proj.py`; the 2-layer MLP input is to be added there. J's files are not edited; every option defaults to off.
- More negatives: `--negative_ctxs n > 1` (frozen, plain Hungarian) is supported by F's entry script (the shared collator / doc split / loss assume one negative per gold); the frozen doc encoder encodes passages in chunks of 1024. Each question has ~30 mined hard negatives, so at 15 per gold most extra slots are random negatives.
- Long runs: 10,000 steps, warmup 400, checkpoints every 1000 steps plus 2500, 16 h walltime (`f_queue.sh` 5th field).
- Every run is added to `f_queue.sh`'s QUEUE, so J's deferred runs wait for it.
- Each run reports: full dev / test, the paper's large / small distance thirds, best single query, query–query cosine, copy statistics, and round-robin next to RRF.
