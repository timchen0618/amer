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

**Two tracks from Stage B on (user, 2026-10-23).** Track 1 builds on the full-dev best (C_multi). Track 2 builds on the setting best on the paper's large-distance test third, paper cap 0.8 × 10k (its step 10000: 26.0 / 32.77 test, large 23.81 vs single 13.69, query–query cosine 0.68). Every later stage runs on both, doubling the runs. Track 2 reports its **final checkpoint (step 10000)**, a fixed rule: the dev large-distance third (rebuilt with the paper's thresholds; 173 / 165 of 500) sits at 2–5 MRecall for every checkpoint and does not follow test-large, so it cannot select. Track 1 keeps full-dev selection.

**Selection-rule comparison (user, 2026-10-23).** Every run from Stage B on also logs the legacy rule's metric, `eval_loss_sr1` (Hungarian loss on the 3000 held-out dev questions at sampling rate 1.0, every 250 steps); the lowest-loss checkpoint is kept and test-evaluated next to the top 2 by full dev, so the two rules can be compared on test.

Notes:

- The RepLlama-inspired changes (Stage C, added 2026-10-22) run before the deferred ablations (Stage D), so the ablations are done on the final best setting.
- Code: options live in F's own entry script `training/inf_retriever/finetuning_multi_paper.py` (`--sampling_cap`, `--one_label_step`, `--io_proj_init random`) and `src/inbatch_proj.py`; the 2-layer MLP input is to be added there. J's files are not edited; every option defaults to off.
- More negatives: `--negative_ctxs n > 1` (frozen, plain Hungarian) is supported by F's entry script (the shared collator / doc split / loss assume one negative per gold); the frozen doc encoder encodes passages in chunks of 1024. Each question has ~30 mined hard negatives, so at 15 per gold most extra slots are random negatives.
- Long runs: 10,000 steps, warmup 400, checkpoints every 1000 steps plus 2500, 16 h walltime (`f_queue.sh` 5th field).
- Every run is added to `f_queue.sh`'s QUEUE, so J's deferred runs wait for it.
- Each run reports: full dev / test, the paper's large / small distance thirds, best single query, query–query cosine, copy statistics, and round-robin next to RRF.
