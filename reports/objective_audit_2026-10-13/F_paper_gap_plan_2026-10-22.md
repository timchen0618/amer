# Session F plan: closing the gaps to the paper's training (QAMPARI, frozen document encoder)

User-approved 2026-10-22. Run autonomously, in stages; each stage builds on the previous stage's best setting (selected on full dev, MRecall@100, RRF k = 5; test for reporting only; < 2 points = tie). Push notification after Stage A.

**Base recipe (all runs unless changed):** C_multi = frozen INF-Retriever document encoder, INF-1.5B query model (full finetune, bidirectional), plain Hungarian (the paper's loss), temperature 0.05, lr 1e-5, hard negatives (ratio 0.5, 1 negative per gold), linear 0 → 1 sampling ramp, 2500 steps, no input/output layers. References: C_multi 31.4 / 31.07, frozen single 31.8 / 31.26 (dev / test).

| Stage | What changes | Runs | Question |
| --- | --- | --- | --- |
| A | Sampling schedule {linear 0→1, paper cap `min(10·t/T, 0.8)`, constant 1.0} × length {2500, 10000 steps} | 5 new (linear @ 2500 = C_multi) | Does the paper's schedule or longer training help, alone or together? |
| B | On A's best: paper one-label step × random-init linear input/output layers (2 × 2) | 3 new | Do the paper's extra one-gold step and its projection layers help? |
| C | On B's best, single changes: temperature 0.01 / 0.02 / 0.03; more negatives per gold (largest that fits in GPU memory, > 3) | 4 | RepLlama-style contrastive settings: sharper softmax, more hard negatives |
| D | On the best so far, single ablations: no hard negatives; causal attention (query-model ablation) | 2 | Do hard negatives or bidirectional attention matter? |
| E | Only if nothing beats C_multi: data filter (drop 1-gold questions, or keep only 5–8 golds as the paper did) | 1–2 | Does training on the paper's gold-count range help? |
| F | After everything, on the final best: re-test input layers: linear input + output, linear input only, **2-layer MLP input, no output layer** | 3 | Do learned input layers help once the rest of the recipe is right? |

Notes:

- The RepLlama-inspired changes (Stage C, added 2026-10-22) run before the deferred ablations (Stage D), so the ablations are done on the final best setting.
- Code: options live in F's own entry script `training/inf_retriever/finetuning_multi_paper.py` (`--sampling_cap`, `--one_label_step`, `--io_proj_init random`) and `src/inbatch_proj.py`; the 2-layer MLP input is to be added there. J's files are not edited; every option defaults to off.
- Long runs: 10,000 steps, warmup 400, checkpoints every 1000 steps plus 2500, 16 h walltime (`f_queue.sh` 5th field).
- Every run is added to `f_queue.sh`'s QUEUE, so J's deferred runs wait for it.
- Each run reports: full dev / test, the paper's large / small distance thirds, best single query, query–query cosine, copy statistics, and round-robin next to RRF.
