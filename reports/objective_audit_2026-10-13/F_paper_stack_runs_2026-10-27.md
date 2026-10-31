# Session F: paper-stack runs with INF-Retriever as the query model (P1–P3, single + multi)

Planned 2026-10-27 with the user. Goal: run the paper's recipe with `infly/inf-retriever-v1-1.5b` as the query model, then isolate data (P1 vs P2) and training stack (P2 vs P3). Each multi-query run has a single-query counterpart trained the same way.

## Runs

| Run | Code | Data | Mode |
| --- | --- | --- | --- |
| P1-multi | paper (`train_distributed.py`) | R's production QAMPARI data (31,917 q; test leak removed, golds stripped) | multi_SS |
| P1-single | paper | same as P1-multi | single |
| P2-multi | paper | seeded split s0 (22,134 train / 2,459 dev, 5–8 golds), embedded with R's script | multi_SS |
| P2-single | paper | same as P2-multi | single |
| P3-multi | F stack (`finetuning_multi_paper.py`) | seeded split s0 (JSONL; frozen INF encodes on the fly) | multi_SS |
| P3-single | F stack | same as P3-multi | single |

**Shared recipe (paper, except batch):**
- batch 50/GPU × 2 GPUs; LoRA r64 α16 dropout 0.1; lr 5e-5; warmup 5%; 120 epochs;
- causal attention; random-init linear input/output layers;
- temperature 0.05; plain Hungarian (multi) / contrastive (single);
- random negatives only, 1 per gold;
- multi: one-gold step (`mix_one_label_shuffled`), sampling rate `min(1 · step/T, 0.8)` (multiplier 1, user 2026-10-27);
- prompt "Instruct: Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: …" in training and inference;
- checkpoint = lowest dev loss (sampling rate 1.0), evaluated every 500 steps.

**Differences by design:**
- P1/P2: P1 dev = paper's internal 10% split (seed 42); P2 dev = the s0 dev file (`--valid_path`). In-batch negatives: the paper's `HungarianContrastiveLoss` also gathers golds and negatives across GPUs (`dist_utils.varsize_gather`), so P1/P2 and P3 match here (corrected 2026-10-27; the earlier "per GPU" note was wrong).
- P3: F's in-batch negatives (gathered across GPUs), F's own gold/negative encoding and negatives (user: no matching of vectors or negatives), dev = s0 dev file.
- P3 also keeps F-stack internals: bf16 autocast (paper: fp16), AdamW beta2 0.98 / eps 1e-6 and no gradient clipping (paper: clip 5.0), F's one-gold step reuses the current batch's questions (paper: a separately shuffled loader), and 26,500 steps (≈ 120 epochs of 22,134 at 100/step; eval/save interval 500).

**Eval (all six):** full 25.86M index (`shared_frozen_base_512`), 5 queries (single: 1), top 500 per query, MRecall@100 / @10 with round-robin and RRF (smoothing 60; this is F's headline "RRF, k = 5", where k = 5 is the number of queries), with the paper's scorer (R's `score.py`), on QAMPARI test (531 q, same file and order as F's test) and the low-similarity subset; plus pairwise cosine of each question's 5 query vectors (`qsim.py`). P1/P2 query vectors from the paper's `gen_ret_and_eval.py`; P3 from `retrieval_inf.py` (paper prompt read from the checkpoint). Script: `scripts/f_inf/eval.sbatch` (worktree `f-paper-inf`).

## Results (QAMPARI test, 531 q; lowsim = 177 q)

MRecall (%) with the paper's scorer; RR = round-robin, RRF = reciprocal-rank fusion (smoothing 60) over the 5 queries. Single runs have one query, so RR = RRF. Query cosine = mean over questions of the mean pairwise cosine of the 5 query vectors.

| Run | @100 all RR | @100 all RRF | @100 lowsim RR | @100 lowsim RRF | @10 all RR | @10 all RRF | @10 lowsim RR | @10 lowsim RRF | query cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1-multi | 21.66 | 23.92 | 14.12 | 14.69 | 2.07 | 2.45 | 0.56 | 0.56 | 0.838 |
| P1-single | 23.54 | 23.54 | 12.43 | 12.43 | 2.82 | 2.82 | 0.56 | 0.56 | – |
| P2-multi | 20.15 | 22.60 | 9.60 | 10.17 | 1.51 | 2.82 | 0.56 | 0.56 | 0.847 |
| P2-single | 21.09 | 21.09 | 8.47 | 8.47 | 2.45 | 2.45 | 0.56 | 0.56 | – |
| P3-multi | 21.47 | 22.98 | 11.86 | 12.43 | 2.07 | 2.82 | 0.56 | 0.00 | 0.860 |
| P3-single | 21.28 | 21.28 | 9.04 | 9.04 | 3.58 | 3.58 | 0.56 | 0.56 | – |

Reference: untrained INF-Retriever on the same test and full index, MRecall@100 = 24.48 (F0 in PLAN.md, F's prompt and eval script, not this scorer).

## Work items

| # | Item | Needs GPU | Status |
| --- | --- | --- | --- |
| 1 | F branch `f-paper-inf` + worktree `.claude/worktrees/f-paper-inf` from R's `paper-repro` (84c0fc2) | no | done |
| 2 | `inf` in data builder (`repro_build_data.py`), launcher `scripts/f_inf/train.sbatch`, eval `scripts/f_inf/eval.sbatch` | no | done |
| 3 | P1 data: R's vectors tokenized for INF → `f_paper_data/training_datasets/inf/qampari/...` (31,917 rows; prompt checked) | no | done (job 21141126) |
| 4 | P2 data: `f_prep` (train 9,001/5,726/3,971/3,436; dev 1,010/648/438/363 by gold count 5–8), embed, `hf` | 1 GPU | done (embed 21141130, hf 21141132: 22,134 train / 2,459 dev rows) |
| 5 | `--valid_path` in the paper trainer (`DataHandler.use_external_dev`) | no | done |
| 6 | P3 (main checkout): `--query_instruct_task paper` (training; retrieval reads it from the checkpoint), `--save_best_dev_loss`, single-query output layer (`EmbeddingModelFrozenDocEncSingleQueryProj`), eval_loss_sr1 for single and io_proj runs, `FREQ_OVERRIDE` in phaseC_train.sbatch | no | done |
| 7 | Smoke tests: P3 multi 21141555, P3 single 21141556 (40 steps); paper stack tested by the P1-multi run itself | 2 GPU | P3 multi passed; single passed on re-run 21158045 after the dev fix |
| 8 | Train: P1-multi 21141576 (48 h walltime: the cluster caps > 48 h jobs at 4 GPUs/user); then P2-multi, P3-multi; singles after | 2 GPU each | multi: P1 21141576, P2 21169371, P3 21158043; single: P1 21176778, P2 21176779 (after P3-multi), P3 21176780 (after P2-multi) |
| 9 | Eval + report + push notification | 1 GPU | todo |

**GPU gate:** these runs do NOT wait for R's release cue once the user consents. The Stage A re-run (S0_*) still waits for it.

## Status log

- 2026-10-27: plan agreed; seeded split rebuilt (5–8 unique golds, train golds de-duplicated: 26,369 → 24,593 pooled, 1,776 dropped for < 5 golds). Waiting for the user's consent to start.
- 2026-10-27: user go-ahead (compete with R for GPUs; P runs do not wait for R's cue). P1 data built; P2 embed + P3 smoke tests + P1-multi submitted.
- 2026-10-27: P2 data built (22,134 / 2,459 rows, paper prompt); P2-multi submitted (21149407).
- 2026-10-27: P3 multi smoke passed (paper prompt, LoRA r64, batch 50/GPU, dev loss 6.27 → 5.38 over 40 steps, best_dev_loss saves; ~3.6 s/step, ~5 min per eval → ~31 h for 26,500 steps). P3-multi submitted (21157410, 2-day walltime).
- 2026-10-27: P3 single smoke (21141556) crashed in its first dev eval: the s0 dev records kept their 30 mined hard negatives (the old clean dev sets have none), and the single-query eval needs exactly one negative per question. The multi dev loss would also have been scored against hard negatives. Fix: `build_seed_split.py` writes dev with `hard_negative_ctxs = []` (train file byte-identical, md5 4680fb3d…). Cancelled the pending P3-multi (21157410, had not started) and resubmitted it (21158043); single smoke re-run (21158045). P2's dev is unaffected (it uses R-style random negatives).
- 2026-10-27: P1-multi running (21141576; step-0 dev loss 13.99 on its 32-batch internal dev). P2-multi (21149407) crashed in its step-0 dev-loss pass: the paper's `src/model.py` asserts `loss_mask.float().mean(dim=0).sum() == output_len` exactly, and the last s0 dev batch's float mean gave 4.9999995 vs 5. Made those mean-based asserts tolerant (|diff| < 1e-4; branch `f-paper-inf`, no change to any computation); P2-multi resubmitted (21169371). P3-multi (21158043) running.
- 2026-10-27: P1-multi speed: 575 iterations/epoch (288 batches + the one-gold step), 2.28 s/iteration incl. eval → ≈ 44 h for 120 epochs, inside the 48 h walltime with little margin. If it times out, its best_model (lowest dev loss so far) is already saved and is what gets evaluated. P2 (443 it/epoch) ≈ 34 h.
- 2026-10-27: dev losses are not comparable across stacks. P1/P2 log `eval loss` = (sum of per-batch losses over both GPUs) / (batches per GPU), i.e. 2× the mean batch loss, and each batch is scored against golds + negatives gathered from both GPUs (~100 questions). P3's `eval_loss_sr1` runs on the main process only with the gather disabled, a per-question mean over 50-question batches (half the candidate pool, ≈ ln 2 lower for an uninformed scorer). The dev sets and document vectors also differ (P1: internal 10% of R's data; P3: s0 dev, F-encoded). Each loss is only used to pick its own run's checkpoint.
- 2026-10-27: P3 single smoke passed on the fixed dev (dev loss 4.66 → 4.01 over 40 steps, best_dev_loss saves). Singles submitted: P1-single 21176778 (now), P2-single 21176779 (afterany P3-multi), P3-single 21176780 (afterany P2-multi, 26,500 steps), keeping F at ≤ 8 GPUs. P3-multi dev loss 6.27 → 3.76 at step 500; P1-multi 13.99 → 8.23 at step 1000.
- 2026-10-28: P1-single finished (21176778, 5.4 h: 34,560 steps at ~0.56 s/step). Dev loss 10.58 → 1.84 (best ≈ step 30,500), flat after ~25k. Test eval submitted (21203037, tag p1_single); this is also the first end-to-end run of `scripts/f_inf/eval.sbatch`.
- 2026-10-28: P1-single test eval done (21203037, 8.7 min; eval pipeline works end to end): MRecall@100 23.54 all / 12.43 lowsim, @10 2.82 / 0.56.
- 2026-10-28: P2-multi finished (21169371, 17.3 h, 120 epochs; best dev loss 5.79 at 16:12, the last evals still improving slowly). Test eval submitted (21258505, tag p2_multi). P3-single (21176780) dependency met, waiting for GPUs. Progress at 16:40: P1-multi epoch 103/120 (19.1 h), P3-multi step 14,500/26,500 (dev loss 2.08).
- 2026-10-28: P2-multi test eval done (21258505, 17.6 min): MRecall@100 RR 20.15 / RRF 22.60 all, 9.60 / 10.17 lowsim; @10 RRF 2.82 all. Query cosine 0.847 (adjacent steps 0.88 / 0.76 / 0.84 / 0.87; 0.2% of questions > 0.95). Below P1-single (23.54) and the untrained base (24.48).
- 2026-10-28: P1-multi finished (21141576, 22.3 h, 120 epochs; best dev loss 4.80 at 19:31, near the end). Test eval submitted (21295883, tag p1_multi). Its 2 GPUs freed, so P2-single was moved up: cancelled the pending 21176779 (waiting on P3-multi) and resubmitted without the dependency (21295886). F training GPUs: P3-multi, P3-single, P2-single, S0_10k = 8.
- 2026-10-28: P1-multi test eval done (21295883, 9.5 min): MRecall@100 RR 21.66 / RRF 23.92 all, 14.12 / 14.69 lowsim; @10 RRF 2.45 all. Query cosine 0.838 (adjacent 0.87 / 0.73 / 0.83 / 0.84; 0.8% of questions > 0.95). vs P1-single: +0.4 all, +2.3 lowsim (RRF); still below the untrained base (24.48).
- 2026-10-28: P3-single finished (21176780, 4.9 h, 26,500 steps; best dev loss 0.745 at step 24,000, flat from ~20k). Test retrieval from best_dev_loss with saved query vectors (21302246), then the paper-scorer eval (21302247, tag p3_single, afterok).
- 2026-10-28: P3-single test retrieval 21302246 failed at load: retrieve_ckpt's nli environment has no peft (first LoRA checkpoint through it). Added RET_ENV (default nli; RET_ENV=div uses the training environment: peft 0.14, faiss 1.9, transformers 4.56) to retrieve_ckpt.sbatch; resubmitted retrieval 21302826 and eval 21302827.
- 2026-10-28: retry 21302826 failed in query embedding: retrieval moves only model.encoder to GPU, but P3-single's output layer (model.out_proj, called from the encoder's forward hook) stayed on CPU. load_retriever now also registers out_proj on the returned encoder (src/inference_utils.py). Resubmitted retrieval 21303421 + eval.
- 2026-10-28: retry 21303421 embedded the test queries (paper prompt read from the checkpoint; saved results/f_paper/P3_single/test_qemb.npy, 531 × 1536) and then failed at the FAISS step: the div environment's faiss has no GPU support (index_cpu_to_gpu). The paper-scorer eval needs only the saved query vectors (it searches with search.py in the nli environment), so it was submitted directly. For P3-multi: embed with RET_ENV=div, search via eval.sbatch.
- 2026-10-28: P3-single test eval done (21303859, 10.4 min): MRecall@100 21.28 all / 9.04 lowsim; @10 3.58 / 0.56.
- 2026-10-29: P2-single finished (21295886, 4.0 h; dev loss 10.58 → 2.60, best at the last eval). Test eval submitted (21318797, tag p2_single).
- 2026-10-29: P2-single test eval done (21318797, 9.4 min): MRecall@100 21.09 all / 8.47 lowsim; @10 2.45 / 0.56. P2-multi vs P2-single (RRF): +1.5 all, +1.7 lowsim. P3-multi at step 21,500/26,500 (dev loss 2.03).
- 2026-10-29: P3-multi finished (21158043, 33.6 h, 26,500 steps; best dev loss 2.004 at step 26,000). Query embedding (RET_ENV=div, K=5, best_dev_loss) 21332094, then eval (afterany; needs only the saved vectors).
- 2026-10-29: P3-multi test eval done (21332095, 9.7 min): MRecall@100 RR 21.47 / RRF 22.98 all, 11.86 / 12.43 lowsim; @10 RRF 2.82 all. Query cosine 0.860 (adjacent 0.91 / 0.78 / 0.85 / 0.87; 0.6% of questions > 0.95). All six P runs evaluated.
- 2026-10-29 summary (RRF, MRecall@100 all / lowsim): multi beats single in every pair on lowsim (P1 +2.3, P2 +1.7, P3 +3.4) and overall (P1 +0.4, P2 +1.5, P3 +1.7); with round-robin, multi trails single overall in P1/P2 and is level in P3. Data (P1 vs P2, paper code): R's production data ahead by 1.3–4.5 (multi) and 2.5–4.0 (single). Code (P3 vs P2, same s0 data): F stack ahead for multi (+0.4 all, +2.3 lowsim) and level for single (+0.2 / +0.6). Query vectors stay close in every multi run (mean pairwise cosine 0.84–0.86; step 2→3 the most distinct, 0.73–0.78). No run beats the untrained INF-Retriever (24.48, F's script; same-scorer base not yet run).

## Follow-up (user 2026-10-29): hard negatives vs LoRA, on P3-multi first

Gap to F's Stage A runs on the same s0 split (S0_multi test 30.1–31.1, S0_single 32.4, F's scorer) vs P3-multi 22.98 / P3-single 21.28 (paper scorer). S0_multi vs P3-multi options differ in: hard negatives (0.5 vs 0), full fine-tuning vs LoRA r64, lr 1e-5 vs 5e-5, 2,500 vs 26,500 steps, bidirectional vs causal attention, no output layer vs random-init layers, no one-gold step vs one-gold step, sampling 0→1 linear vs cap 0.8, F's prompt vs the paper prompt, and the scorer.

| Run | Job | Change vs P3-multi |
| --- | --- | --- |
| P3_multi_hn | 21339901 | `--negative_hard_ratio 0.5` (as F's runs: half of each question's negatives from its 30 mined hard negatives) |
| P3_multi_sft | 21339902 | full fine-tuning (no `--use_lora`), lr 1e-5 (F's full-FT lr; 5e-5 is the LoRA lr) |
| P3_multi_hn_sft | 21339903 | both |

Same 26,500 steps, eval every 500, lowest-dev-loss checkpoint. Calibration (same paper scorer, test): S0_multi step 2250 (21339910 → 21339916), S0_single step 2500 (21339912 → 21339917), untrained INF-Retriever (cal_base). Purpose: separate the scorer from the recipe in the 23 vs 30 gap.

Single-query counterparts (user 2026-10-29: "the same things for P3-single too"), each queued `afterany` its multi run so F stays at ≤ 8 GPUs: P3_single_hn 21340020 (after 21339901; both resubmitted, see below), P3_single_sft 21340021 (after 21339902), P3_single_hn_sft 21340022 (after 21339903). Same changes vs P3-single as above.
- 2026-10-29: calibration, untrained INF-Retriever through the paper scorer (21339920): MRecall@100 24.48 all / 9.60 lowsim, @10 2.26 — identical to F's script (24.48), so the two scorers agree on the base model.
- 2026-10-29: calibration, S0_single step 2500 through the paper scorer (21339917): MRecall@100 32.39 all / 16.38 lowsim, @10 5.27 / 1.13 — identical to F's script (32.39). The scorer is not the gap: P3-single 21.28 vs S0_single 32.39 on the same s0 data, F stack and scorer is a recipe difference.
- 2026-10-29: calibration, S0_multi step 2250 through the paper scorer (21339916): MRecall@100 RR 29.76 / RRF 30.32 all (F's script: 30.32), 15.25 / 14.69 lowsim; @10 RRF 4.14. Query cosine 0.883 (adjacent 0.92 / 0.96 / 0.97 / 0.97). Calibration done: scorers agree on all three; the P3 gap (22.98 vs 30.32 multi, 21.28 vs 32.39 single) is the recipe.
- 2026-10-29: P3_multi_hn 21339901 killed by the system (CANCELLED by uid 0, SIGTERM, node gh125) at step 2,150 after 3.2 h (dev loss 2.62 → 2.50 at steps 1500/2000, no fault in the run). Resubmitted from scratch as 21350659; cancelled P3_single_hn 21340020 (its afterany would have started it now) and re-chained it as 21350660 after 21350659.
- 2026-10-29 16:08: every job under this account was cancelled by root at once (CANCELLED by uid 0), including R's repro_llama runs, F's S0_const_10k and all six P3 variants (multi_sft / multi_hn_sft at ~5.1 h, multi_hn resubmit at 2.0 h; singles had not started). No reservation; partitions up. Looks administrative; not resubmitted pending the user.

- 2026-10-29 23:20: the 16:08 cancel was the retirement of account torch_pr_152_courant. Resubmitted on torch_pr_152_echoi from step 0: P3_multi_hn 21396266, P3_multi_sft 21396267, P3_multi_hn_sft 21396268, P3_single_hn 21396269, P3_single_sft 21396270, P3_single_hn_sft 21396271 (singles afterany their multi).

### Follow-up results (test, paper scorer, best_dev_loss checkpoint)

| Run | @100 all RR / RRF | @100 lowsim RR / RRF | @10 all RR / RRF | @10 lowsim RR / RRF | query cosine (adjacent) |
|---|---|---|---|---|---|
| P3-multi (LoRA, no HN; reference) | 21.47 / 22.98 | 11.86 / 12.43 | 2.07 / 2.82 | 0.56 / 0.00 | 0.860 |
| P3_multi_hn_sft (full FT, HN 0.5, lr 1e-5) | 18.64 / 22.03 | 19.21 / 19.21 | 7.53 / 2.64 | 7.91 / 1.13 | 0.607 (0.77/0.43/0.43/0.55) |
| P3_multi_sft (full FT, no HN, lr 1e-5) | 19.59 / 21.28 | 20.90 / 20.34 | 7.16 / 3.01 | 7.91 / 1.69 | 0.614 (0.78/0.44/0.43/0.57) |
| P3_multi_hn (LoRA, HN 0.5, lr 5e-5) | 18.27 / 22.60 | 7.91 / 11.30 | 2.26 / 2.82 | 0.00 / 0.00 | 0.854 (0.91/0.77/0.82/0.88) |
| P3-single (LoRA, no HN; reference) | 21.28 | 9.04 | 3.58 | 0.56 | – |
| P3_single_hn_sft (full FT, HN 0.5, lr 1e-5) | 19.96 | 15.25 | 5.27 | 4.52 | – |
| P3_single_sft (full FT, no HN, lr 1e-5) | 21.09 | 17.51 | 5.08 | 4.52 | – |

- P3_multi_hn_sft (21396268, 26,500 steps, best dev loss 2.07 near the end; f_ret 21481859 FAILED at the CPU-FAISS step after saving test_qemb.npy, as expected; f_eval 21481860): @100 all slightly below P3-multi (RRF -0.95, RR -2.8), but lowsim +6.8 RRF / +7.4 RR and @10 RR +5.5; query vectors far more diverse (cosine 0.607 vs 0.860). Still below S0_multi (RRF 30.32) and the untrained base (24.48).
- P3_multi_sft (21396267, f_eval 21484687): within ~1 point of P3_multi_hn_sft on every metric (RRF @100 all 21.28 vs 22.03, lowsim 20.34 vs 19.21; cosine 0.614 vs 0.607). So the shift vs P3-multi (lowsim +8, @10 RR +5, cosine 0.86 -> 0.61) comes from full fine-tuning at lr 1e-5, not from hard negatives; HN adds +0.75 RRF @100 all. P3_multi_hn (LoRA + HN) still training.
- P3_single_hn_sft (21396271, 4.2 h; f_eval 21497338): same pattern as multi: @100 all -1.3 vs P3-single, lowsim +6.2, @10 +1.7. Multi vs single with full FT + HN (RRF): all 22.03 vs 19.96 (+2.1), lowsim 19.21 vs 15.25 (+4.0).
- P3_single_sft (21396270, 4.3 h; f_eval 21501380): @100 all 21.09 (= P3-single 21.28), lowsim 17.51 (+8.5), @10 5.08 (+1.5). HN slightly hurts single (all -1.1, lowsim -2.3). Multi vs single with full FT, no HN (RRF): all 21.28 vs 21.09 (+0.2), lowsim 20.34 vs 17.51 (+2.8).
- P3_multi_hn (21396266, 34.5 h; f_eval 21508667): ~ P3-multi (RRF all 22.60 vs 22.98, lowsim 11.30 vs 12.43, cosine 0.854 vs 0.860). HN does nothing under LoRA either. Multi conclusion: LoRA -> full FT is the only change that matters (lowsim +8, cosine 0.86 -> 0.61); neither closes the gap to S0 (~30) or base (24.48).
- 2026-10-31 13:30: P3_single_hn 21396269 CANCELLED by uid 0 on gh129 at step 18,550 (other jobs unaffected); resubmitted from step 0 as 21529219.
