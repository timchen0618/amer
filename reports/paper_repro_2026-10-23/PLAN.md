# Paper reproduction with fixes (AMER, ICML'26 draft): plan

Status (2026-10-23): plan written, **awaiting user go**. Nothing launched.

Goal: rerun the paper's real-data experiments, using the paper's own code with the four data/eval problems fixed. The user also chose some recipe changes (below). Then check whether the paper's claims still hold.

## Background (what was established, 2026-10-22/23)

- **Paper code:** the paper's real-data results come from the old main stack: `train_distributed.py`, `EmbeddingModelSSVariableLeftPad` in `src/model.py`, `gen_ret_and_eval.py` and `eval.py`.
  - The best snapshot of that code is commit `fe0bd78`. HEAD versions of these files are rewrites and cannot rerun the paper.
  - The `training/inf_retriever/` stack is unrelated.
- **Not present in `fe0bd78`, no fix needed (verified in code 2026-10-23):**
  - LR scheduler double-step: the schedule is built on total batches before sharding (lines 160–166).
  - Precision: fp32 master weights with fp16 autocast. The saved adapters are F32.
  - Seeding: `set_seed` is called at line 80. Data order and the 10% val split use a fixed seed 42.
  - Sampler skip/duplicate: absent. Batches are length-sorted, and the collator drops minority-length examples at group boundaries.
- **Present in the paper pipeline, fixed here:**
  1. The eval corpus was the incomplete 21.12M-passage `/scratch/hc3337/embeddings/inf/qampari_embeddings`.
  2. QAMPARI train gold passages have a title ending in a space and text starting with a space. They were embedded unstripped (`gen_embed_for_auto_training.py:128`).
  3. 44/531 QAMPARI test questions have a near-duplicate in train (gold-set Jaccard ≥ 0.8; 28 identical).
  4. 67/827 AmbigQA test questions are in train (`src/dataset.py` adds AmbigQA dev).
- **Reference results** (paper Llama-8B checkpoints, all 531 QAMPARI test questions, MRecall@100, mean of 3 seeds):

  | Index | single | AMER, round-robin | AMER, RRF | low-sim: single / AMER |
  |---|---|---|---|---|
  | Old 21M index | 11.68 | 11.86 | — | — |
  | Full index | 31.45 | 30.45 | 32.64 | 26.74 / 28.63 |

  Files: session scratchpad `rerun8b/`.

## Recipe (user decisions, 2026-10-23)

| Setting | Paper as run | This round |
|---|---|---|
| Code | `fe0bd78` | `fe0bd78` + data/eval fixes only (branch `paper-repro`) |
| Eval corpus | old 21M index | full 25.86M `wikipedia_embeddings/qampari/shared_frozen_base_512` |
| QAMPARI train gold text | leading/trailing whitespace | stripped, re-embedded |
| QAMPARI test leak (44) | in train | removed from train |
| AmbigQA test leak (67) | in train | removed from train |
| Random negatives | sampled once, unseeded | re-sampled with a fixed seed |
| Batch | 8/GPU × 4 GPUs (1B: 32 × 4) | **50/GPU × 2 GPUs (100)** |
| `sample_rate_multiplier` (QAMPARI) | 10 | **1** (AmbigQA already 1) |
| Extra one-gold step (`mix_one_label_shuffled`) for multi runs | yes | yes (faithful) |
| lr / temperature / epochs / warmup / LoRA | 5e-5 / 0.05 / 120 / 0.05 / r64 α16 drop0.1 | unchanged |
| Selection | val loss at sampling rate 1.0 on a 10% train split | unchanged |
| Seeds | 3 | **1 (seed 0)** |
| Aggregation | round-robin | **round-robin and RRF (k = 60)**, both reported |
| Test sets | QAMPARI 531, AmbigQA 827, low-sim subsets | unchanged |

Caveats:
- Batch size and `sample_rate_multiplier` differ from the paper, so a change in results cannot be attributed to the fixes alone.
- With one seed, differences of about 2 points are within seed noise. The 8B seed spread on the full index was about 1–2.5 points. Use a paired bootstrap over questions.

## Steps

1. **Code.** Make a new git worktree on a new branch `paper-repro` from `fe0bd78`, outside the shared checkout.
   - Commit only:
     - (a) strip title/text before embedding;
     - (b) a seed for negative sampling;
     - (c) a leak-filter script;
     - (d) the eval scripts from the 8B rerun (paper-code query embeddings, then full-index search, then round-robin + RRF scoring; AmbigQA uses the paper's answer matching).
   - Launch config: batch 50, 2 GPUs, `sample_rate_multiplier` 1.
   - All outputs go to new paths (`results/paper_repro/...`, `training_datasets/paper_repro/...`). The paper's `results/<lm>/` directories are read-only.
   - No HF token in committed files; use the cached token.
2. **Data rebuild.**
   - Filter leaks: QAMPARI by gold-set Jaccard ≥ 0.8 against test; AmbigQA by normalized question.
   - Strip, then embed golds and seeded random negatives with inf-retriever (1 GPU job per dataset).
   - Build per-LM tokenized HF datasets.
   - Checks: 0 whitespace golds, 0 leaked questions, counts logged.
3. **Smoke test.** One short run per base model at 50/GPU × 2 GPUs: memory, step time, sampling-rate log. This sets the time estimates and walltime.
4. **Training.** Order: Llama-8B QAMPARI, then 8B AmbigQA, then 1B, 3B, Qwen3-4B. Stay within the 16-GPU user cap together with sessions F/J, keeping GPUs free for evals.
5. **Evaluation.** Per run:
   - Query embeddings via `fe0bd78` code.
   - Top 500 per vector on the full index.
   - Round-robin and RRF.
   - MRecall@100, MRecall@10, Recall@100 on the whole test set and the low-sim subset.
6. **Report.** Per-model tables (single / AMER / always-predicted × round-robin / RRF) and a paired bootstrap of AMER vs single, next to:
   - the paper's numbers;
   - the paper checkpoints on the full index.

## Runs (1 seed per setting)

| Base model | Dataset | single | AMER (scheduled sampling) | always-predicted | runs |
|---|---|---|---|---|---|
| Llama-8B | QAMPARI | 1 | 1 | 1 | 3 |
| Llama-8B | AmbigQA | 1 | 1 | 1 | 3 |
| Llama-1B | QAMPARI + AmbigQA | 2 | 2 | 2 | 6 |
| Llama-3B | QAMPARI + AmbigQA | 2 | 2 | 2 | 6 |
| Qwen3-4B | QAMPARI + AmbigQA | 2 | 2 | 2 | 6 |
| **Total** | | 8 | 8 | 8 | **24** |

Plus 4 smoke tests, 2 embedding jobs and 24 eval jobs (about 15 min each).

**Compute, rough until the smoke test:**
- 8B: about 230 GPU-h.
- All models: about 700–850 GPU-h.

**Disk:**

| Item | 8B only | All models |
|---|---|---|
| Runs (best adapter + linear) | ~4.2 GB | ~11 GB |
| HF datasets | ~2.6 GB | ~10.5 GB |
| Re-embedded arrays | ~3.2 GB | ~3.2 GB |
| Base model downloads (8B cached) | 0 | ~17 GB |
| Retrieval id lists | <0.5 GB | ~0.5 GB |
| **Total** | **~10 GB** | **~42 GB** |

## Status log (append: date, item, result)
- 2026-10-23: plan written; waiting for go.
- 2026-10-23: user approved; branch `paper-repro` (worktree `.claude/worktrees/paper-repro`, commit b2749df).
  - Paper settings recovered from wandb (`timchen0618/diverse_retrieval`):
    - QAMPARI: lr 5e-5, save/eval every 500.
    - AmbigQA: lr 2e-4, bs 32×4, srm 1, save/eval every 200.
    - Both used FSDP over 4 processes.
  - Filter: pre-filter counts equal the paper's. Removed QAMPARI 106 and AmbigQA 67 train questions.
  - Scorer validated: it is the paper's `eval_retrieve_docs` and reproduces 11.68 / 11.86 on the old index.
  - Embedding job 20759191 is pending: J's evals hold the 16-GPU cap. Asked J in the objective_audit Status log.
- 2026-10-23 (late):
  - Embedding finished (1 h 21 m). All 193,415 QAMPARI training golds had the whitespace artifact and were stripped; AmbigQA golds had none.
  - Size of the whitespace shift (paper .npy vs stripped, same golds):
    - cosine 0.9989;
    - along the shift direction, the old golds sat 0.023 above random negatives, against 0.002 after the strip;
    - so the leak gave a small, consistent gold-vs-negative cue.
  - HF datasets built for all 4 LMs: QAMPARI 31,917 rows, AmbigQA 4,977.
- 2026-10-23: 8B smoke test (20766576) hit CUDA OOM at 50/GPU (inputs padded to 257 tokens).
- 2026-10-25: turned on FSDP activation checkpointing (identical results, slower; gradient accumulation would change in-batch negatives); re-smoke 20938764.
- 2026-10-25: **New paper-code finding.** `train_distributed.py` never calls `model.train()`.
  - The model loads in eval mode, the first validation (step 1) calls `model.eval()`, and nothing switches it back.
  - So all paper training ran in eval mode, and **the reported LoRA dropout 0.1 was never active** (effectively 0).
  - This also explains why HF gradient checkpointing and FSDP activation checkpointing did nothing: smokes 20938764 and 20940964 OOM'd at the same 138.03 GiB.
- 2026-10-25: memory diagnosis.
  - FSDP at the paper's batch 8 runs but uses 132–135 GB/GPU: it keeps activations for 1 + k full 8B passes per step.
  - Fix: opt-in `--gradient_checkpointing` = `torch.utils.checkpoint` around every decoder layer. Mode-independent, and with dropout inactive the math is identical.
  - The repro keeps the paper's eval-mode behaviour. Re-smoke at batch 50: FSDP 20944495, DDP 20944496.
- 2026-10-26: smoke tests E/F, 8B QAMPARI multi_SS at 50/GPU.
  - E (FSDP 20945230): runs at ~120 GB/GPU, 4.5 s/it. Estimate: 288 it/epoch × 120 epochs ≈ 43 h.
  - F (DDP 20945231): fails on unused parameters, so all runs use FSDP.
  - KV cache turned off (a checkpoint recompute appended to the in-place cache twice; outputs unchanged). Commit 84c0fc2.
- 2026-10-26: **launched the 6 Llama-8B runs** (job ids in `paper_repro_data/logs/jobs_llama-8b.txt`).
  - Cluster rule: at most 4 GPUs per user for jobs over 48 h. So only the two QAMPARI multi runs have a 4-day walltime; QAMPARI single has 30 h and AmbigQA 24 h.
- 2026-10-26 05:45 — Llama-8B AmbigQA: single (1h44m), multi_SS (4h17m), multi_sampling (4h13m) all COMPLETED 120/120 epochs. Eval 20968913 → score 20968914 (afterok) submitted. QAMPARI 8B runs (20945965/66/77) running ~5h in.
- 2026-10-26 05:50 — Llama-1B 6 runs submitted early on freed GPUs (AmbigQA 12h, QAMPARI 40h walltime): ids in paper_repro_data/logs/jobs_llama-1b.txt. Some will pend under the 16-GPU cap.
- 2026-10-26 — **Llama-8B AmbigQA results** (one seed, full corpus, all 827 test q; MRecall@100 all / lowsim, MRecall@10 all): single 65.90 / 49.82 / 35.55; multi_SS RR 69.89 / 56.36 / 40.27, RRF 70.01 / 57.45 / 41.11; multi_sampling RR 67.35 / 51.64 / 38.33, RRF 67.23 / 51.64 / 38.57. scores: paper_repro_data/eval/llama-8b/ambiguous_qe/scores.json.
- 2026-10-26 08:15 — Llama-1B AmbigQA 3 runs COMPLETED 120/120 (33m/1h30/1h25). Eval 20977601 → score 20977603. Llama-3B 6 runs submitted (jobs_llama-3b.txt; AmbigQA 12h, QAMPARI 40h).
- 2026-10-26 08:55 — **Llama-1B AmbigQA results** (MRecall@100 all / lowsim, @10 all): single 60.10 / 47.27 / 30.47; multi_SS RR 63.72 / 52.36 / 32.53, RRF 64.09 / 53.45 / 34.10; multi_sampling RR 61.43 / 49.82 / 32.65, RRF 61.43 / 48.73 / 32.89.
- 2026-10-26 11:55 — Llama-3B AmbigQA 3 runs COMPLETED 120/120; eval 20993463 → score 20993466. Qwen3-4B 6 runs submitted (jobs_qwen3-4b.txt; AmbigQA 12h, QAMPARI single 40h, multi 47h). 3B QAMPARI multi resubmitted at 47h earlier (20977614/18).
- 2026-10-26 12:15 — **Llama-3B AmbigQA results** (MRecall@100 all / lowsim, @10 all): single 66.38 / 50.91 / 37.00; multi_SS RR 68.20 / 54.18 / 38.45, RRF 67.84 / 53.45 / 38.21; multi_sampling RR 65.78 / 50.91 / 36.15, RRF 65.66 / 50.18 / 35.79. Selected ckpts (lowest dev loss) are early on AmbigQA: steps 401–1601 of ~5400 for all models.
- 2026-10-26 — Pairwise cosine of generated query embeddings (AmbigQA k=2, one pair/question; mean): repro 1B SS 0.916 / samp 0.842; 3B SS 0.947 / samp 0.947 (≈half of questions >0.95); 8B SS 0.889 / samp 0.876. Reference, paper 8B QAMPARI ckpts (k=5, mean over 10 pairs): SS 0.58 (3 seeds 0.579–0.584), samp 0.51; step1↔2 0.70 / 0.61, later adjacent steps ~0.45–0.50. Script: scratchpad qsim.py.
- 2026-10-27 09:50 — Llama-1B QAMPARI 3 runs COMPLETED 120/120 (best steps 32501/34001/32501); Qwen3-4B AmbigQA 3 runs COMPLETED (best step 801 all). Evals: 1B qampari 21103084→21103086; qwen ambig 21103088→21103093. Progress: 3B qampari multi ep 97–98 (~30h total of 47h); Qwen qampari multi ep 48–50 (~42h projected of 47h, tight); 8B qampari multi ep 72–73 (~55h of 96h).
- 2026-10-27 10:15 — **Qwen3-4B AmbigQA results** (MRecall@100 all / lowsim, @10 all): single 64.09 / 51.64 / 34.22; multi_SS RR 65.90 / 53.82 / 36.15, RRF 66.02 / 52.36 / 35.31; multi_sampling RR 65.66 / 50.55 / 36.76, RRF 64.93 / 50.18 / 37.12. Walltime check: steady rates; projected totals 3B ~29.5h, Qwen ~41.6h (limit 47h), 8B ~54h (limit 96h) — no extension needed.
- 2026-10-27 10:30 — **Llama-1B QAMPARI results** (full corpus, all 531 q; MRecall@100 all / lowsim, @10 all): single 13.18 / 3.95 / 1.69; multi_SS RR 16.01 / 8.47 / 3.95, RRF 16.95 / 8.47 / 2.07; multi_sampling RR 14.69 / 8.47 / 4.14, RRF 13.94 / 6.78 / 1.13. Ratio AMER/single 1.29 (RRF) = paper Table 10 ratio on old index (8.41/6.53). Query-emb pairwise cos (k=5): SS 0.849, samp 0.636 (paper 8B ckpts 0.58/0.51). Qwen AmbigQA pairwise: SS 0.949 (69% q >0.95), samp 0.946.
- 2026-10-27 15:45 — Llama-3B QAMPARI 3 runs COMPLETED 120/120 (best steps 34001/34001/32501). Eval 21134428 → score 21134429.
- 2026-10-27 16:15 — **Llama-3B QAMPARI results** (MRecall@100 all / lowsim, @10 all): single 19.21 / 10.17 / 2.07; multi_SS RR 18.08 / 12.43 / 2.26, RRF 19.59 / 12.99 / 2.45; multi_sampling RR 16.76 / 9.60 / 3.39, RRF 18.27 / 10.17 / 2.07. Paper Table 9 (old index): 8.53 → 10.55 (+24%); here +2% RRF whole, −6% RR whole, +28% lowsim RRF. Query-emb pairwise cos: SS 0.820, samp 0.658.
- 2026-10-28 07:45 — Llama-8B QAMPARI multi_SS/multi_sampling COMPLETED 120/120 (2d06h33, 2d06h47; best steps 34001/32501). All 6 8B runs trained → posted F release cue in objective_audit PLAN.md. Eval 21211596 → score 21211597.
- 2026-10-28 08:05 — **Llama-8B QAMPARI results** (MRecall@100 all / lowsim, @10 all): single 24.48 / 20.34 / 6.21; multi_SS RR 25.99 / 20.90 / 7.53, RRF 26.74 / 20.34 / 4.33; multi_sampling RR 22.60 / 18.08 / 5.65, RRF 22.79 / 14.12 / 2.82. Query-emb pairwise cos SS 0.749, samp 0.595. Paired bootstrap (10k, one-sided, per-q MRecall@100 all) AMER−single: AmbigQA SS RR +3.99 p=.0002, RRF +4.11 p=.0001; samp +1.45 p=.067 / +1.33 p=.088; QAMPARI SS RR +1.51 p=.13, RRF +2.26 p=.040; samp −1.88 / −1.69. vs paper ckpts on same full index (3-seed mean): single 31.45, AMER RR 30.45 / RRF 32.64 → repro ~7 pts lower in absolute terms. Push notification sent. Script: scratchpad boot.py.
- 2026-10-28 — **Why QAMPARI repro is ~7 pts below paper ckpts (full index): the 44 leaked test questions.** Paper 8B ckpts score 84–91% MRecall@100 on the leaked 44 (single 86.4 all 3 seeds) vs repro 11–18%; 44×~75pts/531 ≈ 6 pts. On the 487 clean questions repro ≈ paper: single 25.67 vs paper 26.49 (seeds 26.49/25.87/27.10); multi_SS RRF 27.52 vs 28.00 (26.90/27.52/29.57), RR 26.69 vs 25.12; multi_sampling RRF 23.61 vs 24.57, RR 23.82 vs 21.70. Clean AMER−single gain: paper RRF +1.5 / RR −1.4; repro RRF +1.9 / RR +1.0. Leak idx: scratchpad qampari_leak_idx.json (leak_split.py). Bootstrap (all 531): 1B SS RR +2.82 p=.015, RRF +3.77 p=.0008; 3B SS RR −1.13, RRF +0.38 p=.40; 3B/1B multi_sampling n.s.
- 2026-10-28 — Investigating QAMPARI diversity/low-sim gap vs paper. Paper Fig 5 values (plot_output_similarity.py @fe0bd78, cos sim): QAMPARI target 0.86, 1B .828, 3B .661, Qwen .671, 8B .63; AmbigQA target .90, 1B .948, 3B .894, Qwen .891, 8B .843. Repro SS: QAMPARI 1B .849, 3B .820, 8B .749; AmbigQA 1B .916, 3B .947, Qwen .949, 8B .889. Hypotheses: srm 1 vs 10 (QAMPARI SS only), batch/step count (paper 3B/Qwen/8B QAMPARI 8×4 → ~3.5× more steps; 1B 32×4 ≈ ours and 1B matches).
  - Old-21M-index re-scoring of all repro runs (like-for-like with paper tables): search 21219856 (AmbigQA) / 21219857 (QAMPARI) → scores_old21m.json (eval_score.sbatch now takes [index]).
  - User-requested: QAMPARI multi_SS with `--sample_rate_multiplier 10` (else identical, batch 50×2): 3B 21220365 (47h), 8B 21220366 (4d). Names *_multi_SS_srm10_0. jobs_srm10.txt.
- 2026-10-28 — **Repro on the paper's old 21M index** (scores_old21m.json; MRecall@100 whole / lowsim, RR; paper Tables 7–10 3-seed mean in brackets):
  - AmbigQA single→SS: 1B 57.80/45.82 → 60.94/50.91 [58.57/46.06 → 60.94/49.33]; 3B 62.64/49.45 → 65.54/52.73 [62.80/49.33 → 65.06/51.64]; Qwen 59.37/47.64 → 62.52/50.18 [61.79/48.85 → 63.81/50.43]; 8B 61.67/46.55 → 66.63/54.91 [62.64/49.58 → 66.26/52.00]. → AmbigQA reproduces.
  - QAMPARI single→SS (RR; RRF): 1B 6.03/1.13 → 7.34/2.26; 7.72/2.82 [6.53/1.50 → 8.41/5.08]; 3B 7.53/3.95 → 7.53/4.52; 8.29/4.52 [8.53/2.82 → 10.55/7.53]; 8B 9.79/5.65 → 9.60/6.21; 9.79/6.21 [11.68/8.47 → 11.86/8.85]. Lowsim n=177 (1 q = 0.56 pt; 14 leaked q inside). → QAMPARI whole-set and lowsim gains for 1B/3B do not reproduce at srm 1 / batch 100.
- 2026-10-28 10:55 — Qwen3-4B QAMPARI multi_SS/multi_sampling COMPLETED 120/120 (42.1h/41.8h; best step 34001). All 24 runs trained. Eval 21222946 → score 21222947; old-index 21222948 → 21222949.
- 2026-10-28 — srm10 runs started; config verified (sample_rate_multiplier=10.0, names llama-{3b,8b}_multi_SS_srm10_0).
- 2026-10-28 — All 24 runs scored (Qwen QAMPARI last). Final report: REPORT.md in this folder. Push sent. srm10 runs pending.
- 2026-10-29 16:08 — srm10 runs 21220365 (3B, epoch 120/120 batch 201/288) and 21220366 (8B, epoch ~59/120) CANCELLED by uid 0 (root), together with every other running hc3337 job (F S0, P3 phaseC, interactive). Nothing resubmitted pending user decision. Saved best checkpoints: 3B step 34001 (of 34500; usable), 8B step 17501 (half-trained).
- 2026-10-29 — Cause of the 16:08 cancellation: account torch_pr_152_courant was removed from hc3337; the new accounts are torch_pr_152_echoi (h200_courant + h200_public) and torch_pr_152_general (h200_public only). scripts/paper_repro/*.sbatch now use echoi. 8B srm10 resubmitted from scratch (21396599; the half-trained dir was moved to *_cancelled_1029). 3B srm10 is not retrained: training had reached epoch 120 and its best checkpoint (step 34001 of 34500) is evaluated as is (21396600).
- 2026-10-29 — 3B srm10 evaluated (ckpt step 34001): full RRF 19.96 (srm1 19.59, p=.41), cosine 0.761 (srm1 0.820, paper 0.661). REPORT.md section added.
- 2026-10-30 — User: run srm10 for the remaining LMs too (QAMPARI multi_SS only; single/forced-sampling don't use the multiplier, AmbigQA paper used srm 1). Submitted 1B 21425784 (24h), Qwen3-4B 21425785 (48h). 8B 21396599 running.
- 2026-10-31 — 1B srm10 done + evaluated (ckpt 34001): full RRF 18.64 vs srm1 16.95 (p=.053; RR p=.042), cosine 0.770 vs 0.849 (paper 0.828). REPORT.md multiplier section now covers 1B + 3B.
- 2026-11-01 — Qwen + 8B srm10 done and evaluated. RRF vs srm1: Qwen 18.27 vs 16.57 (p=.08), 8B 25.24 vs 26.74 (n.s., lower); cosine Qwen 0.772 (0.835), 8B 0.688 (0.749). REPORT.md multiplier section complete for all 4 LMs.
- 2026-11-01 — Diversity-gap investigation (3 subagents): measurement identical (paper's saved Sep-2025 qemb reproduce Fig. 5 exactly with our script); data identical up to leak removal + whitespace offset (target pairwise cos .848 vs .849); paper QAMPARI global batch 128/64/32/32 (1B/3B/Qwen/8B) → 27k/54k/108k/108k steps vs our 34.5k, selection at end in both → step count is the leading cause. Launched 3B paper-recipe test: batch 32×2 = 64, srm10, ~54k steps (job 21602109, name llama-3b_multi_SS_srm10_b64_0).
- 2026-11-01 18:10 — 3B b64 run (21602109) started on gh129; log confirms batch_size_training=32, sample_rate_multiplier=10.0, 449 batches/epoch (898 its incl. one-label) → ~53.9k iterations. ~2.0 s/it early; estimated finish ~36h (≈ 2026-11-03 06:00), within 48h walltime. Completion watcher running.
- 2026-11-01 20:00 — dropout ablation (user request): new flag --train_mode (commit 59b710f) calls model.train() so LoRA dropout 0.1 is active; disables HF checkpointing to avoid double checkpointing. Llama-1B AmbigQA multi_SS → 21610971 (compare llama-1b_multi_SS_0); Llama-1B QAMPARI multi_SS srm10 → 21610973 (compare llama-1b_multi_SS_srm10_0).
  - User instruction (2026-11-01): report dropout-ablation results regardless; if multi results change a lot (vs eval-mode counterpart), also queue the single-query counterpart with --train_mode (same dataset, Llama-1B) and run follow-ups automatically.
- 2026-11-01 — **dropout ablation, AmbigQA Llama-1B multi_SS** (train_mode vs eval-mode, full index, MRecall@100 all/lowsim): RRF 64.09/53.45 → 63.48/52.73; RR 63.72/52.36 → 63.60/52.73; cos .916 → .918. Paired bootstrap n.s. (p≥.28 either direction). Change < 1 pt → no single-query follow-up. QAMPARI srm10 run 21610973 still training.
- 2026-11-02 08:21 — 3B b64 run 21602109 CANCELLED by root at 2026-11-02 08:17 (epoch 47/120, 14h): admins cleared every job on node gh129 (now RESERVED); not an account issue. No resumable state (save_only_improve keeps weights only), so resubmitted from scratch: 21656008 (--exclude is blocked by site policy). Old dir → llama-3b_multi_SS_srm10_b64_0_cancelled_1102.
- 2026-11-02 09:33 — user wants the step-matched control for the batch ablation: 3B QAMPARI multi_SS srm10, batch 50/GPU (global 100), --num_epochs 188 (288×188 ≈ 54.1k it, matches b64's 53.9k): 21660962, 60h walltime. Compare cos/MRecall: b100×120ep (0.761/19.96) vs b64×120ep (21656008) vs b100×188ep (21660962). User: report all in the artifact when done.
- 2026-11-02 21:54 — **dropout ablation, QAMPARI Llama-1B multi_SS srm10** (full index MRecall@100 all/lowsim): RRF 18.64/10.17 → 18.46/11.86; RR 17.89/12.43 → 16.76/12.43; cos .770 → .807 (less diverse). Bootstrap n.s. (RRF p=.46/.61, RR p=.17). Low-sim RRF +1.7 and RR all −1.1 exceed the 1-pt rule → queued single-query counterpart with --train_mode: 21708978.
- 2026-11-02 21:53 — 3B b64 (21656008) at epoch 39/120 (12h); control e188 (21660962) at epoch 52/188 (12h).
- 2026-11-03 10:05 — single-query dropout (21708978): QAMPARI 1B single 13.18/3.95 → 12.62/2.82 (p=.17); AMER−single gap RRF +5.46 → +5.84, RR +4.71 → +4.14. Dropout ablation closed: no effect. Artifact v8 + REPORT.md updated.
- 2026-11-03 — **seeds 1, 2** (user approved plan): AmbigQA single + multi_SS (srm 1) × 4 LMs × seeds 1,2 = 16 runs submitted with per-LM eval/score chains; ids in paper_repro_data/logs/jobs_seeds.txt. Run names <lm>_<cond>_<seed>; eval_embed_search takes SEED env. QAMPARI seeds on hold: user decides after the 3B batch-64 result. User: no always-predicted, no QAMPARI multiplier 1. Report 3-seed mean ± sd, bootstrap on per-question seed means.
- 2026-11-04 09:59 — 3B batch 64 result: cos .761 → .718 (paper .661); RRF 19.96 → 21.09, RR 18.64 → 19.59 (n.s.); vs single RRF p=.11. Artifact v9. Control 21660962 ~10h left. Reported to user; QAMPARI seeds await user decision.
- 2026-11-04 15:55 — 3B step-matched control result: cos .727 (b64 .718, b100×120 .761), RRF 20.90 (b64 21.09). Steps drive ~80% of the diversity gain; batch size little. Artifact v10. AmbigQA seed jobs still PENDING (queue full).
- 2026-11-04 16:41 — user: keep recipe (batch 100, 120 ep). QAMPARI seeds 1,2 submitted: single + multi_SS srm10 × 4 LMs (16 runs; 8B multi 72h walltime, under the 4-GPU >48h cap) with per-LM eval/score chains; ids in logs/jobs_seeds.txt.
- 2026-11-06 10:32 — seed jobs sat PENDING ~3 days: sbatch headers pinned --partition=h200_courant,h200_public + --constraint=h200, so h200_echoi was never offered. Fixed headers (commit 5b0b2a8: --gres=gpu:h200:N, no partition), cancelled all 72 seed jobs, resubmitted via scripts/paper_repro/submit_seeds.sh (ids in logs/jobs_seeds.txt; old list jobs_seeds_cancelled_1106.txt).
- 2026-11-09 (seeds): /scratch quota hit 100% ~04:58, killing all running seed trains + queued evals. Moved partial dirs to *_failed_1109; resubmitted 5 trains (3b srm10 s2, qwen srm10 s1/s2, 8b single s1/s2) + all eval/score chains via scripts/paper_repro/resubmit_seeds_1109.sh. Kept llama-3b srm10 s1 best-dev ckpt (died at epoch 119.4/120).
- 2026-11-09 (seeds): AmbigQA 3-seed stats done → REPORT.md 'Three seeds' + artifact v11.
- 2026-11-09 (seeds): QAMPARI Llama-1B 3-seed done → REPORT.md + artifact v12.
- 2026-11-09 (seeds): QAMPARI Llama-8B 3-seed done → REPORT.md + artifact v13.
- 2026-11-11 (seeds): QAMPARI Llama-3B 3-seed done → REPORT.md + artifact v14. Qwen remaining.
- 2026-11-12 (seeds): all 3-seed results done (QAMPARI Qwen last) → REPORT.md + artifact v15.
- 2026-11-12: 3-seed figures + diversity table in artifact v18; deleted *_failed_1109 partial ckpts (2.9G); report committed + pushed on paper-repro.
