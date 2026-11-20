# Joint (trainable) document encoder runs on QAMPARI: consolidated record (2026-11-20)

Every QAMPARI run trained with a **jointly trained document encoder** (query and documents through the same fine-tuned INF-Retriever encoder) on the clean v3 recipe: 31 runs in three rounds. Round 1 (`clean_v3_abl`) is also in `reports/ablation_clean_v3_2026-10-07.md`, and the J runs are in the "Session J summary" of `reports/objective_audit_2026-10-13/PLAN.md`. **Round 2 (`clean_v3_abl2`) and the causal cell-B runs were never written up**: the ablation report still lists them as "pending". This report records all of them in one place, before the runs are redone on F's recipe.

## Shared recipe

- **Model:** `infly/inf-retriever-v1-1.5b`, full fine-tune, query and document encoder shared and trained jointly (cell D: bidirectional query attention; cell B: `--force_causal`).
- **Data:** `data/training/clean_v3_hn` (QAMPARI train, every gold count, per-slot hard negatives, ratio 0.5, one negative per gold).
- **Dev:** `data/phaseA/qampari_cleanv2dev500.jsonl` (500 questions). **Test:** `data/amer_data/eval_data/qampari.jsonl` (531 questions, the AMER test set; the same test set as every F run, so test is comparable with F, dev is not).
- **Optimisation:** LR 1e-5 (unless noted), 2,500 steps, warmup 100, 50 queries per GPU on 2 GPUs (FSDP, fp32 gradient reduce), temperature 0.05, seed 0, bf16.
- **Multi-query objective:** plain Hungarian unless noted, k = number of golds, scheduled sampling rate rising linearly to 1 at step 2500 unless noted.
- **Code / launchers:** `training/inf_retriever/finetuning_multi.py` (`finetuning_multi_proj.py` for io_proj), driven by `tools/phaseC_run.sh` via `tools/abl_launch.sh` (round 1), `tools/abl2_launch.sh` (round 2, cell B = `wave3 B`), `tools/j_launch.sh` (J). Training wall time 3.3–4.7 h on 2 H200.
- **Selection:** saves every 250 steps (J: 500/750/1000/1250/1500/2000/2500); cheap dev on a reduced corpus ranks steps; top 2 by cheap dev (+ step 2500 from round 2 on) get a full-corpus evaluation; the reported step is the best **full dev**. Test is never used for selection.
- **Metric:** MRecall@100. Multi-query: RRF over k = 5 queries (headline); single-query: plain MRecall@100. One paired SE on test ≈ 2 points; < 2 dev points is a tie.

## Results (all evaluated checkpoints)

Generated from `results/phaseC/full/qampari/<run>_s<step>/{dev,test}_kagg.json` (multi) and `{dev,test}/eval_metrics.txt` (single). `abl_` = round 1, `abl2_` = round 2, `j_` = session J. "hps" = `--loss_fn hungarian_plus_single`.

| run | change (on top of the shared recipe) | dev-selected step | dev | test | evaluated steps: dev / test |
|---|---|---:|---:|---:|---|
| `abl_qampari_single` | single-query (`training_mode=standard_org_q`) | 2000 | 64.80 | 53.30 | 500: 63.20 / 53.48; 2000: 64.80 / 53.30; 2250: 64.20 / 52.17 |
| `abl_qampari_multi` | plain `hungarian` | 1500 | 61.20 | 38.42 | 750: 58.40 / 40.30; 1500: 61.20 / 38.42; 1750: 61.00 / 39.36 |
| `abl_qampari_masked` | `--loss_fn hungarian_masked` | 750 | 65.80 | 47.65 | 750: 65.80 / 47.65; 1000: 63.80 / 48.59; 1500: 64.00 / 48.40 |
| `abl_qampari_fullsamp` | `--full_sampling` | 750 | 56.80 | 38.04 | 250: 55.80 / 40.30; 500: 56.40 / 40.30; 750: 56.80 / 38.04 |
| `abl_qampari_hps` | `--loss_fn hungarian_plus_single --single_loss_weight 0.5` | 750 | 63.00 | 46.14 | 250: 58.40 / 41.62; 750: 63.00 / 46.14; 1000: 62.60 / 47.27 |
| `abl2_qampari_sat25` | `--sampling_saturate_frac 0.25` | 500 | 57.20 | 41.24 | 250: 53.60 / 38.98; 500: 57.20 / 41.24; 2500: 34.60 / 21.28 |
| `abl2_qampari_sat50` | `--sampling_saturate_frac 0.5` | 1000 | 60.40 | 43.31 | 750: 60.20 / 40.30; 1000: 60.40 / 43.31; 2500: 44.80 / 23.92 |
| `abl2_qampari_hps_w025` | hps, `--single_loss_weight 0.25` | 1000 | 62.80 | 40.49 | 750: 60.80 / 44.26; 1000: 62.80 / 40.49; 2500: 54.00 / 40.49 |
| `abl2_qampari_hps_w010` | hps, `--single_loss_weight 0.10` | 1500 | 62.60 | 44.44 | 1500: 62.60 / 44.44; 1750: 61.60 / 41.81; 2500: 54.20 / 35.97 |
| `abl2_qampari_single_lr3e-5` | single-query, LR 3e-5 | 1250 | 61.40 | 48.21 | 1250: 61.40 / 48.21; 2000: 60.40 / 45.57; 2500: 60.00 / 46.52 |
| `abl2_qampari_multi_lr3e-5` | plain `hungarian`, LR 3e-5 | 1750 | 55.80 | 39.92 | 750: 54.80 / 37.10; 1750: 55.80 / 39.92; 2500: 44.40 / 30.32 |
| `abl2_qampari_masked_div05` | `--loss_fn hungarian_masked_div --div_weight 0.5` | 750 | 61.40 | 43.13 | 750: 61.40 / 43.13; 1250: 58.80 / 44.82; 2500: 52.80 / 39.55 |
| `abl2_qampari_masked_div2` | `--loss_fn hungarian_masked_div --div_weight 2.0` | 750 | 60.00 | 44.63 | 750: 60.00 / 44.63; 1000: 58.00 / 43.50; 2500: 54.00 / 38.04 |
| `abl2_qampari_hps_fullsamp` | hps 0.5 + `--full_sampling` | 750 | 62.80 | 46.33 | 750: 62.80 / 46.33; 1000: 61.00 / 45.39; 2500: 49.80 / 35.03 |
| `abl2_qampari_hps_sat50` | hps 0.5 + `--sampling_saturate_frac 0.5` | 1000 | 63.80 | 46.14 | 750: 62.40 / 46.52; 1000: 63.80 / 46.14; 2500: 50.00 / 35.40 |
| `abl2_qampari_B_single` | single-query + `--force_causal` | 2250 | 64.00 | 51.79 | 500: 62.80 / 51.79; 2250: 64.00 / 51.79; 2500: 63.60 / 51.98 |
| `abl2_qampari_B_multi` | plain `hungarian` + `--force_causal` | 1500 | 62.00 | 43.31 | 1250: 58.80 / 42.94; 1500: 62.00 / 43.31; 2500: 52.00 / 31.26 |
| `j_qampari_multi_seq` | `--loss_fn hungarian_seq` | 1250 | 64.00 | 47.27 | 1000: 63.80 / 47.65; 1250: 64.00 / 47.27; 2500: 52.40 / 35.78 |
| `j_qampari_multi_seq_ioproj` | `--loss_fn hungarian_seq` + `--io_proj` (identity init; `finetuning_multi_proj.py`) | 750 | 66.40 | 49.53 | 750: 66.40 / 49.53; 1000: 64.60 / 45.76; 2500: 51.00 / 34.84 |
| `j_qampari_multi_seq_detach` | `--loss_fn hungarian_seq --detach_fed_output` | 1000 | 63.40 | 45.01 | 1000: 63.40 / 45.01; 1500: 61.20 / 43.88; 2500: 54.40 / 37.48 |
| `j_qampari_multi_detach` | `--detach_fed_output` | 1250 | 59.80 | 42.75 | 1250: 59.80 / 42.75; 1500: 59.60 / 39.92; 2500: 50.20 / 32.58 |
| `j_qampari_multi_fednorm` | `--fed_norm token` | 1250 | 61.00 | 40.87 | 1000: 60.00 / 42.56; 1250: 61.00 / 40.87; 2500: 50.20 / 29.57 |
| `j_qampari_hps_seq` | hps 0.5 + `--hps_multi_loss hungarian_seq` | 1000 | 65.20 | 48.40 | 750: 62.40 / 44.26; 1000: 65.20 / 48.40; 2500: 57.20 / 41.81 |
| `j_qampari_hps_detach` | hps 0.5 + `--detach_fed_output` | 1000 | 64.80 | 46.33 | 750: 64.40 / 45.01; 1000: 64.80 / 46.33; 2500: 58.00 / 40.49 |
| `j_qampari_hps_fednorm` | hps 0.5 + `--fed_norm token` | 1000 | 63.00 | 46.52 | 1000: 63.00 / 46.52; 1250: 60.80 / 43.31; 2500: 56.40 / 38.98 |
| `j_qampari_hps_seq_detach` | hps 0.5 + `--hps_multi_loss hungarian_seq --detach_fed_output` | 1000 | 64.20 | 45.76 | 750: 63.60 / 44.44; 1000: 64.20 / 45.76; 2500: 57.20 / 41.05 |
| `j_qampari_hps_seq_docstop` | hps 0.5 + `--hps_multi_loss hungarian_seq --doc_grad_until 1250` | 1000 | 65.20 | 48.40 | 750: 62.40 / 44.26; 1000: 65.20 / 48.40; 2500: 61.60 / 48.59 |
| `j_qampari_sat50_docstop` | `--sampling_saturate_frac 0.5 --doc_grad_until 1250` | 750 | 60.60 | 39.92 | 750: 60.60 / 39.92; 1000: 60.20 / 43.31; 2500: 49.20 / 31.64 |

Secondary scores for multi-query runs at each evaluated step (test): round-robin k=5 / RRF k=1.

| run | step: test RR k=5 / RRF k=1 |
|---|---|
| `abl_qampari_multi` | 750: 37.10 / 31.26; 1500: 33.90 / 16.57; 1750: 33.52 / 10.73 |
| `abl_qampari_masked` | 750: 49.34 / 49.34; 1000: 49.34 / 47.46; 1500: 49.91 / 47.08 |
| `abl_qampari_fullsamp` | 250: 37.10 / 4.52; 500: 34.84 / 6.78; 750: 33.71 / 9.60 |
| `abl_qampari_hps` | 250: 46.14 / 51.60; 750: 45.20 / 54.05; 1000: 43.69 / 51.98 |
| `abl2_qampari_sat25` | 250: 37.10 / 20.53; 500: 29.57 / 7.53; 2500: 17.14 / 0.75 |
| `abl2_qampari_sat50` | 750: 35.40 / 13.18; 1000: 35.78 / 6.59; 2500: 19.77 / 0.38 |
| `abl2_qampari_hps_w025` | 750: 43.31 / 53.67; 1000: 39.74 / 46.52; 2500: 34.09 / 37.29 |
| `abl2_qampari_hps_w010` | 1500: 35.78 / 36.35; 1750: 34.46 / 33.90; 2500: 28.63 / 26.37 |
| `abl2_qampari_multi_lr3e-5` | 750: 32.58 / 25.61; 1750: 34.09 / 10.36; 2500: 24.86 / 2.07 |
| `abl2_qampari_masked_div05` | 750: 47.65 / 48.59; 1250: 45.57 / 42.56; 2500: 37.48 / 29.19 |
| `abl2_qampari_masked_div2` | 750: 43.69 / 45.01; 1000: 41.62 / 40.68; 2500: 36.72 / 29.76 |
| `abl2_qampari_hps_fullsamp` | 750: 41.62 / 50.09; 1000: 42.75 / 48.40; 2500: 32.02 / 40.11 |
| `abl2_qampari_hps_sat50` | 750: 39.55 / 49.91; 1000: 39.92 / 49.34; 2500: 32.20 / 38.23 |
| `abl2_qampari_B_multi` | 1250: 36.35 / 15.07; 1500: 38.04 / 16.57; 2500: 25.05 / 1.88 |
| `j_qampari_multi_seq` | 1000: 41.81 / 47.46; 1250: 41.24 / 47.27; 2500: 30.13 / 23.54 |
| `j_qampari_multi_seq_ioproj` | 750: 42.37 / 48.02; 1000: 39.17 / 43.13; 2500: 30.89 / 24.67 |
| `j_qampari_multi_seq_detach` | 1000: 41.24 / 43.88; 1500: 38.42 / 43.13; 2500: 32.39 / 26.18 |
| `j_qampari_multi_detach` | 1250: 38.42 / 19.02; 1500: 34.27 / 13.37; 2500: 25.61 / 1.69 |
| `j_qampari_multi_fednorm` | 1000: 38.23 / 24.48; 1250: 33.90 / 11.11; 2500: 24.67 / 1.13 |
| `j_qampari_hps_seq` | 750: 42.00 / 50.85; 1000: 45.39 / 51.41; 2500: 36.53 / 42.75 |
| `j_qampari_hps_detach` | 750: 48.78 / 54.99; 1000: 46.89 / 52.35; 2500: 36.16 / 42.75 |
| `j_qampari_hps_fednorm` | 1000: 43.13 / 52.73; 1250: 37.29 / 48.21; 2500: 33.90 / 42.18 |
| `j_qampari_hps_seq_detach` | 750: 45.39 / 51.41; 1000: 44.44 / 48.21; 2500: 37.66 / 42.56 |
| `j_qampari_hps_seq_docstop` | 750: 42.00 / 50.66; 1000: 45.39 / 51.60; 2500: 41.43 / 48.40 |
| `j_qampari_sat50_docstop` | 750: 35.22 / 13.37; 1000: 35.78 / 6.59; 2500: 26.74 / 4.52 |

`j_sat50_dense` (D4(b): plain sat50 with an in-training eval every 50 steps) has cheap-dev scores only: 69.6 at step 1000 → 67.6 at 1250 (rate reaches 1) → 66.4 at 1500 → 56.2 at 2500.

Not run: `abl2` optional `single_lr2e-5` / `multi_lr2e-5`. AmbigQA round 1: only `single` was evaluated (step 150: 82.40 / 76.18) before the AmbigQA full evals were cancelled on 2026-10-07.

## Findings

1. **No multi-query run beats single-query on test.** Best multi: `j_multi_seq_ioproj` 66.4 / 49.53, against `abl_single` 64.8 / 53.30. The causal single (`B_single`, 64.0 / 51.79) ties bidirectional single.
2. **`hungarian_seq` is the only consistent objective fix** (+8.9 test on plain Hungarian, +2.3 on hps). Detach, fed_norm and the diversity penalty tie their baselines. Learned input/output layers add +2.3 test on joint (they tie on frozen).
3. **Late collapse needs a trainable document encoder.** Every run falls 8–13 dev points between its peak and step 2500 (sat25: 57.2 → 34.6; frozen runs fall ≈ 1). Earlier saturation of the sampling rate makes it earlier and larger. Stopping document gradients at step 1250 (`docstop`) removes most of it under hps_seq (test 48.59 at 2500), not under plain sat50. Dev selection already avoids it (selected steps 500–1750, except the single runs).
4. **Higher LR (3e-5) hurts both modes** (single −5.1, multi +1.5 test vs 1e-5 but −5.4 dev).
5. **Single-query weight:** hps 0.5 (46.14) ≥ 0.10 (44.44) ≥ 0.25 (40.49) on test; dev ties.

Details and diagnostics (copy shortcut, input-scale, D1–D5, E1–E5): `reports/objective_audit_2026-10-13/PLAN.md`.

## What is left on disk (2026-11-20, after cleanup)

- Checkpoints: none. The last ones (one per run, 6.7 GB each, for the 11 `clean_v3_j_*` runs and `abl2_B_{single,multi}`, ≈ 87 GB) were deleted on 2026-11-20 at the user's request; each run's `checkpoints/qampari/<run>/opt.txt` (the full option list) is kept. Corpus embeddings had been deleted earlier.
- Results (`results/phaseC/full/qampari/`, ≈ 5 GB for these runs): scores, query embeddings and retrieval lists. Training logs: `sbatch_outputs/phaseC_<run>.out`; driver logs `results/phaseC/<run>.driver.out`; all launches in `results/phaseC/phaseC.log`.

## Caveats for comparison with later runs

- Training data differ from F's current runs (`clean_v3_hn_s0`, 5–8 unique golds, seeded split). Compare test only.
- Old 2,500-step recipe: no 10k steps, no one-gold step, old batch-order sampler (`Random(seed + epoch)`).
