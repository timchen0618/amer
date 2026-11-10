#!/bin/bash
# Session F launch queue (objective audit 2026-10-13): keeps at most MAXRUN frozen F training runs queued or running
# (2 GPUs each; F's share is <= 8 GPUs once J's joint runs are in, leaving room for F's 1-GPU checkpoint retrievals)
# and starts the next queued run with f_run.sh when one finishes training. Runs already started (a .trainjob file)
# are skipped, so restarting this script is safe.
# Usage (repo root, detached): setsid nohup bash training/inf_retriever/tools/f_queue.sh > results/phaseC/f_queue.out 2>&1 &
cd /scratch/hc3337/projects/autoregressive
MAXRUN=${MAXRUN:-3}
WT=/scratch/hc3337/projects/autoregressive/.claude/worktrees/f-objective
LONG="STEPS_OVERRIDE=10000 WARMUP_OVERRIDE=400 F_STEPS=1000,2000,2500,3000,4000,5000,6000,7000,8000,9000,10000 F_FINAL=10000 TRAIN_TIME=16:00:00"
LONG20=${LONG/TRAIN_TIME=16:00:00/TRAIN_TIME=20:00:00}
LONG20K="STEPS_OVERRIDE=20000 WARMUP_OVERRIDE=800 F_STEPS=2000,4000,6000,8000,10000,12000,14000,16000,18000,20000 F_FINAL=20000 TRAIN_TIME=30:00:00"
S0="--train_data data/training/clean_v3_hn_s0/qampari/train_data.jsonl"
S0ENV="F_DEV=data/phaseA/qampari_clean_v3_dev_s0.jsonl F_DATA_ROOT=data/training/clean_v3_hn_s0"
BASE="--train_data data/training/clean_v3_hn/qampari/train_data.jsonl --negative_hard_ratio 0.5 --freeze_doc_encoder"
# order: F5, D4(a) test, rest of F2, F3b, then the 2026-10-15 additions.
# name|extra|code root ("" = main checkout)[|training script]
QUEUE=(
  "C_fednorm|--fed_norm token|$WT"
  "C_sat50|--sampling_saturate_frac 0.5|"
  "C_sat75|--sampling_saturate_frac 0.75|"
  "C_masked|--loss_fn hungarian_masked|"
  "C_hps05|--loss_fn hungarian_plus_single --single_loss_weight 0.5|"
  "C_multi_t01|--temperature 0.1|"
  "C_seq_used|--loss_fn hungarian_seq_used|$WT"
  # 2026-10-15 user-approved additions: learned input/output layers (src/inbatch_proj.py, vs C_seq) and the
  # frozen hps weight sweep (vs C_hps05). 4th field: training entry point (TRAIN_SCRIPT, default finetuning_multi.py).
  "C_seq_ioproj|--loss_fn hungarian_seq --io_proj||finetuning_multi_proj.py"
  "C_hps025|--loss_fn hungarian_plus_single --single_loss_weight 0.25|"
  "C_hps010|--loss_fn hungarian_plus_single --single_loss_weight 0.1|"
  # 2026-10-17 user-approved ioproj follow-ups: #1 ioproj + hps 0.25 (multi term hungarian_seq), vs C_seq_ioproj and
  # C_hps025; #4 ioproj on plain Hungarian (vs C_multi); #5 one layer only on plain Hungarian (vs #4).
  "C_hps025_seq_ioproj|--loss_fn hungarian_plus_single --single_loss_weight 0.25 --hps_multi_loss hungarian_seq --io_proj||finetuning_multi_proj.py"
  "C_multi_ioproj|--loss_fn hungarian --io_proj||finetuning_multi_proj.py"
  "C_multi_inproj|--loss_fn hungarian --io_proj --io_proj_mode in||finetuning_multi_proj.py"
  "C_multi_outproj|--loss_fn hungarian --io_proj --io_proj_mode out||finetuning_multi_proj.py"
  # 2026-10-22 user-approved paper-gap plan, Stage A: sampling schedule x length on plain Hungarian (C_multi = linear
  # 0->1 at 2500 steps). cap = legacy min(10 t / T, 0.8); const = legacy multi_sampling (rate 1.0). 5th field: env
  # for f_run.sh / phaseC_train.sbatch (10k steps: warmup 400 = 4% as at 2500, eval every 1000 + 2500, 16 h walltime).
  "C_multi_cap|--loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8||finetuning_multi_paper.py|"
  "C_multi_const|--loss_fn hungarian --full_sampling||finetuning_multi_paper.py|"
  "C_multi_10k|--loss_fn hungarian||finetuning_multi_paper.py|$LONG"
  "C_multi_cap_10k|--loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8||finetuning_multi_paper.py|$LONG"
  "C_multi_const_10k|--loss_fn hungarian --full_sampling||finetuning_multi_paper.py|$LONG"
  # Stage B (2026-10-23): Stage A winner = C_multi itself (linear @ 2500), so the 2 x 2 is on C_multi.
  "C_multi_onelabel|--loss_fn hungarian --one_label_step||finetuning_multi_paper.py|"
  "C_multi_ioprojrand|--loss_fn hungarian --io_proj --io_proj_init random||finetuning_multi_paper.py|"
  "C_multi_onelabel_ioprojrand|--loss_fn hungarian --one_label_step --io_proj --io_proj_init random||finetuning_multi_paper.py|"
  # Stage B, Track 2 (user 2026-10-23): the same 2 x 2 on the large-distance base, paper cap 0.8 x 10k
  # (Track 2 reports its final checkpoint, step 10000). One-label runs get a 20 h walltime.
  # cancelled 2026-10-23 (Stage A re-run first): "C_cap10k_onelabel|--loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step||finetuning_multi_paper.py|$LONG20"
  # cancelled 2026-10-23 (Stage A re-run first): "C_cap10k_ioprojrand|--loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --io_proj --io_proj_init random||finetuning_multi_paper.py|$LONG"
  # cancelled 2026-10-23 (Stage A re-run first): "C_cap10k_onelabel_ioprojrand|--loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step --io_proj --io_proj_init random||finetuning_multi_paper.py|$LONG20"
  # 2026-10-23 (user): Stage B cancelled; Stage A re-run on the seeded 90/10 split (tools/build_seed_split.py --seed 0:
  # 23732 train / 2637 dev), single-query baseline included. Runs whose name has "single" use mode single.
  "S0_single|$S0||finetuning_multi.py|$S0ENV"
  "S0_multi|$S0 --loss_fn hungarian||finetuning_multi_paper.py|$S0ENV"
  "S0_cap|$S0 --loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8||finetuning_multi_paper.py|$S0ENV"
  "S0_const|$S0 --loss_fn hungarian --full_sampling||finetuning_multi_paper.py|$S0ENV"
  "S0_10k|$S0 --loss_fn hungarian||finetuning_multi_paper.py|$S0ENV $LONG"
  "S0_cap_10k|$S0 --loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8||finetuning_multi_paper.py|$S0ENV $LONG"
  "S0_const_10k|$S0 --loss_fn hungarian --full_sampling||finetuning_multi_paper.py|$S0ENV $LONG"
  # 2026-10-31 (user): S0_10k with sample_rate_multiplier 10 and no cap (rate min(10 t/T, 1.0)); compared with S0_cap_10k
  # (multiplier 10, cap 0.8) on full dev, and the better one is the Stage B base.
  "S0_mult10_10k|$S0 --loss_fn hungarian --sampling_saturate_frac 0.1||finetuning_multi_paper.py|$S0ENV $LONG"
  # 2026-11-01 Stage B on S0_cap_10k (test 31.07 at its dev-selected step vs S0_mult10_10k 29.76; user: pick by test): one-label step x random-init io layers.
  "S0_cap10k_onelabel|$S0 --loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step||finetuning_multi_paper.py|$S0ENV $LONG20"
  "S0_cap10k_ioprojrand|$S0 --loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --io_proj --io_proj_init random||finetuning_multi_paper.py|$S0ENV $LONG"
  "S0_cap10k_onelabel_ioprojrand|$S0 --loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step --io_proj --io_proj_init random||finetuning_multi_paper.py|$S0ENV $LONG20"
  # 2026-11-01 (user): best Stage B setting (one-label step on S0_cap_10k; dev still rising at 10k) run longer: 20k steps,
  # warmup 800, ramp saturates at 2k (0.1 T) under the 0.8 cap, ~23 h of training at 10k = 11.4 h.
  "S0_cap20k_onelabel|$S0 --loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step||finetuning_multi_paper.py|$S0ENV $LONG20K"
  # 2026-11-03 (user): step-matched single-query baselines for the one-gold run (S0_cap10k_onelabel, 10k steps, two
  # optimizer updates per step). _2step: --one_label_step in single mode = a second update on an independent redraw
  # (new gold / negatives) of the same questions (finetuning_multi_paper._install_single_redraw).
  "S0_single_10k|$S0||finetuning_multi_paper.py|$S0ENV $LONG"
  "S0_single_10k_2step|$S0 --one_label_step||finetuning_multi_paper.py|$S0ENV $LONG20"
  # 2026-11-04 (user): Stage D before C, on the best multi setting (S0_cap10k_onelabel): no hard negatives (random
  # negatives only, ratio 0.0 overrides BASE 0.5); causal query-encoder attention (--force_causal, also used at retrieval).
  "S0_cap10k_onelabel_nohn|$S0 --loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step --negative_hard_ratio 0.0||finetuning_multi_paper.py|$S0ENV $LONG20"
  "S0_cap10k_onelabel_causal|$S0 --loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step --force_causal||finetuning_multi_paper.py|$S0ENV $LONG20"
  # Single-query partners of the D runs (user 2026-11-04), on S0_single_10k_2step (same two updates per step).
  "S0_single_10k_2step_nohn|$S0 --one_label_step --negative_hard_ratio 0.0||finetuning_multi_paper.py|$S0ENV $LONG20"
  "S0_single_10k_2step_causal|$S0 --one_label_step --force_causal||finetuning_multi_paper.py|$S0ENV $LONG20"
  # 2026-11-06 (user): causal kept; J's sequential-matching losses (E1) on the multi recipe S0_cap10k_onelabel_causal.
  # hungarian_seq: output t may not match a gold fed in before it; _used: only already-used golds are negatives.
  "S0_cap10k_onelabel_causal_seq|$S0 --loss_fn hungarian_seq --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step --force_causal||finetuning_multi_paper.py|$S0ENV $LONG20"
  "S0_cap10k_onelabel_causal_seqused|$S0 --loss_fn hungarian_seq_used --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step --force_causal||finetuning_multi_paper.py|$S0ENV $LONG20"
  # 2026-11-06 (user): Stage F run 1 pulled forward: learned input + output layers, identity init (default), plain Hungarian.
  "S0_cap10k_onelabel_causal_ioproj|$S0 --loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step --force_causal --io_proj||finetuning_multi_paper.py|$S0ENV $LONG20"
  # 2026-11-09 Stage C (user 2026-11-06): temperature 0.01 / 0.02 / 0.03 (base 0.05), single-query partners on
  # S0_single_10k_2step_causal first (their base does not depend on the pending seq / ioproj results).
  "S0_single_10k_2step_causal_t001|$S0 --one_label_step --force_causal --temperature 0.01||finetuning_multi_paper.py|$S0ENV $LONG20"
  "S0_single_10k_2step_causal_t002|$S0 --one_label_step --force_causal --temperature 0.02||finetuning_multi_paper.py|$S0ENV $LONG20"
  "S0_single_10k_2step_causal_t003|$S0 --one_label_step --force_causal --temperature 0.03||finetuning_multi_paper.py|$S0ENV $LONG20"
  # 15 negatives per question (single mode: finetuning_multi_paper._enable_single_negatives, smoke 22489815 passed).
  "S0_single_10k_2step_causal_neg15|$S0 --one_label_step --force_causal --negative_ctxs 15||finetuning_multi_paper.py|$S0ENV $LONG20"
)
log() { echo "[$(date '+%F %T')] [F queue] $*" | tee -a results/phaseC/phaseC.log; }
for e in "${QUEUE[@]}"; do
  IFS='|' read -r r extra root script envs <<< "$e"; run=clean_v3_f_qampari_$r
  mode=multi; case $r in *single*) mode=single ;; esac
  [ -s results/phaseC/$run.trainjob ] && continue
  while [ "$(squeue -h -u $USER -o %j | grep -c '^phaseC_clean_v3_f_qampari_')" -ge $MAXRUN ]; do sleep 300; done
  env $envs CODE_ROOT=$root TRAIN_SCRIPT=${script:-finetuning_multi.py} setsid nohup bash training/inf_retriever/tools/f_run.sh $mode 1e-5 $run "$BASE $extra" \
    >> results/phaseC/$run.driver.out 2>&1 < /dev/null &
  sleep 30; log "started $run ($(cat results/phaseC/$run.trainjob 2>/dev/null))"
done
log "queue empty"
