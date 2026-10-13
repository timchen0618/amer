#!/bin/bash
# Session F launch queue (objective audit 2026-10-13): keeps at most MAXRUN frozen F training runs queued or running
# (2 GPUs each; F's share is <= 8 GPUs once J's joint runs are in, leaving room for F's 1-GPU checkpoint retrievals)
# and starts the next queued run with f_run.sh when one finishes training. Runs already started (a .trainjob file)
# are skipped, so restarting this script is safe.
# Usage (repo root, detached): setsid nohup bash training/inf_retriever/tools/f_queue.sh > results/phaseC/f_queue.out 2>&1 &
cd /scratch/hc3337/projects/autoregressive
MAXRUN=${MAXRUN:-3}
WT=/scratch/hc3337/projects/autoregressive/.claude/worktrees/f-objective
BASE="--train_data data/training/clean_v3_hn/qampari/train_data.jsonl --negative_hard_ratio 0.5 --freeze_doc_encoder"
# order: F5, D4(a) test, rest of F2, F3b.  name|extra|code root ("" = main checkout)
QUEUE=(
  "C_fednorm|--fed_norm token|$WT"
  "C_sat50|--sampling_saturate_frac 0.5|"
  "C_sat75|--sampling_saturate_frac 0.75|"
  "C_masked|--loss_fn hungarian_masked|"
  "C_hps05|--loss_fn hungarian_plus_single --single_loss_weight 0.5|"
  "C_multi_t01|--temperature 0.1|"
  "C_seq_used|--loss_fn hungarian_seq_used|$WT"
)
log() { echo "[$(date '+%F %T')] [F queue] $*" | tee -a results/phaseC/phaseC.log; }
for e in "${QUEUE[@]}"; do
  IFS='|' read -r r extra root <<< "$e"; run=clean_v3_f_qampari_$r
  [ -s results/phaseC/$run.trainjob ] && continue
  while [ "$(squeue -h -u $USER -o %j | grep -c '^phaseC_clean_v3_f_qampari_')" -ge $MAXRUN ]; do sleep 300; done
  CODE_ROOT=$root setsid nohup bash training/inf_retriever/tools/f_run.sh multi 1e-5 $run "$BASE $extra" \
    >> results/phaseC/$run.driver.out 2>&1 < /dev/null &
  sleep 30; log "started $run ($(cat results/phaseC/$run.trainjob 2>/dev/null))"
done
log "queue empty"
