#!/bin/bash
# Session F, item F2 (objective audit 2026-10-13): frozen-doc-encoder (cell C) loss / schedule batch, same recipe as
# clean_v3_abl2_qampari_C_multi (clean v3 + hard negatives, ratio 0.5, LR 1e-5, 2,500 steps, bidirectional) with one
# change each. Each run is driven by f_run.sh (full dev on every checkpoint, test on top 2 + final).
#   C_sat50     plain Hungarian, rate reaches 1 at 50% (step 1250)   (D4(a): joint clean_v3_abl2_qampari_sat50 exists)
#   C_sat75     plain Hungarian, rate reaches 1 at 75% (step 1875)
#   C_masked    hungarian_masked
#   C_hps05     hungarian_plus_single, single-query weight 0.5
#   C_multi_t01 plain Hungarian, temperature 0.1 (C_multi: 0.05)
# Usage (repo root): bash training/inf_retriever/tools/f2_launch.sh [run suffix ...]   (default: all five)
cd /scratch/hc3337/projects/autoregressive
P=clean_v3_f_qampari
BASE="--train_data data/training/clean_v3_hn/qampari/train_data.jsonl --negative_hard_ratio 0.5 --freeze_doc_encoder"
declare -A V=(
  [C_sat50]="--sampling_saturate_frac 0.5"
  [C_sat75]="--sampling_saturate_frac 0.75"
  [C_masked]="--loss_fn hungarian_masked"
  [C_hps05]="--loss_fn hungarian_plus_single --single_loss_weight 0.5"
  [C_multi_t01]="--temperature 0.1"
)
for r in ${@:-C_sat50 C_sat75 C_masked C_hps05 C_multi_t01}; do
  run=${P}_$r
  pgrep -f "f_run.sh multi .* $run( |$)" >/dev/null && { echo "$run driver already running"; continue; }
  setsid nohup bash training/inf_retriever/tools/f_run.sh multi 1e-5 $run "$BASE ${V[$r]}" > results/phaseC/$run.driver.out 2>&1 < /dev/null &
  echo "launched $run: $BASE ${V[$r]}"; sleep 2
done
