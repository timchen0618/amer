#!/bin/bash
# Session J (joint doc encoder) runs of the objective audit (reports/objective_audit_2026-10-13/PLAN.md).
# Trains this worktree's code (CODE_ROOT) with the clean_v3 QAMPARI recipe; data and checkpoints stay
# under the main checkout. Training only: evaluation runs after the branch is merged.
#   E1 x E2 on the hps 0.5 recipe (off/off = round 1's clean_v3_abl_qampari_hps):
#     j_hps_seq         --hps_multi_loss hungarian_seq   (no matching to a gold fed in before)
#     j_hps_detach      --detach_fed_output
#     j_hps_seq_detach  both
#   E5: j_hps_fednorm  --fed_norm token (fed vectors are 80-300x a token embedding, probe D2)
#   D4(b): j_sat50_dense  plain Hungarian, saturate 50% (as clean_v3_abl2_qampari_sat50), in-training eval
#     every 50 steps with the new logging, to order the events around the switch at step 1250.
# Usage (from anywhere): bash <worktree>/training/inf_retriever/tools/j_launch.sh [run ...]
set -euo pipefail
WT=$(cd "$(dirname "$0")/../../.." && pwd)
MAIN=/scratch/hc3337/projects/autoregressive
cd "$MAIN"
HN="--train_data data/training/clean_v3_hn/qampari/train_data.jsonl --negative_hard_ratio 0.5"   # as abl/abl2_launch.sh
HPS="--loss_fn hungarian_plus_single --single_loss_weight 0.5 --select_metric cover5"
SAVE="500 750 1000 1250 1500 2000 2500"
declare -A EXTRA SAVES
EXTRA[j_hps_seq]="$HPS --hps_multi_loss hungarian_seq";                         SAVES[j_hps_seq]=$SAVE
EXTRA[j_hps_detach]="$HPS --detach_fed_output";                                 SAVES[j_hps_detach]=$SAVE
EXTRA[j_hps_seq_detach]="$HPS --hps_multi_loss hungarian_seq --detach_fed_output"; SAVES[j_hps_seq_detach]=$SAVE
EXTRA[j_hps_fednorm]="$HPS --fed_norm token";                                  SAVES[j_hps_fednorm]=$SAVE
EXTRA[j_sat50_dense]="--sampling_saturate_frac 0.5 --eval_freq 50 --select_metric cover5"; SAVES[j_sat50_dense]="1000 1250 1500 2500"
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(j_hps_seq j_hps_detach j_hps_seq_detach j_hps_fednorm j_sat50_dense)
for r in "${RUNS[@]}"; do
  run=clean_v3_j_qampari_${r#j_}
  [ -n "${EXTRA[$r]+x}" ] || { echo "unknown run $r" >&2; exit 1; }
  if [ -e "checkpoints/qampari/$run" ]; then echo "$run exists; skipping" >&2; continue; fi
  jid=$(DS=qampari MODE=multi LR=1e-5 RUN_NAME=$run EXTRA="$HN ${EXTRA[$r]}" SAVE_AT_OVERRIDE="${SAVES[$r]}" \
        DATA_ROOT=data/training/clean_v3 CODE_ROOT=$WT \
        sbatch --parsable --export=ALL --job-name=phaseC_$run "$WT/training/inf_retriever/tools/phaseC_train.sbatch")
  echo "$(date '+%F %T') $run job $jid code $(git -C "$WT" rev-parse --short HEAD) extra='${EXTRA[$r]}'" | tee -a results/phaseC/j_runs.log
done
