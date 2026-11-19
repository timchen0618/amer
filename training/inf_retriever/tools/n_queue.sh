#!/bin/bash
# Session N (2026-11-13): run the Nemotron gap-closing round with at most MAXRUN training jobs at once (2 GPUs each;
# user budget 8 GPUs). Each entry: <suffix>|<flags>|<env for n_launch.sh>|<file that must exist first, or empty>.
# A run is launched once (marker: results/phaseC/clean_v3_n_qampari_<suffix>.driver.out). Detached:
#   setsid nohup bash .claude/worktrees/n-nemotron/training/inf_retriever/tools/n_queue.sh >> results/phaseC/n_queue.out 2>&1 &
cd /scratch/hc3337/projects/autoregressive
WT=/scratch/hc3337/projects/autoregressive/.claude/worktrees/n-nemotron/training/inf_retriever/tools
MAXRUN=${MAXRUN:-4}
LIN="--loss_fn hungarian --one_label_step"
CAP="--loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step"
SGL="--one_label_step"
NHN="N_DATA_ROOT=data/training/clean_v3_hn_s0_nemo"
HNREADY=data/training/clean_v3_hn_s0_nemo/qampari/train_data.jsonl
PROJOK=results/phaseC/n_ioproj_smoke_ok
QUEUE=(
  # (2) lower learning rate: 5e-6 (F's 1e-5 / 2)
  "nemo_S0_10k_onelabel_lr5e6|$LIN|N_LR=5e-6|"
  "nemo_S0_single_10k_2step_lr5e6|$SGL|N_LR=5e-6|"
  "nemo_S0_cap10k_onelabel_lr5e6|$CAP|N_LR=5e-6|"
  # (3) learned input / output layers (F's --io_proj, identity init); single has the output layer only
  "nemo_S0_10k_onelabel_ioproj|$LIN --io_proj||$PROJOK"
  "nemo_S0_single_10k_2step_ioproj_out|$SGL --io_proj --io_proj_mode out||$PROJOK"
  "nemo_S0_cap10k_onelabel_ioproj|$CAP --io_proj||$PROJOK"
  # (1) hard negatives mined with base Nemotron
  "nemo_S0nhn_10k_onelabel|$LIN|$NHN|$HNREADY"
  "nemo_S0nhn_single_10k_2step|$SGL|$NHN|$HNREADY"
  "nemo_S0nhn_cap10k_onelabel|$CAP|$NHN|$HNREADY"
)
log() { echo "[$(date '+%F %T')] $*"; }
while true; do
  left=0
  for e in "${QUEUE[@]}"; do
    IFS='|' read -r suf flags envs req <<< "$e"
    [ -f results/phaseC/clean_v3_n_qampari_$suf.driver.out ] && continue
    left=$((left + 1))
    [ -n "$req" ] && [ ! -e "$req" ] && continue
    active=$(squeue -u $USER -h -o "%j" | grep -c "^phaseC_clean_v3_n_qampari_")
    [ "$active" -ge "$MAXRUN" ] && break
    log "launch $suf ($envs) [$active active]"
    env $envs bash $WT/n_launch.sh $suf nemotron "$flags"
    left=$((left - 1)); sleep 90   # let the driver submit its training job before counting again
  done
  [ $left -eq 0 ] && { log "queue empty"; exit 0; }
  sleep 300
done
