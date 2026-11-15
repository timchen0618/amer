#!/bin/bash
# Session N (2026-11-15): the Qwen3-Embedding-0.6B replication (round 1 + the three round-2 fixes) with at most MAXRUN training jobs at once (2 GPUs each;
# user budget 8 GPUs). Each entry: <suffix>|<flags>|<env for n_launch.sh>|<file that must exist first, or empty>.
# A run is launched once (marker: results/phaseC/clean_v3_n_qampari_<suffix>.driver.out). Detached:
#   setsid nohup bash .claude/worktrees/n-nemotron/training/inf_retriever/tools/q_queue.sh >> results/phaseC/q_queue.out 2>&1 &
cd /scratch/hc3337/projects/autoregressive
WT=/scratch/hc3337/projects/autoregressive/.claude/worktrees/n-nemotron/training/inf_retriever/tools
MAXRUN=${MAXRUN:-4}
LIN="--loss_fn hungarian --one_label_step"
CAP="--loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step"
SGL="--one_label_step"
QHN="N_DATA_ROOT=data/training/clean_v3_hn_s0_qwen3"
HNREADY=data/training/clean_v3_hn_s0_qwen3/qampari/train_data.jsonl
READY=results/phaseC/q_smoke_ok   # written by hand after the 30-step smokes pass (corpus embedded, retrieval works)
QUEUE=(
  # round 1: F's recipe (inf-mined S0 hard negatives, lr 1e-5)
  "qw3_S0_single_10k_2step|$SGL||$READY"
  "qw3_S0_10k_onelabel|$LIN||$READY"
  "qw3_S0_cap10k_onelabel|$CAP||$READY"
  # round 2a: lr 5e-6
  "qw3_S0_10k_onelabel_lr5e6|$LIN|N_LR=5e-6|$READY"
  "qw3_S0_single_10k_2step_lr5e6|$SGL|N_LR=5e-6|$READY"
  "qw3_S0_cap10k_onelabel_lr5e6|$CAP|N_LR=5e-6|$READY"
  # round 2b: hard negatives mined with base Qwen3-Embedding
  "qw3_S0qhn_10k_onelabel|$LIN|$QHN|$HNREADY"
  "qw3_S0qhn_single_10k_2step|$SGL|$QHN|$HNREADY"
  "qw3_S0qhn_cap10k_onelabel|$CAP|$QHN|$HNREADY"
  # round 2c: learned input / output layers (identity init); single has the output layer only
  "qw3_S0_10k_onelabel_ioproj|$LIN --io_proj||$READY"
  "qw3_S0_single_10k_2step_ioproj_out|$SGL --io_proj --io_proj_mode out||$READY"
  "qw3_S0_cap10k_onelabel_ioproj|$CAP --io_proj||$READY"
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
    env $envs bash $WT/n_launch.sh $suf qwen3 "$flags"
    left=$((left - 1)); sleep 90   # let the driver submit its training job before counting again
  done
  [ $left -eq 0 ] && { log "queue empty"; exit 0; }
  sleep 300
done
