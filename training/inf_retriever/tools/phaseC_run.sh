#!/bin/bash
# Step 5 / phase C: one training run + cheap dev evaluation (and parameter drift) of every
# saved checkpoint. Cheap evals use the phase A reduced corpora on the DEV queries
# (qampari_dev500 / ambigqa_dev300); they screen runs and catch collapse. Full-corpus test
# evals of the top candidates are run separately (tools/gen_embed_ckpt -> retrieve -> eval).
#
# Usage (repo root; lightweight, run detached):
#   bash training/inf_retriever/tools/phaseC_run.sh <qampari|ambigqa> <multi|single> <lr> <run_name> ["<extra training args>"]
# Log: results/phaseC/phaseC.log ("RESULT" lines per checkpoint, "ERROR" on failure).
# Optional environment:
#   DENSE=1     save + cheap-eval every 250 steps (QAMPARI) / every 50 steps (AmbigQA)
#   KEEP_TOP=N  after each cheap eval (and its drift) finishes, keep only the N best checkpoints
#               by cheap dev MRecall@100 and delete the rest; step-0 is deleted once all drift
#               jobs are done. Checkpoints listed in results/phaseC/full_queue_*.txt are never
#               deleted. Default 0 keeps everything.
#   DEV_TAG     dev set for cheap evals: "clean" (default: data/phaseA/<ds>_cleandev500.jsonl and its
#               reduced corpus) or "" (the phase A dev sets, as used before the 2026-09-27 audit).
#               Training data comes from DATA_ROOT (see phaseC_train.sbatch; default data/training/clean).
set -u
cd /scratch/hc3337/projects/autoregressive
DS=$1; MODE=$2; LR=$3; RUN=$4; EXTRA=${5:-}
T=training/inf_retriever/tools
mkdir -p results/phaseC results/param_drift
LOG=results/phaseC/phaseC.log
log() { echo "[$(date '+%F %T')] [$RUN] $*" | tee -a "$LOG" >&2; }
DENSE=${DENSE:-0}; KEEP_TOP=${KEEP_TOP:-0}
DEV_TAG=${DEV_TAG-clean}   # clean dev sets (500 each); DEV_TAG="" = the phase A dev sets
case $DS in
  qampari) STEPS="250 500 1000 2500"; SET=qampari_${DEV_TAG}dev500; K_MULTI=5
           [ "$DENSE" = 1 ] && STEPS=$(seq -s ' ' 250 250 2500) ;;
  ambigqa) STEPS="30 100 150 250 400"; SET=ambigqa_${DEV_TAG}dev500; K_MULTI=2
           [ -z "$DEV_TAG" ] && SET=ambigqa_dev300
           [ "$DENSE" = 1 ] && STEPS=$(seq -s ' ' 50 50 400) ;;
esac
SAVE_AT_OVERRIDE=""; [ "$DENSE" = 1 ] && SAVE_AT_OVERRIDE="$STEPS"
K=""; [ "$MODE" = multi ] && K=$K_MULTI
# squeue can come back empty while the controller is busy, so confirm with sacct before calling a
# job finished; an unknown state counts as active (worst case: one more poll).
JOB_ACTIVE_STATES='^(PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED)'
job_active() {
  squeue -h -j "$1" 2>/dev/null | grep -q . && return 0
  local st; st=$(sacct -j "$1" -X -n -o State 2>/dev/null | tr -d ' ')
  [ -z "$st" ] && return 0
  echo "$st" | grep -qE "$JOB_ACTIVE_STATES"
}

state=results/phaseC/$RUN.trainjob
if [ -s "$state" ]; then TJ=$(cat "$state"); else
  # A fresh submission under a reused run name: move the old training log, results and
  # checkpoints aside, so their "Saving model" lines and eval_metrics.txt are not mistaken for
  # this run's.
  stamp=$(date +%Y%m%d_%H%M%S)
  for old in sbatch_outputs/phaseC_$RUN.out results/phaseC/$RUN checkpoints/$DS/$RUN; do
    [ -e "$old" ] && mv -- "$old" "$old.old_$stamp" && log "moved stale $old to $old.old_$stamp"
  done
  TJ=$(DS=$DS MODE=$MODE LR=$LR RUN_NAME=$RUN EXTRA="$EXTRA" SAVE_AT_OVERRIDE="$SAVE_AT_OVERRIDE" sbatch --parsable --export=ALL --job-name=phaseC_$RUN $T/phaseC_train.sbatch) \
    || { log "ERROR: training submit failed"; exit 1; }
  echo "$TJ" > "$state"; log "submitted training job $TJ (ds=$DS mode=$MODE lr=$LR extra='$EXTRA' dense=$DENSE keep_top=$KEEP_TOP)"
fi
TLOG=sbatch_outputs/phaseC_$RUN.out

declare -A EJ DJ
# True if a full-eval queue lists this checkpoint (paths compared after resolving ./, absolute
# paths and trailing slashes).
queued() {
  local want p
  want=$(realpath -m -- "$1")
  for p in $(awk 'NF >= 2 && $1 != "END" {print $2}' results/phaseC/full_queue_*.txt 2>/dev/null); do
    [ "$(realpath -m -- "$p")" = "$want" ] && return 0
  done
  return 1
}
# Keep only the KEEP_TOP best finished checkpoints (by cheap dev MRecall@100); see header.
prune() {
  [ "$KEEP_TOP" -gt 0 ] 2>/dev/null || return 0
  local s m ranked keep ck
  ranked=$(for s in $STEPS; do
      m=$(grep -m1 -o 'MRecall: [0-9.]*' results/phaseC/$RUN/step$s/eval_metrics.txt 2>/dev/null | cut -d' ' -f2)
      [ -z "$m" ] && continue
      { [ -n "${EJ[$s]:-}" ] && job_active "${EJ[$s]}"; } && continue
      { [ -n "${DJ[$s]:-}" ] && job_active "${DJ[$s]}"; } && continue
      echo "$s $m"; done | sort -k2,2nr -k1,1n)
  keep=$(echo "$ranked" | head -n "$KEEP_TOP" | cut -d' ' -f1)
  for s in $(echo "$ranked" | tail -n +$((KEEP_TOP+1)) | cut -d' ' -f1); do
    ck=checkpoints/$DS/$RUN/checkpoint/step-$s
    [ -d "$ck" ] || continue
    queued "$ck" && continue
    rm -rf -- "$ck" && log "pruned step $s (not in top $KEEP_TOP by cheap dev; kept: $(echo $keep))"
  done
}
for s in $STEPS; do
  ck=checkpoints/$DS/$RUN/checkpoint/step-$s
  out=results/phaseC/$RUN/step$s
  if grep -q MRecall $out/eval_metrics.txt 2>/dev/null; then continue; fi
  while ! grep -q "Saving model to $ck\$" "$TLOG" 2>/dev/null; do
    if ! job_active "$TJ"; then
      grep -q "Saving model to $ck\$" "$TLOG" 2>/dev/null && break
      log "ERROR: training job $TJ ended ($(sacct -j $TJ -X -n -o State | head -1 | tr -d ' ')) without saving step $s"; exit 1
    fi
    prune; sleep 120
  done
  # On a driver restart, reuse this step's eval / drift jobs if they are still queued or running.
  EJ[$s]=$(squeue -h -u "$USER" --name=phaseC_eval_${RUN}_s$s -o %i 2>/dev/null | head -1)
  DJ[$s]=$(squeue -h -u "$USER" --name=phaseC_drift_${RUN}_s$s -o %i 2>/dev/null | head -1)
  if [ -n "${EJ[$s]}" ] || [ -n "${DJ[$s]}" ]; then
    log "step $s: reusing queued jobs (eval ${EJ[$s]:-none}, drift ${DJ[$s]:-none})"
    [ -n "${EJ[$s]}" ] && [ -n "${DJ[$s]}" ] && continue
  fi
  [ -z "${EJ[$s]}" ] && EJ[$s]=$(sbatch --parsable --job-name=phaseC_eval_${RUN}_s$s \
      --export=ALL,CKPT=$ck,SET=$SET,DATA=data/phaseA/$SET.jsonl,OUT_DIR=$out,K=$K $T/cheap_eval_ckpt.sbatch)
  [ -z "${DJ[$s]}" ] && DJ[$s]=$(sbatch --parsable --account=torch_pr_152_courant --time=00:40:00 --mem=48G --cpus-per-task=2 \
      --job-name=phaseC_drift_${RUN}_s$s --output=sbatch_outputs/phaseC_drift_${RUN}_s$s.out \
      --wrap "singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif bash -c 'source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python $T/param_drift.py --run_dir checkpoints/$DS/$RUN --steps $s --out results/param_drift/phaseC_$DS.jsonl'")
  log "step $s saved; cheap dev eval ${EJ[$s]}, drift ${DJ[$s]}"
done

for s in $STEPS; do
  out=results/phaseC/$RUN/step$s
  [ -n "${EJ[$s]:-}" ] && while job_active "${EJ[$s]}" || job_active "${DJ[$s]}"; do sleep 60; done
  m=$(grep -m1 MRecall $out/eval_metrics.txt 2>/dev/null | cut -d'|' -f1-2)
  d=$(grep "\"run\": \"$RUN\", \"step\": $s," results/param_drift/phaseC_$DS.jsonl 2>/dev/null | tail -1 | \
      python3 -c 'import json,sys; r=json.loads(sys.stdin.read()); print("changed %.3f%%, rel. distance %.2e" % (100*r["frac_changed"], r["rel_l2"]))' 2>/dev/null)
  [ -z "$m" ] && { log "ERROR: cheap dev eval for step $s produced no metrics (see sbatch_outputs/cheap_eval_phaseC_eval_${RUN}_s$s.out)"; continue; }
  log "RESULT step $s: dev(cheap) $m | drift: $d"
done
prune
if [ "$KEEP_TOP" -gt 0 ] 2>/dev/null && [ -d checkpoints/$DS/$RUN/checkpoint/step-0 ]; then
  rm -rf -- checkpoints/$DS/$RUN/checkpoint/step-0 && log "deleted step-0 (drift done)"
fi
log "RUN DONE"
