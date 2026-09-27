#!/bin/bash
# Step 5 / phase C: one training run + cheap dev evaluation (and parameter drift) of every
# saved checkpoint. Cheap evals use the phase A reduced corpora on the DEV queries
# (qampari_dev500 / ambigqa_dev300); they screen runs and catch collapse. Full-corpus test
# evals of the top candidates are run separately (tools/gen_embed_ckpt -> retrieve -> eval).
#
# Usage (repo root; lightweight, run detached):
#   bash training/inf_retriever/tools/phaseC_run.sh <qampari|ambigqa> <multi|single> <lr> <run_name> ["<extra training args>"]
# Log: results/phaseC/phaseC.log ("RESULT" lines per checkpoint, "ERROR" on failure).
set -u
cd /scratch/hc3337/projects/autoregressive
DS=$1; MODE=$2; LR=$3; RUN=$4; EXTRA=${5:-}
T=training/inf_retriever/tools
mkdir -p results/phaseC results/param_drift
LOG=results/phaseC/phaseC.log
log() { echo "[$(date '+%F %T')] [$RUN] $*" | tee -a "$LOG" >&2; }
case $DS in
  qampari) STEPS="250 500 1000 2500"; SET=qampari_dev500; K_MULTI=5 ;;
  ambigqa) STEPS="30 100 150 250 400"; SET=ambigqa_dev300; K_MULTI=2 ;;
esac
K=""; [ "$MODE" = multi ] && K=$K_MULTI
job_active() { squeue -h -j "$1" 2>/dev/null | grep -q .; }

state=results/phaseC/$RUN.trainjob
if [ -s "$state" ]; then TJ=$(cat "$state"); else
  TJ=$(DS=$DS MODE=$MODE LR=$LR RUN_NAME=$RUN EXTRA="$EXTRA" sbatch --parsable --export=ALL --job-name=phaseC_$RUN $T/phaseC_train.sbatch) \
    || { log "ERROR: training submit failed"; exit 1; }
  echo "$TJ" > "$state"; log "submitted training job $TJ (ds=$DS mode=$MODE lr=$LR extra='$EXTRA')"
fi
TLOG=sbatch_outputs/phaseC_$RUN.out

declare -A EJ DJ
for s in $STEPS; do
  ck=checkpoints/$DS/$RUN/checkpoint/step-$s
  out=results/phaseC/$RUN/step$s
  if grep -q MRecall $out/eval_metrics.txt 2>/dev/null; then continue; fi
  while ! grep -q "Saving model to $ck\$" "$TLOG" 2>/dev/null; do
    if ! job_active "$TJ"; then
      grep -q "Saving model to $ck\$" "$TLOG" 2>/dev/null && break
      log "ERROR: training job $TJ ended ($(sacct -j $TJ -X -n -o State | head -1 | tr -d ' ')) without saving step $s"; exit 1
    fi
    sleep 120
  done
  EJ[$s]=$(sbatch --parsable --job-name=phaseC_eval_${RUN}_s$s \
      --export=ALL,CKPT=$ck,SET=$SET,DATA=data/phaseA/$SET.jsonl,OUT_DIR=$out,K=$K $T/cheap_eval_ckpt.sbatch)
  DJ[$s]=$(sbatch --parsable --account=torch_pr_152_courant --time=00:40:00 --mem=48G --cpus-per-task=2 \
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
log "RUN DONE"
