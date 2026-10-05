#!/bin/bash
# 24-run grid on the clean splits (data_creation/build_clean_splits.py):
#   dataset {qampari, ambigqa} x mode {single, multi} x data {no hard neg., hard neg.}
#   x LR {1e-6, 3e-6, 1e-5}; loss hungarian (multi) / contrastive (single), scheduled sampling.
# Each run goes through phaseC_run.sh (cheap dev eval + drift of every saved checkpoint):
#   QAMPARI: saves every 250 steps (DENSE=1), keeps the top 3 by cheap dev (KEEP_TOP=3);
#   AmbigQA: default saves (30/100/150/250/400); KEEP_STEPS="150 400" deletes every other step once its
#   cheap eval and drift are done (added after the grid ran; the grid itself kept everything).
# When a run is done, its full-eval candidates are appended to results/phaseC/full_queue_<ds>.txt
# (QAMPARI: top 2 by cheap dev; AmbigQA: steps 150 and 400), which full_eval_queue.sh consumes.
#
# Usage (repo root, run detached): bash training/inf_retriever/tools/grid_launch.sh [filter]
#   filter: optional grep pattern on run names, e.g. "qampari_multi".
# Environment (defaults reproduce the 24-run grid of 2026-09-28):
#   DATA_ROOT  training data, default data/training/clean (hard-negative runs use ${DATA_ROOT}_hn);
#              data/training/clean_v2 for the v2 splits (data_creation/build_clean_v2.py)
#   DEV_TAG    dev set for cheap evals, default "clean"; "cleanv2" for the v2 dev sets
#   PREFIX     run-name prefix, default "grid". Any other prefix also gets its own full-eval queues,
#              results/phaseC/full_queue_<ds>_<PREFIX>.txt; start full_eval_queue.sh on them with
#              QUEUE=<that file> and the same DEV_TAG.
set -u
cd /scratch/hc3337/projects/autoregressive
T=training/inf_retriever/tools
LOG=results/phaseC/phaseC.log
FILTER=${1:-.}
DATA_ROOT=${DATA_ROOT:-data/training/clean}; DEV_TAG=${DEV_TAG-clean}; PREFIX=${PREFIX:-grid}
export DATA_ROOT DEV_TAG   # read by phaseC_run.sh (cheap-eval dev set) and phaseC_train.sbatch (data)
QSUF=""; [ "$PREFIX" != grid ] && QSUF="_$PREFIX"
log() { echo "[$(date '+%F %T')] [grid] $*" | tee -a "$LOG" >&2; }

runs=()
for ds in qampari ambigqa; do
  for mode in single multi; do
    for data in nohn hn; do
      for lr in 1e-6 3e-6 1e-5; do
        runs+=("$ds $mode $data $lr ${PREFIX}_${ds}_${mode}_${data}_lr${lr}")
      done
    done
  done
done

for r in "${runs[@]}"; do
  set -- $r; ds=$1; mode=$2; data=$3; lr=$4; run=$5
  echo "$run" | grep -q -- "$FILTER" || continue
  [ -e results/phaseC/$run.trainjob ] && { log "$run already submitted; skipping"; continue; }
  extra=""
  [ "$data" = hn ] && extra="--train_data ${DATA_ROOT}_hn/$ds/train_data.jsonl --negative_hard_ratio 0.5"
  if [ "$ds" = qampari ]; then dense=1; keep=3; ksteps=""; else dense=0; keep=0; ksteps="150 400"; fi
  DENSE=$dense KEEP_TOP=$keep KEEP_STEPS="$ksteps" setsid nohup bash $T/phaseC_run.sh $ds $mode $lr $run "$extra" \
      > results/phaseC/$run.driver.out 2>&1 < /dev/null &
  log "launched $run (ds=$ds mode=$mode data=$data lr=$lr dense=$dense keep_top=$keep keep_steps='$ksteps' data_root=$DATA_ROOT dev_tag=$DEV_TAG)"
  sleep 2
done

# Follower: queue each finished run's full-eval candidates.
(
  while true; do
    pending=0
    for r in "${runs[@]}"; do
      set -- $r; ds=$1; mode=$2; run=$5
      echo "$run" | grep -q -- "$FILTER" || continue
      marker=results/phaseC/$run.queued
      [ -e "$marker" ] && continue
      last=$(grep -E "\[$run\] RUN (DONE|FAILED)" "$LOG" | tail -1)
      if ! echo "$last" | grep -q "RUN DONE"; then
        pending=1
        # A failed driver never writes RUN DONE: say so once, then keep waiting for a restart (G7).
        if echo "$last" | grep -q "RUN FAILED" && [ ! -e results/phaseC/$run.failed_noted ]; then
          touch results/phaseC/$run.failed_noted
          log "WARNING: $run driver failed; its full evals are not queued until it is restarted and finishes"
        fi
        continue
      fi
      rm -f results/phaseC/$run.failed_noted
      k=-; [ "$mode" = multi ] && { [ "$ds" = qampari ] && k=5 || k=2; }
      if [ "$ds" = qampari ]; then
        missing=$(for d in results/phaseC/$run/step*; do grep -q MRecall $d/eval_metrics.txt 2>/dev/null || echo -n "${d##*step} "; done)
        [ -n "$missing" ] && log "WARNING: $run has no cheap-dev metrics for steps $missing; they are not candidates for full eval"
        steps=$(for d in results/phaseC/$run/step*; do
                  m=$(grep -m1 -o 'MRecall: [0-9.]*' $d/eval_metrics.txt 2>/dev/null | cut -d' ' -f2)
                  [ -n "$m" ] && echo "${d##*step} $m"; done | sort -k2,2nr -k1,1n | head -2 | cut -d' ' -f1)
      else
        steps="150 400"
      fi
      for s in $steps; do
        echo "${run}_s$s checkpoints/$ds/$run/checkpoint/step-$s $k" >> results/phaseC/full_queue_$ds$QSUF.txt
      done
      if [ -z "$steps" ]; then log "WARNING: $run has no cheap-dev metrics at all; nothing queued for full eval"
      else log "queued full evals for $run: steps $(echo $steps)"; fi
      touch "$marker"
    done
    [ $pending = 0 ] && { log "all grid runs done and queued"; break; }
    sleep 300
  done
) > results/phaseC/grid_follower.out 2>&1 < /dev/null &
disown
log "follower started"
