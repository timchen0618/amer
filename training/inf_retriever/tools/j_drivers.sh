#!/bin/bash
# Attach a phaseC_run.sh driver (cheap dev eval + drift of every saved step, prune to the top 3 + the
# final step) to each Session J training job submitted by tools/j_launch.sh. Run from the main checkout
# AFTER objective-fixes is merged there: the cheap evals load checkpoints with the main checkout's code,
# which must know the new loss names and --fed_norm.
#   bash training/inf_retriever/tools/j_drivers.sh [run ...]      (default: every run in results/phaseC/j_runs.log)
set -u
cd /scratch/hc3337/projects/autoregressive
T=training/inf_retriever/tools
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=($(awk '{print $3}' results/phaseC/j_runs.log | sort -u))
for run in "${RUNS[@]}"; do
  jid=$(awk -v r="$run" '$3 == r {j = $5} END {print j}' results/phaseC/j_runs.log)
  [ -n "$jid" ] || { echo "no job id for $run in j_runs.log" >&2; continue; }
  if pgrep -f "phaseC_run.sh qampari multi 1e-5 $run" >/dev/null; then echo "$run: driver already running"; continue; fi
  steps=$(case $run in *sat50_dense) echo "1000 1250 1500 2500" ;; *) echo "500 750 1000 1250 1500 2000 2500" ;; esac)
  echo "$jid" > results/phaseC/$run.trainjob
  DEV_TAG=cleanv2 KEEP_TOP=3 KEEP_FINAL=1 STEPS_OVERRIDE="$steps" \
    setsid nohup bash $T/phaseC_run.sh qampari multi 1e-5 $run > results/phaseC/$run.driver.out 2>&1 < /dev/null &
  echo "$run: driver started for training job $jid (steps: $steps)"
done
