#!/bin/bash
# Session J follower: when a run's cheap-eval driver (phaseC_run.sh, started by tools/j_drivers.sh)
# logs "RUN DONE", queue its top 2 steps by cheap dev (select_metrics.txt, RRF at k = 5) plus the
# final step 2500 for full dev + test evaluation in results/phaseC/full_queue_qampari_clean_v3_j.txt
# (workers: full_eval_queue.sh with that QUEUE, WORKER=ja/jb). Exits once every run is queued.
#   setsid nohup bash training/inf_retriever/tools/j_select.sh [run ...] > results/phaseC/j_select.out 2>&1 &
set -u
cd /scratch/hc3337/projects/autoregressive
Q=${QUEUE:-results/phaseC/full_queue_qampari_clean_v3_j.txt}; touch $Q
LOG=results/phaseC/phaseC.log
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=($(awk '{print $3}' results/phaseC/j_runs.log | sort -u))
log() { echo "[$(date '+%F %T')] [j_select] $*" | tee -a "$LOG" >&2; }
# Only log lines from the run's latest launch (j_runs.log) count, so an earlier failed attempt of the same
# run name does not mark a relaunch as failed (2026-10-23, multi_seq_ioproj).
since() { awk -v r="$1" '$3 == r {t = $1 " " $2} END {print t}' results/phaseC/j_runs.log; }
runlog() { awk -v t="[$(since $1)]" 'substr($0, 1, 21) >= t' $LOG 2>/dev/null; }
sel() { grep -m1 -o 'MRecall: [0-9.]*' "$1/select_metrics.txt" 2>/dev/null | cut -d' ' -f2; }
declare -A DONE=([_]=x)   # never empty: "${#DONE[@]}" on an empty array trips set -u in this bash
while [ $(( ${#DONE[@]} - 1 )) -lt ${#RUNS[@]} ]; do
  for run in "${RUNS[@]}"; do
    [ -n "${DONE[$run]:-}" ] && continue
    if runlog $run | grep -q "\[$run\] RUN FAILED" && ! runlog $run | grep -q "\[$run\] RUN DONE"; then
      log "$run: driver failed; not queueing (restart it with j_drivers.sh $run)"; DONE[$run]=failed; continue
    fi
    runlog $run | grep -q "\[$run\] RUN DONE" || continue
    top=$(for d in results/phaseC/$run/step*; do m=$(sel $d); [ -n "$m" ] && echo "${d##*step} $m"; done | sort -k2,2nr | head -2 | cut -d' ' -f1)
    for s in $top 2500; do
      tag=${run}_s$s; ck=checkpoints/qampari/$run/checkpoint/step-$s
      grep -q "^$tag " $Q && continue
      [ -d "$ck" ] || { log "$run: step $s checkpoint missing, not queued"; continue; }
      echo "$tag $ck 5" >> $Q
    done
    log "$run: queued full evals for steps $(echo $top) 2500"
    DONE[$run]=1
  done
  sleep 300
done
log "all runs handled"
