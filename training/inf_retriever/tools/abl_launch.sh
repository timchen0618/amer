#!/bin/bash
# Ablation round after the 24-run grid (2026-10-05): one change at a time against new baselines,
# all on clean v3 data with hard negatives (per-slot draws, ratio 0.5), dev = the v2 dev sets.
# LR per dataset = the grid's best for both modes on hard negatives (QAMPARI 1e-5, AmbigQA 1e-6).
#
#   single    single-query baseline
#   multi     multi-query baseline: hungarian loss, scheduled sampling
#   masked    --loss_fn hungarian_masked (other golds of the example not used as negatives)
#   fullsamp  --full_sampling (no teacher forcing: always feed back the model's own output)
#   hps       --loss_fn hungarian_plus_single --single_loss_weight 0.5
#
# Driver, selection and evaluation as in the grid (phaseC_run.sh): QAMPARI saves every 250 steps,
# keeps the top 3 by cheap dev, queues the top 2; AmbigQA queues steps 150 and 400 and deletes the
# rest after their cheap evals. Full evals go to results/phaseC/full_queue_<ds>_abl.txt; run
#   QUEUE=results/phaseC/full_queue_<ds>_abl.txt DEV_TAG=cleanv2 PER_STEP=1 WORKER=a \
#       bash training/inf_retriever/tools/full_eval_queue.sh <ds>
# so multi-query checkpoints are also scored for every k with round-robin and RRF (k=1 = the
# first embedding alone, the single-query reading of the hps model).
# Usage (repo root, run detached): bash training/inf_retriever/tools/abl_launch.sh [filter]
set -u
cd /scratch/hc3337/projects/autoregressive
T=training/inf_retriever/tools
LOG=results/phaseC/phaseC.log
FILTER=${1:-.}
export DATA_ROOT=data/training/clean_v3 DEV_TAG=cleanv2
log() { echo "[$(date '+%F %T')] [abl] $*" | tee -a "$LOG" >&2; }

runs=()
for ds in qampari ambigqa; do
  lr=$( [ $ds = qampari ] && echo 1e-5 || echo 1e-6 )
  runs+=("$ds single $lr abl_${ds}_single|")
  runs+=("$ds multi $lr abl_${ds}_multi|")
  runs+=("$ds multi $lr abl_${ds}_masked|--loss_fn hungarian_masked")
  runs+=("$ds multi $lr abl_${ds}_fullsamp|--full_sampling")
  runs+=("$ds multi $lr abl_${ds}_hps|--loss_fn hungarian_plus_single --single_loss_weight 0.5")
done

for r in "${runs[@]}"; do
  spec=${r%%|*}; variant=${r#*|}
  set -- $spec; ds=$1; mode=$2; lr=$3; run=$4
  echo "$run" | grep -q -- "$FILTER" || continue
  [ -e results/phaseC/$run.trainjob ] && { log "$run already submitted; skipping"; continue; }
  extra="--train_data ${DATA_ROOT}_hn/$ds/train_data.jsonl --negative_hard_ratio 0.5 $variant"
  if [ "$ds" = qampari ]; then dense=1; keep=3; ksteps=""; else dense=0; keep=0; ksteps="150 400"; fi
  DENSE=$dense KEEP_TOP=$keep KEEP_STEPS="$ksteps" setsid nohup bash $T/phaseC_run.sh $ds $mode $lr $run "$extra" \
      > results/phaseC/$run.driver.out 2>&1 < /dev/null &
  log "launched $run (ds=$ds mode=$mode lr=$lr extra='$extra' data_root=$DATA_ROOT dev_tag=$DEV_TAG)"
  sleep 2
done

# Follower: queue each finished run's full-eval candidates (same rules as grid_launch.sh).
(
  while true; do
    pending=0
    for r in "${runs[@]}"; do
      spec=${r%%|*}; set -- $spec; ds=$1; mode=$2; run=$4
      echo "$run" | grep -q -- "$FILTER" || continue
      marker=results/phaseC/$run.queued
      [ -e "$marker" ] && continue
      last=$(grep -E "\[$run\] RUN (DONE|FAILED)" "$LOG" | tail -1)
      if ! echo "$last" | grep -q "RUN DONE"; then
        pending=1
        if echo "$last" | grep -q "RUN FAILED" && [ ! -e results/phaseC/$run.failed_noted ]; then
          touch results/phaseC/$run.failed_noted
          log "WARNING: $run driver failed; its full evals are not queued until it is restarted and finishes"
        fi
        continue
      fi
      rm -f results/phaseC/$run.failed_noted
      k=-; [ "$mode" = multi ] && { [ "$ds" = qampari ] && k=5 || k=2; }
      if [ "$ds" = qampari ]; then
        steps=$(for d in results/phaseC/$run/step*; do
                  m=$(grep -m1 -o 'MRecall: [0-9.]*' $d/eval_metrics.txt 2>/dev/null | cut -d' ' -f2)
                  [ -n "$m" ] && echo "${d##*step} $m"; done | sort -k2,2nr -k1,1n | head -2 | cut -d' ' -f1)
      else
        steps="150 400"
      fi
      for s in $steps; do
        echo "${run}_s$s checkpoints/$ds/$run/checkpoint/step-$s $k" >> results/phaseC/full_queue_${ds}_abl.txt
      done
      if [ -z "$steps" ]; then log "WARNING: $run has no cheap-dev metrics; nothing queued for full eval"
      else log "queued full evals for $run: steps $(echo $steps)"; fi
      touch "$marker"
    done
    [ $pending = 0 ] && { log "all ablation runs done and queued"; break; }
    sleep 300
  done
) > results/phaseC/abl_follower.out 2>&1 < /dev/null &
disown
log "follower started"
