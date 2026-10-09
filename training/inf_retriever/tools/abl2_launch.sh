#!/bin/bash
# Ablation round clean_v3_abl2 (planned 2026-10-07), QAMPARI only, against the clean_v3_abl runs
# (single / multi / masked / fullsamp / hps at LR 1e-5, rate ramp to the end), which are not rerun.
# Same data, dev set and driver as clean_v3_abl: clean v3 + hard negatives (ratio 0.5), cleanv2 dev500,
# 2,500 steps, a checkpoint every 250 steps, one seed.
#
# wave 1                                                       wave 2 (needs the winning hps weight W)
#   sat25          multi, rate reaches 1 at 25% of training      hps_fullsamp   hps (w=W), --full_sampling
#   sat50          multi, rate reaches 1 at 50%                  hps_sat50      hps (w=W), rate reaches 1 at 50%
#   hps_w025       hps, single-query weight 0.25                 optional: single_lr2e-5, multi_lr2e-5
#   hps_w010       hps, single-query weight 0.10
#   single_lr3e-5  single, LR 3e-5
#   multi_lr3e-5   multi, LR 3e-5
#   masked_div05   hungarian_masked + diversity penalty, weight 0.5
#   masked_div2    hungarian_masked + diversity penalty, weight 2.0
#
# Selection: cheap dev is ranked by select_metrics.txt (RRF at k = 5 for multi-query, from
# cheap_eval_ckpt.sbatch). Keep the top 3 checkpoints by cheap dev plus the final one (KEEP_FINAL=1); full-evaluate
# the top 2 by cheap dev plus the final step (step 2500), so 2 or 3 full evals per run. Full evals go
# to results/phaseC/full_queue_qampari_clean_v3_abl2.txt; run two workers with
#   QUEUE=results/phaseC/full_queue_qampari_clean_v3_abl2.txt DEV_TAG=cleanv2 PER_STEP=1 WORKER=a \
#       bash training/inf_retriever/tools/full_eval_queue.sh qampari
# Usage (repo root, run detached): bash training/inf_retriever/tools/abl2_launch.sh wave1|wave2 [W]
# RESUME=1 restarts the drivers (and the follower) of runs already submitted, e.g. after failed cheap evals.
set -u
cd /scratch/hc3337/projects/autoregressive
T=training/inf_retriever/tools
LOG=results/phaseC/phaseC.log
P=clean_v3_abl2
WAVE=${1:?usage: abl2_launch.sh wave1|wave2 [W]}
export DATA_ROOT=data/training/clean_v3 DEV_TAG=cleanv2
log() { echo "[$(date '+%F %T')] [$P] $*" | tee -a "$LOG" >&2; }

HPS="--loss_fn hungarian_plus_single --single_loss_weight"
DIV="--loss_fn hungarian_masked_div --div_weight"
runs=()
if [ "$WAVE" = wave1 ]; then
  runs+=("multi 1e-5 ${P}_qampari_sat25|--sampling_saturate_frac 0.25")
  runs+=("multi 1e-5 ${P}_qampari_sat50|--sampling_saturate_frac 0.5")
  runs+=("multi 1e-5 ${P}_qampari_hps_w025|$HPS 0.25")
  runs+=("multi 1e-5 ${P}_qampari_hps_w010|$HPS 0.1")
  runs+=("single 3e-5 ${P}_qampari_single_lr3e-5|")
  runs+=("multi 3e-5 ${P}_qampari_multi_lr3e-5|")
  runs+=("multi 1e-5 ${P}_qampari_masked_div05|$DIV 0.5")
  runs+=("multi 1e-5 ${P}_qampari_masked_div2|$DIV 2.0")
elif [ "$WAVE" = wave2 ]; then
  W=${2:?wave2 needs the winning hps weight}
  runs+=("multi 1e-5 ${P}_qampari_hps_fullsamp|$HPS $W --full_sampling")
  runs+=("multi 1e-5 ${P}_qampari_hps_sat50|$HPS $W --sampling_saturate_frac 0.5")
else
  echo "unknown wave $WAVE" >&2; exit 1
fi

for r in "${runs[@]}"; do
  spec=${r%%|*}; variant=${r#*|}
  set -- $spec; mode=$1; lr=$2; run=$3
  if [ -e results/phaseC/$run.trainjob ]; then
    # RESUME=1: restart the driver of an already submitted run (it reuses the training job id and
    # only submits the cheap evals that have no metrics yet, then prunes); otherwise skip it.
    [ "${RESUME:-0}" = 1 ] || { log "$run already submitted; skipping"; continue; }
    pgrep -f "phaseC_run.sh .* $run( |$)" >/dev/null && { log "$run driver still running; not resumed"; continue; }
  fi
  extra="--train_data ${DATA_ROOT}_hn/qampari/train_data.jsonl --negative_hard_ratio 0.5 $variant"
  DENSE=1 KEEP_TOP=3 KEEP_FINAL=1 setsid nohup bash $T/phaseC_run.sh qampari $mode $lr $run "$extra" \
      > results/phaseC/$run.driver.out 2>&1 < /dev/null &
  log "launched $run ($WAVE; mode=$mode lr=$lr extra='$extra' data_root=$DATA_ROOT dev_tag=$DEV_TAG)"
  sleep 2
done

# Follower: queue each finished run's top 2 by cheap dev plus its final step. Only RUN DONE / RUN FAILED
# lines logged after the follower starts count, so a resumed driver's old "RUN DONE" (from a run whose cheap
# evals failed) cannot trigger queuing before the resubmitted cheap evals finish.
START_LINE=$(wc -l < "$LOG")
(
  while true; do
    pending=0
    for r in "${runs[@]}"; do
      spec=${r%%|*}; set -- $spec; mode=$1; run=$3
      marker=results/phaseC/$run.queued
      [ -e "$marker" ] && continue
      last=$(tail -n +$((START_LINE+1)) "$LOG" | grep -E "\[$run\] RUN (DONE|FAILED)" | tail -1)
      if ! echo "$last" | grep -q "RUN DONE"; then
        pending=1
        if echo "$last" | grep -q "RUN FAILED" && [ ! -e results/phaseC/$run.failed_noted ]; then
          touch results/phaseC/$run.failed_noted
          log "WARNING: $run driver failed; its full evals are not queued until it is restarted and finishes"
        fi
        continue
      fi
      rm -f results/phaseC/$run.failed_noted
      k=-; [ "$mode" = multi ] && k=5
      top=$(for d in results/phaseC/$run/step*; do
              f=$d/select_metrics.txt; [ -s $f ] || f=$d/eval_metrics.txt   # RRF k=5 for multi-query
              m=$(grep -m1 -o 'MRecall: [0-9.]*' $f 2>/dev/null | cut -d' ' -f2)
              [ -n "$m" ] && echo "${d##*step} $m"; done | sort -k2,2nr -k1,1n | head -2 | cut -d' ' -f1)
      steps=$(echo $top 2500 | tr ' ' '\n' | awk '!seen[$0]++' | xargs)
      for s in $steps; do
        echo "${run}_s$s checkpoints/qampari/$run/checkpoint/step-$s $k" >> results/phaseC/full_queue_qampari_${P}.txt
      done
      log "queued full evals for $run: steps $steps (top 2 by cheap dev: $(echo $top); final: 2500)"
      touch "$marker"
    done
    [ $pending = 0 ] && { log "all $WAVE runs done and queued"; break; }
    sleep 300
  done
) > results/phaseC/${P}_follower_$WAVE.out 2>&1 < /dev/null &
disown
log "$WAVE follower started"
