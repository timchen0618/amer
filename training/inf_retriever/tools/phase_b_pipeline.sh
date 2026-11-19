#!/bin/bash
# Step 5 / phase B driver for one dataset: train multi-query then single-query (FSDP, clean
# code), and for every listed checkpoint, in order: full-corpus embed -> retrieve -> eval +
# parameter drift -> delete that checkpoint's embeddings once its results file exists.
# At most one embedding set per dataset exists at a time. Idempotent: rerunning skips
# trainings already submitted/finished and checkpoints that already have eval metrics.
# Stops with an "ERROR" line on any failure.
#
# Usage (from the repo root; lightweight -- only sbatch/squeue/sacct/rm):
#   bash training/inf_retriever/tools/phase_b_pipeline.sh qampari
#   bash training/inf_retriever/tools/phase_b_pipeline.sh ambigqa
# Tracker: https://claude.ai/code/artifact/1c7621cd-7d9d-4137-8ff4-71a8891e17bd
set -u
cd /scratch/hc3337/projects/autoregressive
DS=$1
T=training/inf_retriever
STATE=results/phaseB_state; mkdir -p $STATE results/param_drift
LOG=$STATE/${DS}_pipeline.log
log() { echo "[$(date "+%F %T")] [$DS] $*" | tee -a "$LOG" >&2; }

case $DS in
  qampari) STEPS="250 500 1000 2500"; K_MULTI=5
           MULTI=(finetune_qampari_joint_fsdp.sh qampari_joint_fsdp_mixed qampari_joint_fsdp_mixed_steps2500_t0.05_lr0.00001_ws100_bs50_hungarian)
           SINGLE=(finetune_qampari_single_fsdp.sh qampari_single_fsdp_mixed qampari_single_fsdp_mixed_steps2500_t0.05_lr0.00001_ws100_bs50_contrastive) ;;
  ambigqa) STEPS="30 150 400"; K_MULTI=2
           MULTI=(finetune_ambigqa_joint_fsdp.sh ambigqa_joint_fsdp_mixed ambigqa_joint_fsdp_mixed_steps400_t0.05_lr0.00001_ws15_bs50_hungarian)
           SINGLE=(finetune_ambigqa_single_fsdp.sh ambigqa_single_fsdp_mixed ambigqa_single_fsdp_mixed_steps400_t0.05_lr0.00001_ws15_bs50_contrastive) ;;
  *) echo "dataset must be qampari or ambigqa"; exit 1 ;;
esac

job_state() { sacct -j "$1" -X -n -o State 2>/dev/null | head -1 | tr -d ' '; }
job_active() { squeue -h -j "$1" 2>/dev/null | grep -q .; }
wait_job() {  # wait until a job leaves the queue; return 0 only if every task COMPLETED
  local j=$1
  while job_active "$j"; do sleep 120; done
  sleep 30
  local bad; bad=$(sacct -j "$j" -X -n -o State 2>/dev/null | tr -d ' ' | grep -vc '^COMPLETED$')
  [ "$bad" = "0" ]
}

# --- trainings: multi first, single chained after it ------------------------------------
submit_training() {  # $1 script $2 jobname $3 dependency-jobid(optional)
  local f=$STATE/${DS}_train_$2.jobid
  if [ -s "$f" ]; then cat "$f"; return; fi
  local dep=""; [ -n "${3:-}" ] && dep="--dependency=afterany:$3"
  local j; j=$(sbatch --parsable $dep $T/$1) || { log "ERROR: sbatch $1 failed"; exit 1; }
  echo "$j" > "$f"; log "submitted training $2 as job $j ${dep}"; echo "$j"
}
MJOB=$(submit_training "${MULTI[0]}" "${MULTI[1]}")
SJOB=$(submit_training "${SINGLE[0]}" "${SINGLE[1]}" "$MJOB")
[ -n "$MJOB" ] && [ -n "$SJOB" ] || { log "ERROR: training submission failed (MJOB=$MJOB SJOB=$SJOB)"; exit 1; }

# --- per-checkpoint evaluation --------------------------------------------------------------
process() {  # $1 mode $2 train-jobname $3 run_name $4 train-jobid $5 K $6 step
  local mode=$1 jname=$2 run=$3 tjob=$4 k=$5 step=$6
  local tag=phaseB_${DS}_${mode}_step${step}
  local ckpt=checkpoints/$DS/$run/checkpoint/step-$step
  local emb=wikipedia_embeddings/$DS/$tag
  local out=results/finetuned/$DS/$tag
  local tlog=sbatch_outputs/$jname.out
  if grep -q "MRecall" "$out/eval_metrics.txt" 2>/dev/null; then log "$mode step $step: already done, skipping"; return; fi

  # 1) wait for the checkpoint ("Saving model" is logged after torch.save returns)
  while ! grep -q "Saving model to $ckpt\$" "$tlog" 2>/dev/null; do
    if ! job_active "$tjob"; then
      grep -q "Saving model to $ckpt\$" "$tlog" 2>/dev/null && break
      log "ERROR: training $jname (job $tjob, state $(job_state $tjob)) ended without saving step $step"; exit 1
    fi
    sleep 120
  done
  log "$mode step $step: checkpoint ready ($ckpt)"

  # 2) embed -> retrieve -> eval+drift, chained with afterok
  local ej rj dj
  ej=$(sbatch --parsable --job-name=${tag} --export=ALL,CKPT=$ckpt,EMB_DIR=$emb $T/tools/gen_embed_ckpt.sbatch) || { log "ERROR: embed submit failed"; exit 1; }
  rj=$(sbatch --parsable --dependency=afterok:$ej --job-name=${tag} --export=ALL,CKPT=$ckpt,EMB_DIR=$emb,DATA_NAME=$DS,OUT_DIR=$out,K=$k $T/tools/retrieve_ckpt.sbatch) || { log "ERROR: retrieve submit failed"; exit 1; }
  dj=$(sbatch --parsable --dependency=afterok:$rj --job-name=${tag} --export=ALL,DATA_NAME=$DS,OUT_DIR=$out,RUN_DIR=checkpoints/$DS/$run,STEP=$step,DRIFT_OUT=results/param_drift/phaseB_${DS}.jsonl $T/tools/eval_drift_ckpt.sbatch) || { log "ERROR: eval submit failed"; exit 1; }
  log "$mode step $step: submitted embed $ej -> retrieve $rj -> eval+drift $dj"

  wait_job "$ej" || { log "ERROR: embed $ej failed ($(sacct -j $ej -X -n -o State | sort | uniq -c | tr -s ' ' | tr '\n' ';')); embeddings kept at $emb"; exit 1; }
  log "$mode step $step: embedding done ($(ls $emb | wc -l) shard files)"
  wait_job "$rj" || { log "ERROR: retrieve $rj failed ($(job_state $rj)); embeddings kept at $emb"; exit 1; }

  # 3) delete this checkpoint's embeddings once its results file exists (standing rule)
  if [ -s "$out/$DS.jsonl" ]; then
    rm -rf "$emb" && log "$mode step $step: results file written ($out/$DS.jsonl); deleted embeddings $emb"
  else
    log "ERROR: retrieve finished but $out/$DS.jsonl is missing; embeddings kept at $emb"; exit 1
  fi

  wait_job "$dj" || { log "ERROR: eval+drift $dj failed ($(job_state $dj))"; exit 1; }
  local m; m=$(grep "MRecall" "$out/eval_metrics.txt" | head -2 | tr '\n' ' ' | tr -s ' ')
  local d; d=$(grep "\"step\": $step," results/param_drift/phaseB_${DS}.jsonl | grep "\"run\": \"$run\"" | tail -1 | python3 -c 'import json,sys; r=json.loads(sys.stdin.read()); print("weights changed %.3f%%, rel. distance %.3e" % (100*r["frac_changed"], r["rel_l2"]))' 2>/dev/null)
  log "RESULT $mode step $step: @100/@10 -> $m | drift: $d"
}

for step in $STEPS; do process multi  "${MULTI[1]}"  "${MULTI[2]}"  "$MJOB" "$K_MULTI" "$step"; done
for step in $STEPS; do process single "${SINGLE[1]}" "${SINGLE[2]}" "$SJOB" ""        "$step"; done
log "PHASE B DONE for $DS"
