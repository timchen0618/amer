#!/bin/bash
# Session F (objective audit 2026-10-13, reports/objective_audit_2026-10-13/PLAN.md): one frozen-doc-encoder QAMPARI run,
# trained with phaseC_train.sbatch (DENSE: a checkpoint every 250 steps) and evaluated with the shared frozen corpus
# embedding, so every eval is retrieval only (no corpus re-embedding, no cheap dev):
#   1. full dev on every saved checkpoint as it is saved (multi-query: RRF k = 5 from kagg_eval.py; single: MRecall@100);
#   2. as dev results land, delete every scored checkpoint that is not in the top 2 by full dev (the top 2 can only
#      improve, so a dropped step never comes back) and is not the final step 2500; step-0 is deleted at once;
#   3. after training: full test on the top 2 by full dev + step 2500, with per-step lists, k/aggregation scores,
#      per-embedding scores and query/gold geometry (dev and test), as in full_eval_queue.sh;
#   4. keep only the best-test checkpoint, plus the dev-selected one if different (user decision 2026-10-13).
# Results: results/phaseC/full/qampari/<run>_s<step>/{dev,test}/... (same layout as full_eval_queue.sh, so
# weekly_collect.py reads them). Log: results/phaseC/phaseC.log, prefix [<run>].
# Usage (repo root, detached): setsid nohup bash training/inf_retriever/tools/f_run.sh <multi|single> <lr> <run> "<extra>" &
# Restarting the same command resumes (training job id in results/phaseC/<run>.trainjob; per-step job ids in
# results/phaseC/full/qampari/<run>_s<step>/{dev,test}.job).
set -u
cd /scratch/hc3337/projects/autoregressive
MODE=$1; LR=$2; RUN=$3; EXTRA=${4:-}
DS=qampari; T=training/inf_retriever/tools
LOG=results/phaseC/phaseC.log
EMB=wikipedia_embeddings/qampari/shared_frozen_base_512
# F_DEV / F_DATA_ROOT (optional, 2026-10-23): full-dev file and training data root (its dev_data.jsonl is the in-training
# eval set) for the seeded 90/10 splits (tools/build_seed_split.py), e.g. data/phaseA/qampari_clean_v3_dev_s0.jsonl and
# data/training/clean_v3_hn_s0.
DEV=${F_DEV:-data/phaseA/qampari_cleanv2dev500.jsonl}; TEST=data/amer_data/eval_data/qampari.jsonl
F_DATA_ROOT=${F_DATA_ROOT:-data/training/clean_v3}
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
# F_STEPS (comma-separated save/eval steps) and F_FINAL override the 2500-step schedule for longer runs; the training
# job then needs STEPS_OVERRIDE (= F_FINAL) and usually WARMUP_OVERRIDE / TRAIN_TIME (sbatch --time), all from the env.
STEPS=$(seq -s ' ' 250 250 2500); FINAL=2500
[ -n "${F_STEPS:-}" ] && STEPS=$(echo "$F_STEPS" | tr ',' ' ')
FINAL=${F_FINAL:-$FINAL}
K=""; [ "$MODE" = multi ] && K=5
log() { echo "[$(date '+%F %T')] [$RUN] $*" | tee -a "$LOG" >&2; }
trap 'rc=$?; [ $rc -ne 0 ] && log "RUN FAILED (exit $rc); restart this driver to resume"' EXIT
JOB_ACTIVE_STATES='^(PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED)'
active() {
  squeue -h -j "$1" 2>/dev/null | grep -q . && return 0
  local st; st=$(sacct -j "$1" -X -n -o State 2>/dev/null | tr -d ' ')
  [ -z "$st" ] && return 0
  echo "$st" | grep -qE "$JOB_ACTIVE_STATES"
}
case "$EXTRA" in *--freeze_doc_encoder*) ;; *) echo "f_run.sh is for frozen-doc-encoder runs (--freeze_doc_encoder)" >&2; exit 1 ;; esac
[ -n "$(ls $EMB/passages_* 2>/dev/null)" ] || { log "ERROR: shared embedding $EMB missing"; exit 1; }

state=results/phaseC/$RUN.trainjob
if [ -s "$state" ]; then TJ=$(cat "$state"); else
  # CODE_ROOT (optional): a worktree whose code trains (its phaseC_train.sbatch must support CODE_ROOT, J's fd96d3b+) and
  # retrieves (its retrieve_ckpt.sbatch must run $CODE_ROOT/retrieval_inf.py; needed when a flag changes generate(), e.g.
  # --fed_norm token); data, checkpoints, scoring and results stay on the main checkout.
  TS=$T/phaseC_train.sbatch; [ -n "${CODE_ROOT:-}" ] && TS=$CODE_ROOT/$T/phaseC_train.sbatch
  TJ=$(DS=$DS MODE=$MODE LR=$LR RUN_NAME=$RUN EXTRA="$EXTRA" SAVE_AT_OVERRIDE="$STEPS" DATA_ROOT=$F_DATA_ROOT \
       sbatch --parsable --export=ALL ${TRAIN_TIME:+--time=$TRAIN_TIME} --job-name=phaseC_$RUN $TS) || {
    # an sbatch reply timeout can still create the job (2026-11-10): adopt it instead of leaving an orphan
    sleep 60; TJ=$(squeue -h -u $USER -n phaseC_$RUN -o %i | head -1)
    [ -n "$TJ" ] || { log "ERROR: training submit failed"; exit 1; }; log "sbatch reported failure but job $TJ exists; adopting it"; }
  echo "$TJ" > "$state"; log "submitted training job $TJ (frozen F run; mode=$MODE lr=$LR extra='$EXTRA'${CODE_ROOT:+ code=$CODE_ROOT@$(git -C $CODE_ROOT rev-parse --short HEAD)})"
fi
TLOG=sbatch_outputs/phaseC_$RUN.out
ck() { echo checkpoints/$DS/$RUN/checkpoint/step-$1; }
base() { echo results/phaseC/full/$DS/${RUN}_s$1; }

srun_py() {  # $1 log file, rest: python command (CPU job in the nli env)
  local out=$1; shift
  srun --account=torch_pr_152_echoi --time=01:00:00 --mem=48G --cpus-per-task=4 singularity exec --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG \
    bash -c "source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; $*" > $out 2>&1
}
submit() {  # $1 step $2 dev|test
  local s=$1 set=$2 b data ps=""; b=$(base $s); data=$DEV; [ $set = test ] && data=$TEST
  mkdir -p $b/$set; rm -f $b/$set/$DS.jsonl $b/$set/eval_metrics.txt
  [ -n "$K" ] && ps=$b/${set}_steps.jsonl
  sbatch --parsable --job-name=full_${RUN}_s${s}_$set \
    --export=ALL,CKPT=$(ck $s),EMB_DIR=$EMB,DATA_NAME=$DS,DATA=$data,OUT_DIR=$b/$set,K=$K,SAVE_PER_STEP=$ps,SAVE_QEMB=$b/${set}_qemb.npy \
    ${CODE_ROOT:+$CODE_ROOT/}$T/retrieve_ckpt.sbatch > $b/$set.job
}
# score one finished retrieval; returns 0 when the score is on disk
score() {  # $1 step $2 dev|test
  local s=$1 set=$2 b data try; b=$(base $s); data=$DEV; [ $set = test ] && data=$TEST
  [ -s $b/$set/$DS.jsonl ] || { log "ERROR: s$s $set retrieval has no output (job $(cat $b/$set.job))"; return 1; }
  for try in 1 2 3; do
    srun --account=torch_pr_152_echoi --time=00:30:00 --mem=32G --cpus-per-task=4 singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro $IMG \
      bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python eval.py --data_path $data --topk 100 10 --input-file $b/$set/$DS.jsonl" > $b/$set/eval_metrics.txt 2>&1
    grep -q MRecall $b/$set/eval_metrics.txt && break; sleep 60
  done
  grep -q MRecall $b/$set/eval_metrics.txt || { log "ERROR: s$s $set scoring failed"; return 1; }
  if [ -n "$K" ]; then
    for try in 1 2 3; do
      srun_py $b/${set}_kagg.log python $T/kagg_eval.py --ds $DS --per_step $b/${set}_steps.jsonl --data $data --ks 1 2 3 4 5 --aggs round_robin rrf --out $b/${set}_kagg.json
      python3 -c "import json,sys; sys.exit(len(json.load(open('$b/${set}_kagg.json'))['results']) != 10)" 2>/dev/null && break; sleep 60
    done
    python3 -c "import json,sys; sys.exit(len(json.load(open('$b/${set}_kagg.json'))['results']) != 10)" 2>/dev/null \
      || { log "ERROR: s$s $set k/aggregation scoring failed (see $b/${set}_kagg.log)"; return 1; }
    rm -f $b/${set}_steps.jsonl.passages.jsonl
  fi
  return 0
}
devscore() {  # headline full dev score of a step, or empty
  local b; b=$(base $1)
  if [ -n "$K" ]; then python3 -c "import json; print([r['mrecall@100'] for r in json.load(open('$b/dev_kagg.json'))['results'] if r['agg']=='rrf' and r['k']==5][0])" 2>/dev/null
  else grep -m1 -o 'MRecall: [0-9.]*' $b/dev/eval_metrics.txt 2>/dev/null | cut -d' ' -f2; fi
}
testscore() {
  local b; b=$(base $1)
  if [ -n "$K" ]; then python3 -c "import json; print([r['mrecall@100'] for r in json.load(open('$b/test_kagg.json'))['results'] if r['agg']=='rrf' and r['k']==5][0])" 2>/dev/null
  else grep -m1 -o 'MRecall: [0-9.]*' $b/test/eval_metrics.txt 2>/dev/null | cut -d' ' -f2; fi
}
ranked() { for s in $STEPS; do m=$(devscore $s); [ -n "$m" ] && echo "$s $m"; done | sort -k2,2nr -k1,1n; }
# The legacy rule's pick (2026-10-23): the saved step with the lowest dev Hungarian loss at sampling rate 1.0, from the
# "[eval_loss_sr1] step S: v" lines finetuning_multi_paper.py logs; empty if the run logs none. Kept and test-evaluated
# next to the top 2 by full dev, so the two selection rules can be compared; deleted after test as any other step.
lossbest() {
  grep -oE "\[eval_loss_sr1\] step [0-9]+: [0-9.]+" "$TLOG" 2>/dev/null | awk -v steps=" $STEPS " \
    '{s=$3; sub(":","",s); if (index(steps, " " s " ")) print s, $4}' | sort -k2,2g -k1,1n | head -1 | cut -d' ' -f1
}
prune_dev() {  # step 2: drop scored steps outside the current top 2 (never the final step)
  local s keep lb; keep=$(ranked | head -2 | cut -d' ' -f1 | xargs); lb=$(lossbest)
  for s in $(ranked | tail -n +3 | cut -d' ' -f1); do
    [ "$s" = $FINAL ] && continue
    [ "$s" = "$lb" ] && continue
    [ -d $(ck $s) ] && rm -rf -- $(ck $s) && log "pruned step $s (not in top 2 by full dev; top 2 so far: $keep)"
  done
  [ -d $(ck 0) ] && [ -n "$(ranked)" ] && rm -rf -- $(ck 0) && log "deleted step-0"
  return 0
}

# --- phase 1: full dev on every checkpoint as it is saved ---
declare -A DJOB
pending=" $STEPS "
while [ -n "$(echo $pending)" ]; do
  for s in $pending; do
    b=$(base $s)
    if [ -n "$(devscore $s)" ]; then pending=${pending/ $s / }; continue; fi
    if [ -z "${DJOB[$s]:-}" ]; then
      if [ -s $b/dev.job ]; then DJOB[$s]=$(cat $b/dev.job)   # resumed driver
      elif grep -q "Saving model to $(ck $s)\$" "$TLOG" 2>/dev/null && [ -f $(ck $s)/checkpoint.pth ] \
           && [ $(( $(date +%s) - $(stat -c %Y $(ck $s)/checkpoint.pth) )) -gt 120 ]; then
        submit $s dev; DJOB[$s]=$(cat $b/dev.job); log "step $s saved; full dev retrieval ${DJOB[$s]}"
      fi
      continue
    fi
    active ${DJOB[$s]} && continue
    if score $s dev; then
      log "RESULT dev step $s: $(devscore $s) (full dev, $( [ -n "$K" ] && echo 'RRF k=5' || echo 'MRecall@100'))"
      pending=${pending/ $s / }; prune_dev
    else
      log "ERROR: step $s dev eval failed; resubmitting once"; rm -f $b/dev.job
      [ -e $b/dev.retried ] && { log "ERROR: step $s dev eval failed twice; giving up on it"; pending=${pending/ $s / }; continue; }
      touch $b/dev.retried; unset "DJOB[$s]"
    fi
  done
  if [ -n "$(echo $pending)" ] && ! active $TJ; then
    # training ended: steps never saved will never come
    for s in $pending; do
      [ -z "${DJOB[$s]:-}" ] && ! grep -q "Saving model to $(ck $s)\$" "$TLOG" 2>/dev/null && \
        { log "ERROR: training job $TJ ended ($(sacct -j $TJ -X -n -o State | head -1 | tr -d ' ')) without saving step $s"; exit 1; }
    done
  fi
  [ -n "$(echo $pending)" ] && sleep 120
done
log "TRAINED + DEV DONE: full dev ranking $(ranked | tr '\n' ' ')"

# --- phase 2: full test on top 2 + final ---
LOSSBEST=$(lossbest)
TSTEPS=$(echo $(ranked | head -2 | cut -d' ' -f1) $FINAL $LOSSBEST | tr ' ' '\n' | awk '!seen[$0]++' | xargs)
log "full test on steps $TSTEPS (top 2 by full dev + final${LOSSBEST:+ + lowest dev loss at rate 1.0: $LOSSBEST})"
declare -A TJOB
for s in $TSTEPS; do
  [ -n "$(testscore $s)" ] && continue
  if [ -s $(base $s)/test.job ]; then TJOB[$s]=$(cat $(base $s)/test.job); else submit $s test; TJOB[$s]=$(cat $(base $s)/test.job); fi
done
for s in $TSTEPS; do
  [ -n "$(testscore $s)" ] && continue
  while active ${TJOB[$s]}; do sleep 60; done
  score $s test || exit 1
done
for s in $TSTEPS; do
  b=$(base $s)
  for set in dev test; do
    data=$DEV; [ $set = test ] && data=$TEST
    [ -s $b/${set}_geom.json ] || srun --account=torch_pr_152_echoi --time=02:00:00 --mem=96G --cpus-per-task=16 \
      singularity exec --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG bash -c "source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; python $T/geometry_eval.py --qemb $b/${set}_qemb.npy --emb_glob '$EMB/passages_*' --data $data --out $b/${set}_geom.json" > $b/${set}_geom.log 2>&1 \
      || log "ERROR: s$s $set geometry failed (see $b/${set}_geom.log); continuing"
  done
  [ -n "$K" ] && { srun_py $b/perstep.log python $T/perstep_eval.py $b || log "ERROR: s$s per-embedding scoring failed; continuing"; }
  log "RESULT full ${RUN}_s$s: DEV $(devscore $s) | TEST $(testscore $s) ($( [ -n "$K" ] && echo 'RRF k=5' || echo 'MRecall@100'))"
done

# --- phase 3: keep the best-test checkpoint (+ the dev-selected one) ---
best_test=$(for s in $TSTEPS; do echo "$s $(testscore $s)"; done | sort -k2,2nr -k1,1n | head -1 | cut -d' ' -f1)
dev_sel=$(ranked | head -1 | cut -d' ' -f1)
for s in $TSTEPS; do
  [ "$s" = "$best_test" ] || [ "$s" = "$dev_sel" ] && continue
  [ -d $(ck $s) ] && rm -rf -- $(ck $s) && log "deleted step $s after test (kept: best test $best_test, dev-selected $dev_sel)"
done
log "RUN DONE (dev-selected $dev_sel, best test $best_test${LOSSBEST:+, loss-selected $LOSSBEST})"
