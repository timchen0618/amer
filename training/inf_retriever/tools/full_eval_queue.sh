#!/bin/bash
# Full-corpus evaluation queue for one dataset (step 5 / phase C). Processes candidates
# listed in results/phaseC/full_queue_<ds>.txt ("<tag> <checkpoint dir> <k or ->" per line),
# taking the first unclaimed one whose checkpoint is fully written: embed the full corpus once,
# retrieve for BOTH the dev set (selection) and the test set (reporting), score both, then delete
# the embeddings. New lines can be appended while it runs; a line "END" stops it once everything
# before it is done.
#
# Several workers may run on one queue (2026-09-30: two per dataset). Each claims a candidate by
# creating results/phaseC/full/<ds>/.claims/<tag>/ (mkdir is atomic) and records its job ids there,
# so no two workers submit the same checkpoint. Each worker keeps one embedding set (~75 GB) on
# disk at a time.
#
# Usage (repo root, run detached):
#   WORKER=a bash training/inf_retriever/tools/full_eval_queue.sh <qampari|ambigqa>
#   QUEUE=results/phaseC/full_queue_qampari_v2.txt DEV_TAG=cleanv2 WORKER=a bash .../full_eval_queue.sh qampari
#   RESUME="<tag> <embed job> <dev job> <test job>" WORKER=a bash .../full_eval_queue.sh <ds>
#     (adopt an eval that is already submitted: wait for its jobs, score it, then continue)
set -u
cd /scratch/hc3337/projects/autoregressive
DS=$1; T=training/inf_retriever/tools
WORKER=${WORKER:-a}; RESUME=${RESUME:-}
# QUEUE: the queue file, default results/phaseC/full_queue_<ds>.txt (the 24-run grid); other run
# families get their own (grid_launch.sh PREFIX=...), started with the matching DEV_TAG.
Q=${QUEUE:-results/phaseC/full_queue_$DS.txt}; touch $Q
CLAIMS=results/phaseC/full/$DS/.claims; mkdir -p $CLAIMS
LOG=results/phaseC/phaseC.log
# Clean dev sets (500 each; data_creation/build_clean_splits.py). DEV_TAG="" = phase A dev sets.
DEV_TAG=${DEV_TAG-clean}
if [ -n "$DEV_TAG" ]; then DEV=data/phaseA/${DS}_${DEV_TAG}dev500.jsonl
else DEV=$( [ $DS = qampari ] && echo data/phaseA/qampari_dev500.jsonl || echo data/phaseA/ambigqa_dev300.jsonl ); fi
GOLD=""; [ $DS = ambigqa ] && GOLD="--no-gold-id"
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
log() { echo "[$(date '+%F %T')] [fullq_$DS] $*" | tee -a "$LOG" >&2; }
# squeue can come back empty while the controller is busy, so confirm with sacct before calling a
# job finished; an unknown state counts as active (worst case: one more poll).
JOB_ACTIVE_STATES='^(PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED)'
active() {
  squeue -h -j "$1" 2>/dev/null | grep -q . && return 0
  local st; st=$(sacct -j "$1" -X -n -o State 2>/dev/null | tr -d ' ')
  [ -z "$st" ] && return 0
  echo "$st" | grep -qE "$JOB_ACTIVE_STATES"
}
ok_all() { [ "$(sacct -j "$1" -X -n -o State 2>/dev/null | tr -d ' ' | grep -vc '^COMPLETED$')" = 0 ]; }
evalf() {  # $1 data $2 results jsonl $3 metrics out
  # Scoring only needs the retrieval output, which stays on disk, so a failed srun is retried here
  # instead of redoing the 2-3 h embedding (code audit 2026-09-28, G11).
  local try
  for try in 1 2 3; do
    srun --account=torch_pr_152_courant --time=00:30:00 --mem=32G --cpus-per-task=4 singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro $IMG \
      bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python eval.py --data_path $1 --topk 100 10 $GOLD --input-file $2" > $3 2>&1
    grep -q MRecall $3 && return 0
    log "WARNING: scoring $2 failed (attempt $try of 3)"; sleep 60
  done
  return 1
}
# Wait for one submitted eval, score it, delete its embeddings. Returns 1 on failure (logged).
finish() {  # $1 tag $2 embed job $3 dev job $4 test job
  local tag=$1 ej=$2 rd=$3 rt=$4
  local base=results/phaseC/full/$DS/$tag emb=wikipedia_embeddings/$DS/full_$tag
  while active $ej; do sleep 120; done
  ok_all $ej || { log "ERROR: $tag embedding failed; embeddings kept at $emb"; return 1; }
  while active $rd || active $rt; do sleep 60; done
  if [ -s $base/dev/$DS.jsonl ] && [ -s $base/test/$DS.jsonl ]; then rm -rf $emb
  else log "ERROR: $tag retrieval missing output; embeddings kept at $emb"; return 1; fi
  evalf $DEV $base/dev/$DS.jsonl $base/dev/eval_metrics.txt || { log "ERROR: $tag dev scoring failed"; return 1; }
  evalf data/amer_data/eval_data/$DS.jsonl $base/test/$DS.jsonl $base/test/eval_metrics.txt || { log "ERROR: $tag test scoring failed"; return 1; }
  log "RESULT full $tag: DEV $(grep -m1 MRecall $base/dev/eval_metrics.txt | cut -d'|' -f1-2) | TEST $(grep -m1 MRecall $base/test/eval_metrics.txt | cut -d'|' -f1-2)"
}

if [ -n "$RESUME" ]; then
  read -r rtag rej rrd rrt <<< "$RESUME"
  mkdir -p $CLAIMS/$rtag; echo "$rej $rrd $rrt worker=$WORKER" > $CLAIMS/$rtag/jobs
  log "worker $WORKER: resuming $rtag (embed $rej -> dev $rrd, test $rrt)"
  finish $rtag $rej $rrd $rrt || exit 1
fi

while true; do
  line=""
  while read -r tag ck k; do
    [ -z "$tag" ] && continue
    [ "$tag" = END ] && { line=END; break; }
    grep -q MRecall results/phaseC/full/$DS/$tag/test/eval_metrics.txt 2>/dev/null && continue
    [ -d $CLAIMS/$tag ] && continue
    f=$ck/checkpoint.pth   # ready = fully written: exists and unchanged for 2+ minutes
    [ -f "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 120 ] || continue
    mkdir $CLAIMS/$tag 2>/dev/null || continue   # another worker claimed it first
    line="$tag $ck $k"; break
  done < $Q
  if [ -z "$line" ]; then sleep 300; continue; fi
  if [ "$line" = END ]; then log "worker $WORKER: queue END reached"; exit 0; fi
  read -r tag ck k <<< "$line"; [ "$k" = "-" ] && k=""
  base=results/phaseC/full/$DS/$tag; emb=wikipedia_embeddings/$DS/full_$tag
  # Remove outputs of any earlier attempt, so finish() can only see this attempt's files (a
  # stale retrieval output would otherwise pass its "-s" check, code audit 2026-09-28, R5).
  rm -f $base/dev/$DS.jsonl $base/test/$DS.jsonl $base/dev/eval_metrics.txt $base/test/eval_metrics.txt
  ej=$(sbatch --parsable --job-name=full_$tag --export=ALL,CKPT=$ck,EMB_DIR=$emb $T/gen_embed_ckpt.sbatch)
  rd=$(sbatch --parsable --dependency=afterok:$ej --job-name=full_${tag}_dev --export=ALL,CKPT=$ck,EMB_DIR=$emb,DATA_NAME=$DS,DATA=$DEV,OUT_DIR=$base/dev,K=$k $T/retrieve_ckpt.sbatch)
  rt=$(sbatch --parsable --dependency=afterok:$ej --job-name=full_${tag}_test --export=ALL,CKPT=$ck,EMB_DIR=$emb,DATA_NAME=$DS,OUT_DIR=$base/test,K=$k $T/retrieve_ckpt.sbatch)
  echo "$ej $rd $rt worker=$WORKER" > $CLAIMS/$tag/jobs
  log "$tag: full-corpus eval submitted by worker $WORKER (embed $ej -> dev $rd, test $rt)"
  finish $tag $ej $rd $rt || exit 1
done
