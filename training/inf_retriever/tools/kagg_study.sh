#!/bin/bash
# C2 / aggregation study (code audit 2026-09-28): for one grid checkpoint, embed the full corpus,
# retrieve once with KMAX query embeddings (saving each embedding's own top-500 list), and score
# every k <= KMAX with round-robin and RRF aggregation, on the clean dev set and the test set.
# k values: QAMPARI 1 3 5 8 (trained gold counts 1-8, default 5); AmbigQA 1 2 3 5 (trained 2-5,
# default 2). Retrieval generates max(k) embeddings once.
# Usage (repo root, run detached): bash training/inf_retriever/tools/kagg_study.sh <ds> <full-eval tag> [embed job id to reuse]
#   e.g. bash training/inf_retriever/tools/kagg_study.sh qampari grid_qampari_multi_hn_lr1e-5_s1750
set -u
cd /scratch/hc3337/projects/autoregressive
T=training/inf_retriever/tools
R=results/phaseC
OUT_ROOT=results/kagg
LOG=$OUT_ROOT/kagg_study.log
mkdir -p $OUT_ROOT
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
active() { squeue -h -j "$1" 2>/dev/null | grep -q . && return 0
  local st; st=$(sacct -j "$1" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
  [ -z "$st" ] || echo "$st" | grep -qE '^(PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED)'; }
ok() { [ "$(sacct -j "$1" -X -n -o State 2>/dev/null | tr -d ' ' | grep -vc '^COMPLETED$')" = 0 ]; }

declare -A JOBS
DSET=$1; TAG=$2; EJ_REUSE=${3:-}
for ds in $DSET; do
  if [ $ds = qampari ]; then KMAX=8; KS="1 3 5 8"; else KMAX=5; KS="1 2 3 5"; fi
  DEV=data/phaseA/${ds}_cleandev500.jsonl
  tag=$TAG
  ck=$(awk -v t=$tag '$1 == t {print $2}' $R/full_queue_$ds.txt)
  [ -d "$ck" ] || { log "ERROR: no checkpoint for $tag in $R/full_queue_$ds.txt"; exit 1; }
  out=$OUT_ROOT/$ds/$tag; emb=wikipedia_embeddings/$ds/kagg_$tag
  mkdir -p $out
  log "$ds: $tag ($ck); full-eval dev: $(grep -m1 -o 'MRecall: [0-9.]* | Recall: [0-9.]*' $R/full/$ds/$tag/dev/eval_metrics.txt)"
  if [ -n "$EJ_REUSE" ]; then ej=$EJ_REUSE; log "$ds: reusing embedding job $ej"
  else ej=$(sbatch --parsable --job-name=kagg_$tag --export=ALL,CKPT=$ck,EMB_DIR=$emb $T/gen_embed_ckpt.sbatch); fi
  rj=$(sbatch --parsable --dependency=afterok:$ej --job-name=kagg_ret_$tag --output=sbatch_outputs/kagg_ret_$tag.out \
       --export=ALL,CKPT=$ck,EMB_DIR=$emb,DS=$ds,DEV=$DEV,KMAX=$KMAX,OUT_DIR=$out $T/kagg_retrieve.sbatch)
  vj=$(sbatch --parsable --dependency=afterok:$rj --job-name=kagg_eval_$tag --output=sbatch_outputs/kagg_eval_$tag.out \
       --export=ALL,DS=$ds,DEV=$DEV,KS="$KS",OUT_DIR=$out $T/kagg_eval.sbatch)
  log "$ds: submitted embed $ej -> retrieve $rj -> score $vj"
  JOBS[$ds]="$ej $rj $vj $emb"
done

fail=0
for ds in $DSET; do
  read -r ej rj vj emb <<< "${JOBS[$ds]}"
  while active $ej; do sleep 300; done
  ok $ej || { log "ERROR: $ds embedding $ej failed; embeddings kept at $emb"; scancel $rj $vj 2>/dev/null; fail=1; continue; }
  while active $rj; do sleep 120; done
  if ok $rj; then rm -rf $emb && log "$ds: retrieval done, deleted $emb"
  else log "ERROR: $ds retrieval $rj failed; embeddings kept at $emb"; fail=1; continue; fi
  while active $vj; do sleep 60; done
  ok $vj && log "$ds: scoring done" || { log "ERROR: $ds scoring $vj failed"; fail=1; }
done
log "STUDY DONE $DSET $TAG (fail=$fail)"
exit $fail
