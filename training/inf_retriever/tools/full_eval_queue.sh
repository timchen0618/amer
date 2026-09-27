#!/bin/bash
# Full-corpus evaluation queue for one dataset (step 5 / phase C). Processes candidates
# listed in results/phaseC/full_queue_<ds>.txt ("<tag> <checkpoint dir> <k or ->" per line)
# one at a time, taking the first one whose checkpoint is fully written: embed the full corpus once, retrieve for BOTH the dev set (selection) and
# the test set (reporting), score both, then delete the embeddings. One queue per dataset
# keeps at most two embedding sets on disk. New lines can be appended while it runs; a
# line "END" stops it once everything before it is done.
# Usage (repo root, run detached): bash training/inf_retriever/tools/full_eval_queue.sh <qampari|ambigqa>
set -u
cd /scratch/hc3337/projects/autoregressive
DS=$1; T=training/inf_retriever/tools
Q=results/phaseC/full_queue_$DS.txt; touch $Q
LOG=results/phaseC/phaseC.log
DEV=$( [ $DS = qampari ] && echo data/phaseA/qampari_dev500.jsonl || echo data/phaseA/ambigqa_dev300.jsonl )
GOLD=""; [ $DS = ambigqa ] && GOLD="--no-gold-id"
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
log() { echo "[$(date '+%F %T')] [fullq_$DS] $*" | tee -a "$LOG" >&2; }
active() { squeue -h -j "$1" 2>/dev/null | grep -q .; }
ok_all() { [ "$(sacct -j "$1" -X -n -o State 2>/dev/null | tr -d ' ' | grep -vc '^COMPLETED$')" = 0 ]; }
evalf() {  # $1 data $2 results jsonl $3 metrics out
  srun --account=torch_pr_152_courant --time=00:30:00 --mem=32G --cpus-per-task=4 singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro $IMG \
    bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python eval.py --data_path $1 --topk 100 10 $GOLD --input-file $2" > $3 2>&1
}
while true; do
  line=$(while read -r tag ck k; do [ -z "$tag" ] && continue; [ "$tag" = END ] && { echo END; break; }
         grep -q MRecall results/phaseC/full/$DS/$tag/test/eval_metrics.txt 2>/dev/null && continue
         f=$ck/checkpoint.pth   # ready = fully written: exists and unchanged for 2+ minutes
         [ -f "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 120 ] && { echo "$tag $ck $k"; break; }; done < $Q)
  if [ -z "$line" ]; then sleep 300; continue; fi
  if [ "$line" = END ]; then log "queue END reached"; exit 0; fi
  read -r tag ck k <<< "$line"; [ "$k" = "-" ] && k=""
  base=results/phaseC/full/$DS/$tag; emb=wikipedia_embeddings/$DS/full_$tag
  ej=$(sbatch --parsable --job-name=full_$tag --export=ALL,CKPT=$ck,EMB_DIR=$emb $T/gen_embed_ckpt.sbatch)
  rd=$(sbatch --parsable --dependency=afterok:$ej --job-name=full_${tag}_dev --export=ALL,CKPT=$ck,EMB_DIR=$emb,DATA_NAME=$DS,DATA=$DEV,OUT_DIR=$base/dev,K=$k $T/retrieve_ckpt.sbatch)
  rt=$(sbatch --parsable --dependency=afterok:$ej --job-name=full_${tag}_test --export=ALL,CKPT=$ck,EMB_DIR=$emb,DATA_NAME=$DS,OUT_DIR=$base/test,K=$k $T/retrieve_ckpt.sbatch)
  log "$tag: full-corpus eval submitted (embed $ej -> dev $rd, test $rt)"
  while active $ej; do sleep 120; done
  ok_all $ej || { log "ERROR: $tag embedding failed; embeddings kept at $emb"; exit 1; }
  while active $rd || active $rt; do sleep 60; done
  if [ -s $base/dev/$DS.jsonl ] && [ -s $base/test/$DS.jsonl ]; then rm -rf $emb; else log "ERROR: $tag retrieval missing output; embeddings kept at $emb"; exit 1; fi
  evalf $DEV $base/dev/$DS.jsonl $base/dev/eval_metrics.txt
  evalf data/amer_data/eval_data/$DS.jsonl $base/test/$DS.jsonl $base/test/eval_metrics.txt
  log "RESULT full $tag: DEV $(grep -m1 MRecall $base/dev/eval_metrics.txt | cut -d'|' -f1-2) | TEST $(grep -m1 MRecall $base/test/eval_metrics.txt | cut -d'|' -f1-2)"
done
