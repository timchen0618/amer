#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=00:40:00
#SBATCH --mem=64GB
#SBATCH --account=torch_pr_152_echoi
#SBATCH --constraint=h200|b200
#SBATCH --partition=h200_courant,h200_public,h200,b200
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --gres=gpu:1
#SBATCH --job-name=cheap_eval
#SBATCH --output=sbatch_outputs/cheap_eval_%x.out

# Scheduling (2026-10-08): 86 cheap evals on h200_courant took 19-23 min (median 19.4) but asked for 2 h
# and waited a median 2.5 h (p90 6 h) to start. Now 40 min, and H200 or B200 on the group, public and
# preemptible (h200, b200) partitions; preemptible jobs are protected for their first 30 min, and a
# preempted job is requeued. On B200 the embedding step uses the nli environment (the div environment's
# PyTorch has no sm_100 kernels), as gen_embed_ckpt.sbatch does. A start-up failure with "CUDA-capable
# device(s) is/are busy or unavailable" (2% of B200 tasks, on otherwise healthy nodes) requeues the job
# up to twice instead of failing it.
# Phase A cheap eval: the same embed -> retrieve -> eval pipeline as the full-corpus
# evaluation, but over a reduced corpus (data/phaseA/corpora/<SET>.tsv: base top-1000
# candidates + golds + 100k random passages). Embeddings (~3 GB) are deleted at the end.
#   sbatch --job-name=<tag> --export=ALL,CKPT=<checkpoint dir or HF id>,SET=<qampari_test|ambigqa_test|qampari_dev500|ambigqa_dev300>,DATA=<eval jsonl>,OUT_DIR=<dir>,K=<k or empty> \
#       training/inf_retriever/tools/cheap_eval_ckpt.sbatch
# A HF model id (e.g. infly/inf-retriever-v1-1.5b) is evaluated with retrieval_base.py, as
# the existing base-model results were.
: "${CKPT:?}" "${SET:?}" "${DATA:?}" "${OUT_DIR:?}"
cd /scratch/hc3337/projects/autoregressive
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
CORPUS=data/phaseA/corpora/${SET}.tsv
EMB=wikipedia_embeddings/phaseA/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
DS=${SET%%_*}                      # qampari | ambigqa
GOLD_ARG=""; [ "$DS" = "ambigqa" ] && GOLD_ARG="--no-gold-id"
K_ARG="";    [ -n "${K:-}" ] && K_ARG="--max_new_tokens ${K}"
# Passage truncation (code audit 2026-09-28, R3): 512 tokens = training's --chunk_length. The 24-run
# grid's checkpoints (grid_*) were all evaluated at 1024, the old hard-coded value, and keep it so
# their results stay comparable. PASSAGE_MAXLEN overrides both.
case "$CKPT" in */grid_*|grid_*) PML_DEFAULT=1024 ;; *) PML_DEFAULT=512 ;; esac
PML=${PASSAGE_MAXLEN:-$PML_DEFAULT}
echo "passage max length $PML tokens"
mkdir -p "$EMB" "$OUT_DIR"
trap 'rm -rf "$EMB"' EXIT

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
OVERLAY=/scratch/hc3337/envs/div.ext3; ACTIVATE=""
case "$GPU_NAME" in *B200*) OVERLAY=/scratch/hc3337/envs/nli.ext3; ACTIVATE="conda activate nli;" ;; esac
echo "GPU $GPU_NAME -> embedding environment $OVERLAY"
ELOG=$OUT_DIR/embed.log
singularity exec --nv --overlay $OVERLAY:ro $IMG bash -c "source /ext3/env.sh; $ACTIVATE cd /scratch/hc3337/projects/autoregressive; \
    python gen_embed_new.py --passages $CORPUS --output_dir $EMB --shard_id 0 --num_shards 1 --model_name_or_path $CKPT --per_gpu_batch_size 512 --passage_maxlength $PML" 2>&1 | tee $ELOG
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  if grep -q "busy or unavailable" $ELOG && [ "${SLURM_RESTART_COUNT:-0}" -lt 2 ]; then
    echo "CUDA device busy on $(hostname); requeueing (restart ${SLURM_RESTART_COUNT:-0})"; scontrol requeue $SLURM_JOB_ID; sleep 60
  fi
  exit 1
fi

if [[ "$CKPT" == */* && ! -e "$CKPT" ]]; then RETRIEVER=retrieval_base.py; K_ARG=""; else RETRIEVER=retrieval_inf.py; fi
# Multi-query (K set): also save each embedding's own list, so the checkpoint is ranked by the same
# metric as the full-eval headline, RRF over the K embeddings (2026-10-07; round-robin, in
# eval_metrics.txt, was a weaker proxy: reports/cheap_eval_validity_2026-10-07.md).
PS_ARG=""; [ -n "$K_ARG" ] && PS_ARG="--save_per_step $OUT_DIR/steps.jsonl"
singularity exec --nv --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG bash -c "source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; \
    python $RETRIEVER --model_name_or_path $CKPT --passages $CORPUS --passages_embeddings '$EMB/*' --data $DATA \
        --output_dir $OUT_DIR --projection_size 1536 --per_gpu_batch_size 4 --n_docs 500 --num_shards 1 --use_gpu \
        --output_file results.jsonl $K_ARG $PS_ARG" || exit 1

singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro $IMG bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; \
    python eval.py --data_path $DATA --topk 100 10 $GOLD_ARG --input-file $OUT_DIR/results.jsonl" > $OUT_DIR/eval_metrics.txt 2>&1
grep MRecall $OUT_DIR/eval_metrics.txt
# select_metrics.txt: the number checkpoint selection ranks by (phaseC_run.sh, launchers): RRF at k = K
# for multi-query, the eval_metrics.txt MRecall@100 otherwise. kagg.json has round-robin and RRF for k <= K.
if [ -n "$PS_ARG" ] && [ -s $OUT_DIR/steps.jsonl ]; then
  singularity exec --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG bash -c "source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; \
      python training/inf_retriever/tools/kagg_eval.py --ds $DS --per_step $OUT_DIR/steps.jsonl --data $DATA --ks $(seq -s ' ' 1 $K) --aggs round_robin rrf --out $OUT_DIR/kagg.json" > $OUT_DIR/kagg.log 2>&1 \
    && python3 -c "import json; r=[x for x in json.load(open('$OUT_DIR/kagg.json'))['results'] if x['agg']=='rrf' and x['k']==$K][0]; print('MRecall: %.2f | Recall: %.2f  (RRF, k=$K)' % (r['mrecall@100'], r['recall@100']))" > $OUT_DIR/select_metrics.txt \
    && rm -f $OUT_DIR/steps.jsonl.passages.jsonl
  cat $OUT_DIR/select_metrics.txt 2>/dev/null || echo "WARNING: RRF scoring failed (see $OUT_DIR/kagg.log); selection falls back to round-robin"
else
  grep -m1 'MRecall' $OUT_DIR/eval_metrics.txt | sed 's/^/ /' > $OUT_DIR/select_metrics.txt
fi
