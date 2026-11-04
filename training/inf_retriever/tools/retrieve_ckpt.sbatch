#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=64GB
#SBATCH --account=torch_pr_152_echoi
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=retrieve_ckpt
#SBATCH --output=sbatch_outputs/retrieve_%x.out

# 2026-10-08: b200 back (dropped 2026-10-07). 345 of 352 B200 tasks since 2026-09-29 succeeded; the 7 failures were
# "CUDA-capable device(s) is/are busy or unavailable" at start-up, on nodes (gb006, gb011, gb012) that also ran 18-40
# tasks fine. Such a start-up failure now requeues the task (at most twice) instead of failing it.
# GPU: H200, L40S or B200 (2026-09-29: widens the pool under low priority). Only the b200 and h200 partitions
# in the list above are preemptible; a preempted job is requeued and reruns (retrieval takes <30 min). The nli environment works on B200.
# Retrieve with one checkpoint over its own corpus embeddings, for the test queries of
# DATA_NAME (default) or for DATA=<any eval jsonl> (e.g. the phase A dev sets).
# K = number of query embeddings for multi-query checkpoints
# (QAMPARI 5, AmbigQA 2); leave K unset for single-query checkpoints.
#   sbatch --job-name=<tag> --export=ALL,CKPT=...,EMB_DIR=...,DATA_NAME=qampari,OUT_DIR=...,K=5 \
#       training/inf_retriever/tools/retrieve_ckpt.sbatch
: "${CKPT:?set CKPT}" "${EMB_DIR:?set EMB_DIR}" "${DATA_NAME:?set DATA_NAME}" "${OUT_DIR:?set OUT_DIR}"
K_ARG=""
DATA=${DATA:-data/amer_data/eval_data/${DATA_NAME}.jsonl}
[ -n "${K:-}" ] && K_ARG="--max_new_tokens ${K}"
# Optional: SAVE_PER_STEP=<jsonl> also saves each query embedding's own top list (multi-query only),
# for offline scoring of every k and aggregation with tools/kagg_eval.py.
PS_ARG=""; [ -n "${SAVE_PER_STEP:-}" ] && [ -n "${K:-}" ] && PS_ARG="--save_per_step ${SAVE_PER_STEP}"
# Optional: SAVE_QEMB=<npy> also saves the query embeddings (for tools/geometry_eval.py).
QE_ARG=""; [ -n "${SAVE_QEMB:-}" ] && QE_ARG="--save_query_embeddings ${SAVE_QEMB}"

# Optional: RET_ENV=div runs in the training environment (div.ext3, has peft) for LoRA checkpoints; default nli.
RET_ENV=${RET_ENV:-nli}
ACT="conda activate nli;"; [ "$RET_ENV" = nli ] || ACT=""

ELOG=sbatch_outputs/retrieve_${SLURM_JOB_NAME}.busycheck.log
singularity exec --nv \
            --overlay /scratch/hc3337/envs/${RET_ENV}.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; $ACT cd /scratch/hc3337/projects/autoregressive; python retrieval_inf.py \
                --model_name_or_path $CKPT \
                --passages /scratch/hc3337/wikipedia_chunks/chunks_v5.tsv \
                --passages_embeddings '$EMB_DIR/*' \
                --data $DATA \
                --output_dir $OUT_DIR \
                --projection_size 1536 \
                --per_gpu_batch_size 4 \
                --n_docs 500 \
                --num_shards 16 \
                --use_gpu \
                --output_file ${DATA_NAME}.jsonl \
                $K_ARG $PS_ARG $QE_ARG" 2>&1 | tee $ELOG
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ] && grep -q "busy or unavailable" $ELOG && [ "${SLURM_RESTART_COUNT:-0}" -lt 2 ]; then
  echo "CUDA device busy on $(hostname); requeueing (restart ${SLURM_RESTART_COUNT:-0})"; rm -f $ELOG
  scontrol requeue $SLURM_JOB_ID; sleep 60
fi
rm -f $ELOG
exit $rc
