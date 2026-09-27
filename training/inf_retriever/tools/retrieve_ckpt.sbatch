#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH --job-name=retrieve_ckpt
#SBATCH --output=sbatch_outputs/retrieve_%x.out

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

singularity exec --nv \
            --overlay /scratch/hc3337/envs/nli.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; python retrieval_inf.py \
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
                $K_ARG"
