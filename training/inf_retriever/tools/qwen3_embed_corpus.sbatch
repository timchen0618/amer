#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=96GB
#SBATCH --account=torch_pr_152_echoi
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --constraint=h200|l40s   # not b200: div.ext3 torch 2.5.1+cu121 has no sm_100 kernels
#SBATCH --gres=gpu:1
#SBATCH --job-name=qwen3_embed_corpus
#SBATCH --output=sbatch_outputs/qwen3_embed_corpus_%a.out

# Session N (2026-11-15): base Qwen3-Embedding-0.6B corpus embedding, one shard per array task
# (tools/qwen3_embed.py corpus; skips shards already written, so a requeued / re-submitted task is safe).
#   sbatch --array=0-31%8 training/inf_retriever/tools/qwen3_embed_corpus.sbatch     (from the main checkout)
CODE_ROOT=${CODE_ROOT:-/scratch/hc3337/projects/autoregressive/.claude/worktrees/n-nemotron}
OUT=${OUT:-wikipedia_embeddings/qampari/qwen3emb06_frozen_base_512}
singularity exec --nv --overlay /scratch/hc3337/envs/div.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; export HF_HUB_OFFLINE=1; cd /scratch/hc3337/projects/autoregressive; \
      python $CODE_ROOT/training/inf_retriever/tools/qwen3_embed.py corpus --output_dir $OUT \
        --shard_id $SLURM_ARRAY_TASK_ID --num_shards 32 --maxlen 512 ${EXTRA:-}"
