#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=96GB
#SBATCH --account=torch_pr_152_echoi
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --constraint=h200|l40s|b200
#SBATCH --gres=gpu:1
#SBATCH --job-name=nemo_mine_hn
#SBATCH --output=sbatch_outputs/nemo_mine_hn.out

# Session N (2026-11-13): the S0 training data with hard negatives mined by base Nemotron instead of inf-retriever.
# Same procedure as clean_base_retrieval.sbatch + phaseC_mine_hard_negatives.py (top-100 base retrieval, skip top 3,
# drop golds / gold-title / answer-entity-title / answer-containing passages, keep up to 30), only the retriever changes:
#   1. base Nemotron query embeddings of the S0 training questions (nemotron.ext3),
#   2. faiss top-100 over the Nemotron corpus embedding (nli, retrieval_inf.py --load_query_embeddings),
#   3. the miner (unchanged) -> data/training/clean_v3_hn_s0_nemo/qampari/{train_data,dev_data}.jsonl
#      (dev_data = the in-training eval set, copied unchanged, as the miner always does).
#   sbatch training/inf_retriever/tools/nemo_mine_hn.sbatch      (from the main checkout)
set -e
CODE_ROOT=${CODE_ROOT:-/scratch/hc3337/projects/autoregressive/.claude/worktrees/n-nemotron}
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
SRC=data/training/clean_v3_hn_s0/qampari
DST=data/training/clean_v3_hn_s0_nemo/qampari
M=results/phaseC/mining
QE=$M/qampari_s0train_nemo_qemb.npy
TOP=$M/qampari_s0train_nemo_top100.jsonl
cd /scratch/hc3337/projects/autoregressive
mkdir -p $M
[ -s $QE ] || singularity exec --nv --overlay /scratch/hc3337/envs/nemotron.ext3:ro $IMG /bin/bash -c "source /ext3/env.sh; \
  export HF_HUB_OFFLINE=1; cd /scratch/hc3337/projects/autoregressive; \
  python $CODE_ROOT/training/inf_retriever/tools/nemo_embed_queries.py --data $SRC/train_data.jsonl --out $QE"
[ -s $TOP ] || singularity exec --nv --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG /bin/bash -c "source /ext3/env.sh; \
  conda activate nli; cd /scratch/hc3337/projects/autoregressive; \
  python $CODE_ROOT/retrieval_inf.py --model_name_or_path nemotron_base --load_query_embeddings $QE \
    --passages /scratch/hc3337/wikipedia_chunks/chunks_v5.tsv \
    --passages_embeddings 'wikipedia_embeddings/qampari/nemotron_frozen_base_512/*' --data $SRC/train_data.jsonl \
    --output_dir $M --output_file qampari_s0train_nemo_top100.jsonl --projection_size 2048 --n_docs 100 \
    --num_shards 16 --use_gpu"
singularity exec --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG /bin/bash -c "source /ext3/env.sh; conda activate nli; \
  cd /scratch/hc3337/projects/autoregressive; \
  python training/inf_retriever/tools/phaseC_mine_hard_negatives.py --ds qampari --src $SRC --dst $DST --mined $TOP"
echo ALL_DONE
