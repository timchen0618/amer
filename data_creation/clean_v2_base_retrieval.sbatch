#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200|l40s|b200
#SBATCH --partition=h200_courant,l40s_courant,h200_public,l40s_public,b200,h200
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --gres=gpu:1
#SBATCH --job-name=clean_v2_base_retrieval
#SBATCH --output=sbatch_outputs/%x.out

# Base-retriever (infly/inf-retriever-v1-1.5b, untrained) full-corpus retrieval for the v2 QAMPARI
# splits (data_creation/build_clean_v2.py; AmbigQA is unchanged from v1 and reuses its files):
#   - top-1000 for the v2 dev set -> results/phaseA/base/qampari_cleanv2dev500.jsonl
#     (input to phaseA_build_reduced_corpus.py, the cheap-eval corpus)
#   - top-100 for the v2 train set -> results/phaseC/mining/qampari_cleanv2train_base_top100.jsonl
#     (input to phaseC_mine_hard_negatives.py)
# Same settings as training/inf_retriever/tools/clean_base_retrieval.sbatch (v1), on the wider GPU pool
# of retrieve_ckpt.sbatch (same nli environment and memory). A preempted job is requeued and starts over.
#   sbatch data_creation/clean_v2_base_retrieval.sbatch
singularity exec --nv \
    --overlay /scratch/hc3337/envs/nli.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c '
source /ext3/env.sh
conda activate nli
cd /scratch/hc3337/projects/autoregressive
set -e
EMB="/scratch/hc3337/embeddings/inf/qampari_embeddings/*"
run() {
    python retrieval_base.py --model_name_or_path infly/inf-retriever-v1-1.5b \
        --passages /scratch/hc3337/wikipedia_chunks/chunks_v5.tsv --passages_embeddings "$EMB" \
        --projection_size 1536 --per_gpu_batch_size 16 --num_shards 16 --use_gpu "$@"
}
mkdir -p results/phaseA/base results/phaseC/mining
run --data data/phaseA/qampari_cleanv2dev500.jsonl --output_dir results/phaseA/base \
    --n_docs 1000 --output_file qampari_cleanv2dev500.jsonl
run --data data/training/clean_v2/qampari/train_data.jsonl --output_dir results/phaseC/mining \
    --n_docs 100 --output_file qampari_cleanv2train_base_top100.jsonl
echo ALL_DONE
'
