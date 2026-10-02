#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH --job-name=clean_base_retrieval
#SBATCH --output=sbatch_outputs/%x.out

# Base-retriever retrieval over the full corpus for the clean splits
# (data_creation/build_clean_splits.py):
#   - top-1000 for each clean dev set  -> results/phaseA/base/<ds>_cleandev500.jsonl
#     (input to phaseA_build_reduced_corpus.py, the cheap-eval corpora)
#   - top-100 for each clean train set -> results/phaseC/mining/<ds>_cleantrain_base_top100.jsonl
#     (input to phaseC_mine_hard_negatives.py)
#   sbatch training/inf_retriever/tools/clean_base_retrieval.sbatch
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
for ds in ambigqa qampari; do
    run --data data/phaseA/${ds}_cleandev500.jsonl --output_dir results/phaseA/base \
        --n_docs 1000 --output_file ${ds}_cleandev500.jsonl
    run --data data/training/clean/$ds/train_data.jsonl --output_dir results/phaseC/mining \
        --n_docs 100 --output_file ${ds}_cleantrain_base_top100.jsonl
done
echo ALL_DONE
'
