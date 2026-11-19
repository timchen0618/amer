#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --mem=96GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --job-name=clean_v2_downstream
#SBATCH --output=sbatch_outputs/%x.out

# Steps after data_creation/clean_v2_base_retrieval.sbatch for the v2 QAMPARI data (see
# data/training/clean_v2/README.md): hard-negative mining, the cheap-eval reduced corpus for the v2
# dev set, the AmbigQA links (AmbigQA is unchanged from v1), and the full verification.
#   sbatch --dependency=afterok:<base retrieval job> data_creation/clean_v2_downstream.sbatch
singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c '
source /ext3/env.sh
cd /scratch/hc3337/projects/autoregressive
set -e
# 1. hard negatives: base top-100 minus golds, same-title/answer-entity titles, answer-containing
#    passages and the top 3; at most 30 per example (phaseC_mine_hard_negatives.py defaults)
python training/inf_retriever/tools/phaseC_mine_hard_negatives.py --ds qampari \
    --src data/training/clean_v2/qampari --dst data/training/clean_v2_hn/qampari \
    --mined results/phaseC/mining/qampari_cleanv2train_base_top100.jsonl
ln -sfn ../clean_hn/ambigqa data/training/clean_v2_hn/ambigqa
# 2. reduced corpus: base top-1000 for the v2 dev queries + their golds + the same 100k random
#    passages (seed 0) as every other reduced corpus
python training/inf_retriever/tools/phaseA_build_reduced_corpus.py \
    --spec qampari_cleanv2dev500:results/phaseA/base/qampari_cleanv2dev500.jsonl:data/phaseA/qampari_cleanv2dev500.jsonl \
    --topk 1000 --n_random 100000 --seed 0 --out_dir data/phaseA/corpora
# AmbigQA v2 dev = v1 dev (byte-identical, checked by verify_clean_v2.py): reuse its corpus and base run
ln -sfn ambigqa_cleandev500.tsv data/phaseA/corpora/ambigqa_cleanv2dev500.tsv
ln -sfn ambigqa_cleandev500.jsonl results/phaseA/base/ambigqa_cleanv2dev500.jsonl
# 3. verification
python data_creation/verify_clean_v2.py --stage all
echo ALL_DONE
'
