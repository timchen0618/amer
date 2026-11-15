#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=128GB
#SBATCH --account=torch_pr_152_echoi
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --constraint=h200|l40s   # not b200: div.ext3 torch 2.5.1+cu121 has no sm_100 kernels
#SBATCH --gres=gpu:1
#SBATCH --job-name=qwen3_base_eval
#SBATCH --output=sbatch_outputs/qwen3_base_eval.out

# Session N (2026-11-15): base Qwen3-Embedding-0.6B (no training) on the full S0 dev and the test set, through the same
# query path as a trained checkpoint ('Instruct: <training task>\nQuery:<q>', last token) and the same faiss search /
# eval.py scoring as f_run.sh. Output: results/phaseC/full/qampari/qwen3emb06_base/{dev,test}/eval_metrics.txt
set -e
CODE_ROOT=${CODE_ROOT:-/scratch/hc3337/projects/autoregressive/.claude/worktrees/n-nemotron}
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
B=results/phaseC/full/qampari/qwen3emb06_base
cd /scratch/hc3337/projects/autoregressive
for set in dev test; do
  data=data/phaseA/qampari_clean_v3_dev_s0.jsonl; [ $set = test ] && data=data/amer_data/eval_data/qampari.jsonl
  mkdir -p $B/$set
  [ -s $B/${set}_qemb.npy ] || singularity exec --nv --overlay /scratch/hc3337/envs/div.ext3:ro $IMG /bin/bash -c "source /ext3/env.sh; \
    export HF_HUB_OFFLINE=1; cd /scratch/hc3337/projects/autoregressive; \
    python $CODE_ROOT/training/inf_retriever/tools/qwen3_embed.py queries --data $data --out $B/${set}_qemb.npy"
  singularity exec --nv --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG /bin/bash -c "source /ext3/env.sh; conda activate nli; \
    cd /scratch/hc3337/projects/autoregressive; \
    python $CODE_ROOT/retrieval_inf.py --model_name_or_path qwen3emb_base --load_query_embeddings $B/${set}_qemb.npy \
      --passages /scratch/hc3337/wikipedia_chunks/chunks_v5.tsv \
      --passages_embeddings 'wikipedia_embeddings/qampari/qwen3emb06_frozen_base_512/*' --data $data \
      --output_dir $B/$set --output_file qampari.jsonl --projection_size 1024 --n_docs 500 --num_shards 16 --use_gpu && \
    python eval.py --data_path $data --topk 100 10 --input-file $B/$set/qampari.jsonl > $B/$set/eval_metrics.txt 2>&1"
  echo "$set: $(grep -m1 -o 'MRecall: [0-9.]*' $B/$set/eval_metrics.txt)"
done
