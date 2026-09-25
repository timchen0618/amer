#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=512GB
#SBATCH --job-name=std-tune
#SBATCH --output=/scratch/dq2024/diverse_retriever/training/qwen2/standard/out_files/std-tune-0-6.out
#SBATCH --gres=gpu:h200:4
#SBATCH --account=torch_pr_152_courant


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 

accelerate launch --main_process_port 29501 /scratch/dq2024/diverse_retriever/training/qwen2/finetuning.py --train_data /scratch/dq2024/diverse_retriever/train_data.jsonl \
                                                 --eval_data /scratch/dq2024/diverse_retriever/train_eval_data.jsonl \
                                                 --temperature 0.05 \
                                                 --total_steps 50000 \
                                                 --warmup_steps 1000 \
                                                 --lr 0.000005 \
                                                 --save_freq 2500 \
                                                 --log_freq 25 \
                                                 --eval_freq 2500 \
                                                 --negative_hard_ratio 0.0 \
                                                 --negative_ctxs 1 \
                                                 --per_gpu_batch_size 48 \
                                                 --per_gpu_eval_batch_size 48 \
                                                 --model_path Qwen/Qwen3-Embedding-0.6B \
                                                 --chunk_length 512 \
                                                 --accumulation_steps 2 \
                                                 --run_name "qampari_qwen3_standard_finetuned_only_random_0-6_istr_lr0.000005_temp0.05" \
                                                 --output_dir "/scratch/dq2024/models/diverse_retriever/checkpoints/" \
                                                 --training_mode "standard" \
                                                 --max_positive_documents 1 \
                                                 --sample_length \
                                                 --doc_length 0 1 2 3 4 5 6 \
                                                 --norm_doc \
                                                 --norm_query
"
