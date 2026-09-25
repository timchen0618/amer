#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=512GB
#SBATCH --job-name=tune-0123
#SBATCH --output=/scratch/dq2024/diverse_retriever/training/contriever/standard/out_files/standard-finetune-0123.out
#SBATCH --gres=gpu:4
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

accelerate launch --main_process_port 29502 /scratch/dq2024/diverse_retriever/training/contriever/finetuning.py \
                                                 --train_data /scratch/dq2024/diverse_retriever/train_data_doubled.jsonl  \
                                                 --eval_data /scratch/dq2024/diverse_retriever/train_eval_data.jsonl  \
                                                 --temperature 0.1 \
                                                 --total_steps 100000 \
                                                 --warmup_steps 1000 \
                                                 --lr 0.00005 \
                                                 --save_freq 2500 \
                                                 --log_freq 50 \
                                                 --eval_freq 2500 \
                                                 --negative_hard_ratio 0.0 \
                                                 --negative_ctxs 1 \
                                                 --per_gpu_batch_size 48 \
                                                 --per_gpu_eval_batch_size 48 \
                                                 --model_path facebook/contriever-msmarco \
                                                 --chunk_length 512 \
                                                 --accumulation_steps 2 \
                                                 --run_name "qampari_contriever_standard_only_random_0123_both_objectives_acc2_bs48_lr0.00005_temp0.1" \
                                                 --output_dir "/scratch/dq2024/models/diverse_retriever/checkpoints/" \
                                                 --training_mode "standard" \
                                                 --max_positive_documents 1 \
                                                 --sample_length \
                                                 --doc_length 0 1 2 3

"
