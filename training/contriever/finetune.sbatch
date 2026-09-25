#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=512GB
#SBATCH --job-name=base-tune
#SBATCH --output=outfiles/base-finetune.out
#SBATCH --gres=gpu:h200:4

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 


accelerate launch --main_process_port 29501 ../finetuning.py --train_data ../../../train_data.jsonl  \
                                                 --eval_data ../../../train_eval_data.jsonl  \
                                                 --temperature 0.05 \
                                                 --total_steps 50000 \
                                                 --warmup_steps 1000 \
                                                 --lr 0.0001 \
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
                                                 --run_name "qampari_base_finetuned_only_random_sample_0" \
                                                 --output_dir "/scratch/dq2024/models/diverse_retriever/checkpoints" \
                                                 --training_mode "standard" \
                                                 --max_positive_documents 1 \
                                                 --sample_length \
                                                 --num_workers 4 \
                                                 --doc_length 0
"