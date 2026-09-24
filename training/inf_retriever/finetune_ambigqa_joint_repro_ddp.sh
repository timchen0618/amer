#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=ambigqa_joint_repro_ddp
#SBATCH --output=sbatch_outputs/ambigqa_joint_repro_ddp.out
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:2

# 2x H200, plain DDP (MULTI_GPU), no FSDP.
# AmbigQA analogue of finetune_qampari_joint_repro_ddp.sh. Reproduces the
# original enc_trained_ambigqa_infly_multi_finetuned_steps800_..._hungarian
# run (wandb run-20260507_145357-c9q43js0: sbatch, no FSDP upcast => pure bf16
# DDP). Hyperparameters identical to that run. Its best_model is the step-750
# save; LR hit 0 at step 400 (scheduler double-step on 2 GPUs), so its weights
# equal the step-~400 weights. Saving step-390 (last eval with LR>0 in the
# save_freq=30 grid) and step-750 (matches the original's best_model step).

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh
cd /scratch/hc3337/projects/autoregressive

data_dir=/scratch/hc3337/projects/autoregressive/data/training/filtered/ambigqa
output_dir=checkpoints/ambigqa/
run_name=ambigqa_joint_repro_steps800_t0.05_lr0.00001_ws30_bs50_hungarian_ddp

accelerate launch --config_file training/inf_retriever/accelerate_config_2gpu_ddp.yaml --main_process_port 29523 training/inf_retriever/finetuning_multi.py \
    --train_data \$data_dir/train_data.jsonl \
    --eval_data \$data_dir/dev_data.jsonl \
    --temperature 0.05 \
    --total_steps 800 \
    --warmup_steps 30 \
    --lr 0.00001 \
    --save_freq 30 \
    --log_freq 5 \
    --eval_freq 30 \
    --negative_hard_ratio 0.0 \
    --negative_ctxs 1 \
    --per_gpu_batch_size 50 \
    --per_gpu_eval_batch_size 50 \
    --model_path infly/inf-retriever-v1-1.5b \
    --chunk_length 512 \
    --accumulation_steps 1 \
    --run_name \$run_name \
    --output_dir \$output_dir \
    --training_mode multi \
    --loss_fn hungarian \
    --max_positive_documents 1 \
    --num_workers 2 \
    --norm_query \
    --norm_doc \
    --save_at_steps 390 750
"
