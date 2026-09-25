#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=qampari_joint_fsdp_mixed
#SBATCH --output=sbatch_outputs/qampari_joint_fsdp_mixed.out
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:2

# 2x H200, FSDP (accelerate_config_2gpu_fsdp_nooffload.yaml), joint doc encoder
# (EmbeddingModelDocEncNoProj), QAMPARI. Template for the clean-recipe search, and the
# designated backend for QAMPARI joint training: under mixed precision, DDP / 1 GPU run
# out of memory at 50 examples/GPU on H200, FSDP fits (measured 2026-09-25).
#
# Current code (branch fsdp-clean-recipe) always trains in MIXED precision
# (fp32 weights + fp32 AdamW state, bf16 autocast compute), steps the LR
# scheduler once per training step, and seeds a rank-consistent data order.
#
# --total_steps 2500 --warmup_steps 100 reproduces the *effective* LR schedule
# of the old runs (which used 5000/200 but stepped the scheduler twice per step
# on 2 GPUs, so LR hit 0 at step 2500). Hyperparameters otherwise match the
# fallback recipe (RECIPE_FALLBACK_bf16_ddp.md).

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh
cd /scratch/hc3337/projects/autoregressive

data_dir=/scratch/hc3337/projects/autoregressive/data/training/filtered/qampari
output_dir=checkpoints/qampari/
run_name=qampari_joint_fsdp_mixed_steps2500_t0.05_lr0.00001_ws100_bs50_hungarian

accelerate launch --config_file training/inf_retriever/accelerate_config_2gpu_fsdp_nooffload.yaml --main_process_port 29531 training/inf_retriever/finetuning_multi.py \
    --train_data \$data_dir/train_data.jsonl \
    --eval_data \$data_dir/dev_data.jsonl \
    --temperature 0.05 \
    --total_steps 2500 \
    --warmup_steps 100 \
    --lr 0.00001 \
    --save_freq 250 \
    --log_freq 25 \
    --eval_freq 250 \
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
    --seed 0 \
    --save_at_steps 250 500 1000 2500
"
