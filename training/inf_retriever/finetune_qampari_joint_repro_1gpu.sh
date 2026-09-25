#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=2:45:00
#SBATCH --mem=128GB
#SBATCH --job-name=qampari_joint_repro_1gpu
#SBATCH --output=sbatch_outputs/qampari_joint_repro_1gpu.out
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1

# NOTE (2026-09-25, branch fsdp-clean-recipe): under the current code this script
# trains in MIXED precision (fp32 weights + fp32 AdamW, bf16 autocast) with a
# correctly-stepped LR scheduler and seeded, rank-consistent data order. It is a
# TEMPLATE, not the fallback recipe. The fallback (pure bf16, 2x-compressed LR
# schedule, unseeded shuffling) is only reproducible from git tag
# fallback-bf16-ddp -- see RECIPE_FALLBACK_bf16_ddp.md.
# WARNING: under mixed precision, 50 examples/GPU runs out of memory on H200 with
# DDP or 1 GPU (measured 2026-09-25); DDP fits at 40/GPU, FSDP fits at 50/GPU.
# Decision (2026-09-25): QAMPARI joint training uses FSDP -> finetune_qampari_joint_fsdp.sh.

# 1x H200, single process (distributed_type NO), no FSDP. NOTE: global batch is 50 (April: 2x50), fewer in-batch negatives, and the LR scheduler steps once per step (April's 2-process run stepped it twice).
# Tests the precision hypothesis (2026-09-23): the original with_detach
# checkpoint has bf16 weights + bf16 AdamW states (no FSDP upcast), while the
# FSDP repro holds fp32 master weights. Without FSDP, accelerate does not
# upcast the bf16-loaded model, so training stays pure bf16 like April.
# Hyperparameters identical to the April run (wandb run-20260429_212024-fsrih0ho).
# Only steps 250/500 are needed (original best_model = step 250); the job
# is expected to hit its walltime after step 500.

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh
cd /scratch/hc3337/projects/autoregressive

data_dir=/scratch/hc3337/projects/autoregressive/data/training/filtered/qampari
output_dir=checkpoints/qampari/
run_name=qampari_joint_repro_with_detach_steps5000_t0.05_lr0.00001_ws200_bs50_hungarian_1gpu

accelerate launch --config_file training/inf_retriever/accelerate_config_1gpu.yaml --main_process_port 29522 training/inf_retriever/finetuning_multi.py \
    --train_data \$data_dir/train_data.jsonl \
    --eval_data \$data_dir/dev_data.jsonl \
    --temperature 0.05 \
    --total_steps 5000 \
    --warmup_steps 200 \
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
    --save_at_steps 250 500
"
