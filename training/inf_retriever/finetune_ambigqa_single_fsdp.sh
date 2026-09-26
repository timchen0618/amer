#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=ambigqa_single_fsdp_mixed
#SBATCH --output=sbatch_outputs/ambigqa_single_fsdp_mixed.out
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:2

# Step 5 / phase B clean baseline (2026-09-25): 2x H200, FSDP
# (accelerate_config_2gpu_fsdp_nooffload.yaml), joint doc encoder, mixed precision
# (fp32 weights + fp32 AdamW, bf16 autocast), once-per-step LR schedule, seeded
# rank-consistent data order. Settings are identical to its multi-/single-query
# counterpart; only --training_mode (and the matching loss) differs.
# Tracker: https://claude.ai/code/artifact/1c7621cd-7d9d-4137-8ff4-71a8891e17bd
# AmbigQA single-query (standard_org_q, contrastive loss). 400 steps / 15 warmup is the
# old runs' effective schedule.

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh
cd /scratch/hc3337/projects/autoregressive

data_dir=/scratch/hc3337/projects/autoregressive/data/training/filtered/ambigqa
output_dir=checkpoints/ambigqa/
run_name=ambigqa_single_fsdp_mixed_steps400_t0.05_lr0.00001_ws15_bs50_contrastive

accelerate launch --config_file training/inf_retriever/accelerate_config_2gpu_fsdp_nooffload.yaml --main_process_port 29534 training/inf_retriever/finetuning_multi.py \
    --train_data \$data_dir/train_data.jsonl \
    --eval_data \$data_dir/dev_data.jsonl \
    --temperature 0.05 \
    --total_steps 400 \
    --warmup_steps 15 \
    --lr 0.00001 \
    --save_freq 10 \
    --log_freq 5 \
    --eval_freq 10 \
    --negative_hard_ratio 0.0 \
    --negative_ctxs 1 \
    --per_gpu_batch_size 50 \
    --per_gpu_eval_batch_size 50 \
    --model_path infly/inf-retriever-v1-1.5b \
    --chunk_length 512 \
    --accumulation_steps 1 \
    --run_name \$run_name \
    --output_dir \$output_dir \
    --training_mode standard_org_q \
    --loss_fn auto \
    --max_positive_documents 1 \
    --num_workers 2 \
    --norm_query \
    --norm_doc \
    --seed 0 \
    --save_at_steps 30 150 400
"
