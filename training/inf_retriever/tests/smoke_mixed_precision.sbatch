#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=1:15:00
#SBATCH --mem=128GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --job-name=smoke_mixed
#SBATCH --output=sbatch_outputs/smoke_mixed_%x_%j.out

# Smoke test for the clean training loop (mixed precision, once-per-step LR schedule,
# rank-consistent seeded data order) on one backend. Identical arguments across
# backends so their loss curves can be compared step by step. Stops itself after
# ~55 min (past the step-250 eval + checkpoint); the schedule is still the full
# 2500-step one, so logged LRs follow the real recipe.
#
# Usage (from the repo root; set BACKEND and a matching --gres):
#   sbatch --gres=gpu:1 --export=ALL,BACKEND=1gpu  training/inf_retriever/tests/smoke_mixed_precision.sbatch
#   sbatch --gres=gpu:2 --export=ALL,BACKEND=ddp   training/inf_retriever/tests/smoke_mixed_precision.sbatch
#   sbatch --gres=gpu:2 --export=ALL,BACKEND=fsdp  training/inf_retriever/tests/smoke_mixed_precision.sbatch
# Optional: BATCH (examples per GPU, default 50), RUN_MINUTES (default 55),
# EXTRA_ARGS (appended to the training command, e.g. "--not_save").

case "${BACKEND}" in
  1gpu) CONFIG=accelerate_config_1gpu.yaml;             PORT=29551 ;;
  ddp)  CONFIG=accelerate_config_2gpu_ddp.yaml;         PORT=29552 ;;
  fsdp) CONFIG=accelerate_config_2gpu_fsdp_nooffload.yaml; PORT=29553 ;;
  *) echo "Set BACKEND to 1gpu, ddp or fsdp"; exit 1 ;;
esac
RUN_MINUTES=${RUN_MINUTES:-55}
BATCH=${BATCH:-50}
EXTRA_ARGS=${EXTRA_ARGS:-}

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh
cd /scratch/hc3337/projects/autoregressive

data_dir=/scratch/hc3337/projects/autoregressive/data/training/filtered/qampari
timeout ${RUN_MINUTES}m accelerate launch --config_file training/inf_retriever/${CONFIG} --main_process_port ${PORT} training/inf_retriever/finetuning_multi.py \
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
    --per_gpu_batch_size ${BATCH} \
    --per_gpu_eval_batch_size 50 \
    --model_path infly/inf-retriever-v1-1.5b \
    --chunk_length 512 \
    --accumulation_steps 1 \
    --run_name smoke_mixed_${BACKEND}_bs${BATCH} \
    --output_dir checkpoints/qampari/ \
    --training_mode multi \
    --loss_fn hungarian \
    --max_positive_documents 1 \
    --num_workers 2 \
    --norm_query \
    --norm_doc \
    --seed 0 \
    --save_at_steps 250 ${EXTRA_ARGS}
status=\$?
[ \$status -eq 124 ] && echo 'smoke test stopped by timeout as planned' && exit 0
exit \$status
"
