#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:2
#SBATCH --job-name=phaseC_train
#SBATCH --output=sbatch_outputs/%x.out

# Step 5 / phase C training run: the clean phase B setup (2-GPU FSDP, mixed precision,
# once-per-step LR schedule, seeded data order, joint doc encoder) with one knob changed.
#   sbatch --job-name=phaseC_<run> --export=ALL,DS=qampari,MODE=multi,LR=3e-6,RUN_NAME=<run>[,EXTRA="--flag ..."][,SEED=0] \
#       training/inf_retriever/tools/phaseC_train.sbatch
# Optional: SAVE_AT_OVERRIDE="250 500 ..." replaces the default save steps.
# Schedules are the old runs' effective ones (as in phase B). Checkpoints skip the AdamW state.
: "${DS:?}" "${MODE:?}" "${LR:?}" "${RUN_NAME:?}"
EXTRA=${EXTRA:-}; SEED=${SEED:-0}
case $DS in
  qampari) STEPS=2500; WARMUP=100; FREQ=250; LOGF=25; SAVE_AT="250 500 1000 2500" ;;
  ambigqa) STEPS=400;  WARMUP=15;  FREQ=10;  LOGF=5;  SAVE_AT="30 100 150 250 400" ;;
  *) echo "DS must be qampari or ambigqa"; exit 1 ;;
esac
# Dense schedules (phaseC_run.sh with DENSE=1) override the save steps; FREQ must divide them.
SAVE_AT=${SAVE_AT_OVERRIDE:-$SAVE_AT}
case $MODE in
  multi)  TM="--training_mode multi --loss_fn hungarian" ;;
  single) TM="--training_mode standard_org_q --loss_fn auto" ;;
  *) echo "MODE must be multi or single"; exit 1 ;;
esac
PORT=$((29600 + SLURM_JOB_ID % 300))
# Clean splits (data_creation/build_clean_splits.py); DATA_ROOT=data/training/filtered reproduces
# the phase B/C runs before the 2026-09-27 audit.
DATA_ROOT=${DATA_ROOT:-data/training/clean}
D=/scratch/hc3337/projects/autoregressive/$DATA_ROOT/$DS

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh
cd /scratch/hc3337/projects/autoregressive
accelerate launch --config_file training/inf_retriever/accelerate_config_2gpu_fsdp_nooffload.yaml --main_process_port $PORT training/inf_retriever/finetuning_multi.py \
    --train_data $D/train_data.jsonl --eval_data $D/dev_data.jsonl \
    --temperature 0.05 --total_steps $STEPS --warmup_steps $WARMUP --lr $LR \
    --save_freq $FREQ --log_freq $LOGF --eval_freq $FREQ \
    --negative_hard_ratio 0.0 --negative_ctxs 1 \
    --per_gpu_batch_size 50 --per_gpu_eval_batch_size 50 \
    --model_path infly/inf-retriever-v1-1.5b --chunk_length 512 --accumulation_steps 1 \
    --run_name $RUN_NAME --output_dir checkpoints/$DS/ \
    $TM --max_positive_documents 1 --num_workers 2 --norm_query --norm_doc \
    --seed $SEED --save_at_steps $SAVE_AT --no_save_optimizer --no_save_best_model $EXTRA
"
