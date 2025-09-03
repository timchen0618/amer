#!/bin/bash

# GENERATE ARGS
data_name="qampari_5_to_8"
training_data_name="qampari"
suffix="less_SS_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05"

retriever="inf"
use_best_model=true
compute_loss=false
full_finetuning=true
base_model="llama-1b"
checkpoint_num="30000"

# inference_modes="all first second"
inference_modes="all"
max_new_tokens=5
num_shards="8"
use_gpu=true
machine="torch"

# SLURM CONFIGURATION BEGINS
SBATCH_DIR="sbatch_jobs_eval"
JOB_OUTPUT_DIR="sbatch_outputs_eval"
TIME_LIMIT="4:00:00"
MEMORY="200GB"
CPUS_PER_TASK=20
GPU_STRING="1"
EMAIL="hc3337@nyu.edu"
# Singularity 
if [ "$machine" = "greene" ]; then
    SINGULARITY_IMAGE="/scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
    SLURM_EXTRA_ARGS=""
elif [ "$machine" = "torch" ]; then
    SINGULARITY_IMAGE="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
    SLURM_EXTRA_ARGS="#SBATCH --requeue
#SBATCH --partition=h200"
else
    echo "Invalid machine"
    exit 1
fi
OVERLAY_FILE="/scratch/hc3337/envs/nli.ext3"
HF_TOKEN="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"
WORK_DIR="/scratch/hc3337/projects/autoregressive"
# SLURM CONFIGURATION ENDS
