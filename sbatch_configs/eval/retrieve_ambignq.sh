#!/bin/bash

# GENERATE ARGS
data_name="ambiguous_qe_mmlf"  # ambiguous_qe_query_exp, ambiguous_qe_mmlf
training_data_name="ambiguous_qe"
suffix="llama-1b_single_0"

retriever="inf"
use_best_model=true
compute_loss=false
full_finetuning=false
base_model="llama-1b"  # llama-1b, qwen3-4b, llama-3b, llama-8b
checkpoint_num="70000"

# inference_modes="all first second"
inference_modes="all"
round_robin_percentage=1.0
max_new_tokens=1
num_shards="16"
use_gpu=true
machine="torch"
use_l40s=false

# SLURM CONFIGURATION BEGINS
SBATCH_DIR="sbatch_jobs_eval"
JOB_OUTPUT_DIR="sbatch_outputs_eval"
TIME_LIMIT="1:00:00"
MEMORY="200GB"
CPUS_PER_TASK=10
EMAIL="hc3337@nyu.edu"
# Singularity 
if [ "$machine" = "greene" ]; then
    GPU_STRING="a100:1"
    SINGULARITY_IMAGE="/scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
    SLURM_EXTRA_ARGS=""
elif [ "$machine" = "torch" ]; then
    GPU_STRING="1"
    SINGULARITY_IMAGE="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
    if [ "$use_l40s" = true ]; then
        CONSTRAINT="h200|l40s"
    else
        CONSTRAINT="h200"
    fi
    SLURM_EXTRA_ARGS="#SBATCH --comment=\"preemption=yes;requeue=yes\"
#SBATCH --constraint=\"$CONSTRAINT\"
#SBATCH --account=torch_pr_152_courant"
else
    echo "Invalid machine"
    exit 1
fi
OVERLAY_FILE="/scratch/hc3337/envs/nli.ext3"
HF_TOKEN="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"
WORK_DIR="/scratch/hc3337/projects/autoregressive"
# SLURM CONFIGURATION ENDS
