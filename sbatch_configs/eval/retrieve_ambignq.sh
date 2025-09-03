#!/bin/bash

# GENERATE ARGS
data_name="ambiguous_qe_query_exp"
training_data_name="nq"
suffix="toy_contrastive"

retriever="inf"
use_best_model=false
compute_loss=false
full_finetuning=false
base_model="llama-1b"
checkpoint_num="70000"

# inference_modes="all first second"
inference_modes="all"
max_new_tokens=1
num_shards="8"
use_gpu=true
machine="greene"

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
