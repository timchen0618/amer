#!/bin/bash

# Hyperparameter Search Configuration
# Edit this file to customize your hyperparameter search space

# =================================================================
# HYPERPARAMETER SEARCH SPACE
# =================================================================

# Learning rates to test
LEARNING_RATES=(2e-5 1e-5 5e-5 1e-4)
# LEARNING_RATES=(2e-5)

# Temperature values for contrastive loss
TEMPERATURES=(0.05)
# TEMPERATURES=(0.01)

# Batch sizes
BATCH_SIZES=(16 8 32)
# BATCH_SIZES=(16)

# Number of epochs
NUM_EPOCHS_LIST=(20 10 30)
# NUM_EPOCHS_LIST=(20)

# Warmup ratios
# WARMUP_RATIOS=(0.05 0.1)
WARMUP_RATIOS=(0.05)

# Use hard negatives
USE_HARD_NEGATIVES=true

# =================================================================
# BASE CONFIGURATION
# =================================================================

# Project name (will be used in wandb)
BASE_PROJECT="diverse_retrieval"

# Base directory for saving results
BASE_SAVE_PATH="results/qampari_inf"

# Training dataset path
if [ "$USE_HARD_NEGATIVES" = true ]; then
    BASE_TRAIN_PATH="training_datasets/qampari/inf/autoregressive_qampari_inf_train_dataset_1b_contrastive_hard_negative_5_to_8_ctxs/"
else
    BASE_TRAIN_PATH="training_datasets/qampari/inf/autoregressive_qampari_inf_train_dataset_1b_contrastive_5_to_8_ctxs/"
fi

# Model checkpoints
BASE_ADAPTER_PATH="results/qampari_inf/toy_qemb_from_nq/checkpoint_30000"
BASE_LINEAR_CHECKPOINT_PATH="results/qampari_inf/toy_qemb_from_nq/checkpoint_30000_linear.pt"

# =================================================================
# FIXED HYPERPARAMETERS
# =================================================================

# Loss function (options: MSE, Hungarian_MSE, Contrastive, Hungarian_Contrastive)
LOSS_FUNCTION="Hungarian_Contrastive"

# Model embedding dimension
EMBEDDING_MODEL_DIM=1536

# How often to save checkpoints (in steps)
SAVE_EVERY_N_STEPS=500

# Gradient accumulation steps
GRADIENT_ACCUMULATION_STEPS=1

# Weight decay
WEIGHT_DECAY=0.01

# Scheduler type
SCHEDULER="linear"

# Maximum gradient norm for clipping
MAX_GRAD_NORM=1.0

# =================================================================
# SLURM CONFIGURATION
# =================================================================

# Maximum number of concurrent jobs
MAX_CONCURRENT_JOBS=40

# SLURM job time limit
TIME_LIMIT="48:00:00"

# Memory per job
MEMORY="300GB"

# Number of CPUs per task
CPUS_PER_TASK=20

# GPU configuration
GPU_TYPE="a100"
GPUS_PER_NODE=4

# Email for notifications
EMAIL="hc3337@nyu.edu"

# Singularity configuration
SINGULARITY_IMAGE="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY_FILE="/scratch/hc3337/envs/div.ext3"

# HuggingFace token (if needed)
HF_TOKEN="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"

# Working directory
WORK_DIR="/scratch/hc3337/projects/autoregressive"

# =================================================================
# OUTPUT DIRECTORIES
# =================================================================

# Directory for SBATCH files
SBATCH_DIR="sbatch_jobs_qampari"

# Directory for job output logs
JOB_OUTPUT_DIR="sbatch_outputs_qampari"

# =================================================================
# ADVANCED OPTIONS
# =================================================================

# Add delay between job submissions (seconds)
SUBMISSION_DELAY=2

# Whether to create dependency chains (submit jobs sequentially)
# Set to "true" to avoid overwhelming the scheduler
USE_DEPENDENCIES=false

# Dry run mode - generate SBATCH files but don't submit them
DRY_RUN=false

# =================================================================
# FILTERING OPTIONS
# =================================================================

# Skip combinations that are likely to fail or are redundant
# For example, skip very high learning rates with small batch sizes

filter_combinations() {
    local lr=$1
    local temp=$2
    local batch=$3
    local epochs=$4
    local warmup=$5
    
    # Example filters (uncomment and modify as needed):
    
    # Skip high learning rates with small batch sizes
    # if (( $(echo "$lr > 5e-5" | bc -l) )) && (( batch < 32 )); then
    #     return 1  # Skip this combination
    # fi
    
    # Skip very long training with large batch sizes
    # if (( epochs > 20 )) && (( batch > 32 )); then
    #     return 1  # Skip this combination
    # fi
    
    return 0  # Keep this combination
}

# =================================================================
# CUSTOM EXPERIMENT NAMING
# =================================================================

# Function to generate custom experiment names
# You can modify this to include additional information
generate_custom_exp_name() {
    local lr=$1
    local temp=$2
    local batch=$3
    local epochs=$4
    local warmup=$5
    local use_hard_negatives=$6

    # Default naming scheme
    if [ "$use_hard_negatives" = true ]; then
        echo "hypersearch_lr${lr}_temp${temp}_batch${batch}_ep${epochs}_warmup${warmup}_hn"
    else
        echo "hypersearch_lr${lr}_temp${temp}_batch${batch}_ep${epochs}_warmup${warmup}"
    fi
    
    # Alternative naming schemes (uncomment one if desired):
    # echo "exp_$(date +%Y%m%d)_lr${lr}_t${temp}_b${batch}_e${epochs}_w${warmup}"
    # echo "${loss_function}_lr${lr}_temp${temp}_${batch}batch_${epochs}ep"
} 