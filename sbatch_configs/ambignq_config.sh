#!/bin/bash

# Hyperparameter Search Configuration
# Edit this file to customize your hyperparameter search space

# =================================================================
# HYPERPARAMETER SEARCH SPACE
# =================================================================

# Learning rates to test
# LEARNING_RATES=(1e-5 2e-5 5e-5 1e-4)
LEARNING_RATES=(5e-5)

# Temperature values for contrastive loss
# TEMPERATURES=(0.03 0.1)
TEMPERATURES=(0.05)

# Batch sizes
# BATCH_SIZES=(8 16 32)
BATCH_SIZES=(16)

# Number of epochs
# NUM_EPOCHS_LIST=(10 20 30)
NUM_EPOCHS_LIST=(30)

# Warmup ratios
# WARMUP_RATIOS=(0.05 0.1)
WARMUP_RATIOS=(0.05)


# LR min ratios
LR_MIN_RATIO=0.1

# MODES -> 
# 1. hungarian_contrastive
# 2. contrastive_first_label
# 3. contrastive_one_label_shuffled
# 4. contrastive_all_labels_ordered
# 5. contrastive_all_labels_shuffled
# 6. mse_first_label

MODE="mse_all_labels"
# # Loss function (options: MSE, Hungarian_MSE, Contrastive, Hungarian_Contrastive)
# LOSS_FUNCTION="Hungarian_Contrastive"

if [ "$MODE" == "hungarian_contrastive" ]; then
    LOSS_FUNCTION="Hungarian_Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_first_label" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST="--take_first"
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_one_label_shuffled" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST="--take_first"
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_all_labels_ordered" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_all_labels_ordered_wo_seq" ]; then
    LOSS_FUNCTION="Contrastive_wo_seq"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_all_labels_shuffled" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "mse_first_label" ]; then
    LOSS_FUNCTION="MSE"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY="--first_label_only"
elif [ "$MODE" == "mse_all_labels" ]; then
    LOSS_FUNCTION="MSE"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""
fi


SAVE_ONLY_IMPROVE="--save_only_improve"
SAVE_BEST_MODEL="--save_best_model"

# =================================================================
# BASE CONFIGURATION
# =================================================================

# Project name (will be used in wandb)
BASE_PROJECT="diverse_retrieval"

# Experiment prefix
EXP_PREFIX="ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_${MODE}"
MODEL_TYPE="EmbeddingModelSSVariableLeftPad"
FULL_FINETUNING="--full_finetuning" # FULL_FINETUNING=""
TRAIN_ON_ALL_DATA="" # TRAIN_ON_ALL_DATA="--train_on_all_data"
SCHEDULE_SAMPLING="--schedule_sampling" # SCHEDULE_SAMPLING="--schedule_sampling"
LEFT_PADDING="--left_padding" # LEFT_PADDING="--left_padding"

# Base directory for saving results
BASE_SAVE_PATH="results/ambiguous_qe_inf/"

# Training dataset path
BASE_TRAIN_PATH="training_datasets/ambiguous_qe/inf/autoregressive_ambiguous_qe_inf_train_dataset_1b_contrastive_2_to_5_ctxs/"

# Model checkpoints
BASE_ADAPTER_PATH="results/nq_inf/toy_contrastive/checkpoint_70000"
BASE_LINEAR_CHECKPOINT_PATH="results/nq_inf/toy_contrastive/checkpoint_70000_linear.pt"

# =================================================================
# FIXED HYPERPARAMETERS
# =================================================================

# Model embedding dimension
EMBEDDING_MODEL_DIM=1536

# How often to save checkpoints (in steps)
SAVE_EVERY_N_STEPS=250

# Gradient accumulation steps
GRADIENT_ACCUMULATION_STEPS=1

# Weight decay
WEIGHT_DECAY=0.01

# Scheduler type
SCHEDULER="linear"

# Maximum gradient norm for clipping
MAX_GRAD_NORM=5.0

# =================================================================
# SLURM CONFIGURATION
# =================================================================

# Maximum number of concurrent jobs
MAX_CONCURRENT_JOBS=100

# SLURM job time limit
TIME_LIMIT="4:00:00"

# Memory per job
MEMORY="300GB"

# Number of CPUs per task
CPUS_PER_TASK=40

# GPU configuration
GPU_TYPE="a100"
GPUS_PER_NODE=4
GPU_STRING="a100:4"

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
SBATCH_DIR="sbatch_jobs"

# Directory for job output logs
JOB_OUTPUT_DIR="sbatch_outputs"

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
    local prefix=$7
    
    # Default naming scheme    
    if [ "$use_hard_negatives" = true ]; then
        echo "${prefix}_lr${lr}_temp${temp}_batch${batch}_ep${epochs}_warmup${warmup}_hn"
    else
        echo "${prefix}_lr${lr}_temp${temp}_batch${batch}_ep${epochs}_warmup${warmup}"
    fi
} 