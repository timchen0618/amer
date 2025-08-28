#!/bin/bash

# Hyperparameter Search Configuration
# Edit this file to customize your hyperparameter search space

# =================================================================
# HYPERPARAMETER SEARCH SPACE
# =================================================================

# LEARNING_RATES=(2e-5 1e-5 5e-5 1e-4)
LEARNING_RATES=(2e-5)
# TEMPERATURES=(0.03 0.1)
TEMPERATURES=(0.05)
# BATCH_SIZES=(16 8 32)
BATCH_SIZES=(32)
# NUM_EPOCHS_LIST=(20 10 30 40)
NUM_EPOCHS_LIST=(120)
# WARMUP_RATIOS=(0.05 0.1)
WARMUP_RATIOS=(0.05)
# Use hard negatives
USE_HARD_NEGATIVES=false
# LR min ratios
LR_MIN_RATIO=0.0

# =================================================================
# BASE CONFIGURATION
# =================================================================

# Project name (will be used in wandb)
BASE_PROJECT="diverse_retrieval"
full_finetuning=true            # whether to use full finetuning
all_data=false                  # whether to train on all data
multiple_gpus=false              # whether to use multiple GPUs
save_only_improve=true          # whether to save only improve
save_best_model=true            # whether to save best model
normalize=true
LOG_WITH="wandb"
use_inf_base_model=false
machine="torch" # greene, torch
less_ss=true

MODEL_TYPE="EmbeddingModelSSVariableLeftPad"
MODE="hungarian_contrastive_woseq"

# MODES -> 
# 1. hungarian_contrastive
# 2. contrastive_first_label
# 3. contrastive_one_label_shuffled
# 4. contrastive_all_labels_ordered
# 5. contrastive_all_labels_shuffled
# 6. mse_first_label

# # Loss function (options: MSE, Hungarian_MSE, Contrastive, Hungarian_Contrastive)
# LOSS_FUNCTION="Hungarian_Contrastive"


if [ "$MODE" == "hungarian_contrastive" ]; then
    LOSS_FUNCTION="Hungarian_Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_all_labels_ordered" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_one_label_shuffled" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST="--take_first"
    QUESTION_ONLY=""
elif [ "$MODE" == "hungarian_contrastive_woseq" ]; then
    LOSS_FUNCTION="Hungarian_Contrastive_woseq"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_all_labels_ordered_woseq" ]; then
    LOSS_FUNCTION="Contrastive_woseq"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_first_label" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST="--take_first"
    QUESTION_ONLY=""
elif [ "$MODE" == "mse_all_labels" ]; then
    LOSS_FUNCTION="MSE"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_all_labels_shuffled" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "mse_all_labels_shuffled" ]; then
    LOSS_FUNCTION="MSE"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "hungarian_contrastive_no_shuffle" ]; then
    LOSS_FUNCTION="Hungarian_Contrastive"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "mse_one_label_shuffled" ]; then
    LOSS_FUNCTION="MSE"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY="--first_label_only"
elif [ "$MODE" == "mse_first_label" ]; then
    LOSS_FUNCTION="MSE"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY="--first_label_only"
fi


# Experiment prefix
if [ "$MODEL_TYPE" == "EmbeddingModel" ]; then
    MODEL_STR=""
    SCHEDULE_SAMPLING=""
    LEFT_PADDING=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSS" ]; then
    MODEL_STR="_SS"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSSVariable" ]; then
    MODEL_STR="_SSVariable"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSSVariableLeftPad" ]; then
    MODEL_STR="_SSVariableLeftPad"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING="--left_padding"
fi

# =================================================================
if [ "$full_finetuning" = true ]; then
    FULL_FINETUNING="--full_finetuning" # FULL_FINETUNING=""
    FINETUNING_STR="_full_finetuning"
else
    FINETUNING_STR=""
    FULL_FINETUNING=""
fi

if [ "$all_data" = true ]; then
    TRAIN_ON_ALL_DATA="--train_on_all_data"
else
    TRAIN_ON_ALL_DATA=""
fi

if [ "$multiple_gpus" = true ]; then
    GPUS_PREFIX="_4gpu"
else
    GPUS_PREFIX=""
fi

if [ "$save_only_improve" = true ]; then
    SAVE_ONLY_IMPROVE="--save_only_improve"
else
    SAVE_ONLY_IMPROVE=""
fi

if [ "$save_best_model" = true ]; then
    SAVE_BEST_MODEL="--save_best_model"
else
    SAVE_BEST_MODEL=""
fi

if [ "$normalize" = true ]; then
    normalize_prefix="normalized_"
    NORMALIZE_STR="--normalize_embeddings"
else
    normalize_prefix=""
    NORMALIZE_STR=""
fi

if [ "$use_inf_base_model" = true ]; then
    base_prefix="inf_"
else
    base_prefix=""
fi  

if [ "$less_ss" = true ]; then
    LESS_SS="--less_ss"
    less_ss_prefix="less_SS_"
else
    LESS_SS=""
    less_ss_prefix=""
fi

# Experiment prefix
EXP_PREFIX="${less_ss_prefix}${base_prefix}${normalize_prefix}qampari${GPUS_PREFIX}${FINETUNING_STR}${MODEL_STR}_${MODE}"

# Base directory for saving results
if [ "$use_inf_base_model" = true ]; then
    BASE_SAVE_PATH="results/inf/qampari_inf"
else
    BASE_SAVE_PATH="results/llama-1b/qampari_inf"
fi

# Training dataset path
if [ "$use_inf_base_model" = true ]; then
    if [ "$USE_HARD_NEGATIVES" = true ]; then
        BASE_TRAIN_PATH="training_datasets/inf/qampari/inf/autoregressive_qampari_inf_train_dataset_1b_contrastive_hard_negative_5_to_8_ctxs/"
    else
        BASE_TRAIN_PATH="training_datasets/inf/qampari/inf/autoregressive_qampari_inf_train_dataset_1b_contrastive_5_to_8_ctxs/"
    fi
else
    if [ "$USE_HARD_NEGATIVES" = true ]; then
        BASE_TRAIN_PATH="training_datasets/llama-1b/qampari/inf/autoregressive_qampari_inf_train_dataset_1b_contrastive_hard_negative_5_to_8_ctxs/"
    else
        BASE_TRAIN_PATH="training_datasets/llama-1b/qampari/inf/autoregressive_qampari_inf_train_dataset_1b_contrastive_5_to_8_ctxs/"
    fi
fi



# Model checkpoints
if [ "$use_inf_base_model" = true ]; then
    MODEL_ID="infly/inf-retriever-v1-1.5b"
    BASE_ADAPTER_PATH=None
    BASE_LINEAR_CHECKPOINT_PATH=None
else
    MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
    BASE_ADAPTER_PATH="results/llama-1b/qampari_inf/toy_qemb_from_nq/checkpoint_30000"
    BASE_LINEAR_CHECKPOINT_PATH="results/llama-1b/qampari_inf/toy_qemb_from_nq/checkpoint_30000_linear.pt"
fi



# =================================================================
# FIXED HYPERPARAMETERS
# =================================================================
# Model embedding dimension
EMBEDDING_MODEL_DIM=1536

# How often to save checkpoints (in steps)
SAVE_EVERY_N_STEPS=100

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
MAX_CONCURRENT_JOBS=40

if [ "$multiple_gpus" = true ]; then
    # SLURM job time limit
    TIME_LIMIT="48:00:00"
    # Memory per job
    MEMORY="300GB"
    # Number of CPUs per task
    CPUS_PER_TASK=40
    # GPU configuration
    GPU_TYPE="a100"
    GPUS_PER_NODE=4
    GPU_STRING="4"
else
    # SLURM job time limit
    TIME_LIMIT="24:00:00"
    # Memory per job
    MEMORY="200GB"
    # Number of CPUs per task
    CPUS_PER_TASK=20
    # GPU configuration
    GPU_TYPE="a100"
    GPUS_PER_NODE=1
    GPU_STRING="1"
fi

# Email for notifications
EMAIL="hc3337@nyu.edu"

# Singularity configuration
if [ "$machine" = "greene" ]; then
    SINGULARITY_IMAGE="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
elif [ "$machine" = "torch" ]; then
    SINGULARITY_IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
else
    echo "Invalid machine"
    exit 1
fi
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
    local prefix=$7
    
    # Default naming scheme    
    if [ "$use_hard_negatives" = true ]; then
        echo "${prefix}_lr${lr}_temp${temp}_batch${batch}_ep${epochs}_warmup${warmup}_hn"
    else
        echo "${prefix}_lr${lr}_temp${temp}_batch${batch}_ep${epochs}_warmup${warmup}"
    fi
} 