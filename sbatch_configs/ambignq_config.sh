#!/bin/bash

# Hyperparameter Search Configuration
# Edit this file to customize your hyperparameter search space

# =================================================================
# HYPERPARAMETER SEARCH SPACE
# =================================================================

# Learning rates to test
# LEARNING_RATES=(5e-4 2e-4 1e-4)
LEARNING_RATES=(5e-5)

# Temperature values for contrastive loss
# TEMPERATURES=(0.04 0.03 0.02 0.01)
TEMPERATURES=(0.05)

# Batch sizes
# BATCH_SIZES=(8 16 32)
BATCH_SIZES=(8)

# Number of epochs
# NUM_EPOCHS_LIST=(10 20 30)
NUM_EPOCHS_LIST=(120)

# Warmup ratios
# WARMUP_RATIOS=(0.05 0.1)
WARMUP_RATIOS=(0.05)

SAMPLE_RATE_MULTIPLIERS=(1)
# LR min ratios
LR_MIN_RATIO=0.0



# =================================================================
# BASE CONFIGURATION
# =================================================================

# Project name (will be used in wandb)
BASE_PROJECT="diverse_retrieval"
full_finetuning=false            # whether to use full finetuning
all_data=false                  # whether to train on all data
multiple_gpus=true              # whether to use multiple GPUs
save_only_improve=true          # whether to save only improve
save_best_model=true            # whether to save best model
normalize=true
LOG_WITH="wandb"
machine="torch" # greene, torch
pred_length_labels=false

doc_encoder="inf"
base_model_name="llama-8b" # llama-1b, qwen3-4b, llama-3b, llama-8b

mix_one_label_shuffled=false
resume_from_checkpoint=false
use_stateful_dataloader=false
use_l40s=false


MODEL_TYPE="EmbeddingModelSSVariableLeftPad"
MODE="contrastive_one_label_shuffled"

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
elif [ "$MODE" == "contrastive_all_labels_shuffled" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
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
elif [ "$MODE" == "contrastive_all_labels_shuffled_woseq" ]; then
    LOSS_FUNCTION="Contrastive_woseq"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
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
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSS" ]; then
    MODEL_STR="_SS"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING=""
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSSVariable" ]; then
    MODEL_STR="_SSVariable"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING=""
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSSVariableLeftPad" ]; then
    MODEL_STR="_SSVariableLeftPad"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING="--left_padding"
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSSPredLength" ]; then
    MODEL_STR="_SSPredLength"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING="--left_padding"
    PRED_LENGTH="--pred_length"
elif [ "$MODEL_TYPE" == "EmbeddingModelFixed" ]; then
    MODEL_STR="_Fixed"
    SCHEDULE_SAMPLING=""
    LEFT_PADDING=""
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSSVariableLeftPadPredLength" ]; then
    MODEL_STR="_SSVariableLeftPadPredLength"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING="--left_padding"
    PRED_LENGTH="--pred_length"
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

if [ "$base_model_name" = "llama-1b" ]; then
    base_model_prefix=""
else
    base_model_prefix="${base_model_name}_"
fi



if [ "$resume_from_checkpoint" = true ]; then
    RESUME_FROM_CHECKPOINT="--resume_from_checkpoint"
else
    RESUME_FROM_CHECKPOINT=""
fi

if [ "$use_stateful_dataloader" = true ]; then
    USE_STATEFUL_DATALOADER="--use_stateful_dataloader"
else
    USE_STATEFUL_DATALOADER=""
fi

if [ "$mix_one_label_shuffled" = true ]; then
    MIX_ONE_LABEL_SHUFFLED="--mix_one_label_shuffled"
    mix_one_label_shuffled_prefix="mix_one_"
else
    MIX_ONE_LABEL_SHUFFLED=""
    mix_one_label_shuffled_prefix=""
fi

if [ "$pred_length_labels" = true ]; then
    pred_length_labels_str="_pred_length"
else
    pred_length_labels_str=""
fi


# Experiment prefix
EXP_PREFIX="${mix_one_label_shuffled_prefix}${base_model_prefix}${normalize_prefix}ambiguous_qe${GPUS_PREFIX}${FINETUNING_STR}${MODEL_STR}_${MODE}"

# Base directory for saving results
BASE_SAVE_PATH="results/${base_model_name}/ambiguous_qe_${doc_encoder}"

# Training dataset path
BASE_TRAIN_PATH="training_datasets/${base_model_name}/ambiguous_qe/${doc_encoder}/autoregressive_ambiguous_qe_${doc_encoder}_train_dataset_1b_contrastive_2_to_5_ctxs${pred_length_labels_str}/"


# Model checkpoints
if [ "$base_model_name" = "inf" ]; then
    MODEL_ID="infly/inf-retriever-v1-1.5b"
    BASE_ADAPTER_PATH=None
    BASE_LINEAR_CHECKPOINT_PATH=None
elif [ "$base_model_name" = "llama-1b" ]; then
    # Model checkpoints
    MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
    BASE_ADAPTER_PATH="results/llama-1b/nq_${doc_encoder}/toy_contrastive/checkpoint_70000"
    BASE_LINEAR_CHECKPOINT_PATH="results/llama-1b/nq_${doc_encoder}/toy_contrastive/checkpoint_70000_linear.pt"
    # BASE_ADAPTER_PATH="results/llama-1b/nq_inf/toy_qemb/checkpoint_30000"
    # BASE_LINEAR_CHECKPOINT_PATH="results/llama-1b/nq_inf/toy_qemb/checkpoint_30000_linear.pt"
elif [ "$base_model_name" = "qwen3-4b" ]; then
    MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
    BASE_ADAPTER_PATH=None
    BASE_LINEAR_CHECKPOINT_PATH=None
elif [ "$base_model_name" = "llama-3b" ]; then
    MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
    BASE_ADAPTER_PATH=None
    BASE_LINEAR_CHECKPOINT_PATH=None
elif [ "$base_model_name" = "llama-8b" ]; then
    MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
    BASE_ADAPTER_PATH=None
    BASE_LINEAR_CHECKPOINT_PATH=None
else
    echo "Invalid base model name"
    exit 1
fi


# =================================================================
# FIXED HYPERPARAMETERS
# =================================================================
# Model embedding dimension
if [ "$doc_encoder" == "inf" ]; then
    EMBEDDING_MODEL_DIM=1536
else
    EMBEDDING_MODEL_DIM=1024
fi

# How often to save checkpoints (in steps)
SAVE_EVERY_N_STEPS=200

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


if [ "$multiple_gpus" = true ]; then
    # SLURM job time limit
    TIME_LIMIT="8:00:00"
    # Memory per job
    MEMORY="128GB"
    # Number of CPUs per task
    CPUS_PER_TASK=8
    # GPU configuration
    GPU_TYPE="a100"
    GPUS_PER_NODE=4
    GPU_STRING="4"
    PYTHON_COMMAND="accelerate launch train_distributed.py"
else
    # SLURM job time limit
    TIME_LIMIT="4:00:00"
    # Memory per job
    MEMORY="200GB"
    # Number of CPUs per task
    CPUS_PER_TASK=10
    # GPU configuration
    GPU_TYPE="a100"
    GPUS_PER_NODE=1
    GPU_STRING="1"
    PYTHON_COMMAND="python train.py"
fi

# Email for notifications
EMAIL="hc3337@nyu.edu"

# Singularity configuration
if [ "$machine" = "greene" ]; then
    SINGULARITY_IMAGE="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
    CONSTRAINT="h100|a100"
elif [ "$machine" = "torch" ]; then
    SINGULARITY_IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
    if [ "$use_l40s" = true ]; then
        CONSTRAINT="h200|l40s"
    else
        CONSTRAINT="h200"
    fi
    PREEMPTION="#SBATCH --comment=\"preemption=yes;requeue=yes\""
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
    local srm=$8
    
    # Default naming scheme    
    if [ "$use_hard_negatives" = true ]; then
        echo "${prefix}_lr${lr}_temp${temp}_batch${batch}_ep${epochs}_warmup${warmup}_srm${srm}_hn"
    else
        echo "${prefix}_lr${lr}_temp${temp}_batch${batch}_ep${epochs}_warmup${warmup}_srm${srm}"
    fi
} 