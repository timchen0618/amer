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
# BATCH_SIZES=(32)
BATCH_SIZES=(128)

# Number of epochs
# NUM_EPOCHS_LIST=(400)
NUM_EPOCHS_LIST=(3000)

# Warmup ratios
# WARMUP_RATIOS=(0.05 0.1)
WARMUP_RATIOS=(0.05)

# LR min ratios
LR_MIN_RATIO=0.0


# Project name (will be used in wandb)
BASE_PROJECT="diverse_retrieval"
full_finetuning=true

# dataset configurations
transformation_type="diverse_mlps" # linear, diverse_mlps
small=false
hard_strategy="ood" #  "", multi_query, ood, sample_transformation
xlarge=false
normalize=true

# dataset_name="diverse_mlps_multi_query_sm"
multiple_gpus=false              # whether to use multiple GPUs
save_only_improve=true          # whether to save only improve
save_best_model=true            # whether to save best model
all_data=$small
force_sampling=false

MODEL_TYPE="EmbeddingModelSS"


MODE="contrastive_first_label"
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
elif [ "$MODE" == "contrastive_one_label_shuffled" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST="--take_first"
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
elif [ "$MODE" == "contrastive_all_labels_ordered_wo_seq" ]; then
    LOSS_FUNCTION="Contrastive_wo_seq"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""
fi



# Experiment prefix
if [ "$MODEL_TYPE" == "EmbeddingModel" ]; then
    MODEL_STR=""
    SCHEDULE_SAMPLING=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSS" ]; then
    MODEL_STR="_SS"
    SCHEDULE_SAMPLING="--schedule_sampling"
elif [ "$MODEL_TYPE" == "EmbeddingModelSSVariable" ]; then
    MODEL_STR="_SSVariable"
    SCHEDULE_SAMPLING="--schedule_sampling"
elif [ "$MODEL_TYPE" == "EmbeddingModelSSVariableLeftPad" ]; then
    MODEL_STR="_SSVariableLeftPad"
    SCHEDULE_SAMPLING="--schedule_sampling"
elif [ "$MODEL_TYPE" == "EmbeddingModelSSAddQ" ]; then
    MODEL_STR="_SSAddQ"
    SCHEDULE_SAMPLING="--schedule_sampling"
elif [ "$MODEL_TYPE" == "EmbeddingModelSSAvgQ" ]; then
    MODEL_STR="_SSAvgQ"
    SCHEDULE_SAMPLING="--schedule_sampling"
fi

# =================================================================
# BASE CONFIGURATION
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


if [ "$small" = true ]; then
    data_small_suffix="_sm"
    exp_name_large_prefix=""
else
    if [ "$xlarge" = true ]; then
        data_small_suffix="_xl"
        exp_name_large_prefix="xl_"
    else
        data_small_suffix=""
        exp_name_large_prefix="large_"
    fi
fi


if [ "$transformation_type" = "diverse_mlps" ]; then
    transformation_type_suffix="mlps"
elif [ "$transformation_type" = "linear" ]; then
    transformation_type_suffix="linear"
else
    echo "Invalid transformation type"
    exit 1
fi

if [ "$hard_strategy" = "" ]; then
    hard_strategy_suffix=""
elif [ "$hard_strategy" = "multi_query" ]; then
    hard_strategy_suffix="_multi_query"
elif [ "$hard_strategy" = "ood" ]; then
    hard_strategy_suffix="_ood"
elif [ "$hard_strategy" = "sample_transformation" ]; then
    hard_strategy_suffix="_sample_transformation"
else
    echo "Invalid hard strategy"
    exit 1
fi

if [ "$normalize" = true ]; then
    normalize_prefix="normalized_"
    NORMALIZE_STR="--normalize_embeddings"
else
    normalize_prefix=""
    NORMALIZE_STR=""
fi

if [ "$force_sampling" = true ]; then
    FORCE_SAMPLING="--force_sampling"
    force_sampling_prefix="fsampling_"
else
    FORCE_SAMPLING=""
    force_sampling_prefix=""
fi

dataset_name="${transformation_type}${hard_strategy_suffix}${data_small_suffix}"
BASE_SAVE_PATH="results/gaussian_${transformation_type}${hard_strategy_suffix}_inf/"
BASE_TRAIN_PATH="training_datasets/gaussian_${transformation_type}${hard_strategy_suffix}/inf/gaussian_${transformation_type}${hard_strategy_suffix}_train_dataset_1b_contrastive${data_small_suffix}"
EXP_DATA_PREFIX="${force_sampling_prefix}${normalize_prefix}${exp_name_large_prefix}${transformation_type_suffix}_gaussian${hard_strategy_suffix}"


# elif [ "$transformation_type" = "linear" ]; then
#     if [ "$small" = true ]; then
#         BASE_SAVE_PATH="results/gaussian_synthetic_inf/"
#         BASE_TRAIN_PATH="training_datasets/gaussian_synthetic/inf/gaussian_synthetic_train_dataset_1b_contrastive_sm"
#         EXP_DATA_PREFIX="sm_gaussian"
#     else
#         BASE_SAVE_PATH="results/gaussian_synthetic_inf/"
#         BASE_TRAIN_PATH="training_datasets/gaussian_synthetic/inf/gaussian_synthetic_train_dataset_1b_contrastive"
#         EXP_DATA_PREFIX="gaussian"
#     fi
# else
#     echo "Invalid transformation type"
#     exit 1
# fi

# Experiment prefix
EXP_PREFIX="${EXP_DATA_PREFIX}${GPUS_PREFIX}${FINETUNING_STR}${MODEL_STR}_${MODE}"

# Model checkpoints
MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_ID="results/gaussian_diverse_mlps_inf/mlps_gaussian_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep16000_warmup0.05/best_model"

# BASE_ADAPTER_PATH="results/gaussian_diverse_mlps_inf/mlps_gaussian_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep32000_warmup0.05/best_model"
# BASE_LINEAR_CHECKPOINT_PATH="results/gaussian_diverse_mlps_inf/mlps_gaussian_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep16000_warmup0.05/best_model_linear.pt"
BASE_ADAPTER_PATH=None
BASE_LINEAR_CHECKPOINT_PATH=None

# =================================================================
# FIXED HYPERPARAMETERS
# =================================================================

# Model embedding dimension
EMBEDDING_MODEL_DIM=1024

# How often to save checkpoints (in steps)
SAVE_EVERY_N_STEPS=250

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
MAX_CONCURRENT_JOBS=100


if [ "$multiple_gpus" = true ]; then
    # SLURM job time limit
    TIME_LIMIT="24:00:00"
    # Memory per job
    MEMORY="600GB"
    # Number of CPUs per task
    CPUS_PER_TASK=40
    # GPU configuration
    GPU_TYPE="a100"
    GPUS_PER_NODE=4
    GPU_STRING="4"
else
    # SLURM job time limit
    TIME_LIMIT="144:00:00"
    # Memory per job
    MEMORY="300GB"
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
SINGULARITY_IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
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