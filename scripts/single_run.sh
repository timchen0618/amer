#!/bin/bash

# Single Training Run Script
# This script runs a single training job without SLURM
# Uses default parameters from the hyperparameter search configuration

# Load configuration
CONFIG_FILE="sbatch_configs/qampari_config.sh"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
    echo "Loaded configuration from $CONFIG_FILE"
else
    echo "Error: Configuration file $CONFIG_FILE not found!"
    echo "Please create $CONFIG_FILE or copy from qampari_config.sh"
    exit 1
fi

# Set default hyperparameters (using first values from arrays)
LEARNING_RATE=${LEARNING_RATES[0]}
TEMPERATURE=${TEMPERATURES[0]}
BATCH_SIZE=${BATCH_SIZES[0]}
NUM_EPOCHS=${NUM_EPOCHS_LIST[0]}
WARMUP_RATIO=${WARMUP_RATIOS[0]}

# Generate experiment name
EXP_NAME="${EXP_PREFIX}_lr${LEARNING_RATE}_temp${TEMPERATURE}_batch${BATCH_SIZE}_ep${NUM_EPOCHS}_warmup${WARMUP_RATIO}"

# Set paths
SAVE_PATH="${BASE_SAVE_PATH}"
OUTPUT_LOG="single_run_output.log"

# Create save directory
mkdir -p "$SAVE_PATH"

echo "Starting single training run..."
echo "Experiment name: $EXP_NAME"
echo "Save path: $SAVE_PATH"
echo "Output log: $OUTPUT_LOG"
echo ""


# Build training arguments
ARGS="--project ${BASE_PROJECT} \
      --save_path ${SAVE_PATH} \
      --name ${EXP_NAME} \
      --train_path ${BASE_TRAIN_PATH} \
      --adapter_path ${BASE_ADAPTER_PATH} \
      --linear_checkpoint_path ${BASE_LINEAR_CHECKPOINT_PATH} \
      --model_id ${MODEL_ID} \
      --lr ${LEARNING_RATE} \
      --temperature ${TEMPERATURE} \
      --batch_size_training ${BATCH_SIZE} \
      --num_epochs ${NUM_EPOCHS} \
      --warmup_ratio ${WARMUP_RATIO} \
      --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
      --weight_decay ${WEIGHT_DECAY} \
      --scheduler ${SCHEDULER} \
      --max_grad_norm ${MAX_GRAD_NORM} \
      --loss_function ${LOSS_FUNCTION} \
      --lr_min_ratio ${LR_MIN_RATIO} \
      ${SHUFFLE_SEQUENCE} \
      ${SAVE_ONLY_IMPROVE} \
      ${TAKE_FIRST} \
      ${QUESTION_ONLY} \
      --embedding_model_dim ${EMBEDDING_MODEL_DIM} \
      --save_every_n_steps ${SAVE_EVERY_N_STEPS} \
      --model_type ${MODEL_TYPE} \
      ${FULL_FINETUNING} \
      ${SAVE_BEST_MODEL} \
      ${SCHEDULE_SAMPLING} \
      ${TRAIN_ON_ALL_DATA} \
      ${LEFT_PADDING} \
      ${NORMALIZE_STR} \
      ${FORCE_SAMPLING} \
      --log_with ${LOG_WITH}"

echo "Training arguments:"
echo "$ARGS"
echo ""

# Set HuggingFace token if provided
if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN="$HF_TOKEN"
fi

# Run training
echo "Starting training at $(date)"
echo "Logging output to: $OUTPUT_LOG"

# Use accelerate launch for distributed training support
HF_TOKEN="$HF_TOKEN" PYTHONPATH=. python train.py $ARGS 2>&1 | tee "$OUTPUT_LOG"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "Training completed at $(date)"
echo "Exit code: $EXIT_CODE"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Training completed successfully!"
    echo "Results saved in: $SAVE_PATH"
    echo "Output log saved as: $OUTPUT_LOG"
else
    echo "Training failed with exit code: $EXIT_CODE"
    echo "Check the output log for details: $OUTPUT_LOG"
fi