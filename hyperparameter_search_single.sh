#!/bin/bash

# Hyperparameter Search Script
# This script generates SBATCH files for different hyperparameter combinations
# and submits them as separate jobs

# Load configuration
CONFIG_FILE="sbatch_configs/gaussian_config.sh"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
    echo "Loaded configuration from $CONFIG_FILE"
else
    echo "Error: Configuration file $CONFIG_FILE not found!"
    echo "Please create $CONFIG_FILE or copy from gaussian_config.sh"
    exit 1
fi

# Create directories if they don't exist
mkdir -p "$JOB_OUTPUT_DIR"
mkdir -p "$SBATCH_DIR"
mkdir -p "$BASE_SAVE_PATH"

# Function to generate experiment name (use custom function if defined)
generate_exp_name() {
    local lr=$1
    local temp=$2
    local batch=$3
    local epochs=$4
    local warmup=$5
    local use_hard_negatives=$6
    local prefix=$7

    if declare -f generate_custom_exp_name > /dev/null; then
        generate_custom_exp_name "$lr" "$temp" "$batch" "$epochs" "$warmup" "$use_hard_negatives" "$prefix"
    else
        echo "hypersearch_lr${lr}_temp${temp}_batch${batch}_ep${epochs}_warmup${warmup}_hn${use_hard_negatives}"
    fi
}

# Function to create SBATCH file
create_sbatch_file() {
    local exp_name=$1
    local lr=$2
    local temp=$3
    local batch=$4
    local epochs=$5
    local warmup=$6

    local sbatch_file="${SBATCH_DIR}/run_${exp_name}.SBATCH"
    local output_file="${JOB_OUTPUT_DIR}/run_${exp_name}.out"
    local save_path="${BASE_SAVE_PATH}"
    
    # Create SBATCH file based on configuration
    cat > "$sbatch_file" << EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEMORY}
#SBATCH --job-name=hypersearch_${exp_name}
#SBATCH --mail-type=END
#SBATCH --mail-user=${EMAIL}
#SBATCH --output=${output_file}
#SBATCH --gres=gpu:${GPU_STRING}

SINGULARITY_IMAGE=${SINGULARITY_IMAGE}
OVERLAY_FILE=${OVERLAY_FILE}

# Hyperparameter configuration
ARGS="--project ${BASE_PROJECT} \\
      --save_path ${save_path} \\
      --name ${exp_name} \\
      --train_path ${BASE_TRAIN_PATH} \\
      --adapter_path ${BASE_ADAPTER_PATH} \\
      --linear_checkpoint_path ${BASE_LINEAR_CHECKPOINT_PATH} \\
      --lr ${lr} \\
      --temperature ${temp} \\
      --batch_size_training ${batch} \\
      --num_epochs ${epochs} \\
      --warmup_ratio ${warmup} \\
      --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \\
      --weight_decay ${WEIGHT_DECAY} \\
      --scheduler ${SCHEDULER} \\
      --max_grad_norm ${MAX_GRAD_NORM} \\
      --loss_function ${LOSS_FUNCTION} \\
      ${SHUFFLE_SEQUENCE} \\
      ${SAVE_ONLY_IMPROVE} \\
      ${TAKE_FIRST} \\
      ${QUESTION_ONLY} \\
      --embedding_model_dim ${EMBEDDING_MODEL_DIM} \\
      --save_every_n_steps ${SAVE_EVERY_N_STEPS} \\
      --model_type ${MODEL_TYPE} \\
      ${FULL_FINETUNING} \\
      ${SAVE_BEST_MODEL} \\
      ${SCHEDULE_SAMPLING} \\
      ${TRAIN_ON_ALL_DATA} \\
      ${LEFT_PADDING}"

singularity exec --nv --overlay \${OVERLAY_FILE}:ro \$SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd ${WORK_DIR}; (trap 'kill 0' SIGINT; HF_TOKEN=${HF_TOKEN} python train.py \$ARGS & wait)"
EOF
    
    echo "$sbatch_file"
}

# Function to submit job with dependency (optional)
submit_job() {
    local sbatch_file=$1
    local dependency_job_id=$2
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN: Would submit $sbatch_file"
        return 0
    fi
    
    if [[ -n "$dependency_job_id" && "$USE_DEPENDENCIES" == "true" ]]; then
        job_id=$(sbatch --dependency=afterany:$dependency_job_id "$sbatch_file" | awk '{print $4}')
    else
        job_id=$(sbatch "$sbatch_file" | awk '{print $4}')
    fi
    
    echo "$job_id"
}

# Function to calculate total combinations
calculate_total_combinations() {
    local total=0
    for lr in "${LEARNING_RATES[@]}"; do
        for temp in "${TEMPERATURES[@]}"; do
            for batch in "${BATCH_SIZES[@]}"; do
                for epochs in "${NUM_EPOCHS_LIST[@]}"; do
                    for warmup in "${WARMUP_RATIOS[@]}"; do
                        if filter_combinations "$lr" "$temp" "$batch" "$epochs" "$warmup"; then
                            ((total++))
                        fi
                    done
                done
            done
        done
    done
    echo $total
}

# Main hyperparameter search loop
echo "Starting hyperparameter search..."
echo "Configuration loaded from: $CONFIG_FILE"
echo "Using hard negatives: $USE_HARD_NEGATIVES"

total_combinations=$(calculate_total_combinations)
echo "Total combinations to run: $total_combinations"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY_RUN mode enabled - SBATCH files will be created but not submitted"
fi

job_counter=0
submitted_jobs=()

for lr in "${LEARNING_RATES[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do
        for batch in "${BATCH_SIZES[@]}"; do
            for epochs in "${NUM_EPOCHS_LIST[@]}"; do
                for warmup in "${WARMUP_RATIOS[@]}"; do
                    # Apply filtering
                    if ! filter_combinations "$lr" "$temp" "$batch" "$epochs" "$warmup"; then
                        echo "Skipping filtered combination: lr=$lr, temp=$temp, batch=$batch, epochs=$epochs, warmup=$warmup"
                        continue
                    fi
                    
                    exp_name=$(generate_exp_name "$lr" "$temp" "$batch" "$epochs" "$warmup" "$USE_HARD_NEGATIVES" "$EXP_PREFIX")
                    
                    echo "Creating experiment: $exp_name"
                    
                    # Create SBATCH file
                    sbatch_file=$(create_sbatch_file "$exp_name" "$lr" "$temp" "$batch" "$epochs" "$warmup")
                    
                    # Submit job (if not dry run)
                    if [[ "$DRY_RUN" != "true" ]]; then
                        if [[ ${#submitted_jobs[@]} -lt $MAX_CONCURRENT_JOBS ]]; then
                            job_id=$(submit_job "$sbatch_file")
                            submitted_jobs+=("$job_id")
                            echo "  Submitted job $job_id for $exp_name"
                        else
                            # Wait for one of the jobs to finish before submitting new one
                            if [[ "$USE_DEPENDENCIES" == "true" ]]; then
                                oldest_job=${submitted_jobs[0]}
                                submitted_jobs=("${submitted_jobs[@]:1}")  # Remove first element
                                job_id=$(submit_job "$sbatch_file" "$oldest_job")
                                submitted_jobs+=("$job_id")
                                echo "  Submitted job $job_id for $exp_name (waiting for $oldest_job)"
                            else
                                job_id=$(submit_job "$sbatch_file")
                                submitted_jobs+=("$job_id")
                                echo "  Submitted job $job_id for $exp_name"
                            fi
                        fi
                    fi
                    
                    ((job_counter++))
                    
                    # Add delay between submissions
                    if [[ "$DRY_RUN" != "true" && $SUBMISSION_DELAY -gt 0 ]]; then
                        sleep $SUBMISSION_DELAY
                    fi
                done
            done
        done
    done
done

echo ""
echo "Hyperparameter search setup complete!"
echo "Total experiments created: $job_counter"
echo "SBATCH files stored in: $SBATCH_DIR"
echo "Job outputs will be in: $JOB_OUTPUT_DIR"
echo "Results will be saved in: $BASE_SAVE_PATH"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "DRY_RUN mode - no jobs were submitted."
    echo "To actually submit jobs, set DRY_RUN=false in $CONFIG_FILE"
else
    echo ""
    echo "Monitor jobs with: squeue -u \$USER"
    echo "Cancel all jobs with: scancel -u \$USER"
    echo ""
    echo "Submitted job IDs: ${submitted_jobs[*]}"
fi

echo ""
echo "To analyze results after completion, run:"
echo "python analyze_hyperparameter_results.py --results_dir $BASE_SAVE_PATH --outputs_dir $JOB_OUTPUT_DIR" 