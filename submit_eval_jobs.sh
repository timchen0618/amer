#!/bin/bash

# Evaluation Job Submission Script
# This script submits a single evaluation job for all hyperparameter search experiments
# Uses eval.sh as template and populates suffix_list with all completed experiments

# Load configuration
CONFIG_FILE="hypersearch_config_qampari.sh"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
    echo "Loaded configuration from $CONFIG_FILE"
else
    echo "Error: Configuration file $CONFIG_FILE not found!"
    echo "Please create $CONFIG_FILE or copy from hypersearch_config.sh"
    exit 1
fi

# Evaluation-specific configuration
EVAL_DATA_NAME="ambiguous_qe"  # Change this based on your evaluation dataset
EVAL_TRAINING_DATA_NAME="ambiguous_qe"  # Change this based on your training dataset
RETRIEVER_LIST="inf"
USE_GPU="--use_gpu"
NUM_SHARDS="16"
CHECKPOINT_NUM="4001"  # You may want to make this configurable

# Evaluation arguments
MAX_NEW_TOKENS=""  # or "--max_new_tokens 2" if needed
USE_BEST_MODEL="--use_best_model"
COMPUTE_LOSS="--compute_loss"
TOPK_LIST="100 10"
HAS_GOLD_ID="--has-gold-id"

# SLURM configuration for evaluation job
EVAL_TIME_LIMIT="12:00:00"  # Longer time for evaluating multiple experiments
EVAL_MEMORY="200GB"
EVAL_CPUS_PER_TASK=20
EVAL_GPUS_PER_NODE=2

# Output directories for evaluation job
EVAL_SBATCH_DIR="sbatch_eval_jobs"
EVAL_JOB_OUTPUT_DIR="sbatch_eval_outputs"

# Create directories if they don't exist
mkdir -p "$EVAL_SBATCH_DIR"
mkdir -p "$EVAL_JOB_OUTPUT_DIR"

# Function to generate experiment name (same as hyperparameter search)
generate_exp_name() {
    local lr=$1
    local temp=$2
    local batch=$3
    local epochs=$4
    local warmup=$5
    local use_hard_negatives=$6

    if declare -f generate_custom_exp_name > /dev/null; then
        generate_custom_exp_name "$lr" "$temp" "$batch" "$epochs" "$warmup" "$use_hard_negatives"
    else
        echo "hypersearch_lr${lr}_temp${temp}_batch${batch}_ep${epochs}_warmup${warmup}_hn${use_hard_negatives}"
    fi
}

# Function to check if experiment directory exists and has checkpoints
check_experiment_exists() {
    local exp_name=$1
    local exp_dir="${BASE_SAVE_PATH}/${exp_name}"
    
    if [[ -d "$exp_dir" ]]; then
        # Check if there are any checkpoint files
        if ls "$exp_dir"/checkpoint_* 1> /dev/null 2>&1; then
            return 0  # Experiment exists and has checkpoints
        fi
    fi
    return 1  # Experiment doesn't exist or has no checkpoints
}

# Find all completed experiments
echo "Scanning for completed hyperparameter search experiments..."
echo "Results directory: $BASE_SAVE_PATH"

completed_experiments=()
suffix_list=""

# Loop through all hyperparameter combinations to find completed experiments
for lr in "${LEARNING_RATES[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do
        for batch in "${BATCH_SIZES[@]}"; do
            for epochs in "${NUM_EPOCHS_LIST[@]}"; do
                for warmup in "${WARMUP_RATIOS[@]}"; do
                    # Apply same filtering as hyperparameter search
                    if ! filter_combinations "$lr" "$temp" "$batch" "$epochs" "$warmup"; then
                        continue
                    fi
                    
                    # Generate experiment name (same as hyperparameter search)
                    exp_name=$(generate_exp_name "$lr" "$temp" "$batch" "$epochs" "$warmup" "$USE_HARD_NEGATIVES")
                    
                    # Check if experiment exists and has been completed
                    if check_experiment_exists "$exp_name"; then
                        echo "Found completed experiment: $exp_name"
                        completed_experiments+=("$exp_name")
                        
                        # Add to suffix list with hypersearch_ prefix
                        if [[ -z "$suffix_list" ]]; then
                            suffix_list="hypersearch_${exp_name}"
                        else
                            suffix_list="${suffix_list} hypersearch_${exp_name}"
                        fi
                    else
                        echo "Skipping $exp_name - experiment not found or incomplete"
                    fi
                done
            done
        done
    done
done

if [[ ${#completed_experiments[@]} -eq 0 ]]; then
    echo "No completed experiments found!"
    exit 1
fi

echo ""
echo "Found ${#completed_experiments[@]} completed experiments:"
for exp in "${completed_experiments[@]}"; do
    echo "  - $exp"
done

# Create single evaluation SBATCH file
sbatch_file="${EVAL_SBATCH_DIR}/eval_all_experiments.SBATCH"
output_file="${EVAL_JOB_OUTPUT_DIR}/eval_all_experiments.out"

echo ""
echo "Creating evaluation SBATCH file: $sbatch_file"

cat > "$sbatch_file" << EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${EVAL_CPUS_PER_TASK}
#SBATCH --time=${EVAL_TIME_LIMIT}
#SBATCH --mem=${EVAL_MEMORY}
#SBATCH --job-name=eval_hypersearch_all
#SBATCH --mail-type=END
#SBATCH --mail-user=${EMAIL}
#SBATCH --output=${output_file}
#SBATCH --gres=gpu:${EVAL_GPUS_PER_NODE}
#SBATCH --constraint="a100|h100"

SINGULARITY_IMAGE=${SINGULARITY_IMAGE}
OVERLAY_FILE="/scratch/hc3337/envs/nli.ext3"

# Evaluation configuration
data_name="${EVAL_DATA_NAME}"
training_data_name="${EVAL_TRAINING_DATA_NAME}"
suffix_list="${suffix_list}"
retriever_list="${RETRIEVER_LIST}"
use_gpu="${USE_GPU}"
num_shards="${NUM_SHARDS}"
checkpoint_num="${CHECKPOINT_NUM}"
max_new_tokens="${MAX_NEW_TOKENS}"
use_best_model="${USE_BEST_MODEL}"
compute_loss="${COMPUTE_LOSS}"
topk_list="${TOPK_LIST}"
has_gold_id="${HAS_GOLD_ID}"

# Run evaluation inside singularity container
singularity exec --nv --overlay \${OVERLAY_FILE}:ro \$SINGULARITY_IMAGE /bin/bash -c "
source /ext3/env.sh
cd ${WORK_DIR}

echo \"Starting evaluation for experiments: \$suffix_list\"

# Generate retrieval results and evaluate
python gen_ret_and_eval.py --data_name \$data_name \\
                            --training_data_name \$training_data_name \\
                            --suffix_list \$suffix_list \\
                            --retriever_list \$retriever_list \\
                            \$use_gpu --num_shards \$num_shards \\
                            --checkpoint_num \$checkpoint_num \\
                            \$max_new_tokens \$use_best_model \$compute_loss

# Evaluate retrieval results
for suffix in \$suffix_list
do
    echo \"Evaluating retrieval results for \$suffix\"
    for retriever in \$retriever_list
    do  
        ROOT_DIR=\"${WORK_DIR}/results/\$suffix/\"
        echo \"Evaluating retrieval results for \$retriever in \$ROOT_DIR\"
        python eval.py --data-type \$data_name \\
            --root \$ROOT_DIR \\
            --topk \$topk_list \$has_gold_id
    done
done

echo \"Evaluation complete for all experiments\"
"
EOF

echo "SBATCH file created successfully!"

# Submit the job
if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "DRY_RUN mode - job would be submitted but not actually running"
    echo "SBATCH file: $sbatch_file"
    echo "Output file: $output_file"
else
    echo ""
    echo "Submitting evaluation job..."
    job_id=$(sbatch "$sbatch_file" | awk '{print $4}')
    echo "Submitted evaluation job: $job_id"
    echo "Monitor with: squeue -j $job_id"
    echo "Output file: $output_file"
fi

echo ""
echo "Evaluation setup complete!"
echo "Experiments to evaluate: ${#completed_experiments[@]}"
echo "Suffix list: $suffix_list" 