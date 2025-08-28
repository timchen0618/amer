#!/bin/bash

# Hyperparameter Search Script
# This script generates SBATCH files for different hyperparameter combinations
# and submits them as separate jobs

# Load configuration
CONFIG_FILE="sbatch_configs/eval/retrieve_ambignq.sh"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
    echo "Loaded configuration from $CONFIG_FILE"
else
    echo "Error: Configuration file $CONFIG_FILE not found!"
    echo "Please create $CONFIG_FILE or copy from gaussian_config.sh"
    exit 1
fi
source sbatch_configs/eval/eval_utils.sh
echo "Loading utils from sbatch_configs/eval/eval_utils.sh"

# Create directories if they don't exist
mkdir -p "$JOB_OUTPUT_DIR"
mkdir -p "$SBATCH_DIR"


# Function to create SBATCH file
create_sbatch_file() {
    local sbatch_file="${SBATCH_DIR}/run_${exp_name}.SBATCH"
    local output_file="${JOB_OUTPUT_DIR}/run_${exp_name}.out"
    
    # Create SBATCH file based on configuration
    cat > "$sbatch_file" << EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEMORY}
#SBATCH --job-name=retrieve_${exp_name}
#SBATCH --mail-type=END
#SBATCH --mail-user=${EMAIL}
#SBATCH --output=${output_file}
#SBATCH --gres=gpu:${GPU_STRING}
${SLURM_EXTRA_ARGS}

SINGULARITY_IMAGE=${SINGULARITY_IMAGE}
OVERLAY_FILE=${OVERLAY_FILE}

# Hyperparameter configuration
ARGS="--data_name $data_name \\
    --training_data_name $training_data_name \\
    --split $split \\
    --retriever $retriever \\
    --dev_data_path $dev_data_path \\
    $gpu_str --num_shards $num_shards \\
    --base_model_id $base_model_id \\
    --adapter_path $adapter_path \\
    --linear_checkpoint_path $linear_checkpoint_path \\
    --base_model_type $base_model \\
    $max_new_tokens_str $compute_loss_str \\
    --top_k_per_query 500 \\
    --top_k 500 \\
    --inference_modes $inference_modes \\
    --output_path $output_path \\
    $google_api"

singularity exec --nv --overlay \${OVERLAY_FILE}:ro \$SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd ${WORK_DIR}; (trap 'kill 0' SIGINT; HF_TOKEN=${HF_TOKEN} python gen_ret_and_eval.py \$ARGS & wait)"
EOF
    
    echo "$sbatch_file"
}



# Create SBATCH file
sbatch_file=$(create_sbatch_file)
# Submit job 
job_id=$(sbatch "$sbatch_file" | awk '{print $4}')
echo "Submitted job with ID: $job_id"
echo "Job output will be in: $JOB_OUTPUT_DIR/run_${exp_name}.out"

echo ""
echo "Evaluation setup complete!"
echo "SBATCH files stored in: $SBATCH_DIR"
echo "Job outputs will be in: $JOB_OUTPUT_DIR"
