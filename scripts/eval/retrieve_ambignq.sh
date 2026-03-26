#!/bin/bash

# GENERATE ARGS
data_name="ambiguous_qe"
training_data_name="nq"
suffix_list="toy_contrastive"

retriever="inf"
use_best_model=false
compute_loss=false
full_finetuning=false
base_model="llama-1b"
checkpoint_num="70000"

# inference_modes="all first second"
inference_modes="all"
max_new_tokens=2
num_shards="8"
use_gpu=true

# SLURM CONFIGURATION BEGINS
SBATCH_DIR="sbatch_jobs_eval/"
JOB_OUTPUT_DIR="sbatch_outputs_eval/"
TIME_LIMIT="4:00:00"
MEMORY="200GB"
CPUS_PER_TASK=20
GPU_STRING="1"
EMAIL="hc3337@nyu.edu"
# Singularity 
if [ "$machine" = "greene" ]; then
    SINGULARITY_IMAGE="/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
    SLURM_EXTRA_ARGS=""
elif [ "$machine" = "torch" ]; then
    SINGULARITY_IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
    SLURM_EXTRA_ARGS="#SBATCH --requeue
#SBATCH --partition=h200"
else
    echo "Invalid machine"
    exit 1
fi
OVERLAY_FILE="/scratch/hc3337/envs/div.ext3"
HF_TOKEN="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"
WORK_DIR="/scratch/hc3337/projects/autoregressive"
# SLURM CONFIGURATION ENDS


# Set dev_data_path based on data_name
split="dev"  # Define split variable (assuming 'dev' as default)
if [[ "$data_name" == "ambiguous" || "$data_name" == "ambiguous_qe" ]]; then
    dev_data_path="data_creation/raw_data/${data_name}_${split}_question_only_2_to_5_ctxs.jsonl"
elif [[ "$data_name" == "qampari" ]]; then
    dev_data_path="data_creation/raw_data/${data_name}_${split}_question_only_5_to_8_ctxs.jsonl"
elif [[ "$data_name" == "nq" || "$data_name" == "msmarco" || "$data_name" == "wsd_distinct" ]]; then
    dev_data_path="data_creation/raw_data/${data_name}_${split}_question_only.jsonl"
else
    dev_data_path="data_creation/raw_data/${data_name}_question_only.jsonl"
fi

# set up model paths
if [ "$use_best_model" = true ]; then
    if [ "$full_finetuning" = true ]; then
        base_model_id="results/${base_model}/${training_data_name}_${retriever}/${suffix}/best_model"
        adapter_path=None
    else
        adapter_path="results/${base_model}/${training_data_name}_${retriever}/${suffix}/best_model"
    fi
    linear_checkpoint_path="results/${base_model}/${training_data_name}_${retriever}/${suffix}/best_model_linear.pt"
else
    if [ "$full_finetuning" = true ]; then
        base_model_id="results/${base_model}/${training_data_name}_${retriever}/${suffix}/checkpoint_{checkpoint_num}"
        adapter_path=None
    else
        adapter_path="results/${base_model}/${training_data_name}_${retriever}/${suffix}/checkpoint_{checkpoint_num}"
    fi
    linear_checkpoint_path="results/${base_model}/${training_data_name}_${retriever}/${suffix}/checkpoint_{checkpoint_num}_linear.pt"
fi
exp_name="${base_model}|${training_data_name}_${retriever}|${suffix}"
# set up output path
output_path="results/${base_model}/${training_data_name}_${retriever}/${suffix}/retrieval_out_${split}_${data_name}.jsonl"


#
###############################
### Define strings for args ###
###############################

if [ "$use_gpu" = true ]; then
    gpu_str="--use_gpu"
else
    gpu_str=""
fi

if [ "$max_new_tokens" = 0 ]; then
    max_new_tokens_str=""
else
    max_new_tokens_str="--max_new_tokens $max_new_tokens"
fi
echo "max_new_tokens", $max_new_tokens_str

if [ "$use_best_model" = true ]; then
    use_best_model_str="--use_best_model"
else
    use_best_model_str=""
fi

if [ "$compute_loss" = true ]; then
    compute_loss_str="--compute_loss"
else
    compute_loss_str=""
fi

if [ "$full_finetuning" = true ]; then
    full_finetuning_str="--full_finetuning"
else
    full_finetuning_str=""
fi


##############################
### End Definition of args ###
##############################

# python gen_ret_and_eval.py --data_name $data_name \
#                             --training_data_name $training_data_name \
#                             --split $split \
#                             --retriever $retriever \
#                             --dev_data_path $dev_data_path \
#                             $gpu_str --num_shards $num_shards \
#                             --base_model_id $base_model_id \
#                             --adapter_path $adapter_path \
#                             --linear_checkpoint_path $linear_checkpoint_path \
#                             --base_model_type $base_model \
#                             $max_new_tokens_str $compute_loss_str \
#                             --top_k_per_query 500 \
#                             --top_k 500 \
#                             --inference_modes $inference_modes \
#                             --output_path $output_path 
