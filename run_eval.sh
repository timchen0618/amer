#!/bin/bash
# Hyperparameter Search Script
# This script generates SBATCH files for different hyperparameter combinations
# and submits them as separate jobs
use_sbatch=false
data_type="qampari" # qampari, qampari+ambiguous_qe, ambignq, berds
save_embeddings_str="" # "--save_embeddings"
save_before_aggregation_str="" # "--save_before_aggregation"

data_name="ambiguous_qe"
training_data_name="ambiguous_qe"
suffix="[suffix]"

retriever="inf"
use_best_model=true
compute_loss=false
full_finetuning=false
base_model="llama-1b"  # llama-1b, qwen3-4b, llama-3b, llama-8b
checkpoint_num="70000"

# inference_modes="all first second"
inference_modes="all"
round_robin_percentage=1.0
max_new_tokens=2
num_shards="16"
use_gpu=true



# Set dev_data_path based on data_name
split="dev"  # Define split variable (assuming 'dev' as default)

if [[ "$data_name" == "ambiguous" || "$data_name" == "ambiguous_qe" ]]; then
    dev_data_path="data/questions/${data_name}_${split}_question_only_2_to_5_ctxs.jsonl"
elif [[ "$data_name" == "ambiguous_qe_query_exp" ]]; then
    dev_data_path="data/questions/ambiguous_qe_query_exp_2_to_5_ctxs.jsonl"
elif [[ "$data_name" == "qampari_5_to_8" ]]; then
    dev_data_path="data/questions/qampari_${split}_question_only_5_to_8_ctxs.jsonl"
elif [[ "$data_name" == "qampari_query_exp_5_to_8" ]]; then
    dev_data_path="data/questions/qampari_query_exp_5_to_8_ctxs.jsonl"
elif [[ "$data_name" == "qampari" ]]; then
    dev_data_path="data/questions/qampari_${split}_question_only.jsonl"
elif [[ "$data_name" == "qampari_query_exp" ]]; then
    dev_data_path="data/questions/qampari_query_exp.jsonl"
elif [[ "$data_name" == "nq" || "$data_name" == "msmarco"]]; then
    dev_data_path="data/questions/${data_name}_${split}_question_only.jsonl"
else
    dev_data_path="data/questions/${data_name}_question_only.jsonl"
fi

# set up model paths
if [ "$base_model" = "llama-1b" ]; then
    base_model_id="meta-llama/Llama-3.2-1B-Instruct"
elif [ "$base_model" = "inf" ]; then
    base_model_id="infly/inf-retriever-v1-1.5b"
elif [ "$base_model" = "qwen3-4b" ]; then
    base_model_id="Qwen/Qwen3-4B-Instruct-2507"
elif [ "$base_model" = "llama-3b" ]; then
    base_model_id="meta-llama/Llama-3.2-3B-Instruct"
elif [ "$base_model" = "llama-8b" ]; then
    base_model_id="meta-llama/Llama-3.1-8B-Instruct"
else
    echo "Invalid base model"
    exit 1
fi

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
        base_model_id="results/${base_model}/${training_data_name}_${retriever}/${suffix}/checkpoint_${checkpoint_num}"
        adapter_path=None
    else
        adapter_path="results/${base_model}/${training_data_name}_${retriever}/${suffix}/checkpoint_${checkpoint_num}"
    fi
    linear_checkpoint_path="results/${base_model}/${training_data_name}_${retriever}/${suffix}/checkpoint_${checkpoint_num}_linear.pt"
fi
exp_name="${base_model}_${training_data_name}_${retriever}_${suffix}"

# set up output path
if [ "$round_robin_percentage" != 1.0 ]; then
    output_path="results/${base_model}/${training_data_name}_${retriever}/${suffix}/retrieval_out_${split}_${data_name}_rr_${round_robin_percentage}.jsonl"
else
    output_path="results/${base_model}/${training_data_name}_${retriever}/${suffix}/retrieval_out_${split}_${data_name}.jsonl"
fi
echo "output_path", $output_path

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

if [ "$pred_length" = true ]; then
    pred_length_str="--pred_length"
else
    pred_length_str=""
fi



python gen_ret_and_eval.py --data_name $data_name \
            --training_data_name $training_data_name \
            --split $split \
            --retriever $retriever \
            --dev_data_path $dev_data_path \
            $gpu_str --num_shards $num_shards \
            --base_model_id $base_model_id \
            --adapter_path $adapter_path \
            --linear_checkpoint_path $linear_checkpoint_path \
            --base_model_type $base_model \
            $max_new_tokens_str $compute_loss_str $pred_length_str --round_robin_percentage $round_robin_percentage \
            --top_k_per_query 500 \
            --top_k 500 \
            --inference_modes $inference_modes \
            --output_path $output_path \
            $save_embeddings_str $save_before_aggregation_str