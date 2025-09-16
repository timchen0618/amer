# Set dev_data_path based on data_name
split="dev"  # Define split variable (assuming 'dev' as default)
sanity_check=true

if [ "$sanity_check" = true ]; then
    sanity_check_str="fixed_model/"
else
    sanity_check_str=""
fi

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
elif [[ "$data_name" == "nq" || "$data_name" == "msmarco" || "$data_name" == "wsd_distinct" ]]; then
    dev_data_path="data/questions/${data_name}_${split}_question_only.jsonl"
else
    dev_data_path="data/questions/${data_name}_question_only.jsonl"
fi

# set up model paths
if [ "$base_model" = "llama-1b" ]; then
    base_model_id="meta-llama/Llama-3.2-1B-Instruct"
elif [ "$base_model" = "inf" ]; then
    base_model_id="infly/inf-retriever-v1-1.5b"
else
    echo "Invalid base model"
    exit 1
fi

if [ "$use_best_model" = true ]; then
    if [ "$full_finetuning" = true ]; then
        base_model_id="results/${base_model}/${training_data_name}_${retriever}/${sanity_check_str}${suffix}/best_model"
        adapter_path=None
    else
        adapter_path="results/${base_model}/${training_data_name}_${retriever}/${sanity_check_str}${suffix}/best_model"
    fi
    linear_checkpoint_path="results/${base_model}/${training_data_name}_${retriever}/${sanity_check_str}${suffix}/best_model_linear.pt"
else
    if [ "$full_finetuning" = true ]; then
        base_model_id="results/${base_model}/${training_data_name}_${retriever}/${sanity_check_str}${suffix}/checkpoint_${checkpoint_num}"
        adapter_path=None
    else
        adapter_path="results/${base_model}/${training_data_name}_${retriever}/${sanity_check_str}${suffix}/checkpoint_${checkpoint_num}"
    fi
    linear_checkpoint_path="results/${base_model}/${training_data_name}_${retriever}/${sanity_check_str}${suffix}/checkpoint_${checkpoint_num}_linear.pt"
fi
exp_name="${base_model}_${training_data_name}_${retriever}_${suffix}"

# set up output path
if [ "$round_robin_percentage" != 1.0 ]; then
    output_path="results/${base_model}/${training_data_name}_${retriever}/${sanity_check_str}${suffix}/retrieval_out_${split}_${data_name}_rr_${round_robin_percentage}.jsonl"
else
    output_path="results/${base_model}/${training_data_name}_${retriever}/${sanity_check_str}${suffix}/retrieval_out_${split}_${data_name}.jsonl"
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