#!/bin/bash

# GENERATE ARGS
data_name="qampari"
training_data_name="qampari"
suffix_list=$1
# inf_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch8_ep60_warmup0.05
# inf_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch8_ep60_warmup0.05
# inf_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch8_ep60_warmup0.05


retriever_list="inf"
use_gpu=true
use_best_model=true
compute_loss=true
full_finetuning=true

# EVALUATE ARGS
has_gold_id=false
topk_list="100 10"
inference_modes="all first second"
select_indices_file=""
max_new_tokens=0
num_shards="16"
checkpoint_num="70000"

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

if [ "$has_gold_id" = true ]; then
    has_gold_id_str="--has-gold-id"
else
    has_gold_id_str=""
fi

if [ "$full_finetuning" = true ]; then
    full_finetuning_str="--full_finetuning"
else
    full_finetuning_str=""
fi

if [ "$select_indices_file" != "" ]; then
    select_indices_file_str="--selected-indices-file $select_indices_file"
else
    select_indices_file_str=""
fi

##############################
### End Definition of args ###
##############################

python gen_ret_and_eval.py --data_name $data_name \
                            --training_data_name $training_data_name \
                            --suffix_list $suffix_list \
                            --retriever_list $retriever_list \
                            $gpu_str --num_shards $num_shards \
                            --checkpoint_num $checkpoint_num \
                            $max_new_tokens_str $use_best_model_str $compute_loss_str $full_finetuning_str \
                            --inference_modes $inference_modes \
                            --top_k_per_query 500 \
                            --top_k 500

for suffix in $suffix_list
do
    echo "Evaluating retrieval results for $suffix"
    for retriever in $retriever_list
    do  
        ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${training_data_name}_${retriever}/${suffix}/"
        # ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${suffix}/"
        echo "Evaluating retrieval results for $retriever"
        python eval.py --data-type $data_name \
            --root $ROOT_DIR \
            --topk $topk_list $has_gold_id_str $select_indices_file_str
    done
done