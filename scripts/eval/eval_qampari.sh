#!/bin/bash
# GENERATE ARGS
data_name="qampari_5_to_8"
training_data_name="qampari"
suffix_list="less_SS_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_woseq_lr2e-5_temp0.05_batch32_ep120_warmup0.05/"
file_list="retrieval_out_dev_qampari_5_to_8_max_new_tokens_5.jsonl"
retriever="inf"
base_model="llama-1b"

# EVALUATE ARGS
has_gold_id=false
topk_list="100 10"
# inference_modes="all first second"
inference_modes="all"
# select_indices_file="data/qampari/large_distance_indices_inf.txt"
select_indices_file=""

###############################
### Define strings for args ###
###############################

if [ "$has_gold_id" = true ]; then
    has_gold_id_str="--has-gold-id"
else
    has_gold_id_str=""
fi

if [ "$select_indices_file" != "" ]; then
    select_indices_file_str="--selected-indices-file $select_indices_file"
else
    select_indices_file_str=""
fi

if [ "$file_list" != "" ]; then
    file_list_str="--file-list $file_list"
else
    file_list_str=""
fi


for suffix in $suffix_list
do
    echo "Evaluating retrieval results for $suffix"
    ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${base_model}/${training_data_name}_${retriever}/${suffix}/"
    # ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${suffix}/"
    echo "Evaluating retrieval results for $retriever"
    python eval.py --data-type $data_name \
        --root $ROOT_DIR \
        --topk $topk_list $has_gold_id_str $select_indices_file_str $file_list_str
done

for suffix in $suffix_list
do
    echo "Evaluating retrieval results for $suffix"
    ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${base_model}/${training_data_name}_${retriever}/${suffix}/"
    # ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${suffix}/"
    echo "Evaluating retrieval results for $retriever"
    python eval.py --data-type $data_name \
        --root $ROOT_DIR \
        --topk $topk_list --has-gold-id $select_indices_file_str $file_list_str

done