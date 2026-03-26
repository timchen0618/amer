#!/bin/bash
# GENERATE ARGS
data_name="ambiguous_single"
training_data_name="ambiguous_qe"  # ambiguous_qe, qampari+ambiguous_qe

suffix_list=(
    "llama-8b_multi_SS_0"
    "llama-8b_multi_SS_1"
    "llama-8b_multi_SS_2"
)
file_list="retrieval_out_dev_ambigqa_single_max_new_tokens_2.jsonl" # retrieval_out_dev_ambiguous_qe_max_new_tokens_2.jsonl


retriever="inf"
# base_model="llama-8b" # llama-1b, qwen3-4b, llama-3b, llama-8b

# EVALUATE ARGS
has_gold_id=false
topk_list="100 10"
# inference_modes="all first second"
inference_modes="all"
# select_indices_file="data/ambiguous/qampari_embeddings_data/large_distance_indices_inf.txt"
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

for base_model in "llama-8b" 
do
    for suffix in ${suffix_list[@]}
    do
        # base_model="${suffix%%_*}"
        echo "Evaluating retrieval results for $suffix | base_model: $base_model"
        ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${base_model}/${training_data_name}_${retriever}/${suffix}/"
        # ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/base_retrievers/$suffix/"
        python eval.py --data-type $data_name \
            --root $ROOT_DIR \
            --topk $topk_list $has_gold_id_str $select_indices_file_str $file_list_str
    done
done