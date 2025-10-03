#!/bin/bash
# GENERATE ARGS
data_name="qampari"
training_data_name="qampari"  # ambiguous_qe, qampari+ambiguous_qe
sanity_check_dir=""

suffix_list=(
    "[suffix]"
)
file_list="retrieval_out.jsonl" 


retriever="inf"
base_model="llama-1b" # llama-1b, qwen3-4b, llama-3b, llama-8b

# EVALUATE ARGS
has_gold_id=false
topk_list="100 10"

inference_modes="all"
select_indices_file=""

project_dir="/path/to/project"

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


for suffix in ${suffix_list[@]}
do
    echo "Evaluating retrieval results for $suffix"
    ROOT_DIR="${project_dir}/results/${base_model}/${training_data_name}_${retriever}/${sanity_check_dir}/${suffix}/"

    python eval.py --data-type $data_name \
        --root $ROOT_DIR \
        --topk $topk_list $has_gold_id_str $select_indices_file_str $file_list_str
done