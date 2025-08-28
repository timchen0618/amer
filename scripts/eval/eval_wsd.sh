#!/bin/bash

# GENERATE ARGS
data_name="wsd_distinct"
training_data_name="qampari"
suffix_list="toy_qemb_from_nq"

retriever="inf"
base_model="llama-1b"

# EVALUATE ARGS
has_gold_id=true
topk_list="100 10"
inference_modes="all" # all, first, second
# select_indices_file="data/ambiguous/qampari_embeddings_data/small_distance_indices_inf.txt"
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
    python eval.py --data-type $data_name \
        --root $ROOT_DIR \
        --topk $topk_list $has_gold_id_str $select_indices_file_str
done