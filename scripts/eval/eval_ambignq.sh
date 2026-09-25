#!/bin/bash
# GENERATE ARGS
data_name="ambiguous_qe"
training_data_name="ambigqa"  # ambiguous_qe, qampari+ambiguous_qe

suffix_list=(
    "base_model"
    "frozenDocEnc_singlequery"
    "frozenDocEnc_causal"
    "frozenDocEnc_singlequery_lora"
    "frozenDocEnc_causal_lora"
    "standard"
    "multi_hungarian"
)
retriever="inf"

# EVALUATE ARGS
has_gold_id=false
topk_list="100 10"
# select_indices_file="data/ambiguous/qampari_embeddings_data/large_distance_indices_inf.txt"
select_indices_file=""

### Define strings for args ###
if [ "$has_gold_id" = true ]; then
    has_gold_id_str=""
else
    has_gold_id_str="--no-gold-id"
fi

if [ "$select_indices_file" != "" ]; then
    select_indices_file_str="--selected-indices-file $select_indices_file"
else
    select_indices_file_str=""
fi

### Evaluate ###
for suffix in ${suffix_list[@]}
do
    echo "Evaluating retrieval results for $suffix"
    if [ "$suffix" == "base_model" ]; then
        # Untrained infly/inf-retriever-v1-1.5b, already computed by scripts/retrieve_base_model.sh
        ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/base_retrievers/inf/amer_data/"
    else
        ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/finetuned/${training_data_name}/${suffix}/"
    fi
    python eval.py --data_path data/amer_data/eval_data/${training_data_name}.jsonl \
    --topk $topk_list $has_gold_id_str $select_indices_file_str \
    --input-file $ROOT_DIR/${training_data_name}.jsonl
    # --input-file results/base_retrievers/inf/amer_data/${training_data_name}.jsonl
        
done
