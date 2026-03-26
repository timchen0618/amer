#!/bin/bash

# GENERATE ARGS
compute_only=false

training_data_name="qampari"
suffix_list="toy_qemb_from_nq"    
retriever="inf"
base_model="llama-1b"

# EVALUATE ARGS
has_gold_id=false
topk_list="100 10"
inference_modes="all"
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

if [ "$compute_only" = true ]; then
    compute_only_str="--compute_only"
else
    compute_only_str=""
fi


# for data_name in "opinionqa" "kialo" "arguana_generated"
for data_name in "kialo"
do
    for suffix in $suffix_list
    do
        ROOT="/scratch/hc3337/projects/autoregressive/results/${base_model}/${training_data_name}_${retriever}/${suffix}/"
        PORT=29500
        TOPK=10

        # for DATA in "retrieval_out_dev_arguana_generated.jsonl.mistralpred" "retrieval_out_dev_arguana_generated_single.jsonl.mistralpred" "retrieval_out_dev_arguana_generated_from_2nd_to_3rd.jsonl.mistralpred"
        for DATA in  "retrieval_out_dev_${data_name}.jsonl" 
        do
            MODEL_SHORT="mistral"
            MODEL_NAME="timchen0618/Mistral_BERDS_evaluator_full"
            OUTPUT="${ROOT}/${DATA}.${MODEL_SHORT}pred"

            PYTHONPATH=.. python berds_eval/eval_vllm.py \
                    --data ${ROOT}/${DATA} \
                    --output_file ${OUTPUT}   \
                    --instructions berds_eval/instructions_chat.txt \
                    --model ${MODEL_NAME}  \
                    --model_type ${MODEL_SHORT} \
                    --topk ${TOPK} \
                    $compute_only_str
        done
    done
done

