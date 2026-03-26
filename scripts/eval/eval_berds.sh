#!/bin/bash

# GENERATE ARGS
compute_only=true

training_data_name="qampari"  # ambiguous_qe, qampari
suffix_list=(
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    
    "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm1"

)   
retriever="inf"
base_model="llama-1b"
sanity_check_dir="fixed_model"



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


for data_name in "arguana_generated" "kialo" "opinionqa"
# for data_name in "kialo"
do
    for suffix in "${suffix_list[@]}"
    do
        ROOT="/scratch/hc3337/projects/autoregressive/results/${base_model}/${training_data_name}_${retriever}/${sanity_check_dir}/${suffix}/"
        PORT=29500
        TOPK=10

        # for DATA in "retrieval_out_dev_arguana_generated.jsonl.mistralpred" "retrieval_out_dev_arguana_generated_single.jsonl.mistralpred" "retrieval_out_dev_arguana_generated_from_2nd_to_3rd.jsonl.mistralpred"
        for DATA in  "retrieval_out_dev_${data_name}_max_new_tokens_2.jsonl.mistralpred" 
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

