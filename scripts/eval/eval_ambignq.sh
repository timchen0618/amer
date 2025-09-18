#!/bin/bash
# GENERATE ARGS
data_name="ambiguous_qe"
training_data_name="ambiguous_qe"  # ambiguous_qe, qampari+ambiguous_qe
sanity_check_dir="fixed_model"
suffix_list=(
    # Only Linear
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr1e-4_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"

    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch8_ep120_warmup0.05_srm1"
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch8_ep120_warmup0.05_srm1"
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch8_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch8_ep120_warmup0.05_srm1"

    # Fixed -> LoRA
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"

    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"

    # fixed model - AmbigNQ  -> LoRA (SRM 10)
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm3"
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm3"
    # "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm3"


    # fixed full finetuning - AmbigNQ
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"

    # Pred Length
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_hungarian_contrastive_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1"
    # ambiguous_qe
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPadPredLength_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"

)
# file_list="retrieval_out_dev_ambiguous_qe_max_new_tokens_1.jsonl" # retrieval_out_dev_ambiguous_qe_max_new_tokens_2.jsonl
file_list="ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.json"
retriever="inf"
base_model="llama-1b"

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


for suffix in ${suffix_list[@]}
do
    echo "Evaluating retrieval results for $suffix"
    # ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${base_model}/${training_data_name}_${retriever}/${sanity_check_dir}/${suffix}/"
    ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/base_retrievers/inf/"
    python eval.py --data-type $data_name \
        --root $ROOT_DIR \
        --topk $topk_list $has_gold_id_str $select_indices_file_str $file_list_str
done