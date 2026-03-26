#!/bin/bash
# GENERATE ARGS
data_name="qampari_5_to_8"
training_data_name="qampari"  # qampari, qampari+ambiguous_qe
sanity_check_str=""
suffix_list=(
    # best model before detach()
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep60_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep60_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"

    
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch8_ep60_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch8_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch8_ep60_warmup0.05_srm10"

    # SSVariableLeftPad, fixing detach() -> Only Linear
    # 4.8 "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch8_ep60_warmup0.05_srm10"
    # 5.7 "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch8_ep60_warmup0.05_srm10"
    # 5.1 "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch8_ep60_warmup0.05_srm10"
    # 5.1 # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch8_ep60_warmup0.05_srm10"

    # "normalized_qampari_4gpu_full_finetuning_Fixed_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch8_ep60_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_Fixed_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch8_ep60_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_Fixed_hungarian_contrastive_lr2e-5_temp0.05_batch8_ep60_warmup0.05_srm10"

    # Fixed Model -> LoRA
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm1"

    # Fixed Model - QAMPARI  -> LoRA (SRM 10) #####
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    
    # "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"


    # Fixed full finetuning - QAMPARI
    # "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    
    # Pred Length
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_hungarian_contrastive_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1"
    # qampari
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    
    # STELLA
    # "mix_one_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"

    # qwen3-4b, qampari
    # "qwen3-4b_normalized_qampari_4gpu_SSVariableLeftPad_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch8_ep120_warmup0.05_srm10"
    # "mix_one_qwen3-4b_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch8_ep120_warmup0.05_srm10"
    
    # # llama-3b, qampari
    # "llama-3b_normalized_qampari_4gpu_SSVariableLeftPad_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch8_ep120_warmup0.05_srm10"
    # "mix_one_llama-3b_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch8_ep120_warmup0.05_srm10"

    # # llama-8b, qampari
    # "llama-8b_normalized_qampari_4gpu_SSVariableLeftPad_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch8_ep120_warmup0.05_srm10"
    # "mix_one_llama-8b_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch8_ep120_warmup0.05_srm10"

    # # FROM BASE Inf, QAMPARI
    # "normalized_qampari_4gpu_SSVariableLeftPad_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # # FROM BASE Stella, QAMPARI
    # "normalized_qampari_4gpu_SSVariableLeftPad_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"

    # "mix_one_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr2e-4_temp0.05_batch32_ep150_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr2e-4_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr1e-4_temp0.05_batch32_ep150_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr1e-4_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "mix_one_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
    # normalized_qampari_4gpu_SSVariableLeftPad_contrastive_one_label_shuffled_lr2e-4_temp0.05_batch32_ep150_warmup0.05_srm10
    # "single"

    
    # Force Sampling
    # llama-1b
    # "force_sampling_mix_one_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"

    # # llama-3b
    # "force_sampling_mix_one_llama-3b_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch8_ep120_warmup0.05_srm10"
    # "force_sampling_mix_one_llama-3b_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr1e-4_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "force_sampling_mix_one_llama-3b_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"

    # # qwen3-4b
    # "force_sampling_mix_one_qwen3-4b_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch8_ep120_warmup0.05_srm10"
    # "force_sampling_mix_one_qwen3-4b_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr1e-4_temp0.05_batch32_ep120_warmup0.05_srm10"
    # "force_sampling_mix_one_qwen3-4b_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10"

    # # llama-8b
    # "force_sampling_mix_one_llama-8b_normalized_qampari_4gpu_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch8_ep120_warmup0.05_srm10"

    # "llama-1b_multi_SS_0"
    # "llama-1b_multi_SS_1"
    # "llama-1b_multi_SS_2"
    # "llama-1b_multi_sampling_0"
    # "llama-1b_multi_sampling_1"
    # "llama-1b_multi_sampling_2"

    # "llama-3b_multi_SS_0"
    # "llama-3b_multi_SS_1"
    # "llama-3b_multi_SS_2"
    # "llama-3b_multi_sampling_0"
    # "llama-3b_multi_sampling_1"
    # "llama-3b_multi_sampling_2"

    # "qwen3-4b_multi_SS_0"
    # "qwen3-4b_multi_SS_1"
    # "qwen3-4b_multi_SS_2"
    # "qwen3-4b_multi_sampling_0"
    # "qwen3-4b_multi_sampling_1"
    # "qwen3-4b_multi_sampling_2"

    # "llama-8b_multi_SS_0"
    # "llama-8b_multi_SS_1"
    # "llama-8b_multi_SS_2"
    # "llama-8b_multi_sampling_0"
    # "llama-8b_multi_sampling_1"
    # "llama-8b_multi_sampling_2"

    # "llama-1b_single_0"
    # "llama-1b_single_1"
    # "llama-1b_single_2"

    # "llama-3b_single_0"
    # "llama-3b_single_1"
    # "llama-3b_single_2"

    # "qwen3-4b_single_0"
    # "qwen3-4b_single_1"
    # "qwen3-4b_single_2"

    # "llama-8b_single_0"
    # "llama-8b_single_1"
    # "llama-8b_single_2"

    # "multi_SS"
    # "multi_sampling"
    "inf"

)

file_list="dev_data_gt_qampari_corpus_5_to_8_ctxs.jsonl"  # retrieval_out_dev_qampari_5_to_8_max_new_tokens_1.jsonl
# file_list="retrieval_out_dev_qampari_5_to_8_max_new_tokens_1_reranked_l0.5.jsonl retrieval_out_dev_qampari_5_to_8_max_new_tokens_1_reranked_l0.75.jsonl retrieval_out_dev_qampari_5_to_8_max_new_tokens_1_reranked_l0.9.jsonl"
retriever="inf"
# base_model="qwen3-4b" # llama-1b, qwen3-4b, llama-3b, llama-8b

# EVALUATE ARGS
has_gold_id=false
topk_list="100 10"
# inference_modes="all first second"
inference_modes="all"
# select_indices_file="data/qampari_5_to_8/large_distance_indices_inf.txt"
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


# for suffix in "${suffix_list[@]}"
# do
#     echo "Evaluating retrieval results for $suffix"
#     ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${base_model}/${training_data_name}_${retriever}/sanity_check/${suffix}/"
#     # ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${suffix}/"
#     echo "Evaluating retrieval results for $retriever"
#     python eval.py --data-type $data_name \
#         --root $ROOT_DIR \
#         --topk $topk_list $has_gold_id_str $select_indices_file_str $file_list_str
# done


for suffix in "${suffix_list[@]}"
do
    base_model="${suffix%%_*}"
    echo "Evaluating retrieval results for $suffix | base_model: $base_model"
    # ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${base_model}/${training_data_name}_${retriever}/${sanity_check_str}${suffix}/"
    ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/base_retrievers/inf/"
    echo "Evaluating retrieval results for $retriever"
    python eval.py --data-type $data_name \
        --root $ROOT_DIR \
        --topk $topk_list --has-gold-id $select_indices_file_str $file_list_str

done