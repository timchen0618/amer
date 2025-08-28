#!/bin/bash

# GENERATE ARGS
# for data_name in "opinionqa" "kialo" "arguana_generated"
for data_name in "kialo"
do
    training_data_name="nq"
    # suffix_list="normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05 normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05"
    # suffix_list="normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05"
    suffix_list="toy_contrastive"


    compute_only=false

    retriever_list="inf"
    use_gpu=true
    use_best_model=true
    compute_loss=false
    full_finetuning=true
    base_model="llama-1b"

    # EVALUATE ARGS
    has_gold_id=false
    topk_list="100 10"
    inference_modes="all"
    # select_indices_file="data/ambiguous/qampari_embeddings_data/small_distance_indices_inf.txt"
    select_indices_file=""

    max_new_tokens=2
    num_shards="8"
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

    if [ "$base_model" = "llama-1b" ]; then
        base_model_type_str="--base_model_type llama-1b"
    elif [ "$base_model" = "inf" ]; then
        base_model_type_str="--base_model_type inf"
    fi

    if [ "$compute_only" = true ]; then
        compute_only_str="--compute_only"
    else
        compute_only_str=""
    fi

    ##############################
    ### End Definition of args ###
    ##############################

    # python gen_ret_and_eval.py --data_name $data_name \
    #                             --training_data_name $training_data_name \
    #                             --suffix_list $suffix_list \
    #                             --retriever_list $retriever_list \
    #                             $gpu_str --num_shards $num_shards \
    #                             --checkpoint_num $checkpoint_num \
    #                             $max_new_tokens_str $use_best_model_str $compute_loss_str $full_finetuning_str \
    #                             --inference_modes $inference_modes \
    #                             --top_k_per_query 500 \
    #                             --top_k 500 \
    #                             --google_api \
    #                             $base_model_type_str

    
    for suffix in $suffix_list
    do
        ROOT="/scratch/hc3337/projects/autoregressive/results/${base_model}/${training_data_name}_${retriever_list}/${suffix}/"
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

