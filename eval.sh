#!/bin/bash

# GENERATE ARGS
data_name="ambiguous_qe"
training_data_name="ambiguous_qe"
# suffix_list="hypersearch_lr1e-4_temp0.05_batch16_ep20_warmup0.05 hypersearch_lr1e-4_temp0.05_batch16_ep10_warmup0.05"
suffix_list="toy_contrastive_from_stage2_lr2e5_ep20_temp0.05_warmup0.05"
retriever_list="inf"
use_gpu="--use_gpu"
num_shards="16"
checkpoint_num="1501"


# max_new_tokens="--max_new_tokens 2"
max_new_tokens=""

# use_best_model="--use_best_model"
use_best_model=""

compute_loss="--compute_loss"
# compute_loss=""

# EVALUATE ARGS
topk_list="100 10"
# has_gold_id="--has-gold-id"
has_gold_id=""


python gen_ret_and_eval.py --data_name $data_name \
                            --training_data_name $training_data_name \
                            --suffix_list $suffix_list \
                            --retriever_list $retriever_list \
                            $use_gpu --num_shards $num_shards \
                            --checkpoint_num $checkpoint_num \
                            $max_new_tokens $use_best_model $compute_loss

for suffix in $suffix_list
do
    echo "Evaluating retrieval results for $suffix"
    for retriever in $retriever_list
    do  
        ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${training_data_name}_${retriever}/${suffix}/"
        # ROOT_DIR="/scratch/hc3337/projects/autoregressive/results/${suffix}/"
        echo "Evaluating retrieval results for $retriever"
        python eval.py --data-type $data_name \
            --root $ROOT_DIR \
            --topk $topk_list $has_gold_id
    done
done