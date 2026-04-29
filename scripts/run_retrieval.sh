#!/bin/bash

# python gen_ret_and_eval.py --split dev \
#             --dev_data_path [dev_data_path] \
#             --output_path [output_path]
#             --base_model_id meta-llama/Llama-3.2-1B-Instruct \
#             --adapter_path [adapter_path] \
#             --linear_checkpoint_path [linear_checkpoint_path] \
#             --base_model_type llama-1b \
#             --num_shards 16 \
#             --max_new_tokens 5 \
#             --top_k_per_query 500 \
#             --top_k 500 \
    
data_dir="/scratch/hc3337/embeddings"
rootdir="/scratch/hc3337"        
python retrieval_inf.py \
     --model_name_or_path checkpoints/enc_trained_qampari_infly_standard_org_q_finetuned_steps5000_t0.05_lr0.00001_ws200_bs256_gradchkpt_refiltered_5to8/checkpoint/best_model/ \
     --passages "/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv" \
     --passages_embeddings "wikipedia_embeddings/standard/*" \
     --data "data/amer_data/eval_data/qampari.jsonl"  \
     --output_dir results/finetuned/standard_org_q/ \
     --projection_size 1536 \
     --per_gpu_batch_size 4 \
     --n_docs 500 \
     --num_shards 16 \
     --use_gpu \
     --output_file qampari.jsonl


