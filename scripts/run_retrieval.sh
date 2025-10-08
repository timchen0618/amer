#!/bin/bash

python gen_ret_and_eval.py --split dev \
            --dev_data_path [dev_data_path] \
            --output_path [output_path]
            --base_model_id meta-llama/Llama-3.2-1B-Instruct \
            --adapter_path [adapter_path] \
            --linear_checkpoint_path [linear_checkpoint_path] \
            --base_model_type llama-1b \
            --num_shards 16 \
            --max_new_tokens 5 \
            --top_k_per_query 500 \
            --top_k 500 \
            



