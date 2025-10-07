python test.py --model_paths  [model_paths] \
               --raw_data_dir [raw_data_dir] \
               --embedding_data_dir [embedding_data_dir] \
               --checkpoint_name best_model \
               --max_new_tokens_list 5  \
               --split test \
               --k_values 1 5 10 20 50 100 500 \
               --model_type EmbeddingModel \
               --embedding_model_dim 1024 \
               --full_finetuning 
