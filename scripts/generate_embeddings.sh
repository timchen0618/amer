python generate_embeddings.py \
    --model_name_or_path [document_encoder_id] \
    --passages [passages_path]  \
    --output_dir [output_dir] \
    --per_gpu_batch_size 8 \
    --shard_id $1 \
    --shard_size 100000
