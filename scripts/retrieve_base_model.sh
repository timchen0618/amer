rootdir="/path/to/project"
data_dir="/path/to/data"

python retrieval_base.py \
     --model_name_or_path nvidia/NV-Embed-v2 \
     --passages "[path/to/passsages]" \
     --passages_embeddings "${data_dir}/NV-Embed/qampari_embeddings/*" \
     --data "[path/to/data]"  \
     --output_dir results/base_retrievers/nv-embed/ \
     --projection_size 4096 \
     --per_gpu_batch_size 4 \
     --n_docs 500 \
     --num_shards 2 \
     --output_file "dev_data_gt_qampari_corpus_5_to_8_ctxs.jsonl" 


python retrieval_base.py \
     --model_name_or_path nvidia/NV-Embed-v2 \
     --passages "[path/to/passsages]" \
     --passages_embeddings "${data_dir}/NV-Embed/qampari_embeddings/*" \
     --data "[path/to/data]"  \
     --output_dir results/base_retrievers/nv-embed/ \
     --projection_size 4096 \
     --per_gpu_batch_size 4 \
     --n_docs 500 \
     --num_shards 2 \
     --output_file "ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.json"