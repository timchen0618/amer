
data_dir="/scratch/hc3337/embeddings"
rootdir="/scratch/hc3337"


# python retrieval_base.py \
#      --model_name_or_path /scratch/hc3337/models/iterative_retrieval/infly-finetuned/ \
#      --passages "${rootdir}/wikipedia_chunks/chunks_v5.tsv" \
#      --passages_embeddings "/scratch/hc3337/embeddings/iterative_retrieval/infly-finetuned/qampari_embeddings/*" \
#      --data "/scratch/hc3337/projects/Search-R1/outputs_qampari/last_query/output_infly_finetuned_RL.jsonl" \
#      --output_dir results/base_retrievers/infly/ \
#      --projection_size 1536 \
#      --per_gpu_batch_size 4 \
#      --n_docs 500 \
#      --num_shards 16 \
#      --use_gpu \
#      --output_file qampari_last_queries_infly_finetuned_searchr1_RL.jsonl

python retrieval_base.py \
     --model_name_or_path /scratch/hc3337/models/iterative_retrieval/qwen3-finetuned/ \
     --passages "${rootdir}/wikipedia_chunks/chunks_v5.tsv" \
     --passages_embeddings "/scratch/hc3337/embeddings/iterative_retrieval/qwen3-finetuned/qampari_embeddings/*" \
     --data "/scratch/hc3337/projects/Search-R1/outputs_qampari/last_query/output_qwen3-0.6b_finetuned_RL.jsonl" \
     --output_dir results/base_retrievers/qwen3-0.6b/ \
     --projection_size 1024 \
     --per_gpu_batch_size 4 \
     --n_docs 500 \
     --num_shards 16 \
     --use_gpu \
     --output_file qampari_last_queries_qwen3-0.6b_finetuned_searchr1_RL.jsonl


# python retrieval_base.py \
#      --model_name_or_path infly/inf-retriever-v1-1.5b \
#      --passages "${rootdir}/wikipedia_chunks/chunks_v5.tsv" \
#      --passages_embeddings "${data_dir}/inf/qampari_embeddings/*" \
#      --data "/scratch/hc3337/projects/BrowseComp-Plus/webqsp_runs/infly/tongyi_multi/webqsp_last_queries.jsonl" \
#      --output_dir results/base_retrievers/infly/ \
#      --projection_size 1536 \
#      --per_gpu_batch_size 4 \
#      --n_docs 500 \
#      --num_shards 16 \
#      --use_gpu \
#      --output_file webqsp_last_queries_tongyi_multi.jsonl

# python retrieval_base.py \
#      --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
#      --passages "${rootdir}/wikipedia_chunks/chunks_v5.tsv" \
#      --passages_embeddings "${data_dir}/qwen3-0.6b/qampari_embeddings/*" \
#      --data "/scratch/hc3337/projects/BrowseComp-Plus/webqsp_runs/qwen3-0.6b/tongyi_multi/webqsp_last_queries.jsonl"  \
#      --output_dir results/base_retrievers/qwen3-0.6b/ \
#      --projection_size 1024 \
#      --per_gpu_batch_size 4 \
#      --n_docs 500 \
#      --num_shards 16 \
#      --use_gpu \
#      --output_file webqsp_last_queries_tongyi_multi.jsonl


# python retrieval_base.py \
#      --model_name_or_path infly/inf-retriever-v1-1.5b \
#      --passages "${rootdir}/wikipedia_chunks/chunks_v5.tsv" \
#      --passages_embeddings "${data_dir}/inf/qampari_embeddings/*" \
#      --data "/scratch/hc3337/projects/Search-R1/outputs_webqsp/last_query/output_infly_webqsp.jsonl"  \
#      --output_dir results/base_retrievers/infly/ \
#      --projection_size 1536 \
#      --per_gpu_batch_size 4 \
#      --n_docs 500 \
#      --num_shards 16 \
#      --use_gpu \
#      --output_file webqsp_last_queries_searchr1.jsonl

# python retrieval_base.py \
#      --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
#      --passages "${rootdir}/wikipedia_chunks/chunks_v5.tsv" \
#      --passages_embeddings "${data_dir}/qwen3-0.6b/qampari_embeddings/*" \
#      --data "/scratch/hc3337/projects/Search-R1/outputs_webqsp/last_query/output_qwen3-0.6b_webqsp.jsonl"  \
#      --output_dir results/base_retrievers/qwen3-0.6b/ \
#      --projection_size 1024 \
#      --per_gpu_batch_size 4 \
#      --n_docs 500 \
#      --num_shards 16 \
#      --use_gpu \
#      --output_file webqsp_last_queries_searchr1.jsonl

