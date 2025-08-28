#!/bin/bash
data_dir="/scratch/hc3337/embeddings"
rootdir="/scratch/hc3337"

# python retrieval_base.py \
#      --model_name_or_path infly/inf-retriever-v1-1.5b \
#      --passages "${rootdir}/wikipedia_chunks/chunks_v5.tsv" \
#      --passages_embeddings "${data_dir}/inf/qampari_embeddings/*" \
#      --data "${rootdir}/projects/diverse_response/data/qampari_data/nqformat_data/dev_data_gt_qampari_corpus_5_to_8_ctxs.json"  \
#      --output_dir results/base_retrievers/inf/ \
#      --projection_size 1536 \
#      --per_gpu_batch_size 4 \
#      --n_docs 500 \
#      --use_gpu \
#      --num_shards 16 \
#      --output_file "dev_data_gt_qampari_corpus_5_to_8_ctxs.json" \
#      --save_embeddings


# python retrieval_base.py \
#      --model_name_or_path infly/inf-retriever-v1-1.5b \
#      --passages "${rootdir}/wikipedia_chunks/chunks_v5.tsv" \
#      --passages_embeddings "${data_dir}/inf/qampari_embeddings/*" \
#      --data "${rootdir}/projects/autoregressive/data/ambiguous/qampari_embeddings_data/nqformat_data/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.json"  \
#      --output_dir results/base_retrievers/inf/ \
#      --projection_size 1536 \
#      --per_gpu_batch_size 4 \
#      --n_docs 500 \
#      --use_gpu \
#      --num_shards 16 \
#      --output_file "ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.json" \
#      --save_embeddings



# python retrieval_base.py \
#      --model_name_or_path infly/inf-retriever-v1-1.5b \
#      --passages "${rootdir}/projects/autoregressive/data/wsd/distinct/corpus.tsv" \
#      --passages_embeddings "/scratch/hc3337/embeddings/inf/wsd_distinct/*" \
#      --data "${rootdir}/projects/autoregressive/data/wsd/distinct/dev.json"  \
#      --output_dir results/base_retrievers/inf/ \
#      --projection_size 1536 \
#      --per_gpu_batch_size 4 \
#      --n_docs 500 \
#      --use_gpu \
#      --num_shards 1 \
#      --output_file "dev_wsd_distinct.json"




# python retrieval_base.py \
#      --model_name_or_path infly/inf-retriever-v1-1.5b \
#      --passages "${rootdir}/wikipedia_chunks/chunks_v5.tsv" \
#      --passages_embeddings "${data_dir}/inf/qampari_embeddings/*" \
#      --data "${rootdir}/projects/diverse_response/data/qampari_data/nqformat_data/dev_data_gt_qampari_corpus_5_to_8_ctxs_query_exp.json"  \
#      --output_dir results/base_retrievers/inf/ \
#      --projection_size 1536 \
#      --per_gpu_batch_size 4 \
#      --n_docs 500 \
#      --use_gpu \
#      --num_shards 8 \
#      --output_file "dev_data_gt_qampari_corpus_5_to_8_ctxs_query_exp.jsonl"


# python retrieval_base.py \
#      --model_name_or_path infly/inf-retriever-v1-1.5b \
#      --passages "${rootdir}/wikipedia_chunks/chunks_v5.tsv" \
#      --passages_embeddings "${data_dir}/inf/qampari_embeddings/*" \
#      --data "${rootdir}/projects/autoregressive/data/ambiguous/qampari_embeddings_data/nqformat_data/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs_query_exp.json"  \
#      --output_dir results/base_retrievers/inf/ \
#      --projection_size 1536 \
#      --per_gpu_batch_size 4 \
#      --n_docs 500 \
#      --use_gpu \
#      --num_shards 8 \
#      --output_file "ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs_query_exp.jsonl"




# EVALUATE ARGS
has_gold_id=false
# select_indices_file="data/ambiguous/qampari_embeddings_data/small_distance_indices_inf.txt"
topk_list="100 10"
# ['qampari', 'ambiguous', 'ambiguous_qe', 'wsd_distinct']
data_name="qampari"
file_list="dev_data_gt_qampari_corpus_5_to_8_ctxs_reranked_l0.5.jsonl dev_data_gt_qampari_corpus_5_to_8_ctxs_reranked_l0.75.jsonl dev_data_gt_qampari_corpus_5_to_8_ctxs_reranked_l0.9.jsonl dev_data_gt_qampari_corpus_5_to_8_ctxs_reranked_l0.95.jsonl"

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


python eval.py --data-type $data_name \
    --root /scratch/hc3337/projects/autoregressive/results/base_retrievers/inf/ \
    --topk $topk_list $has_gold_id_str $select_indices_file_str --file-list ${file_list}


    