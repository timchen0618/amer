import json
from collections import defaultdict

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def write_jsonl(file_path, data_list):
    with open(file_path, 'w') as f:
        for data in data_list:
            f.write(json.dumps(data) + '\n')
    


def reciprocal_rank_fusion(ranked_lists, k=60, id_keys=("doc_id", "document_id", "id")):
    """
    Aggregate multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Args:
        ranked_lists (List[List[Any]]): Each inner list contains documents ordered
            from highest to lowest relevance. A document can be any hashable object or
            a dict that exposes an identifier under one of `id_keys`.
        k (int): RRF constant that dampens the contribution of lower ranked docs.
        id_keys (Tuple[str, ...]): Keys to inspect when extracting identifiers from
            dict-based document representations.

    Returns:
        List[Any]: Documents sorted by their fused RRF scores (highest first).

    Raises:
        ValueError: If a document is unhashable and no identifier can be resolved.
    """

    if not ranked_lists:
        return []
    if k <= 0:
        raise ValueError("Parameter 'k' must be a positive integer.")

    def resolve_doc_id(doc):
        if isinstance(doc, dict):
            for key in id_keys:
                if key in doc:
                    return doc[key]
            raise ValueError(f"Document {doc} does not have an identifier under one of {id_keys}.")
        else:
            raise ValueError(f"Document {doc} is not a dictionary.")
        # try:
        #     hash(doc)
        #     return doc
        # except TypeError as exc:
        #     raise ValueError(
        #         "Document objects must be hashable or expose an identifier "
        #         f"under one of {id_keys}."
        #     ) from exc

    doc_scores = defaultdict(float)
    canonical_docs = {}
    appearance_order = {}
    seen_counter = 0

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        for rank, document in enumerate(ranked_list, start=1):
            doc_id = resolve_doc_id(document)
            doc_scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in canonical_docs:
                canonical_docs[doc_id] = document
                appearance_order[doc_id] = seen_counter
                seen_counter += 1

    sorted_doc_ids = sorted(
        doc_scores.items(),
        key=lambda item: (-item[1], appearance_order[item[0]]),
    )
    return [canonical_docs[doc_id] for doc_id, _ in sorted_doc_ids]

if __name__ == "__main__":
    # for data_type in ["ambiguous_qe", "qampari"]:
    for data_type in ["qampari"]:
        if data_type == "ambiguous_qe":
            questions = read_jsonl(f"data/questions/ambiguous_qe_dev_question_only_2_to_5_ctxs.jsonl")
        elif data_type == "qampari":
            questions = read_jsonl(f"data/questions/qampari_dev_question_only_5_to_8_ctxs.jsonl")
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        lens = [int(l.strip('\n')) for l in open(f'data/mmlf/{data_type}_lens.txt', 'r')]
        for model in ["llama-1b", "llama-3b", "llama-8b", "qwen3-4b"]:
        # for model in ["qwen3-4b"]:
            print(f"Aggregating {data_type} with {model}")
            root_dir = f"results/{model}/{data_type}_inf/"
            results = read_jsonl(f"{root_dir}/single/retrieval_out_dev_{data_type}_mmlf_max_new_tokens_1.jsonl")
            if data_type == "ambiguous_qe":
                org_query_results = read_jsonl(f"{root_dir}/single/retrieval_out_dev_{data_type}_max_new_tokens_1.jsonl")
            elif data_type == "qampari":
                org_query_results = read_jsonl(f"{root_dir}/single/retrieval_out_dev_{data_type}_5_to_8_max_new_tokens_1.jsonl")
            else:
                raise ValueError(f"Invalid data type: {data_type}")
            
            assert len(results) == sum(lens), (len(results), sum(lens))
            assert len(org_query_results) == len(lens), (len(org_query_results), len(lens))
            results_list = []
            out_results = []
            aggregated_results = []
            start_idx = 0
            for i in range(len(lens)):
                # results_list.append([inst['ctxs'] for inst in results[start_idx:start_idx+lens[i]]])
                aggregated_results.append(reciprocal_rank_fusion([org_query_results[i]['ctxs'][:100]] + [inst['ctxs'][:100] for inst in results[start_idx:start_idx+lens[i]]]))
                out_results.append(results[start_idx])
                start_idx += lens[i]

            assert len(aggregated_results) == len(questions), (len(aggregated_results), len(questions))
            for i in range(len(aggregated_results)):
                out_results[i]['ctxs'] = aggregated_results[i]
                out_results[i]['question'] = questions[i]['question'] if 'question' in questions[i] else questions[i]['question_text']
            assert len(aggregated_results) == len(lens), (len(aggregated_results), len(lens))
            
            write_jsonl(f"{root_dir}/single/retrieval_out_dev_{data_type}_mmlf_max_new_tokens_1_aggregated.jsonl", out_results)