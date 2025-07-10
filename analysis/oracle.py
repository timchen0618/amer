import numpy as np

import os
import json
import glob
import time

from pathlib import Path

from retrieval_utils import Indexer, add_passages, load_passages, index_encoded_data

import structlog
logger = structlog.get_logger()



def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def aggregate_different_queries_by_length(top_ids_and_scores, lengths=None, MAX_LATENTS=None, top_k=100, aggregate_start_idx=0, aggregate_end_idx=None):
    # aggregate top_ids_and_scores for different queries
    if MAX_LATENTS is not None:
        assert len(top_ids_and_scores) % (MAX_LATENTS) == 0, (len(top_ids_and_scores), MAX_LATENTS)
        assert lengths is None, (lengths, MAX_LATENTS)
        lengths = [MAX_LATENTS] * (len(top_ids_and_scores) // MAX_LATENTS)
    else:
        assert len(top_ids_and_scores) == sum(lengths), (len(top_ids_and_scores), sum(lengths))
    
    aggregated_top_ids_and_scores = []
    start_idx = 0
    for i in range(len(lengths)):
        aggregated_top_ids_and_scores_per_inst = []
        ids_and_scores_to_aggregate = top_ids_and_scores[start_idx:start_idx+lengths[i]]
        if aggregate_end_idx is not None:
            ids_and_scores_to_aggregate = ids_and_scores_to_aggregate[aggregate_start_idx:aggregate_end_idx]
        else:
            ids_and_scores_to_aggregate = ids_and_scores_to_aggregate[aggregate_start_idx:]
        # print('lens', len(ids_and_scores_to_aggregate))
        start_idx += lengths[i]
        # aggregate ids and scores to be a single list, and avoid duplicates
        # take from each list in a round-robin manner until reaches top_k
        # put them into aggregated_top_ids_and_scores, which follows the format of top_ids_and_scores
        seen_ids = set()
    
        # Find the maximum length among all lists
        max_len = max(len(list(zip(*sublist))) for sublist in ids_and_scores_to_aggregate)
        # Round-robin aggregation
        for idx in range(max_len):
            for query_results in ids_and_scores_to_aggregate:
                query_results = list(zip(*query_results))
                # Skip if we've processed all items from this query
                if idx >= len(query_results):
                    continue
                # print(query_results[idx])
                current_id, current_score = query_results[idx]
                
                # Only add if we haven't seen this ID before
                if current_id not in seen_ids:
                    aggregated_top_ids_and_scores_per_inst.append((current_id, current_score))
                    seen_ids.add(current_id)
                
                if len(aggregated_top_ids_and_scores_per_inst) >= top_k:
                    break
            if len(aggregated_top_ids_and_scores_per_inst) >= top_k:
                break
        aggregated_top_ids_and_scores_per_inst = list(zip(*aggregated_top_ids_and_scores_per_inst))
        aggregated_top_ids_and_scores.append(aggregated_top_ids_and_scores_per_inst)
    
    assert start_idx == len(top_ids_and_scores), (start_idx, len(top_ids_and_scores))
    return aggregated_top_ids_and_scores

        
def main_test(passages_embeddings, passages_path, output_path, 
              raw_data_path = '/scratch/hc3337/projects/autoregressive/data/wsd/distinct/train.jsonl', 
              data_path = 'out.npy', lengths_path = "", embedding_size = 4096, top_k_per_query = 100, top_k = 100,
              start_idx = 0, end_idx = None, MAX_LATENTS = None, aggregate_start_idx = 0, aggregate_end_idx = None):
    
    logger.info('loading question embeddings and attempt to retrieve from %s', data_path)
    question_embeddings = np.load(data_path)
    logger.info('question embeddings shape: %s', question_embeddings.shape)
    

    logger.info('loading data from %s', raw_data_path)
    if end_idx is None:
        data = read_jsonl(raw_data_path)[start_idx:]
    else:
        data = read_jsonl(raw_data_path)[start_idx:end_idx]
    logger.info('length of the data to be retrieved: %s', len(data))

    if MAX_LATENTS is None:
        lengths = np.load(lengths_path)
        logger.info('loaded lengths from %s', lengths_path)
        assert len(data) == len(lengths), (len(data), len(lengths))
        assert question_embeddings.shape[0] == sum(lengths), (question_embeddings.shape[0], sum(lengths))
    else:
        lengths = None
    
    logger.info("doing indexing...")
    index = Indexer(embedding_size, 0, 8)

    # index all passages
    input_paths = glob.glob(passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")

    logger.info(f"Indexing passages from files {input_paths}")
    start_time_indexing = time.time()
    index_encoded_data(index, input_paths, 100000)
    logger.info(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

    # load passages
    passages = load_passages(passages_path)
    passage_id_map = {x["id"]: x for x in passages}
    

    # get top k results
    start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(question_embeddings.reshape(-1, embedding_size), top_k_per_query)
    logger.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
    top_ids_and_scores = aggregate_different_queries_by_length(top_ids_and_scores, lengths, MAX_LATENTS, top_k, aggregate_start_idx, aggregate_end_idx)

    logger.info(f"length of the data to be retrieved: {len(data)}, length of the retrieved results: {len(top_ids_and_scores)}")
    add_passages(data, passage_id_map, top_ids_and_scores)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        for ex in data:
            json.dump(ex, fout, ensure_ascii=False)
            fout.write("\n")
    logger.info(f"Saved results to {output_path}")
    
    

    
if __name__ == "__main__":    
    # for data_name in ['nq', 'msmarco', 'qampari']:
    for data_name in ['ambiguous']:
        # for training_data_name in ['nq', 'msmarco', 'qampari']:
        for training_data_name in ['ambiguous']:
            # if data_name != training_data_name:
            #     continue
            retriever_list = ['inf'] # ['stella', 'inf', 'cont']            
            split = 'dev'
            MAX_LATENTS = None
            
            embeddings_dir = 'qampari_embeddings' if data_name in ['qampari'] else data_name
            if data_name == 'ambiguous':
                embeddings_dir = 'nq'
            passage_embeddings_map = {
                'stella': {"embedding_path": f"/datastor1/hungting/stella_en_400M_v5/{embeddings_dir}/*", "embedding_dim": 1024},
                'inf': {"embedding_path": f"/datastor1/hungting/inf/{embeddings_dir}/*", "embedding_dim": 1536},
                'cont': {"embedding_path": f"/datastor1/hungting/Contriever/{embeddings_dir}/*", "embedding_dim": 768}
            }
            
            ### passage embeddings ###
            if data_name in ['qampari']:
                passages_path = f'/datastor1/hungting/wikipedia_chunks/chunks_v5.tsv'
            elif data_name == 'ambiguous':
                passages_path = f'/scratch/cluster/hungting/projects/autoregressive/data/nq/corpus.tsv'
            else:
                passages_path = f'/scratch/cluster/hungting/projects/autoregressive/data/{data_name}/corpus.tsv'
            
            ### load dev data ###
            if data_name == 'ambiguous':
                dev_data_path = f'data_creation/raw_data/{data_name}_{split}_question_only_2_to_5_ctxs.jsonl'
            elif data_name == 'qampari':
                dev_data_path = f'data_creation/raw_data/{data_name}_{split}_question_only_5_to_8_ctxs.jsonl'
            else:
                dev_data_path = f'data_creation/raw_data/{data_name}_{split}_question_only.jsonl'
            

            for retriever in retriever_list:
                dataset_name = f"{data_name}_{retriever}"
                
                ### load embeddings dataset ###
                if data_name == 'ambiguous':
                    dataset_path = f'training_datasets/autoregressive_{dataset_name}_dev_dataset_1b_contrastive_2_to_5_ctxs'
                elif data_name == 'qampari':
                    dataset_path = f'training_datasets/autoregressive_{dataset_name}_dev_dataset_1b_contrastive_5_to_8_ctxs'
                else:
                    dataset_path = f'training_datasets/autoregressive_{dataset_name}_dev_dataset_1b_qemb'
                logger.info('loading embeddings dataset from %s', dataset_path)
        
                ### retrieve and evaluate ###
                main_test(
                        passages_embeddings = passage_embeddings_map[retriever]["embedding_path"], 
                        passages_path = passages_path, 
                        raw_data_path = dev_data_path,
                        embedding_size = passage_embeddings_map[retriever]["embedding_dim"], 
                        lengths_path = f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/{data_name}_dev_oracle_embeddings_{retriever}_lengths.npy',
                        data_path = f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/{data_name}_dev_oracle_embeddings_{retriever}.npy',
                        output_path = f'data/ambiguous/nq_embeddings_data/oracle_retrieval_out_{split}_{data_name}.jsonl',
                        MAX_LATENTS = MAX_LATENTS,
                        )
            
                main_test(
                        passages_embeddings = passage_embeddings_map[retriever]["embedding_path"], 
                        passages_path = passages_path, 
                        raw_data_path = dev_data_path,
                        embedding_size = passage_embeddings_map[retriever]["embedding_dim"], 
                        lengths_path = f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/{data_name}_dev_oracle_embeddings_{retriever}_lengths.npy',
                        data_path = f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/{data_name}_dev_oracle_embeddings_{retriever}.npy',
                        output_path = f'data/ambiguous/nq_embeddings_data/oracle_retrieval_out_{split}_{data_name}_single.jsonl',
                        MAX_LATENTS = MAX_LATENTS,
                        aggregate_start_idx = 0,
                        aggregate_end_idx = 1
                        )
                
                main_test(
                        passages_embeddings = passage_embeddings_map[retriever]["embedding_path"], 
                        passages_path = passages_path, 
                        raw_data_path = dev_data_path,
                        embedding_size = passage_embeddings_map[retriever]["embedding_dim"], 
                        lengths_path = f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/{data_name}_dev_oracle_embeddings_{retriever}_lengths.npy',
                        data_path = f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/{data_name}_dev_oracle_embeddings_{retriever}.npy',
                        output_path = f'data/ambiguous/nq_embeddings_data/oracle_retrieval_out_{split}_{data_name}_from_2nd_to_3rd.jsonl',
                        MAX_LATENTS = MAX_LATENTS,
                        aggregate_start_idx = 1,
                        aggregate_end_idx = 2
                        )
                    
        