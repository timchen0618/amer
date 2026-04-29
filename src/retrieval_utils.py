import numpy as np
import faiss
import json
from tqdm import tqdm
import json
import pickle
from typing import List, Tuple
import csv 
import os
import glob
import time


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids

def shard_and_get_embedding_files(embedding_files, shard_id, num_shards):
    num_files_per_shard = [len(embedding_files) // num_shards for _ in range(num_shards)]
    # evenly distribute the remaining numbers to each shard
    for i in range(len(embedding_files) % num_shards):
        num_files_per_shard[i] += 1
    assert sum(num_files_per_shard) == len(embedding_files)
    start_idx = sum(num_files_per_shard[:shard_id])
    end_idx = start_idx + num_files_per_shard[shard_id]
    return embedding_files[start_idx:end_idx]


def aggregate_sharded_results(all_sharded_ids_and_scores, num_shards):
    if num_shards == 1:
        return all_sharded_ids_and_scores[0]
    
    # len(all_sharded_ids_and_scores) -> num_shards
    # all_sharded_ids_and_scores[0] -> list of top_ids_and_scores for the first shard
    # docs = [passages[doc_id] for doc_id in results_and_scores[0]]
    # scores = [str(score) for score in results_and_scores[1]]

    # Aggregate results from all shards
    top_ids_and_scores = []
    for i in range(len(all_sharded_ids_and_scores[0])):
        top_ids_and_scores.append([])
        for _ in range(2):
            top_ids_and_scores[i].append([])
        for shard_id in range(num_shards):
            top_ids_and_scores[i][1] = np.append(top_ids_and_scores[i][1], all_sharded_ids_and_scores[shard_id][i][1])  # scores
            top_ids_and_scores[i][0].extend(all_sharded_ids_and_scores[shard_id][i][0])  # ids
            
        indices = np.argsort(top_ids_and_scores[i][1])[::-1]
        top_ids_and_scores[i][1] = top_ids_and_scores[i][1][indices]
        top_ids_and_scores[i][0] = [top_ids_and_scores[i][0][j] for j in indices]
            
    return top_ids_and_scores


def index_encoded_data(index, embedding_files, indexing_batch_size, shard_id=0, num_shards=1):
    allids = []
    allembeddings = np.array([])
    print('shard_id', shard_id, 'num_shards', num_shards)
    embedding_files = shard_and_get_embedding_files(embedding_files, shard_id, num_shards)
    print('embedding_files', len(embedding_files))
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")

def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    assert len(data) == len(top_passages_and_scores), (len(data), len(top_passages_and_scores))
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]
        
def add_passages_single_instance(d, passages, top_passages_and_scores):
    results_and_scores = top_passages_and_scores
    docs = [passages[doc_id] for doc_id in results_and_scores[0]]
    scores = [str(score) for score in results_and_scores[1]]
    ctxs_num = len(docs)
    d["ctxs"] = [
        {
            "id": results_and_scores[0][c],
            "title": docs[c]["title"],
            "text": docs[c]["text"],
            "score": scores[c],
        }
        for c in range(ctxs_num)
    ]

# Used for passage retrieval
def load_passages(path):
    if not os.path.exists(path):
        print(f"{path} does not exist")
        return
    print(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        if path.endswith(".jsonl"):
            for k, line in enumerate(fin):
                ex = json.loads(line)
                passages.append(ex)
        else:
            reader = csv.reader(fin, delimiter="\t")
            for k, row in enumerate(reader):
                if not row[0] == "id":
                    ex = {"id": row[0], "title": row[2], "text": row[1]}
                    passages.append(ex)
    return passages


def load_index(embedding_size, passages_embeddings, n_subquantizers=0, n_bits=8, save_or_load_index=False, use_gpu=False, shard_id=0, num_shards=1, logger=None):
    if logger is not None:
        logger.info("doing indexing...")
    index = Indexer(embedding_size, n_subquantizers=n_subquantizers, n_bits=n_bits, use_gpu=use_gpu)

    # index all passages
    input_paths = glob.glob(passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    
    if save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        if logger is not None:
            logger.info(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, 100000, shard_id=shard_id, num_shards=num_shards)
        if logger is not None:
            logger.info(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if save_or_load_index:
            index.serialize(embeddings_dir)
    return index

class Indexer(object):

    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8, use_gpu=False):
        if n_subquantizers > 0:
            quantizer = faiss.IndexFlatIP(vector_sz)  # Inner Product as base index
            nlist = 4096
            self.index = faiss.IndexIVFPQ(quantizer, vector_sz, nlist, n_subquantizers, n_bits)
            self.index.nprobe = min(64, nlist)  # Set a reasonable nprobe value
        else:
            self.index = faiss.IndexFlatIP(vector_sz)
            if use_gpu:
                print('using gpu')
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        self.index_id_to_db_id = []

    def index_data(self, ids, embeddings):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
            
        self.index.add(embeddings)
        print(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048) -> List[Tuple[List[object], List[float]]]:
        query_vectors = query_vectors.astype('float32')
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in tqdm(range(nbatch)):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Serializing index to {index_file}, meta data to {meta_file}')

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        print('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        self.index_id_to_db_id.extend(db_ids)
        
        