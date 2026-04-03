# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

import src.index
import src.inf_retriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text
import threading
import psutil

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class GPUKeepAlive:
    """Runs lightweight GPU operations to keep GPU active during CPU work"""
    
    def __init__(self, interval=0.5, tensor_size=1024):
        self.interval = interval
        self.tensor_size = tensor_size
        self.running = False
        self.thread = None
        
        if torch.cuda.is_available():
            self.dummy_a = torch.randn(tensor_size, tensor_size, device='cuda')
            self.dummy_b = torch.randn(tensor_size, tensor_size, device='cuda')
            print(f"✓ GPUKeepAlive initialized ({tensor_size}x{tensor_size} tensors)", flush=True)
    
    def _worker(self):
        while self.running:
            if torch.cuda.is_available():
                # Aggressive continuous computation - no sleep!
                for _ in range(50):  # More iterations
                    _ = torch.matmul(self.dummy_a, self.dummy_b)
                    _ = torch.matmul(self.dummy_b, self.dummy_a)
                torch.cuda.synchronize()
    
    def start(self):
        if not torch.cuda.is_available() or self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("✓ GPUKeepAlive: GPU now active during CPU operations", flush=True)
    
    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("✓ GPUKeepAlive stopped", flush=True)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction for better retrieval performance"""
    return f'Instruct: {task_description}\nQuery: {query}'

@torch.no_grad()
def embed_batch_texts(args, texts, model, tokenizer, normalize=None):
    embeddings, batch_texts = [], []
    for k, _text in enumerate(texts):
        if args.lowercase:
            _text = _text.lower()
        if args.normalize_text:
            _text = src.normalize_text.normalize(_text)
        batch_texts.append(_text)

        if len(batch_texts) == args.per_gpu_batch_size or k == len(texts) - 1:

            encoded_batch = tokenizer.batch_encode_plus(
                batch_texts,
                return_tensors="pt",
                max_length=args.question_maxlength,
                padding=True,
                truncation=True,
            )
            

            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}

            if normalize is not None and 'normalize' in model.forward.__code__.co_varnames:
                output = model(**encoded_batch, normalize=normalize)
            else:
                output = model(**encoded_batch)

            #output = model(**encoded_batch)
            embeddings.append(output)

            batch_texts = []

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


@torch.no_grad()
def embed_queries(args, queries, input_negatives, model, tokenizer):
    if args.training_mode == "base" or args.training_mode == "standard" or args.training_mode == "standard_concat":
        return embed_queries_standard(args, queries, model, tokenizer)
    elif args.training_mode == "subtraction":
        return embed_queries_subtraction(args, queries, input_negatives, model, tokenizer)
    elif args.training_mode == "gru":
        return embed_queries_gru(args, queries, input_negatives, model, tokenizer)
    elif args.training_mode == "linear_projection":
        return embed_queries_linear_projection(args, queries, input_negatives, model, tokenizer)
    elif args.training_mode == "subtraction_linear":
        return embed_queries_subtraction_linear(args, queries, input_negatives, model, tokenizer)
    elif args.training_mode == "sentence_transformer":
        return embed_queries_sentence_transformer(args, queries, input_negatives, model, tokenizer)
    else:
        raise ValueError(f"Invalid training mode: {args.training_mode}")



@torch.no_grad()
def embed_queries_standard(args, queries, model, tokenizer):
    

    model.eval()

    normalize_query = args.norm_query
    print(f"Using normalize={normalize_query} for queries", flush=True)

    task = 'Given a query, retrieve relevant passages that answer the query'
    queries = [get_detailed_instruct(task, q) for q in queries]

    if args.training_mode == "base":
        if hasattr(model, 'encoder'):
            embeddings = embed_batch_texts(args, queries, model.encoder, tokenizer, normalize=normalize_query)
        else:
            embeddings = embed_batch_texts(args, queries, model, tokenizer)
    else:
        embeddings = embed_batch_texts(args, queries, model, tokenizer)
    print(f"Questions embeddings shape: {embeddings.size()}")
    # if embeddings.dtype == torch.bfloat16:
    #     embeddings = embeddings.float()


    model_name = args.model_name_or_path.replace('/', '_').replace('\\', '_')
    embeddings_file = f"{args.output_dir}/query_embeddings_{args.outfile}.npy"
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(embeddings_file, embeddings.cpu().numpy())
    print(f"Saved query embeddings to: {embeddings_file}")
    
    return embeddings.cpu().numpy()



@torch.no_grad()
def embed_queries_subtraction(args, queries, input_negatives, model, tokenizer):
    model.eval()
    # embeddings, batch_question, inpn_embeddings, batch_inpn = [], [], [], []
    len_input_negatives = [len(inpn) for inpn in input_negatives]
    input_negatives = [inpn for inpn_list in input_negatives for inpn in inpn_list]  # flatten list
        
    embeddings = embed_batch_texts(args, queries, model.encoder, tokenizer)
    print(f"Questions embeddings shape: {embeddings.size()}")
    
    if len(input_negatives) == 0:
        return embeddings.cpu().numpy()
    
    inpn_embeddings = embed_batch_texts(args, input_negatives, model.encoder, tokenizer)
    print(f"Input Negatives embeddings shape: {inpn_embeddings.size()}")
    
    assert len(len_input_negatives) == embeddings.size(0)
    start = 0
    for i, _len in enumerate(len_input_negatives):
        end = start + _len
        embeddings[i] = embeddings[i] - inpn_embeddings[start:end].sum(dim=0)
        start = end
    return embeddings.cpu().numpy()



@torch.no_grad()
def embed_queries_gru(args, queries, input_negatives, model, tokenizer):
    model.eval()
    # embeddings, batch_question, inpn_embeddings, batch_inpn = [], [], [], []
    len_input_negatives = [len(inpn) for inpn in input_negatives]
    input_negatives = [inpn for inpn_list in input_negatives for inpn in inpn_list]  # flatten list

    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [get_detailed_instruct(task, q) for q in queries]
    
    embeddings = embed_batch_texts(args, queries, model.encoder, tokenizer)
    print(f"Questions embeddings shape: {embeddings.size()}")
    
    if len(input_negatives) == 0:
        return embeddings.cpu().numpy()
    
    inpn_embeddings = embed_batch_texts(args, input_negatives, model.encoder, tokenizer)
    print(f"Input Negatives embeddings shape: {inpn_embeddings.size()}")

    
    gru = model.gru
    
    assert len(len_input_negatives) == embeddings.size(0)
    start = 0
    for i, _len in enumerate(len_input_negatives):
        end = start + _len
        _, hn = gru(inpn_embeddings[start:start+_len], embeddings[i].unsqueeze(0))
        embeddings[i] = hn[-1]
        start = end
    return embeddings.cpu().numpy()

@torch.no_grad()
def embed_queries_linear_projection(args, queries, input_negatives, model, tokenizer):
    model.eval()
    # embeddings, batch_question, inpn_embeddings, batch_inpn = [], [], [], []
    len_input_negatives = [len(inpn) for inpn in input_negatives]
    input_negatives = [inpn for inpn_list in input_negatives for inpn in inpn_list]  # flatten list

    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [get_detailed_instruct(task, q) for q in queries]
    
    embeddings = embed_batch_texts(args, queries, model.encoder, tokenizer)
    print(f"Questions embeddings shape: {embeddings.size()}")
    
    if len(input_negatives) == 0:
        return embeddings.cpu().numpy()
    
    inpn_embeddings = embed_batch_texts(args, input_negatives, model.encoder, tokenizer)
    print(f"Input Negatives embeddings shape: {inpn_embeddings.size()}")

    linear = model.linear
    
    assert len(len_input_negatives) == embeddings.size(0)
    start = 0
    for i, _len in enumerate(len_input_negatives):
        end = start + _len
        
        doc_emb = inpn_embeddings[start:start+_len].mean(dim=0)
        q_doc_emb = torch.cat([embeddings[i].unsqueeze(0), doc_emb.unsqueeze(0)], dim=1)
        embeddings[i] = linear(q_doc_emb).squeeze(0)
        start = end
    return embeddings.cpu().numpy()


@torch.no_grad()
def embed_queries_subtraction_linear(args, queries, input_negatives, model, tokenizer):
    model.eval()
    # embeddings, batch_question, inpn_embeddings, batch_inpn = [], [], [], []
    len_input_negatives = [len(inpn) for inpn in input_negatives]
    input_negatives = [inpn for inpn_list in input_negatives for inpn in inpn_list]  # flatten list

    embeddings = embed_batch_texts(args, queries, model.encoder, tokenizer)
    print(f"Questions embeddings shape: {embeddings.size()}")
    
    if len(input_negatives) == 0:
        return embeddings.cpu().numpy()
    
    inpn_embeddings = embed_batch_texts(args, input_negatives, model.encoder, tokenizer)
    print(f"Input Negatives embeddings shape: {inpn_embeddings.size()}")
    

    
    weights = model.weights
    bias = model.bias
    
    assert len(len_input_negatives) == embeddings.size(0)
    start = 0
    for i, _len in enumerate(len_input_negatives):
        end = start + _len
        doc_emb = weights * torch.t(inpn_embeddings[start:start+_len]) * bias # (768, _len)
        embeddings[i] = embeddings[i] - torch.sum(doc_emb, dim=1)
        start = end
    return embeddings.cpu().numpy()


@torch.no_grad()
def embed_queries_sentence_transformer(args, queries, input_negatives, model, tokenizer):
    model.eval()
    # embeddings, batch_question, inpn_embeddings, batch_inpn = [], [], [], []
    input_negatives = ['\n'.join(inpn_list) for inpn_list in input_negatives]  # flatten list
    
    embeddings = embed_batch_texts(args, queries, model.encoder, tokenizer)
    print(f"Questions embeddings shape: {embeddings.size()}")
    
    if len(input_negatives) == 0:
        return embeddings.cpu().numpy()
    
    inpn_embeddings = embed_batch_texts(args, input_negatives, model.encoder, tokenizer)
    print(f"Input Negatives embeddings shape: {inpn_embeddings.size()}")

    linear = model.linear
    embeddings = linear(torch.cat([embeddings, inpn_embeddings], dim=1))             
    
    return embeddings.cpu().numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}", flush=True)
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    print(message)
    return match_stats.questions_doc_hits

def calculate_avg_query_length(queries, tokenizer, question_maxlength=32000):
    """Calculate average query length in tokens"""
    total_tokens = 0
    for query in queries:
        encoded = tokenizer.encode(
            query,
            max_length=question_maxlength,
            truncation=True,
            add_special_tokens=True
        )
        total_tokens += len(encoded)
    
    avg_length = total_tokens / len(queries) if queries else 0
    return avg_length, total_tokens

def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        if 'ctxs' not in d:
            d['ctxs'] = []
        d["ctxs"] += [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    try:
        if data_path.endswith(".json"):
            with open(data_path, "r") as fin:
                data = json.load(fin)
        elif data_path.endswith(".jsonl"):
            data = []
            with open(data_path, "r") as fin:
                for k, example in enumerate(fin):
                    example = json.loads(example)
                    data.append(example)
    except:
        data = []
        with open(data_path, 'r') as fin:
            for line in fin:
                data.append(json.loads(line.strip()))
        return data
    return data


def main(args):

    print(f"Loading model from: {args.model_name_or_path}", flush=True)
    model, tokenizer, _ = src.inf_retriever.load_retriever(args.model_name_or_path)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    torch.cuda.synchronize()
    model_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"✓ Retrieval model memory: {model_memory_gb:.2f} GB", flush=True)
    process = psutil.Process(os.getpid())
    memory_before_index_gb = process.memory_info().rss / 1e9

    index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)

    # index all passages
    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        print(f"Indexing passages from files {input_paths}", flush=True)
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.", flush=True)
        if args.save_or_load_index:
            index.serialize(embeddings_dir)

    # load passages
    passages = src.data.load_passages(args.passages)
    passage_id_map = {x["id"]: x for x in passages}

    # data_paths = args.data
    print(f"Retrieving for file: {args.data}", flush=True)
    data = load_data(args.data)

    skip_queries = 10000
    if len(data) > skip_queries:
        print(f"Skipping first {skip_queries} queries. Processing {len(data) - skip_queries} remaining queries.", flush=True)
        data = data[skip_queries:]
    else:
        print(f"Warning: Data has only {len(data)} queries, which is less than {skip_queries}. Processing all.", flush=True)

    output_path = os.path.join(args.output_dir, os.path.basename(args.outfile))

    # embed the query
    queries = [ex[args.question_key] for ex in data]

    avg_query_length, total_tokens = calculate_avg_query_length(queries, tokenizer, args.question_maxlength)
    print(f"Average query length: {avg_query_length:.2f} tokens",flush=True)
    print(f"Total queries: {len(queries)}",flush=True)
    print(f"Total tokens: {total_tokens}",flush=True)

    print(f'doing {args.training_mode}....', flush=True)
    input_negatives = []
    if args.training_mode != "base" and args.training_mode != "standard" and (not args.autoregressive):
        for example in data:
            input_negatives.append([
                n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in example['input_negative_ctxs']
            ])
    elif args.training_mode == "standard_concat":
        for example in data:
            input_negatives.append([
                n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in example['input_negative_ctxs']
            ])
        queries = ['Question: ' + ex[args.question_key] + '\n\n' + 'Documents:' + '\n'.join(inpn) for ex, inpn in zip(data, input_negatives)]

    torch.cuda.reset_peak_memory_stats()
    embed_start_time = time.time()
        
    questions_embedding = embed_queries(args, queries, input_negatives, model, tokenizer)    

    embed_end_time = time.time()
    embed_time = embed_end_time - embed_start_time
    embed_throughput = len(queries) / embed_time if embed_time > 0 else 0
    print(f"Embedding time: {embed_time:.2f}s", flush=True)
    print(f"Embedding throughput: {embed_throughput:.2f} queries/sec", flush=True)


    if not args.autoregressive:
        # get top k results
        # start_time_retrieval = time.time()
        # top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        # print(f"Search time: {time.time()-start_time_retrieval:.1f} s.", flush=True)
        gpu_keepalive = GPUKeepAlive(interval=0.2, tensor_size=2048)
        with gpu_keepalive:
            start_time_retrieval = time.time()
            top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
            end_time_retrieval = time.time()
            
            search_time = end_time_retrieval - start_time_retrieval
            search_throughput = len(queries) / search_time if search_time > 0 else 0
            
            print(f"Search time: {search_time:.1f} s.", flush=True)
            print(f"Search throughput: {search_throughput:.2f} queries/sec", flush=True)
        add_passages(data, passage_id_map, top_ids_and_scores)
    else:
        # each time retrieve top 1 document
        num_docs = 0
        while True:
            # retrieve one step
            start_time_retrieval = time.time()
            top_ids_and_scores = index.search_knn(questions_embedding, args.step)
            print(f"Search time: {time.time()-start_time_retrieval:.1f} s.", flush=True)
            num_docs += args.step
            if num_docs >= args.n_docs:
                break
            
            # add the passages to ctxs
            add_passages(data, passage_id_map, top_ids_and_scores)
            for inst in data:
                inst['input_negative_ctxs'] = inst['ctxs']
                assert len(inst['input_negative_ctxs']) == num_docs
            
            # re-embed the query
            
            print(f'doing {args.training_mode}....')
            input_negatives = []
            for example in data:
                input_negatives.append([
                    n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in example['input_negative_ctxs']
                ])
            if args.training_mode == "standard":
                assert len(data) == len(input_negatives)
                queries = ['Question: ' + ex[args.question_key] + '\n\n' + 'Documents:' + '\n'.join(inpn) for ex, inpn in zip(data, input_negatives)]
            else:
                queries = [ex[args.question_key] for ex in data]
                
            questions_embedding = embed_queries(args, queries, input_negatives, model, tokenizer)    

    
            
    # hasanswer = validate(data, args.validation_workers)
    # add_hasanswer(data, hasanswer)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        for ex in data:
            json.dump(ex, fout, ensure_ascii=False)
            fout.write("\n")
    print(f"Saved results to {output_path}", flush = True)

    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    total_time = embed_time + search_time
    time_per_query = total_time / len(queries) if len(queries) > 0 else 0

    #process = psutil.Process(os.getpid())
    memory_after_index_gb = process.memory_info().rss / 1e9
    index_memory_gb = memory_after_index_gb - memory_before_index_gb
    print(f"✓ Index memory: {index_memory_gb:.2f} GB", flush=True)


    print("\n" + "="*80)
    print("RETRIEVAL EFFICIENCY STATS")
    print("="*80)
    print(f"Total queries: {len(queries)}")
    print(f"Embedding time: {embed_time:.2f}s")
    print(f"Search time: {search_time:.2f}s")
    print(f"Total retrieval time: {total_time:.2f}s")
    print(f"Time per query: {time_per_query:.4f}s")
    print(f"Throughput: {len(queries)/total_time:.2f} queries/sec")
    print("")
    print(f"Retrieval model memory: {model_memory_gb:.2f} GB")
    print(f"Peak GPU memory: {peak_memory:.2f} GB")
    print(f"Model: {args.model_name_or_path}")
    print("="*80 + "\n", flush=True)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=32000, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=1536)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    
    parser.add_argument("--outfile", type=str, default="output.jsonl")
    parser.add_argument("--training_mode", type=str, default="standard")
    parser.add_argument("--question_key", type=str, default="question")
    parser.add_argument("--autoregressive", action="store_true", help="autoregressive retrieval")
    parser.add_argument("--step", type=int, default=3, help="number of documents to retrieve per step")
    parser.add_argument("--norm_query", action="store_true", help="Apply L2 normalization to query embeddings")

    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)