import os
import argparse
import csv
import json
import pickle
import time
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from sentence_transformers import SentenceTransformer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

from src.retrieval_utils import Indexer, add_passages, load_passages, index_encoded_data, add_passages_single_instance
import structlog
logger = structlog.get_logger()

def normalize_np(x, p=2, dim=1, eps=1e-12):
    """
    NumPy implementation of torch.nn.functional.normalize
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    return x / norm


@torch.no_grad()
def embed_queries(args, queries, model):
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",
                             "example_input": "Given a question and some relevant passages, retrieve passages that answer the question but are not in the input set.",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "

    max_length = 32768
    model.eval()
    embeddings, batch_question = [], []

    for k, q in enumerate(queries):
        if args.lowercase:
            q = q.lower()
        if args.normalize_text:
            q = src.normalize_text.normalize(q)
        batch_question.append(q)
    embeddings = model._do_encode(batch_question, batch_size=args.per_gpu_batch_size, instruction=query_prefix, max_length=max_length, num_workers=32, return_numpy=True)
    embeddings = normalize_np(embeddings, p=2, dim=1)
    print("Questions embeddings shape:", embeddings.shape)
    return embeddings

@torch.no_grad()
def embed_queries_stella(args, queries, model):
    if 'inf-retriever' in args.model_name_or_path:
        query_prompt_name = "query"
    else:
        query_prompt_name = "s2p_query"

    model.eval()
    embeddings, batch_question = [], []

    for k, q in enumerate(queries):
        if args.lowercase:
            q = q.lower()
        if args.normalize_text:
            q = src.normalize_text.normalize(q)
        batch_question.append(q)

        if len(batch_question) == args.per_gpu_batch_size:
            embeddings.append(model.encode(batch_question, prompt_name=query_prompt_name))
            batch_question = []
    if len(batch_question) > 0:
        embeddings.append(model.encode(batch_question, prompt_name=query_prompt_name))
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = normalize_np(embeddings, p=2, dim=1)
    print("Questions embeddings shape:", embeddings.shape)
    return embeddings


@torch.no_grad()
def embed_queries_llm2vec(args, queries, model):
    instruction = (
        "Given a question and some relevant passages, retrieve Wikipedia passages that answer the question but are not in the given passages."
    )

    model.eval()
    embeddings, batch_question = [], []

    for k, q in enumerate(queries):
        if args.lowercase:
            q = q.lower()
        if args.normalize_text:
            q = src.normalize_text.normalize(q)
        batch_question.append([instruction, q])

        if len(batch_question) == args.per_gpu_batch_size:
            print('batch size', len(batch_question))
            embeddings.append(model.encode(batch_question))
            batch_question = []
    if len(batch_question) > 0:
        embeddings.append(model.encode(batch_question))
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = normalize_np(embeddings, p=2, dim=1)
    print("Questions embeddings shape:", embeddings.shape)
    return embeddings

def shard_and_get_embedding_files(embedding_files, shard_id, num_shards):
    num_files_per_shard = [len(embedding_files) // num_shards for _ in range(num_shards)]
    # evenly distribute the remaining numbers to each shard
    for i in range(len(embedding_files) % num_shards):
        num_files_per_shard[i] += 1
    assert sum(num_files_per_shard) == len(embedding_files)
    start_idx = sum(num_files_per_shard[:shard_id])
    end_idx = start_idx + num_files_per_shard[shard_id]
    return embedding_files[start_idx:end_idx]


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


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids




def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
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


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data

def load_llm2vec_model(model_path):
    if 'supervised' in model_path:
        if 'Mistral-7B-Instruct-v0.2' in model_path:
            base_model_path = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
        else:
            base_model_path = '-'.join(model_path.split('-')[:-1])
        print(base_model_path)
        from llm2vec import LLM2Vec
        from peft import PeftModel

        # Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path
        )
        config = AutoConfig.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model = PeftModel.from_pretrained(
            model,
            base_model_path,
        )
        model = model.merge_and_unload()  # This can take several minutes on cpu

        # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
        model = PeftModel.from_pretrained(
            model, model_path
        )

        # Wrapper for encoding and pooling operations
        l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
    return l2v

    
    
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
    
    
    
def load_index(embedding_size, passages_embeddings, n_subquantizers=0, n_bits=8, save_or_load_index=False, use_gpu=False, shard_id=0, num_shards=1):
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
        logger.info(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, 100000, shard_id=shard_id, num_shards=num_shards)
        logger.info(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if save_or_load_index:
            index.serialize(embeddings_dir)
    return index

def retrieve(question_embeddings, num_shards, passages_embeddings, passage_id_map, 
              embedding_size = 4096, top_k_per_query = 100, top_k = 100, save_or_load_index = False, use_gpu = False, 
              n_subquantizers = 0, n_bits = 8):

        
    # Start Retrieving!
    all_sharded_ids_and_scores = []
    for shard_id in range(num_shards):
        # Load index
        logger.info('passages_embeddings', passages_embeddings=passages_embeddings)
        index = load_index(
            embedding_size, 
            passages_embeddings, 
            n_subquantizers=n_subquantizers,
            n_bits=n_bits,
            save_or_load_index=save_or_load_index,
            use_gpu=use_gpu,
            shard_id=shard_id,
            num_shards=num_shards
        )
    
        # Start Search! Get top k results.
        start_time_retrieval = time.time()
        sharded_ids_and_scores = index.search_knn(question_embeddings.reshape(-1, embedding_size), top_k_per_query)
        logger.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
        all_sharded_ids_and_scores.append(sharded_ids_and_scores)
    
    top_ids_and_scores = aggregate_sharded_results(all_sharded_ids_and_scores, num_shards)
    logger.info(f"aggregated top_ids_and_scores for {num_shards} shards")
    return top_ids_and_scores
    # add_passages(data, passage_id_map, top_ids_and_scores)

    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # with open(output_path, "w") as fout:
    #     for ex in data:
    #         json.dump(ex, fout, ensure_ascii=False)
    #         fout.write("\n")
    # logger.info(f"Saved results to {output_path}")
    
    



def main(args):
    data_paths = glob.glob(args.data)
    if len(data_paths) == 0:
        assert False, 'No data paths found'
    print('Loading data from data_paths', data_paths)
    
    print(f"Loading model from: {args.model_name_or_path}")
    
    if ('stella' in args.model_name_or_path) or ('inf-retriever' in args.model_name_or_path):
        model = SentenceTransformer(args.model_name_or_path, trust_remote_code=True)
        if 'inf-retriever' in args.model_name_or_path:
            model.max_seq_length = 8192
    elif ('LLM2Vec' in args.model_name_or_path) or ('llm2vec' in args.model_name_or_path):
        print('loading LLM2Vec Model')
        model = load_llm2vec_model(args.model_name_or_path)
    else:
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    # print("Loading index", args.projection_size, args.n_subquantizers, args.n_bits, args.use_gpu)
    # index = Indexer(args.projection_size, args.n_subquantizers, args.n_bits, use_gpu=args.use_gpu)

    # # index all passages
    # input_paths = glob.glob(args.passages_embeddings)
    # print('input_paths', input_paths)
    # input_paths = sorted(input_paths)
    # embeddings_dir = os.path.dirname(input_paths[0])
    # index_path = os.path.join(embeddings_dir, "index.faiss")
    # if args.save_or_load_index and os.path.exists(index_path):
    #     index.deserialize_from(embeddings_dir)
    # else:
    #     print(f"Indexing passages from files {input_paths}")
    #     start_time_indexing = time.time()
    #     index_encoded_data(index, input_paths, args.indexing_batch_size, args.shard_id, args.num_shards)
    #     print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
    #     if args.save_or_load_index:
    #         index.serialize(embeddings_dir)

    # load passages
    passages = load_passages(args.passages)
    passage_id_map = {x["id"]: x for x in passages}

    
    alldata = []
    for path in data_paths:
        print('loading data from ', path)
        data = load_data(path)

        output_path = os.path.join(args.output_dir, args.output_file)
        print('start embedding queries')
        queries = [ex["question"] if 'question' in ex else ex['question_text'] for ex in data]
        if 'stella' in args.model_name_or_path or ('inf-retriever' in args.model_name_or_path):
            questions_embedding = embed_queries_stella(args, queries, model)
        elif ('LLM2Vec' in args.model_name_or_path) or ('llm2vec' in args.model_name_or_path):
            print('embed llm2vec')
            questions_embedding = embed_queries_llm2vec(args, queries, model)
        else:
            questions_embedding = embed_queries(args, queries, model)
        print('finished embedding queries')
        
        if args.save_embeddings:
            np.save(os.path.join(args.output_dir, f'questions_embeddings_{Path(path).stem}.npy'), questions_embedding)
            exit(0)
        # # get top k results
        # start_time_retrieval = time.time()
        # top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        # print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
        top_ids_and_scores = retrieve(questions_embedding, args.num_shards, args.passages_embeddings, passage_id_map, 
              embedding_size = args.projection_size, top_k_per_query = args.n_docs, top_k = args.n_docs, 
              save_or_load_index = args.save_or_load_index, use_gpu = args.use_gpu,
              n_subquantizers = args.n_subquantizers, n_bits = args.n_bits)
        
        add_passages(data, passage_id_map, top_ids_and_scores)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        print(f"Saved results to {output_path}")
        


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
    parser.add_argument("--per_gpu_batch_size", type=int, default=16, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=4096)
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
    parser.add_argument("--output_file", type=str, default="output.jsonl", help="output file name")
    parser.add_argument("--shard_id", type=int, default=0, help="shard id")
    parser.add_argument("--num_shards", type=int, default=8, help="number of shards")
    parser.add_argument("--use_gpu", action="store_true", help="use gpu")
    parser.add_argument("--use_dummy", action="store_true", help="use dummy model")
    parser.add_argument("--save_embeddings", action="store_true", help="save embeddings")
    args = parser.parse_args()
    main(args)