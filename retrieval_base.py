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

from src.retrieval_utils import add_passages, load_passages, load_index, aggregate_sharded_results
from src.inference_utils import embed_queries_iterative_retrieval, embed_queries, embed_queries_stella
import structlog
logger = structlog.get_logger()


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
            num_shards=num_shards,
            logger=logger
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
    # Load data
    data_paths = glob.glob(args.data)
    if len(data_paths) == 0:
        assert False, 'No data paths found'
    print('Loading data from data_paths', data_paths)
    
    # Load model
    print(f"Loading model from: {args.model_name_or_path}")
    if ('stella' in args.model_name_or_path) or ('inf-retriever' in args.model_name_or_path) or ('Qwen' in args.model_name_or_path):
        model = SentenceTransformer(args.model_name_or_path, trust_remote_code=True)
        if ('inf-retriever' in args.model_name_or_path) or ('Qwen' in args.model_name_or_path):
            model.max_seq_length = 8192
    elif 'NV-Embed' in args.model_name_or_path:
        model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
    elif 'iterative_retrieval' in args.model_name_or_path:
        model = AutoModel.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    # load passages
    passages = load_passages(args.passages)
    passage_id_map = {x["id"]: x for x in passages}

    
    # Inference
    for path in data_paths:
        print('loading data from ', path)
        data = load_data(path)

        output_path = os.path.join(args.output_dir, args.output_file)
        # Embed queries
        print('start embedding queries')
        queries = [ex.get("question") or ex.get("question_text") or "" for ex in data]
        if 'stella' in args.model_name_or_path or ('inf-retriever' in args.model_name_or_path) or ('Qwen' in args.model_name_or_path):
            questions_embedding = embed_queries_stella(args, queries, model)
        elif 'iterative_retrieval' in args.model_name_or_path:
            print('embed iterative retrieval')
            questions_embedding = embed_queries_iterative_retrieval(args, queries, model)
        elif 'NV-Embed' in args.model_name_or_path:
            questions_embedding = embed_queries(args, queries, model)
        else:
            raise ValueError(f"Unsupported model: {args.model_name_or_path}")
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

        # Save results
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
