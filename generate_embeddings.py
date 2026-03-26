import torch
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
import argparse
import os
import pickle
from tqdm import tqdm

import pandas as pd
import json

import numpy as np

def normalize_np(x, p=2, dim=1, eps=1e-12):
    """
    NumPy implementation of torch.nn.functional.normalize
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    return x / norm

def convert_to_tsv(path):
    with open(path, 'r') as jsonl_file, open(path.replace('.jsonl', '.tsv'), 'w') as tsv_file:
        for line in jsonl_file:
            data = json.loads(line)
            id = data.get('id', '')
            title = data.get('title', '')
            raw = data.get('raw', '')
            tsv_file.write(f"{id}\t{title}\t{raw}\n")


def embed_passages(args, passages, model):
    batch_size = args.per_gpu_batch_size
    allids = []
    all_texts = []
    allembeddings = []

    def add_eos(input_examples, eos_token):
        input_examples = [input_example + eos_token for input_example in input_examples]
        return input_examples
            
    for _, row in tqdm(passages.iterrows()):
        allids.append(row.iloc[0])
        all_texts.append(str(row.iloc[2]) + ' ' + str(row.iloc[1]))
    allembeddings = model.encode(add_eos(all_texts, model.tokenizer.eos_token), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

    return allids, allembeddings

@torch.no_grad()
def embed_passages_stella(args, passages, model):
    batch_size = args.per_gpu_batch_size
    allids = []
    batch_ids = []
    batch_texts = []
    allembeddings = []
    
    # Embed the documents
    for _, row in tqdm(passages.iterrows()):
        batch_ids.append(row.iloc[0])
        batch_texts.append(str(row.iloc[2]) + ' ' + str(row.iloc[1]))
        if len(batch_ids) == batch_size:
            docs_vectors = model.encode(batch_texts)
            # add embeddings and ids
            allembeddings.append(docs_vectors)
            allids.extend(batch_ids)
            # reset batch
            batch_ids = []
            batch_texts = []
    # process the last batch
    if len(batch_ids) > 0:
        docs_vectors = model.encode(batch_texts)
        allembeddings.append(docs_vectors)
        allids.extend(batch_ids)
    allembeddings = np.concatenate(allembeddings, axis=0)
    allembeddings = normalize_np(allembeddings, p=2, dim=1)
    return allids, allembeddings    




def main(args):
    model = SentenceTransformer(args.model_name_or_path, trust_remote_code=True)
    if 'inf-retriever' in args.model_name_or_path:
        model.max_seq_length = 8192
    if 'NV-Embed' in args.model_name_or_path:
        model.max_seq_length = 8192
        model.tokenizer.padding_side = 'right'
        
    print('finish loading model')
    
    model.eval()
    model = model.cuda()
    
    if not args.no_fp16:
        model = model.half()

    print('start embedding, shard_size: ', args.shard_size)
    shard_size = args.shard_size
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
        
    passages = pd.read_csv(args.passages,
                chunksize=shard_size,
                skiprows=start_idx,
                delimiter='\t',
                dtype={"id": str, "title": str, "text": str})

    print(f"Embedding generation for {shard_size} passages from idx {start_idx} to {end_idx}.")

    # actually doing the embedding
    for chunk in passages:
        if ('stella' in args.model_name_or_path) or ('inf-retriever' in args.model_name_or_path):
            allids, allembeddings = embed_passages_stella(args, chunk, model)
        else:
            allids, allembeddings = embed_passages(args, chunk, model)
        break

    save_file = os.path.join(args.output_dir, args.prefix + f"_{args.shard_id:02d}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving {len(allids)} passage embeddings to {save_file}.")
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    print(f"Total passages processed {len(allids)}. Written to {save_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--output_dir", type=str, default="wikipedia_embeddings", help="dir path to save embeddings")
    parser.add_argument("--prefix", type=str, default="passages", help="prefix path to save embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=32, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--passage_maxlength", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--no_title", action="store_true", help="title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--instruction", type=str, default="", help="instruction for the model")
    parser.add_argument("--shard_size", type=int, default=2500000, help="shard size")
    
    args = parser.parse_args()


    main(args)