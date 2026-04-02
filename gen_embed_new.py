import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer
import argparse
import os
import pickle
from tqdm import tqdm

import pandas as pd
import json
from pathlib import Path

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

@torch.no_grad()
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
    allembeddings = normalize_np(allembeddings, p=2, dim=1)

    return allids, allembeddings

@torch.no_grad()
def embed_passages_sentence_transformers(args, passages, model):
    import numpy as np
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

@torch.no_grad()
def embed_passages_iterative_retrieval(args, passages, tokenizer, model):
    import torch
    import torch.nn.functional as F

    from torch import Tensor


    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    max_length = 1024
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
            # docs_vectors = model.encode(batch_texts)
            
            # Tokenize the input texts
            batch_dict = tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(model.device)
            outputs = model(**batch_dict)
            docs_vectors = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            docs_vectors = docs_vectors.cpu().numpy()
            # add embeddings and ids
            allembeddings.append(docs_vectors)
            allids.extend(batch_ids)
            # reset batch
            batch_ids = []
            batch_texts = []
    # process the last batch
    if len(batch_ids) > 0:
        # Tokenize the input texts
        batch_dict = tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(model.device)
        outputs = model(**batch_dict)
        docs_vectors = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        docs_vectors = docs_vectors.cpu().numpy()
        
        allembeddings.append(docs_vectors)
        allids.extend(batch_ids)
        
        
    allembeddings = np.concatenate(allembeddings, axis=0)
    allembeddings = normalize_np(allembeddings, p=2, dim=1)
    return allids, allembeddings    

def split_num_passages(num_passages, num_shards):
    # split range(num_passages) into chunks of size num_shards
    return [range(i * num_passages // num_shards, (i + 1) * num_passages // num_shards) for i in range(num_shards)]

def find_checkpoint_dir(model_path):
    for candidate in [
        os.path.join(model_path, 'checkpoint.pth'),
        os.path.join(model_path, 'best_model', 'checkpoint.pth'),
    ]:
        if os.path.exists(candidate):
            return os.path.dirname(candidate)
    return None

def main(args):
    # convert_to_tsv(args.passages)
    # args.passages = args.passages.replace('.jsonl', '.tsv')
    checkpoint_dir = find_checkpoint_dir(args.model_name_or_path)
    use_finetuned = checkpoint_dir is not None
    tokenizer = None

    if use_finetuned:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training', 'inf_retriever'))
        from training.inf_retriever.src.inf_retriever import load_retriever
        print(f'Detected finetuned checkpoint at {checkpoint_dir}, loading via load_retriever')
        retriever, tokenizer, _ = load_retriever(checkpoint_dir)
        # Extract the underlying AutoModel (returns last_hidden_state, compatible with embed_passages_iterative_retrieval)
        if hasattr(retriever, 'inf_model'):        # INFRetriever (standard mode)
            model = retriever.inf_model
        elif hasattr(retriever, 'encoder'):        # InBatch (standard_org_q and others)
            model = retriever.encoder.inf_model
        else:
            raise ValueError(f'Cannot extract underlying AutoModel from {type(retriever)}')
    elif ('stella' in args.model_name_or_path) or ('inf-retriever' in args.model_name_or_path) or ('NV-Embed' in args.model_name_or_path):
        model = SentenceTransformer(args.model_name_or_path, trust_remote_code=True)
        if 'inf-retriever' in args.model_name_or_path:
            model.max_seq_length = 8192
        if 'NV-Embed' in args.model_name_or_path:
            model.max_seq_length = 8192
            model.tokenizer.padding_side = 'right'
    elif 'iterative_retrieval' in args.model_name_or_path:
        model = AutoModel.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        
    print('finish loading model')
    
    model.eval()
    model = model.cuda()
    
    if not args.no_fp16:
        model = model.half()

    import math
    total_passages = sum(1 for _ in open(args.passages)) - 1  # subtract header
    shard_size = math.ceil(total_passages / args.num_shards)
    start_idx = args.shard_id * shard_size
    end_idx = min(start_idx + shard_size, total_passages)
    print(f'total_passages: {total_passages}, num_shards: {args.num_shards}, shard_size: {shard_size}')
    print('start_idx: ', start_idx, 'end_idx: ', end_idx)

    passages = pd.read_csv(args.passages,
                chunksize=shard_size,
                skiprows=start_idx,
                delimiter='\t',
                dtype={"id": str, "title": str, "text": str})

    print(f"Embedding generation for {shard_size} passages from idx {start_idx} to {end_idx}.")

    # actually doing the embedding
    for chunk in passages:
        if use_finetuned:
            allids, allembeddings = embed_passages_iterative_retrieval(args, chunk, tokenizer, model)
        elif ('stella' in args.model_name_or_path) or ('inf-retriever' in args.model_name_or_path) or ('LLM2Vec' in args.model_name_or_path) or ('llm2vec' in args.model_name_or_path):
            allids, allembeddings = embed_passages_sentence_transformers(args, chunk, model)
        elif 'iterative_retrieval' in args.model_name_or_path:
            if 'infly' in args.model_name_or_path:
                tokenizer_path = 'infly/inf-retriever-v1-1.5b'
            elif 'qwen3' in args.model_name_or_path:
                tokenizer_path = 'Qwen/Qwen3-Embedding-0.6B'
            elif 'contriever' in args.model_name_or_path:
                tokenizer_path = 'facebook/contriever-msmarco'
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            allids, allembeddings = embed_passages_iterative_retrieval(args, chunk, tokenizer, model)
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
    
    args = parser.parse_args()


    main(args)
    
    # python gen_embed_new.py --passages /scratch/hc3337/wikipedia_chunks/small.tsv \
    #                           --output_dir wikipedia_embeddings \
    #                           --shard_id 0 \
    #                           --num_shards 32 \
    #                           --model_name_or_path checkpoints/qampari_infly_standard_org_q_finetuned_steps5000_t0.05_lr0.00001_ws200_bs256_gradchkpt/checkpoint/best_model/ \
    #                           --per_gpu_batch_size 512 
