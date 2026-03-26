import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from peft import PeftModel
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


def embed_passages(args, passages, model):
    max_length = 32768
    batch_size = args.per_gpu_batch_size
    allids = []
    all_texts = []
    allembeddings = []
    batch_ids = []
    batch_texts = []

    def add_eos(input_examples, eos_token):
        input_examples = [input_example + eos_token for input_example in input_examples]
        return input_examples
        
    for _, row in tqdm(passages.iterrows()):
        allids.append(row.iloc[0])
        all_texts.append(str(row.iloc[2]) + ' ' + str(row.iloc[1]))
    allembeddings = model.encode(add_eos(all_texts, model.tokenizer.eos_token), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    # allembeddings = normalize_np(allembeddings, p=2, dim=1)

    return allids, allembeddings

@torch.no_grad()
def embed_passages_stella(args, passages, model):
    from sklearn.preprocessing import normalize
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


def embed_passages_transformers(args, passages, model, tokenizer):
    batch_texts = []
    allembeddings = []
    allids = []
    batch_ids = []

    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
     
    max_length = 8192
    batch_size = args.per_gpu_batch_size

    # Embed the documents
    for _, row in tqdm(passages.iterrows()):
        batch_ids.append(row.iloc[0])
        batch_texts.append(str(row.iloc[2]) + ' ' + str(row.iloc[1]))
        if len(batch_texts) == batch_size:
            # docs_vectors = model.encode(batch_texts)
            allids.extend(batch_ids)
            batch_dict = tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
            outputs = model(**batch_dict, output_hidden_states=True)
            embeddings = last_token_pool(outputs.hidden_states[-1], batch_dict['attention_mask'])

            # add embeddings and ids
            allembeddings.append(embeddings)
            # reset batch
            batch_texts = []
            batch_ids = []
    # process the last batch
    if len(batch_texts) > 0:
        batch_dict = tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
        outputs = model(**batch_dict, output_hidden_states=True)
        embeddings = last_token_pool(outputs.hidden_states[-1], batch_dict['attention_mask'])
        allembeddings.append(embeddings)
        allids.extend(batch_ids)
    allembeddings = np.concatenate(allembeddings, axis=0)

    # normalize embeddings
    return allids, normalize_np(allembeddings, p=2, dim=1)
    


def load_llm2vec_model(model_path):
    if 'supervised' in model_path:
        if 'Mistral-7B-Instruct-v0.2' in model_path:
            base_model_path = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
        else:
            base_model_path = '-'.join(model_path.split('-')[:-1])
        
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


def split_num_passages(num_passages, num_shards):
    # split range(num_passages) into chunks of size num_shards
    return [range(i * num_passages // num_shards, (i + 1) * num_passages // num_shards) for i in range(num_shards)]

def main(args):
    # convert_to_tsv(args.passages)
    # args.passages = args.passages.replace('.jsonl', '.tsv')
    tokenizer = None
    if (args.adapter_path is None) and (('stella' in args.model_name_or_path) or ('inf-retriever' in args.model_name_or_path) or ('NV-Embed' in args.model_name_or_path)):
        print('loading model from ', args.model_name_or_path)
        print('adapter path is None', args.adapter_path is None)
        model = SentenceTransformer(args.model_name_or_path, trust_remote_code=True)
        if 'inf-retriever' in args.model_name_or_path:
            model.max_seq_length = 8192
        if 'NV-Embed' in args.model_name_or_path:
            model.max_seq_length = 8192
            model.tokenizer.padding_side = 'right'
    elif 'train_doc_encoder' in args.adapter_path:
        print('loading train_doc_encoder model')
        # base_model = AutoModel.from_pretrained(args.model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        print('loading base model from', args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        print('loading adapter from', args.adapter_path)
        model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=False)
        model = model.merge_and_unload()
    elif 'LLM2Vec' in args.model_name_or_path or 'llm2vec' in args.model_name_or_path:
        model = load_llm2vec_model(args.model_name_or_path)
    else:
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        
    print('finish loading model')
    
    model.eval()
    model = model.cuda()
    
    if not args.no_fp16:
        model = model.half()

    # shard_size = 90
    print('start embedding, shard_size: ', args.shard_size)
    shard_size = args.shard_size
    
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    print('start_idx: ', start_idx, 'end_idx: ', end_idx)
        
    passages = pd.read_csv(args.passages,
                chunksize=shard_size,
                skiprows=start_idx,
                delimiter='\t',
                dtype={"id": str, "title": str, "text": str})

    print(f"Embedding generation for {shard_size} passages from idx {start_idx} to {end_idx}.")

    # actually doing the embedding
    for chunk in passages:
        if (args.adapter_path is None) and (('stella' in args.model_name_or_path) or ('inf-retriever' in args.model_name_or_path) or ('LLM2Vec' in args.model_name_or_path) or ('llm2vec' in args.model_name_or_path)):
            allids, allembeddings = embed_passages_stella(args, chunk, model)
        elif 'train_doc_encoder' in args.adapter_path:
            allids, allembeddings = embed_passages_transformers(args, chunk, model, tokenizer)
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
    parser.add_argument("--adapter_path", type=str, default=None, help="path to adapter weights")
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--no_title", action="store_true", help="title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--instruction", type=str, default="", help="instruction for the model")
    parser.add_argument("--shard_size", type=int, default=2500000, help="shard size")
    parser.add_argument("--use_google", action="store_true", help="use google api to embed passages")
    
    args = parser.parse_args()


    main(args)