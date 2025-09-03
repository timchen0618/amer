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

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

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
def embed_passages_stella(passages, model):
    batch_size = 4
    batch_texts = []
    allembeddings = []
    
    # Embed the documents
    for row in tqdm(passages):
        batch_texts.append(str(row))
        if len(batch_texts) == batch_size:
            docs_vectors = model.encode(batch_texts)
            # add embeddings and ids
            allembeddings.append(docs_vectors)
            # reset batch
            batch_texts = []
    # process the last batch
    if len(batch_texts) > 0:
        docs_vectors = model.encode(batch_texts)
        allembeddings.append(docs_vectors)
    allembeddings = np.concatenate(allembeddings, axis=0)
    allembeddings = normalize_np(allembeddings, p=2, dim=1)
    return allembeddings

def load_model(model_name_or_path):
    if ('stella' in model_name_or_path) or ('inf-retriever' in model_name_or_path):
        model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
        if 'inf-retriever' in model_name_or_path:
            model.max_seq_length = 8192
    else:
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.eval()
    model = model.cuda()
    model = model.half()
    return model

def save_embedding_results_flexible(data, output_dir):
    """Save clustering results in flexible formats that handle variable-length data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save as pickle files (most flexible for Python objects)
    with open(output_dir / f'embeddings.pkl', 'wb') as f:
        pickle.dump(data, f)

    
def compute_l2_distance(query_1, query_2):
    return np.linalg.norm(query_1 - query_2)

def compute_cosine_similarity(query_1, query_2):
    return np.dot(query_1, query_2) / (np.linalg.norm(query_1) * np.linalg.norm(query_2))
    
def compute_averge_target_distance_same_example(target_vectors_list):
    l2_distance_list = []
    cosine_similarity_list = []
    for i in range(len(target_vectors_list)):
        all_target_vectors = target_vectors_list[i]
        for j in range(len(all_target_vectors)):
            for k in range(j+1, len(all_target_vectors)):
                l2_distance_list.append(compute_l2_distance(all_target_vectors[j], all_target_vectors[k]))
                cosine_similarity_list.append(compute_cosine_similarity(all_target_vectors[j], all_target_vectors[k]))
    return np.mean(l2_distance_list), np.mean(cosine_similarity_list)

if __name__ == "__main__":
    model = load_model('infly/inf-retriever-v1-1.5b')
    domains = json.load(open('outputs/domains.json', 'r'))
    # output_dir = 'outputs/q_docs_wctx_1/'
    # for output_dir in ['outputs/q_docs_wctx_1/', 'outputs/q_docs_woctx_1/', 'outputs/q_docs_wctx/', 'outputs/q_docs_woctx/', 'outputs/q_docs_existing/', 'outputs/q_docs_existing_1/']:
    for output_dir in ['vllm_outputs/q_docs_woctx_4_100/', 'vllm_outputs/q_docs_woctx_4_200/', 'vllm_outputs/q_docs_woctx_4_150/']:
        print('Generating embeddings for', output_dir)
        all_data = {"question_embeddings": [], "document_embeddings": []}
        questions = []
        if 'existing' in output_dir:
            data = read_jsonl(f'{output_dir}/existing_q2docs_1k.jsonl')
            # read questions from the data
            # q_data = read_jsonl('../data/eli5+researchy_questions_1k.jsonl')[:200]
            questions += [inst['question'] for inst in data]
            
            for inst in data:
                if len(inst['positive_documents']) == 0 or not isinstance(inst['positive_documents'], list):
                    continue
                embeddings = embed_passages_stella(inst['positive_documents'], model)
                all_data["document_embeddings"].append(embeddings)
        else:
            i = 0
            for domain, sub_domains in tqdm(domains.items()):
                for sub_domain in sub_domains:
                    data = json.load(open(f'{output_dir}/{domain}_{sub_domain}_q_and_docs.json', 'r'))
                    # read questions from the data
                    questions += [inst['question'] for inst in data]                 
                    for inst in data:
                        if len(inst['positive_documents']) == 0 or not isinstance(inst['positive_documents'], list):
                            continue
                        embeddings = embed_passages_stella(inst['positive_documents'], model)
                        all_data["document_embeddings"].append(embeddings)
                i += 1
            
        question_embeddings = embed_passages_stella(questions, model)
        print('question_embeddings.shape', question_embeddings.shape)
        all_data["question_embeddings"] = question_embeddings
        save_embedding_results_flexible(all_data, f'{output_dir}')
        
        
        
        # ## For Target Question ##         
        # target_question = 'What role does transfer learning play in reducing the need for large labeled datasets in AI applications?'
        # all_embeddings = []
        # for domain, sub_domains in tqdm(domains.items()):
        #     for sub_domain in sub_domains:
        #         data = json.load(open(f'{output_dir}/{domain}_{sub_domain}_q_and_docs.json', 'r'))
                
        #         for inst in data:
        #             if inst['question'] != target_question or len(inst['positive_documents']) == 0 or not isinstance(inst['positive_documents'], list):
        #                 continue
        #             embeddings = embed_passages_stella(inst['positive_documents'], model)
        #             all_embeddings.append(embeddings)
        #         i += 1
        # l2, cosine = compute_averge_target_distance_same_example(all_embeddings)
        # print(f'{output_dir} has {l2} l2 distance and {cosine} cosine similarity')