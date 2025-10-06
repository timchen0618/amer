import torch
import torch.nn.functional as F
from transformers import AutoModel
import transformers
from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm

import json
from pathlib import Path
import numpy as np

import random
import os
import csv

import argparse

def write_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for inst in data:
            f.write(json.dumps(inst) + '\n')

def read_tsv(file_path):
    data = []
    with open(file_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader):
            if not row[0] == "id":
                data.append(row)
            if len(data) > 3000000:
                break
    return data

def read_jsonl(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


@torch.no_grad()
def embed_passages_stella(passages, model):
    batch_size = 128
    batch_texts = []
    allembeddings = []
    
    # Embed the documents
    for row in tqdm(passages):
        batch_texts.append(str(row['title']) + ' ' + str(row['text']))
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


@torch.no_grad()
def embed_batch_texts(texts, model, tokenizer, device):
    embeddings, batch_texts = [], []
    for k, _text in enumerate(tqdm(texts)):
        if isinstance(_text, dict):
            _text = _text['title'] + ' ' + _text['text']
        batch_texts.append(_text)

        if len(batch_texts) == 256 or k == len(texts) - 1:

            encoded_batch = tokenizer.batch_encode_plus(
                batch_texts,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True,
            )
            encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
            output = model(**encoded_batch)
            embeddings.append(output)

            batch_texts = []

    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    return embeddings

        
@torch.no_grad()
def embed_passages(passages, model, model_name, tokenizer=None, device=None):
    if 'contriever' in model_name:
        return embed_batch_texts(passages, model, tokenizer, device)
    if ('stella' in model_name) or ('inf-retriever' in model_name):
        return embed_passages_stella(passages, model)
    else:
        raise NotImplementedError
        
    
def load_model(model_name):
    if ('stella' in model_name) or ('inf-retriever' in model_name):
        model = SentenceTransformer(model_name, trust_remote_code=True)
        if 'inf-retriever' in model_name:
            model.max_seq_length = 8192
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return model, None


def normalize_np(x, p=2, dim=1, eps=1e-12):
    """
    NumPy implementation of torch.nn.functional.normalize
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    return x / norm



def get_embeddings_from_data(data, model, model_name, tokenizer=None, device=None, unsqueeze_0=False):
    all_embeddings = []
    all_lens = []
     
    all_contexts = []
    for i, inst in enumerate(tqdm(data)):
        if ('ground_truths' in inst and len(inst['ground_truths']) > 0) or ('positive_ctxs' in inst and len(inst['positive_ctxs']) > 0):
            gt_string = 'ground_truths' if 'ground_truths' in inst else 'positive_ctxs'
            if isinstance(inst[gt_string][0], list):
                contexts = [l[0] for l in inst[gt_string]]
            elif isinstance(inst[gt_string][0], dict):
                contexts = inst[gt_string]
            else:
                print(inst[gt_string][0])
                raise NotImplementedError
        elif 'ctxs' in inst and len(inst['ctxs']) > 0:
            contexts = inst['ctxs']
        else:
            raise NotImplementedError
        all_contexts.extend(contexts)
        all_lens.append(len(contexts))
        
    if len(list(set(all_lens))) > 1 and unsqueeze_0:
        print(all_lens)
        raise ValueError('all_lens are not the same, with unsqueeze_0 is True')

    embeddings = embed_passages(all_contexts, model, model_name, tokenizer=tokenizer, device=device) # (batch*len, dim)
    all_embeddings = embeddings
        
    if unsqueeze_0:
        assert np.unique(all_lens).size == 1, f"all_lens: {all_lens}"
        all_embeddings = all_embeddings.reshape(len(all_lens), all_lens[0], -1)
    
    return all_embeddings, np.array(all_lens)



def check_context_same(positive_ctxs, ctx2):
    if isinstance(positive_ctxs[0], list):
        positive_ctxs = [l for x in positive_ctxs for l in x]
    if ctx2 == '':
        return True
    for ctx in positive_ctxs:
        if (ctx['title'] == ctx2['title'] and ctx['title'] != '') or ctx['text'] == ctx2['text']:
            return True
    return False

def get_random_embeddings(data, model, model_name, corpus, length=5, tokenizer=None, device=None):
    all_embeddings = []
    documents = []
    for inst in tqdm(data):
        negative_ctxs = []
        for _ in range(length):
            negative_ctx = random.choice(corpus)
            if 'positive_ctxs' in inst:
                positive_ctxs = inst['positive_ctxs']
            elif 'ctxs' in inst:
                positive_ctxs = inst['ctxs']
            else:
                positive_ctxs = inst['ground_truths']
            while check_context_same(positive_ctxs, negative_ctx):
                negative_ctx = random.choice(corpus)
            documents.append(negative_ctx)
            negative_ctxs.append(negative_ctx)
        inst['negative_ctxs'] = negative_ctxs
        
    all_embeddings = embed_passages(documents, model, model_name, tokenizer=tokenizer, device=device)
    all_embeddings = all_embeddings.reshape(len(data), length, -1)
    return all_embeddings, data



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file', type=str, default='/path/to/corpus.tsv')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    from pathlib import Path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_dir = str(Path(__file__).parent.parent)

    data_mapping = {
        'qampari': {'train': f'{project_dir}/amer_data/qampari_train.jsonl'},
        'ambiguous_qe': {'train': f'{project_dir}/amer_data/ambigqa_train.jsonl'},
    }
    model_mapping = {
        'inf': 'infly/inf-retriever-v1-1.5b',
        'stella': 'NovaSearch/stella_en_400M_v5',
        'cont': 'facebook/contriever-msmarco',
    }
                

    corpus = read_tsv(args.corpus_file)
    for data_name in ['qampari', 'ambiguous_qe']: 
        rootdir = Path(f'{project_dir}/data_creation/raw_data/')
        rootdir.mkdir(parents=True, exist_ok=True)
        
        for model_name in ['inf']:
            rootdir = Path(f'{project_dir}/data_creation/raw_data/') / f'{data_name}_{model_name}'
            rootdir.mkdir(parents=True, exist_ok=True)
            model, tokenizer = load_model(model_mapping[model_name])
            model = model.to(device)
            
            length_maps = {
                'qampari': [5,6,7,8],
                'ambiguous_qe':[2,3,4,5],
            }
            
            for length in length_maps[data_name]:
                for split in ['train']:
                    data_file = data_mapping[data_name][split]
                    data = read_jsonl(data_file)

                    if data_name == 'ambiguous_qe':
                        data = [l for l in data if len(l['positive_ctxs']) == length]
                    elif data_name == 'qampari':
                        data = [l for l in data if len(l['ground_truths']) == length]
                    else:
                        raise NotImplementedError
                    
                    if 'question_text' in data[0]:
                        write_data = [{"question_text": l['question_text']} for l in data]
                    else:
                        write_data = [{"question": l['question']} for l in data]
                    print(len(write_data))
                    write_jsonl(write_data, f'{project_dir}/data_creation/raw_data/{data_name}_{split}_question_only_{length}_ctxs.jsonl')
                    
                    cid2corpus = {c[0]: {"title": c[2], "text": c[1]} for c in corpus}
                    print('loaded data for {} with {} instances'.format(data_name, len(data)))
                
                    output_numpy_file = rootdir / f'{data_name}_{split}_positive_embeddings_{length}.npy'
                    output_random_embeddings_file = rootdir / f'{data_name}_{split}_random_embeddings_{length}.npy'
                    
                    all_embeddings, all_lens = get_embeddings_from_data(data, model, model_mapping[model_name], tokenizer=tokenizer, device=device, unsqueeze_0=True)
                    np.save(output_numpy_file, all_embeddings)                        
                    random_embeddings, data = get_random_embeddings(data, model, model_mapping[model_name], list(cid2corpus.values()), length=length, tokenizer=tokenizer, device=device)
                    np.save(output_random_embeddings_file, random_embeddings)