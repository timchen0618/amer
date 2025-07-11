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

from src.contriever import load_retriever

def write_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for inst in data:
            f.write(json.dumps(inst) + '\n')

def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


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





@torch.no_grad()
def embed_queries_stella(queries, model, model_name_or_path):
    if 'inf-retriever' in model_name_or_path:
        query_prompt_name = "query"
    else:
        query_prompt_name = "s2p_query"
    per_gpu_batch_size = 4

    model.eval()
    embeddings, batch_question = [], []

    for q in tqdm(queries):
        batch_question.append(q)
        if len(batch_question) == per_gpu_batch_size:
            embeddings.append(model.encode(batch_question, prompt_name=query_prompt_name))
            batch_question = []
    if len(batch_question) > 0:
        embeddings.append(model.encode(batch_question, prompt_name=query_prompt_name))
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = normalize_np(embeddings, p=2, dim=1)
    print("Questions embeddings shape:", embeddings.shape)
    return embeddings

@torch.no_grad()
def embed_queries_nv(queries, model):
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",
                             "example_input": "Given a question and some relevant passages, retrieve passages that answer the question but are not in the input set.",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "

    max_length = 32768
    model.eval()
    embeddings, batch_question = [], []

    for q in queries:
        batch_question.append(q)
    embeddings = model._do_encode(batch_question, batch_size=4, instruction=query_prefix, max_length=max_length, num_workers=32, return_numpy=True)
    embeddings = normalize_np(embeddings, p=2, dim=1)
    print("Questions embeddings shape:", embeddings.shape)
    return embeddings

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

def embed_passages_nv(passages, model):
    max_length = 32768
    batch_size = 4
    all_texts = []
    allembeddings = []


    for row in tqdm(passages):
        all_texts.append(str(row['title']) + ' ' + str(row['text']))
    allembeddings = model._do_encode(all_texts, batch_size=batch_size, instruction="", max_length=max_length, num_workers=32, return_numpy=True)
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
def embed_queries(queries, model, model_name, tokenizer=None, device=None):
    if 'contriever' in model_name:
        return embed_batch_texts(queries, model, tokenizer, device)
    if ('stella' in model_name) or ('inf-retriever' in model_name):
        return embed_queries_stella(queries, model, model_name)
    else:
        return embed_queries_nv(queries, model)
        
@torch.no_grad()
def embed_passages(passages, model, model_name, tokenizer=None, device=None):
    if 'contriever' in model_name:
        return embed_batch_texts(passages, model, tokenizer, device)
    if ('stella' in model_name) or ('inf-retriever' in model_name):
        return embed_passages_stella(passages, model)
    else:
        return embed_passages_nv(passages, model)
        
    
def load_model(model_name):
    if 'contriever' in model_name:
        model, tokenizer, _ = load_retriever(model_name)
        model.eval()
        return model, tokenizer
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



def read_jsonl(data_path):
    import json
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_embeddings_from_data(data, model, model_name, tokenizer=None, device=None, unsqueeze_0=False, question_only=False, no_question=False):
    all_embeddings = []
    question_embeddings = []
    all_lens = []
    
    if not no_question:  # if no_question is True, we don't need to embed the questions
        questions = [inst['question'] if 'question' in inst else inst['question_text'] for inst in data ]
        question_embeddings = embed_queries(questions, model, model_name, tokenizer=tokenizer, device=device)
        print('question_embeddings.shape', question_embeddings.shape)
    
    if question_only:    # if question_only is True, we only need to embed the questions
        assert no_question == False, "question_only and no_question cannot be both True"
        all_lens = [1] * len(question_embeddings)
        print(all_lens[:100])
        return question_embeddings, np.array(all_lens)
    else:                # if question_only is False, we need to embed the contexts     
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
        if not no_question:  # if there is a question, we need to concatenate the question embedding with the context embeddings
            start_idx = 0
            for _len in all_lens:
                embeddings_list = embeddings[start_idx:start_idx+_len]  # get the embeddings for the current batch
                all_embeddings.append(np.concatenate([question_embeddings[i].reshape(1, -1), embeddings_list], axis=0))
                start_idx += _len
            assert start_idx == embeddings.shape[0], f"start_idx: {start_idx}, embeddings.shape[0]: {embeddings.shape[0]}"
            all_lens = [1+_len for _len in all_lens]
            all_embeddings = np.concatenate(all_embeddings, axis = 0)
        else:
            all_embeddings = embeddings
            
        if unsqueeze_0:
            assert np.unique(all_lens).size == 1, f"all_lens: {all_lens}"
            all_embeddings = all_embeddings.reshape(len(all_lens), all_lens[0], -1)
        
        # all_embeddings = np.concatenate(all_embeddings, axis = 0)
        print(all_embeddings.shape)
        print(all_lens[:100])
        return all_embeddings, np.array(all_lens)



def check_context_same(positive_ctxs, ctx2):
    if isinstance(positive_ctxs[0], list):
        positive_ctxs = [l for x in positive_ctxs for l in x]
    if ctx2 == '':
        return True
    for ctx in positive_ctxs:
        # print(ctx)
        if (ctx['title'] == ctx2['title'] and ctx['title'] != '') or ctx['text'] == ctx2['text']:
            print('=========')
            print(ctx['title'], ctx2['title'])
            print(ctx['text'], ctx2['text'])
            return True
    return False

def get_random_embeddings(data, model, model_name, corpus, length=5, tokenizer=None, device=None):
    
    # return size (batch, 1, dim)
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

def load_msmarco_data(split='train', top_k=None, return_corpus=False):
    queries = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/msmarco/queries.jsonl')
    qrels = read_tsv(f'/scratch/cluster/hungting/projects/autoregressive/data/msmarco/qrels/{split}.tsv')[1:]
    corpus = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/msmarco/corpus.jsonl')
    
    id2query = {q['_id']: q['text'] for q in queries}
    cid2corpus = {c['_id']: {"title": c['title'], "text": c['text']} for c in corpus}
    
    if top_k is not None:
        qrels = qrels[:top_k]

    # query-id    corpus-id   score
    data = {}
    for row in qrels:
        query = id2query[row[0]]
        cid = row[1]
        if query not in data:
            data[query] = {'question': query, 'ctxs': []}
        if len(data[query]['ctxs']) < 1:
            data[query]['ctxs'].append(cid2corpus[cid])
    if return_corpus:
        return list(data.values()), cid2corpus
    else:
        return list(data.values())

def load_nq_data(split='train', top_k=None, return_corpus=False):
    if split == 'train':
        qrels = read_tsv(f'/scratch/cluster/hungting/projects/autoregressive/data/nq/nq-train/qrels/train.tsv')[1:]
        queries = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/nq/nq-train/queries.jsonl')
        corpus = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/nq/nq-train/corpus.jsonl')
    elif split == 'dev':
        qrels = read_tsv(f'/scratch/cluster/hungting/projects/autoregressive/data/nq/qrels/test.tsv')[1:]
        queries = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/nq/queries.jsonl')
        corpus = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/nq/corpus.jsonl')
    else:
        raise NotImplementedError
    id2query = {q['_id']: q['text'] for q in queries}
    cid2corpus = {c['_id']: {"title": c['title'], "text": c['text']} for c in corpus}

    if top_k is not None:
        qrels = qrels[:top_k]
    # query-id    corpus-id   score
    data = {}
    for row in qrels:
        query = id2query[row[0]]
        cid = row[1]
        if query not in data:
            data[query] = {'question': query, 'ctxs': []}
        data[query]['ctxs'].append(cid2corpus[cid])
    if return_corpus:
        return list(data.values()), cid2corpus
    else:
        return list(data.values())

def compute_num_ctxs(data):
    lens = [len(inst['ctxs']) for inst in data]
    return sum(lens) / float(len(lens))

"""
Viable Commands:
1. gen_distill: generate data for single question embedding training (distillation)
2. write_question_only: write question only jsonl files
3. gen_contrastive: generate data for contrastive training
4. gen_contrastive_hard_negatives: generate hard negatives data embeddings for contrastive training
5. oracle: generate oracle data embeddings for single question embedding training
6. gen_doc_embeddings_for_clustering: generate document embeddings for clustering

"""



# command = 'gen_distill'
# command = 'write_question_only'
command = 'gen_contrastive'


from pathlib import Path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_mapping = {
    'qampari': {'train': '/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/train_data_gt_qampari_corpus.jsonl',
                'dev': '/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl'},
    'nq': {'train': '/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/nq_train_question_only.jsonl',
        'dev': '/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/nq_dev_question_only.jsonl'},
    'msmarco': {'train': '/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/msmarco_train_question_only.jsonl',
                'dev': '/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/msmarco_dev_question_only.jsonl'},
    'ambiguous': {'train': '/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_train.jsonl',
                'dev': '/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev.jsonl'},
    'ambiguous_qe': {'train': '/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_train.jsonl',
                'dev': '/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev.jsonl'},
    'berds': {'train': '/scratch/cluster/hungting/projects/autoregressive/data/berds/berds_with_gold_docs.jsonl',}
}
model_mapping = {
    'inf': 'infly/inf-retriever-v1-1.5b',
    'stella': 'NovaSearch/stella_en_400M_v5',
    'cont': 'facebook/contriever-msmarco',
}
    
if command == 'gen_distill':
    ########################################################
    ## generate data for single question embedding training (distillation)
    ########################################################
    # for model_name in ['inf', 'stella', 'cont']:
    # for model_name in ['cont']:
    import sys
    # model_name = sys.argv[1]
    length_maps = {
        'ambiguous': [2,3,4,5],
        'ambiguous_qe':[2,3,4,5],
    }
    for split in ['dev', 'train']:
        for model_name in ['inf', 'stella', 'cont']:
            model, tokenizer = load_model(model_mapping[model_name])
            model = model.to(device)
            # for data_name in ['qampari', 'msmarco', 'nq']:
            
            # for data_name in ['msmarco', 'nq']:   
            for data_name in ['ambiguous_qe']:
                # locate and create the directory for the data
                rootdir = Path('../../autoregressive/data_creation/raw_data/') / f'{data_name}_{model_name}'
                rootdir.mkdir(parents=True, exist_ok=True)

                # load the data, and set output file names
                data_file = data_mapping[data_name][split]
                data = read_jsonl(data_file)
                if data_name == 'ambiguous' or data_name == 'ambiguous_qe':
                    lengths = length_maps[data_name]
                    data = [l for l in data if len(l['positive_ctxs']) in lengths]
                            
                print('loaded data for {} with {} instances'.format(data_name, len(data)))
                output_numpy_file = rootdir / f'{data_name}_{split}_question_only.npy'
                output_lens_file = rootdir / f'{data_name}_{split}_question_only_lens.npy'

                # get the embeddings and save them
                all_embeddings, all_lens = get_embeddings_from_data(data, model, model_mapping[model_name], tokenizer=tokenizer, device=device, unsqueeze_0=True, question_only=True)
                np.save(output_numpy_file, all_embeddings)
                np.save(output_lens_file, all_lens)

if command == 'write_question_only':
    ########################################################
    ## Write to question only jsonl files
    ########################################################
    # for data_name in ['qampari', 'nq', 'msmarco']:
    # for data_name in ['nq', 'msmarco']:
    for data_name in ['ambiguous']:
        rootdir = Path('../../autoregressive/data_creation/raw_data/')
        rootdir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'dev']:
            if data_name == 'msmarco':
                data = load_msmarco_data(split=split)
            elif data_name == 'nq':
                data = load_nq_data(split=split)
            else:
                data_file = data_mapping[data_name][split]
                data = read_jsonl(data_file)
            print('loaded data for {} with {} instances'.format(data_name, len(data)))
            keys = list(data[0].keys())
            for inst in data:
                for k in keys:
                    if k not in ['question', 'question_text'] and k in inst:
                        inst.pop(k)
            out_jsonl_file = rootdir / f'{data_name}_{split}_question_only.jsonl'
            write_jsonl(data, out_jsonl_file)
            out_json_file = rootdir / f'{data_name}_{split}_question_only.json'
            for inst in data:
                inst['ctxs'] = []
                inst['answers'] = ['']
            write_json(data, out_json_file)


if command == 'gen_contrastive':
    ########################################################
    #nerate data for contrastive training
    ########################################################
    
    # for data_name in ['nq']:
    for data_name in ['qampari']:
        rootdir = Path('../../autoregressive/data_creation/raw_data/')
        rootdir.mkdir(parents=True, exist_ok=True)
        
        for model_name in ['inf', 'cont', 'stella']:
            rootdir = Path('../../autoregressive/data_creation/raw_data/') / f'{data_name}_{model_name}'
            model, tokenizer = load_model(model_mapping[model_name])
            model = model.to(device)
            # model, tokenizer = None, None
            length_maps = {
                'msmarco': [1],
                'nq': [1],
                'qampari': [5,6,7,8],
                'ambiguous': [2,3,4,5],
                'ambiguous_qe':[2,3,4,5],
                'arguana': [2],
                'opinionqa': [2],
                'kialo': [2,3,4,5],
                'berds': [2],
            }
            
            # for length in length_maps[data_name]:
            for length in [5]:
                for split in ['train']:
                    if data_name == 'msmarco':
                        data, cid2corpus = load_msmarco_data(split=split, return_corpus=True)
                    elif data_name == 'nq':
                        data, cid2corpus = load_nq_data(split=split, return_corpus=True)
                    else:
                        data_file = data_mapping[data_name][split]
                        data = read_jsonl(data_file)

                        if data_name == 'ambiguous' or data_name == 'ambiguous_qe' or data_name == 'berds':
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
                        write_jsonl(write_data, f'/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/{data_name}_{split}_question_only_{length}_ctxs.jsonl')
                        corpus = read_tsv('/scratch/cluster/hungting/chunks_v5.tsv')
                        # corpus = read_tsv('/datastor1/hungting/MassiveDS-140B/massive_ds_140b.tsv')
                        cid2corpus = {c[0]: {"title": c[2], "text": c[1]} for c in corpus}
                    print('loaded data for {} with {} instances'.format(data_name, len(data)))
                
                    if length == 1:
                        output_numpy_file = rootdir / f'{data_name}_{split}_positive_embeddings.npy'
                        output_random_embeddings_file = rootdir / f'{data_name}_{split}_random_embeddings.npy'
                    else:
                        output_numpy_file = rootdir / f'{data_name}_{split}_positive_embeddings_{length}.npy'
                        output_random_embeddings_file = rootdir / f'{data_name}_{split}_random_embeddings_{length}.npy'
                    
                    all_embeddings, all_lens = get_embeddings_from_data(data, model, model_mapping[model_name], tokenizer=tokenizer, device=device, 
                                                                        unsqueeze_0=True, 
                                                                        question_only=False,
                                                                        no_question=True)
                    np.save(output_numpy_file, all_embeddings)

                    # if data_name == 'ambiguous':
                    #     for inst in data:
                    #         # flatten the positive_ctxs
                    #         inst['positive_ctxs'] = [l for x in inst['positive_ctxs'] for l in x]
                        
                    random_embeddings, data = get_random_embeddings(data, model, model_mapping[model_name], list(cid2corpus.values()), length=length, tokenizer=tokenizer, device=device)
                    np.save(output_random_embeddings_file, random_embeddings)


if command == 'gen_contrastive_hard_negatives':
    ########################################################
    #nerate data for contrastive training
    ########################################################
    corpus = read_tsv('/scratch/cluster/hungting/chunks_v5.tsv')
    for data_name in ['qampari']:
        rootdir = Path('../../autoregressive/data_creation/raw_data/')
        rootdir.mkdir(parents=True, exist_ok=True)
        
        for model_name in ['inf', 'cont', 'stella']:
            rootdir = Path('../../autoregressive/data_creation/raw_data/') / f'{data_name}_{model_name}'
            model, tokenizer = load_model(model_mapping[model_name])
            model = model.to(device)
            # model, tokenizer = None, None
            length_maps = {
                'msmarco': [1],
                'nq': [1],
                'qampari': [5,6,7,8],
                'ambiguous': [2,3,4,5],
                'ambiguous_qe': [2,3,4,5],
            }
            
            for length in length_maps[data_name]:
                for split in ['train', 'dev']:
                    # load the hard negative data.
                    if data_name == 'ambiguous':
                        hard_negative_data = read_jsonl(f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/ambignq+nqopen-all_multi_answer_evidence_{split}_hard_negative_{model_name}.jsonl')
                    elif data_name == 'qampari':
                        hard_negative_data = read_jsonl(f'/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/{split}_data_gt_qampari_corpus_hard_negative_{model_name}.jsonl')
                    else:
                        raise NotImplementedError

                    # load the original data.
                    if data_name == 'msmarco':
                        data, cid2corpus = load_msmarco_data(split=split, return_corpus=True)
                    elif data_name == 'nq':
                        data, cid2corpus = load_nq_data(split=split, return_corpus=True)
                    else:
                        data_file = data_mapping[data_name][split]
                        data = read_jsonl(data_file)
                        
                        # select the data with the appropriate length.
                        indices = []
                        for i, l in enumerate(data):
                            if data_name == 'ambiguous':
                                if len(l['positive_ctxs']) == length:
                                    indices.append(i)
                            elif data_name == 'qampari':
                                if len(l['ground_truths']) == length:
                                    indices.append(i)
                            else:
                                raise NotImplementedError
                        data = [data[i] for i in indices]
                        
                        
                        # for ambiguous and qampari, we need to select appropriate hard negative data.
                        # for msmarco and nq, we can directly use the hard negative data.
                        hard_negative_data = [hard_negative_data[i] for i in indices]
                        
                    assert len(data) == len(hard_negative_data), (len(data), len(hard_negative_data))
                    for inst in hard_negative_data:
                        assert 'ctxs' in inst and not ('positive_ctxs' in inst or 'ground_truths' in inst), inst.keys()
                        # assert len(inst['ctxs']) == length, (f'{len(inst["ctxs"])} != {length}; the length of the hard negative data is not correct.')
                        if len(inst['ctxs']) != length:
                            print('length is not correct, falling back to random selection.')
                            lens_to_meet = length - len(inst['ctxs'])
                            negative_ctxs = []
                            for _ in range(lens_to_meet):
                                negative_ctx = random.choice(corpus)
                                if 'positive_ctxs' in inst:
                                    positive_ctxs = inst['positive_ctxs']
                                elif 'ground_truths' in inst:
                                    positive_ctxs = inst['ground_truths']
                                else:
                                    raise NotImplementedError
                                while check_context_same(positive_ctxs, negative_ctx):
                                    negative_ctx = random.choice(corpus)
                                negative_ctxs.append(negative_ctx)
                            inst['ctxs'] += negative_ctxs
                        assert len(inst['ctxs']) == length, (f'{len(inst["ctxs"])} != {length}; the length of the hard negative data is not correct.')
                        
                    # hard negative embeddings name.
                    if length == 1:
                        output_numpy_file = rootdir / f'{data_name}_{split}_hard_negative_embeddings.npy'
                    else:
                        output_numpy_file = rootdir / f'{data_name}_{split}_hard_negative_embeddings_{length}.npy'
                    
                    # actually generate the hard negative embeddings.
                    all_embeddings, all_lens = get_embeddings_from_data(hard_negative_data, model, model_mapping[model_name], tokenizer=tokenizer, device=device, 
                                                                        unsqueeze_0=True, 
                                                                        question_only=False,
                                                                        no_question=True)
                    np.save(output_numpy_file, all_embeddings)
                    
                    
if command == "oracle":
        
    for model_name in ['inf', 'cont', 'stella']:
        model, tokenizer = load_model(model_mapping[model_name])
        model = model.to(device)
        data_name = "ambiguous"
        output_numpy_file = f"/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/{data_name}_dev_oracle_embeddings_{model_name}.npy"
        output_lens_file = f"/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/{data_name}_dev_oracle_embeddings_{model_name}_lengths.npy"
        data_mapping = {"qampari": "/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus_5_to_8_ctxs.jsonl",
                        "ambiguous": "/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.jsonl",}
        
        if data_name == 'msmarco':
            data, cid2corpus = load_msmarco_data(split=split, return_corpus=True)
        elif data_name == 'nq':
            data, cid2corpus = load_nq_data(split=split, return_corpus=True)
        else:
            data_file = data_mapping[data_name]
            data = read_jsonl(data_file)
            
        all_embeddings, all_lens = get_embeddings_from_data(data, model, model_mapping[model_name], tokenizer=tokenizer, device=device, 
                                                            unsqueeze_0=False, 
                                                            question_only=False,
                                                            no_question=True)
        np.save(output_numpy_file, all_embeddings)
        np.save(output_lens_file, all_lens)
        print(all_lens)
            
            
        
if command == 'gen_doc_embeddings_for_clustering':
    ########################################################
    #nerate data for contrastive training
    ########################################################
    
    # for data_name in ['nq']:
    # for data_name in ['squad', 'quora_duplicates', 'nq', 'trivia_qa', 't2ranking', 'eli5_question_answer', 'dureader', 'hotpot_qa', 'msmarco_document', 'msmarco_passage', 'fever', 'miracl', 'mrtydi', 'allnli']:
    for data_name in ['trivia_qa_10']:
        rootdir = Path('/var/local/timchen0618/retrieval_outputs/echo_data/mteb_retriever/stella-400M')
        data = read_jsonl(rootdir / f'{data_name}.json')
        
        for model_name in ['stella']:
            model, tokenizer = load_model(model_mapping[model_name])
            model = model.to(device)        

            output_numpy_file = rootdir / f'{data_name}_doc_embeddings.npy'
            output_lens_file = rootdir / f'{data_name}_doc_embeddings_lengths.npy'
            
            all_embeddings, all_lens = get_embeddings_from_data(data, model, model_mapping[model_name], tokenizer=tokenizer, device=device, 
                                                                unsqueeze_0=True, 
                                                                question_only=False,
                                                                no_question=True)
            np.save(output_numpy_file, all_embeddings)
            np.save(output_lens_file, all_lens)