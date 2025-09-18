import torch
from torch.utils.data import DataLoader
from src.dataset import (
    load_embeddings_dataset,
    contrastive_eval_collator,
    mse_eval_collator,
    DataHandler
)
from functools import partial
import structlog
logger = structlog.get_logger()
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json


"""
    The script is used to group the data by the distance between the target document embeddings.
    The distance is calculated using the L2 distance.
    The data is grouped into two groups: large distance and small distance.
    Decided by the 1/4 and 3/4 quantiles of the distance.
"""


def read_jsonl(data_path):
    import json
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def normalize_np(x, p=2, dim=1, eps=1e-12):
    """
    NumPy implementation of torch.nn.functional.normalize
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    return x / norm


def load_model(model_name):
    if ('stella' in model_name) or ('inf-retriever' in model_name):
        model = SentenceTransformer(model_name, trust_remote_code=True)
        if 'inf-retriever' in model_name:
            model.max_seq_length = 8192
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return model, None


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
def embed_passages(passages, model, model_name):
    if ('stella' in model_name) or ('inf-retriever' in model_name):
        return embed_passages_stella(passages, model)
    
    
    
def get_embeddings_from_data(inst, model, model_name):
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
    
    # contexts = [ctx['title'] + ' ' + ctx['text'] if 'title' in ctx else ctx['text'] for ctx in contexts]

    embeddings = embed_passages(contexts, model, model_name) # (batch*len, dim)
    return embeddings

def load_input_data(loss_function, question_only, batch_size_training, get_split, input_data_path):
    use_contrastive_data = 'Contrastive' in loss_function
    logger.info('use_contrastive_data', more_than_strings=use_contrastive_data)
    # Load dataset
    if use_contrastive_data:
        collator = contrastive_eval_collator
    else:
        collator = partial(mse_eval_collator, question_only=question_only)
    full_dataset = load_embeddings_dataset(dataset_path=input_data_path)
    data_handler = DataHandler(full_dataset, collator, batch_size_training, 'train' if 'train' in get_split else 'dev')
    
    # load the corresponding split
    if get_split == 'train-held-out':
        dataloader = data_handler.get_dev_dataloader()
    elif get_split == 'train':
        dataloader = data_handler.get_sequential_train_dataloader()
    elif get_split == 'dev':
        dataloader = data_handler.get_full_dataloader()
    else:
        raise ValueError(f"Invalid split: {get_split}")
    return dataloader


def write_list_to_file(file_path, list_of_indices):
    with open(file_path, 'w') as f:
        for index in list_of_indices:
            f.write(f"{index}\n")

# load_input_data(loss_function, question_only, batch_size_training, get_split, input_data_path):
data_name = 'qampari'
retriever = 'inf'
dataset_name = f"{data_name}_{retriever}"

project_dir = '/scratch/hc3337/projects'


def average_pairwise_distances(embeddings):
    distances = torch.cdist(embeddings, embeddings)
    distances = torch.triu(distances, diagonal=1)
    if distances[distances.nonzero(as_tuple=True)].numel() == 0:
        return 0
    return torch.mean(distances[distances.nonzero(as_tuple=True)]).item()

avg_distances = []
large_distance_indices = []
small_distance_indices = []


if data_name in ['ambiguous', 'ambiguous_qe', 'qampari_5_to_8']:
    if data_name == 'qampari_5_to_8':
        data_name = 'qampari'
    dataset_path = f'training_datasets/llama-1b/{data_name}/{retriever}/autoregressive_{dataset_name}_dev_dataset_1b_contrastive_2_to_5_ctxs'
    dataloader = load_input_data('Hungarian_Contrastive', False, 1, 'dev', dataset_path)

    for i, batch in enumerate(dataloader):
        avg_distances.append(average_pairwise_distances(batch['positive_embeddings']))

    avg_distances = torch.tensor(avg_distances)
    quantile_1 = torch.quantile(avg_distances, 1/4, interpolation='nearest')
    quantile_2 = torch.quantile(avg_distances, 3/4, interpolation='nearest')
    logger.info('avg_distances', quantile_1=quantile_1, quantile_2=quantile_2)

    for i, batch in enumerate(dataloader):
        if average_pairwise_distances(batch['positive_embeddings']) > quantile_2:
            large_distance_indices.append(i)
        if average_pairwise_distances(batch['positive_embeddings']) < quantile_1:
            small_distance_indices.append(i)

    print(len(large_distance_indices))
    print(len(small_distance_indices))

elif data_name in ['qampari']:
    data = read_jsonl(f'{project_dir}/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl')
    model_name = 'infly/inf-retriever-v1-1.5b'
    model, _ = load_model(model_name)
    model.eval()
    all_embeddings = []
    for i, inst in enumerate(tqdm(data)):
        embeddings = get_embeddings_from_data(inst, model, model_name)
        embeddings = torch.from_numpy(embeddings)
        avg_distances.append(average_pairwise_distances(embeddings))
        all_embeddings.append(embeddings)
        
    avg_distances = torch.tensor(avg_distances)
    quantile_1 = torch.quantile(avg_distances, 1/4, interpolation='nearest')
    quantile_2 = torch.quantile(avg_distances, 3/4, interpolation='nearest')
    logger.info('avg_distances', quantile_1=quantile_1, quantile_2=quantile_2)

    for i, embeddings in enumerate(all_embeddings):
        if average_pairwise_distances(embeddings) > quantile_2:
            large_distance_indices.append(i)
        if average_pairwise_distances(embeddings) < quantile_1:
            small_distance_indices.append(i)

    print(len(large_distance_indices))
    print(len(small_distance_indices))

# write_list_to_file(f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/qampari_embeddings_data/large_distance_indices_{retriever}.txt', large_distance_indices)
# write_list_to_file(f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/qampari_embeddings_data/small_distance_indices_{retriever}.txt', small_distance_indices)

write_list_to_file(f'{project_dir}/autoregressive/data/qampari/large_distance_indices_quarter_{retriever}.txt', large_distance_indices)
write_list_to_file(f'{project_dir}/autoregressive/data/qampari/small_distance_indices_quarter_{retriever}.txt', small_distance_indices)
