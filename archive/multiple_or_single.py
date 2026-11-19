from pathlib import Path
import json
import numpy as np


"""
The script is used to choose whether to use multiple or single question embeddings for the retrieval, 
during inference time. 

The selection criteria is based on the cosine distance between the predicted question embeddings.
"""


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def write_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def normalize_np(x, p=2, dim=1, eps=1e-12):
    """
    NumPy implementation of torch.nn.functional.normalize
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    return x / norm

def average_pairwise_distances(embeddings):
    # Normalize embeddings for cosine similarity
    normalized_embeddings = normalize_np(embeddings, p=2, dim=1)
    # Compute cosine similarity matrix
    cosine_similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
    # Convert to cosine distances (1 - cosine_similarity)
    cosine_distances = 1 - cosine_similarities
    # Get upper triangular part (excluding diagonal)
    distances = np.triu(cosine_distances, k=1)
    # Get non-zero elements (upper triangular part)
    non_zero_distances = distances[distances > 0]
    if len(non_zero_distances) == 0:
        return 0
    return np.mean(non_zero_distances)


rootdir = '/scratch/hc3337/projects/autoregressive/results/llama-1b/qampari_inf/fixed_model/'

suffix_list = [
    "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10",
    "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10",
    "mix_one_normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10",
]

# QAMPARI
for suffix in suffix_list:
    cosine_distance_list = []
    new_data = []
    file_path = Path(rootdir) / suffix / 'retrieval_out_dev_qampari_5_to_8_max_new_tokens_5.jsonl'
    one_file_path = Path(rootdir) / suffix / 'retrieval_out_dev_qampari_5_to_8_max_new_tokens_1.jsonl'
    embedding_path = Path(rootdir) / suffix / 'retrieval_out_dev_qampari_5_to_8.npy'
    retrieved_docs = read_jsonl(file_path)
    one_retrieved_docs = read_jsonl(one_file_path)
    embeddings = np.load(embedding_path)
    assert len(retrieved_docs) == len(embeddings) // 5
    step = 5
    for i in range(len(retrieved_docs)):
        # get the embeddings, and the distance
        embedding_inst = embeddings[i*step:(i+1)*step]
        cosine_distance = average_pairwise_distances(embedding_inst)
        cosine_distance_list.append(cosine_distance)
    
    avg_cosine_distance = np.mean(np.array(cosine_distance_list))
    percentiles = {25: 0, 50: 0, 75: 0}
    for k in [25, 50, 75]:
        percentiles[k] = np.percentile(np.array(cosine_distance_list), k)
        
    for k, v in percentiles.items():
        print(f'{k} percentile: {v}')
    print('avg cosine distance', avg_cosine_distance)
    for k in [25, 50, 75]:
        new_data = []
        for i in range(len(retrieved_docs)):
            inst = retrieved_docs[i]
            one_inst = one_retrieved_docs[i]
            if cosine_distance_list[i] < percentiles[k]:
                new_data.append(one_inst)
            else:
                new_data.append(inst)
        new_file_path = str(file_path).replace('.jsonl', f'_{k}_percent_single.jsonl')
        # print(f'writing to {new_file_path}')
        write_jsonl(new_file_path, new_data)
        
        
rootdir = '/scratch/hc3337/projects/autoregressive/results/llama-1b/ambiguous_qe_inf/fixed_model/'

suffix_list = [    
    "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1",
    "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1",
    "mix_one_normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
]

# AmbigNQ
for suffix in suffix_list:
    cosine_distance_list = []
    new_data = []
    file_path = Path(rootdir) / suffix / 'retrieval_out_dev_ambiguous_qe_max_new_tokens_2.jsonl'
    one_file_path = Path(rootdir) / suffix / 'retrieval_out_dev_ambiguous_qe_max_new_tokens_1.jsonl'
    embedding_path = Path(rootdir) / suffix / 'retrieval_out_dev_ambiguous_qe.npy'
    retrieved_docs = read_jsonl(file_path)
    one_retrieved_docs = read_jsonl(one_file_path)
    embeddings = np.load(embedding_path)
    assert len(retrieved_docs) == len(embeddings) // 2
    step = 2
    for i in range(len(retrieved_docs)):
        # get the embeddings, and the distance
        embedding_inst = embeddings[i*step:(i+1)*step]
        cosine_distance = average_pairwise_distances(embedding_inst)
        cosine_distance_list.append(cosine_distance)
    
    avg_cosine_distance = np.mean(np.array(cosine_distance_list))
    percentiles = {25: 0, 50: 0, 75: 0}
    for k in [25, 50, 75]:
        percentiles[k] = np.percentile(np.array(cosine_distance_list), k)
        
    for k, v in percentiles.items():
        print(f'{k} percentile: {v}')
    print('avg cosine distance', avg_cosine_distance)
    for k in [25, 50, 75]:
        new_data = []
        for i in range(len(retrieved_docs)):
            inst = retrieved_docs[i]
            one_inst = one_retrieved_docs[i]
            if cosine_distance_list[i] < percentiles[k]:
                new_data.append(one_inst)
            else:
                new_data.append(inst)
        new_file_path = str(file_path).replace('.jsonl', f'_{k}_percent_single.jsonl')
        # print(f'writing to {new_file_path}')
        write_jsonl(new_file_path, new_data)