import os
import json
import numpy as np
import random
import prettytable
from src.dataset import DataHandler, contrastive_eval_collator, load_embeddings_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import pickle

def normalize_np(x, p=2, dim=1, eps=1e-12):
    """
    NumPy implementation of torch.nn.functional.normalize
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    return x / norm

@torch.no_grad()
def embed_queries_stella(model_name_or_path, queries, model, per_gpu_batch_size=4):
    if 'inf-retriever' in model_name_or_path:
        query_prompt_name = "query"
    else:
        query_prompt_name = "s2p_query" 

    model.eval()
    embeddings, batch_question = [], []

    for k, q in enumerate(queries):
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
def embed_passages(passages, model, per_gpu_batch_size=4):
    alltext = []
    for k, p in tqdm(enumerate(passages)):
        alltext.append(p)
    
    with torch.no_grad():
        allembeddings = model.encode(alltext, batch_size=per_gpu_batch_size)  # default is 512, but got oom
    return allembeddings

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_synthetic_dataset(data_dir='./synthetic_data', normalize=False):            
    # 1. Load configuration (metadata about the dataset)
    with open(os.path.join(data_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    corpus = np.load(os.path.join(data_dir, 'corpus.npy'))              # Shape: (corpus_size, dimensions)
    queries = np.load(os.path.join(data_dir, 'queries.npy'))            # Shape: (total_queries, dimensions)
    # transformation_matrices = np.load(os.path.join(data_dir, 'transformation_matrices.npy'))  # Shape: (n_rotations, dimensions, dimensions)
    
    # 3. Load query-ground truth mappings
    with open(os.path.join(data_dir, 'query_ground_truth_pairs.json'), 'r') as f:
        pairs_data = json.load(f)
    
    return {
        'config': config,
        'corpus': corpus,
        'queries': queries,
        # # 'transformation_matrices': transformation_matrices,
        'pairs_data': pairs_data
    }
    
# def normalize_vectors(vectors):
#     return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def compute_l2_distance(query_1, query_2):
    return np.linalg.norm(query_1 - query_2)

def compute_cosine_similarity(query_1, query_2):
    return np.dot(query_1, query_2) / (np.linalg.norm(query_1) * np.linalg.norm(query_2))

def compute_averge_query_distance(query_vectors):
    # L2 distance
    
    l2_distance_list = []
    cosine_similarity_list = []
    for i in range(10000):
        two_random_nums = random.sample(range(len(query_vectors)), 2)
        query_1 = query_vectors[two_random_nums[0]]
        query_2 = query_vectors[two_random_nums[1]]
        l2_distance_list.append(compute_l2_distance(query_1, query_2))
        cosine_similarity_list.append(compute_cosine_similarity(query_1, query_2))
    return np.mean(l2_distance_list), np.mean(cosine_similarity_list)
    
def compute_averge_target_distance_same_example(target_vectors_list):
    l2_distance_list = []
    cosine_similarity_list = []
    for i in range(len(target_vectors_list)):
        all_target_vectors = target_vectors_list[i]
        for j in range(len(all_target_vectors)):
            for k in range(j+1, len(all_target_vectors)):
                l2_distance_list.append(compute_l2_distance(all_target_vectors[j], all_target_vectors[k]))
                cosine_similarity_list.append(compute_cosine_similarity(all_target_vectors[j], all_target_vectors[k]))
    print('max', 'l2', np.max(l2_distance_list), 'cosine', np.max(cosine_similarity_list))
    print('min', 'l2', np.min(l2_distance_list), 'cosine', np.min(cosine_similarity_list))
    import matplotlib.pyplot as plt
    plt.hist(l2_distance_list, bins=100)
    plt.savefig('l2_distance_hist.png')
    plt.close()
    plt.hist(cosine_similarity_list, bins=100)
    plt.savefig('cosine_similarity_hist.png')
    plt.close()
    # compute the percentage of cosine similarity greater than 0.9
    print('percentage of cosine similarity greater than 0.93', np.sum(np.array(cosine_similarity_list) > 0.93) / len(cosine_similarity_list))
    return np.mean(l2_distance_list), np.mean(cosine_similarity_list)

def compute_averge_target_distance_different_examples(target_vectors_list, in_example_idx=0, _print=False):
    
    l2_distance_list = []
    cosine_similarity_list = []
    random_numbers = []
    for i in range(10000):
        two_random_nums = random.sample(range(len(target_vectors_list)), 2)
        random_numbers.append(two_random_nums)
        targets_1 = target_vectors_list[two_random_nums[0]]
        targets_2 = target_vectors_list[two_random_nums[1]]
        if in_example_idx != -1:
            query_1 = targets_1[in_example_idx]
            query_2 = targets_2[in_example_idx]
        else:
            if min(len(targets_1), len(targets_2)) == 0:
                continue
            else:
                query_1 = random.choice(targets_1)
                query_2 = random.choice(targets_2)
            # elif min(len(targets_1), len(targets_2)) == 1:
            #     print('one target')
            #     query_1 = targets_1[0]
            #     query_2 = targets_2[0]
            # else:
            #     two_random_in_example_nums = random.sample(range(min(len(targets_1), len(targets_2))), 2)
            #     query_1 = targets_1[two_random_in_example_nums[0]]
            #     query_2 = targets_2[two_random_in_example_nums[1]]
        l2_distance_list.append(compute_l2_distance(query_1, query_2))
        cosine_similarity_list.append(compute_cosine_similarity(query_1, query_2))
    if _print:
        print(random_numbers[:100])
        print(l2_distance_list[:100])
    return np.mean(l2_distance_list), np.mean(cosine_similarity_list)

def main(args):
    import sys
    from pathlib import Path
    
    all_l2_distances = []
    all_cosine_similarities = []
    table = prettytable.PrettyTable()
    table.field_names = ["distance_type", "betw qs", "betw ts (same)", "betw ts (diff), -1", "betw ts (diff), 0", "betw ts (diff), 1", "betw ts (diff), 2", "betw ts (diff), 3", "betw ts (diff), 4"]
        
    data_path = Path(args.data_path)
    if args.data_type == 'synthetic':
        data = load_synthetic_dataset(data_dir=f'./data_creation/gaussian/data/{data_path}/', normalize=False)
        print(f'checking {data_path} data')
        split = 'train'
        pairs = data['pairs_data'][split]
        queries = data['queries']
        corpus = data['corpus']
        
        query_vectors_list = []
        target_vectors_list = []
        
        for i in range(len(pairs)):
            query_vector = queries[pairs[i]['query_idx']]
            ground_truth_indices = pairs[i]['ground_truth_indices']
            target_vectors = corpus[ground_truth_indices]
            query_vectors_list.append(query_vector.reshape(1, -1))
            target_vectors_list.append(normalize_np(target_vectors))
        
        query_vectors = np.concatenate(query_vectors_list, axis=0)
        query_vectors = normalize_np(query_vectors)
    elif args.data_type == 'ambiguous_qe' or args.data_type == 'qampari':
        print(f'checking {args.data_path} data')
        full_dataset = load_embeddings_dataset(dataset_path=args.data_path)
        collator = contrastive_eval_collator
        data_handler = DataHandler(full_dataset, collator, 1, 'dev', 4)
        dataloader = data_handler.get_full_dataloader()
        target_vectors_list = []
        for batch in tqdm(dataloader):
            if args.model_name_or_path == 'infly/inf-retriever-v1-1.5b':
                target_vectors_list.append(batch['positive_embeddings'].reshape(-1, 1536))
            elif args.model_name_or_path == 'NovaSearch/stella_en_400M_v5':
                target_vectors_list.append(batch['positive_embeddings'].reshape(-1, 1024))
            
        # query vector
        rootdir='/scratch/hc3337/projects/autoregressive/results/base_retrievers/inf'
        if args.data_type == 'ambiguous_qe':
            if 'train' in args.data_path.split('/')[-1]:
                query_path = f'{rootdir}/questions_embeddings_ambiguous_qe_train_question_only_2_to_5_ctxs.npy'
            else:
                query_path = f'{rootdir}/questions_embeddings_ambiguous_qe_dev_question_only_2_to_5_ctxs.npy'
        elif args.data_type == 'qampari':
            if 'train' in args.data_path.split('/')[-1]:
                query_path = f'{rootdir}/questions_embeddings_qampari_train_question_only_5_to_8_ctxs.npy'
            else:
                query_path = f'{rootdir}/questions_embeddings_qampari_dev_question_only_5_to_8_ctxs.npy'
        print(f'loading query vectors from {query_path}')
        query_vectors = np.load(query_path)
        
    elif args.data_type == 'multi_source':
        data = read_json(args.data_path)
        
        print(f'loading {args.model_name_or_path} model')
        if ('stella' in args.model_name_or_path) or ('inf-retriever' in args.model_name_or_path):
            model = SentenceTransformer(args.model_name_or_path, trust_remote_code=True)
            if 'inf-retriever' in args.model_name_or_path:
                model.max_seq_length = 1024
        model.eval()
        model = model.cuda().half()
        
        queries = [ex['query_text'] for ex in data]
        query_vectors = embed_queries_stella(args.model_name_or_path, queries, model)
        
        target_vectors_list = []
        for ex in data:
            passages = []
            for category in ex['selected_golden_documents']:
                if ex['selected_golden_documents'][category] and 'text' in ex['selected_golden_documents'][category]:
                    passages.append(ex['selected_golden_documents'][category]['text'])
            if len(passages) <= 1:
                continue
            passages_embeddings = embed_passages(passages, model)
            target_vectors_list.append(passages_embeddings)
    elif args.data_type == 'clustered_data':
        # load raw data
        raw_data_path = str(Path(args.data_path).stem.split('mean_shift')[0].split('dbscan')[0].split('kmeans')[0].strip('_'))+'.jsonl'
        print(f'loading raw data from {str(Path(args.data_path).parent.parent / 'data' / raw_data_path)}')
        data = read_jsonl(Path(args.data_path).parent.parent / 'data' / raw_data_path)
        
        # embed queries
        print(f'loading {args.model_name_or_path} model')
        if ('stella' in args.model_name_or_path) or ('inf-retriever' in args.model_name_or_path):
            model = SentenceTransformer(args.model_name_or_path, trust_remote_code=True)
            if 'inf-retriever' in args.model_name_or_path:
                model.max_seq_length = 1024
        model.eval()
        model = model.cuda().half()
        
        queries = [ex['question'] for ex in data]
        query_vectors = embed_queries_stella(args.model_name_or_path, queries, model)
        
        # load target vectors
        print(f'loading target vectors from {args.data_path}')
        cluster_data = pickle.load(open(args.data_path, 'rb'))
        target_vectors_list = []
        for i in range(len(cluster_data)):
            if cluster_data[i].shape[0] <= 1:
                continue
            target_vectors_list.append(normalize_np(cluster_data[i]))
    elif args.data_type == 'llm_generation':
        print(f'loading {args.model_name_or_path} model')
        data = pickle.load(open(args.data_path, 'rb'))
        target_vectors_list = data['document_embeddings']
        query_vectors = data['question_embeddings']
    
    # compute the average distance between queries
    l2_distance, cosine_similarity = compute_averge_query_distance(query_vectors)
    all_l2_distances.append(l2_distance.round(3))
    all_cosine_similarities.append(cosine_similarity.round(3))
    # all_l2_distances.append(0)
    # all_cosine_similarities.append(0)

    # compute the average distance between target vectors (of same example)
    l2_distance, cosine_similarity = compute_averge_target_distance_same_example(target_vectors_list)
    all_l2_distances.append(l2_distance.round(3))
    all_cosine_similarities.append(cosine_similarity.round(3))
    
    # compute the average distance between target vectors (of different examples)
    l2_distance, cosine_similarity = compute_averge_target_distance_different_examples(target_vectors_list, in_example_idx=-1)
    all_l2_distances.append(l2_distance.round(3))
    all_cosine_similarities.append(cosine_similarity.round(3))
    
    l2_distance, cosine_similarity = compute_averge_target_distance_different_examples(target_vectors_list, in_example_idx=0)
    all_l2_distances.append(l2_distance.round(3))
    all_cosine_similarities.append(cosine_similarity.round(3))
    
    
    if args.data_type != 'synthetic':
        all_l2_distances.extend([0, 0, 0, 0])
        all_cosine_similarities.extend([0, 0, 0, 0])
    else:
        l2_distance, cosine_similarity = compute_averge_target_distance_different_examples(target_vectors_list, in_example_idx=1)
        all_l2_distances.append(l2_distance.round(3))
        all_cosine_similarities.append(cosine_similarity.round(3))
        l2_distance, cosine_similarity = compute_averge_target_distance_different_examples(target_vectors_list, in_example_idx=2)
        all_l2_distances.append(l2_distance.round(3))
        all_cosine_similarities.append(cosine_similarity.round(3))
        l2_distance, cosine_similarity = compute_averge_target_distance_different_examples(target_vectors_list, in_example_idx=3)
        all_l2_distances.append(l2_distance.round(3))
        all_cosine_similarities.append(cosine_similarity.round(3))
        l2_distance, cosine_similarity = compute_averge_target_distance_different_examples(target_vectors_list, in_example_idx=4)
        all_l2_distances.append(l2_distance.round(3))
        all_cosine_similarities.append(cosine_similarity.round(3))
    
    
    table.add_row(["L2"]+all_l2_distances)
    table.add_row(["Cosine"]+all_cosine_similarities)
    print(table)
    
    # write this table to a csv file
    with open(f'sanity_check_{data_path.stem}.csv', 'w') as f:
        f.write(table.get_csv_string())
    
    
    # compute the average distance between target vectors (of same example) and queries
    
    # compute the average distance between target vectors (of different examples) and queries
    # for l in target_vectors_list:
    #     print(l[0][:10])
    
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="gaussian")
    parser.add_argument("--data_type", type=str, default="synthetic")
    parser.add_argument("--model_name_or_path", type=str, default="infly/inf-retriever-v1-1.5b")
    
    args = parser.parse_args()
    main(args)
    
    # qampari, ambignq
    # python sanity_check.py --data_type qampari --data_path training_datasets/llama-1b/qampari/inf/autoregressive_qampari_inf_dev_dataset_1b_contrastive_5_to_8_ctxs/
    # python sanity_check.py --data_type ambiguous_qe --data_path training_datasets/llama-1b/ambiguous_qe/inf/autoregressive_ambiguous_qe_inf_dev_dataset_1b_contrastive_2_to_5_ctxs/
    
    # multi-source
    # python sanity_check.py --data_path golden_retrieval_dataset_per_category.json --data_type multi_source
    
    # clustered data
    # python sanity_check.py --data_type clustered_data --data_path large_scale/clustered_data/small_mean_shift_centroids_flexible.pkl
    
    # llm generation
    # python sanity_check.py --data_type llm_generation --data_path large_scale/llm_generation/outputs/q_docs_woctx/embeddings.pkl
    
    