import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
import torch
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path    
import os
import sys

def safe_from_list_and_save(dataset_dicts, out_dataset_path, batch_size=10000):
    datasets = []
    for i in range(0, len(dataset_dicts), batch_size):
        chunk = dataset_dicts[i:i + batch_size]
        chunk_dataset = Dataset.from_list(chunk)
        datasets.append(chunk_dataset)

    full_dataset = concatenate_datasets(datasets)
    full_dataset.save_to_disk(out_dataset_path)
    return full_dataset


def data_collator(features):
    """
    """
    batch = {'input_ids': [], 'attention_mask':[]}
    for inst in features:
        for k in inst.keys():
            if k in ['input_ids', 'attention_mask']:
                batch[k].append(torch.tensor(inst[k]).unsqueeze(0))
    for k, v in batch.items():
        batch[k] = torch.cat(v, dim=0)
    return batch




@torch.no_grad()
def create_input_embeddings_for_contrastive(model_name = "meta-llama/Llama-3.2-1B-Instruct", 
                                      input_data_path='', 
                                      batch_size=32, 
                                      positive_embeddings_path='', 
                                      negative_embeddings_path='', 
                                      out_dataset_path=''):
    """
        Create a dataset of input embeddings for the query.
        What do we need:
        1. The input data.
        2. The output embeddings. (centroid embeddings)
        3. The length of the output embeddings.
    """
    # Tokenize dataset
    def tokenize_function(examples):
        if 'question' in examples:
            question = examples['question']
        else:
            question = examples['question_text']
        examples['text'] = formulate_text(instruction, question)
        print(examples['text'][0])
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=257, return_tensors='pt')
    
    def formulate_text(instruction, queries):
        return [instruction.replace('[QUERY]', query) for query in queries]

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    if model_name == 'infly/inf-retriever-v1-1.5b':
        print('Using infly/inf-retriever-v1-1.5b')
        instruction_template = "Instruct: "
        response_template = ""
    elif model_name == "meta-llama/Llama-3.2-1B-Instruct" or model_name == "meta-llama/Llama-3.2-3B-Instruct" or model_name == "meta-llama/Llama-3.1-8B-Instruct":
        print('Using model: ', model_name)
        instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif model_name == "Qwen/Qwen3-4B-Instruct-2507":
        instruction_template = "<|im_start|>user\n"
        response_template = "<|im_end|>\n<|im_start|>assistant\n"
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    instruction = (f'{instruction_template}Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: [QUERY]{response_template}').strip('\n')
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(input_data_path))

    # # Define model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    seperator = tokenizer(response_template)[1:]
    
    # tokenize dataset => get the question embedding
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # output size (# data, 128, hidden_size)
    dataloader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    # positive_embeddings
    positive_embeddings = np.load(positive_embeddings_path)
    if len(positive_embeddings.shape) == 2:
        positive_embeddings = positive_embeddings.reshape(positive_embeddings.shape[0], 1, -1)
    # negative_embeddings
    negative_embeddings = np.load(negative_embeddings_path)
    if len(negative_embeddings.shape) == 2:
        negative_embeddings = negative_embeddings.reshape(negative_embeddings.shape[0], 1, -1)
    print('positive_embeddings.shape, negative_embeddings.shape', positive_embeddings.shape, negative_embeddings.shape)

    import time
    start_time = time.time()
    i = 0
    actual_data_size = 0
    actual_data_indices = []
    
    dataset_dicts = []
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)

        # positive embeddings 
        positive = positive_embeddings[i]  # (bsz, k, d)
        # negative embeddings
        negative = negative_embeddings[i]  # (bsz, k, d)
        
        dataset_dicts.append({"input_ids": batch['input_ids'][0].cpu().numpy(), 
                              "attention_mask": batch['attention_mask'][0].cpu().numpy(), 
                              "positive_embeddings": positive, 
                              "negative_embeddings": negative})
                
        actual_data_size += 1
        actual_data_indices.append(i)
        i += 1
        if i >= positive_embeddings.shape[0]:
            break

    # make dataset directory
    Path(out_dataset_path).mkdir(parents=True, exist_ok=True)

    safe_from_list_and_save(dataset_dicts, out_dataset_path, batch_size=2000)

    print('time elapsed: ', (time.time()-start_time)/60.0, 'min.')
    print('actual data size: ', actual_data_size)
    return actual_data_indices
    

if __name__ == '__main__':
    rootdir = Path(__file__).parent
    
    generate_split = sys.argv[1]
    if generate_split not in ['contrastive', 'gaussian_synthetic']:
        raise ValueError(f'Invalid generate_split: {generate_split}')

                
    if generate_split == 'contrastive': 
        base_model_map = {
            'llama-1b': 'meta-llama/Llama-3.2-1B-Instruct',
            'llama-3b': 'meta-llama/Llama-3.2-3B-Instruct',
            'llama-8b': 'meta-llama/Llama-3.1-8B-Instruct',
            'qwen3-4b': 'Qwen/Qwen3-4B-Instruct-2507',
        }
        for base_model_name in base_model_map.keys():
            print(f'Generating contrastive dataset for {base_model_name}')
            length_maps = {
                'qampari': [5,6,7,8],
                'ambiguous_qe':[2,3,4,5],
            }
            
            for split in ['train']:
                for data_name in ['qampari', 'ambiguous_qe']:
                    for length in length_maps[data_name]:
                        for model_name in ['inf']:
                            rootdir = Path('raw_data/')
                            
                            data_indices = create_input_embeddings_for_contrastive(batch_size=1, 
                                                            model_name=base_model_map[base_model_name],
                                                            input_data_path=rootdir / f'{data_name}_{split}_question_only_{length}_ctxs.jsonl', 
                                                            positive_embeddings_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_positive_embeddings_{length}.npy', 
                                                            negative_embeddings_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_random_embeddings_{length}.npy',
                                                            out_dataset_path=f'../training_datasets/{base_model_name}/{data_name}/{model_name}/autoregressive_{data_name}_{model_name}_{split}_dataset_{length}_ctxs')
                            print(f'data size: {len(data_indices)} | model: {base_model_name} | data_name: {data_name} | length: {length} | split: {split}') 
    
    if generate_split == 'gaussian_synthetic':
        
        def load_synthetic_dataset(data_dir='./synthetic_data', normalize=False):
            
            # 1. Load configuration (metadata about the dataset)
            with open(os.path.join(data_dir, 'config.json'), 'r') as f:
                config = json.load(f)
            
            # 2. Load the main data arrays
            if normalize:
                corpus = np.load(os.path.join(data_dir, 'normalized_corpus.npy'))              # Shape: (corpus_size, dimensions)
                queries = np.load(os.path.join(data_dir, 'normalized_queries.npy'))            # Shape: (total_queries, dimensions)
            else:
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
                'pairs_data': pairs_data
            }
            
        @torch.no_grad()
        def create_synthetic_dataset(out_dataset_path, pairs, queries, corpus, LENGTH):
            batch = {'inputs_embeds': [], 'attention_mask':[], 'positive_embeddings': [], 'negative_embeddings': []}
            
            for i in range(len(pairs)):
                query_vector = queries[pairs[i]['query_idx']]
                ground_truth_indices = pairs[i]['ground_truth_indices']

                # create random negative indices that are not in ground_truth_indices
                random_indices = np.random.choice(len(corpus), size=len(ground_truth_indices), replace=False)
                while np.any(np.isin(random_indices, ground_truth_indices)):
                    random_indices = np.random.choice(len(corpus), size=len(ground_truth_indices), replace=False)

                batch['inputs_embeds'].append(query_vector)
                batch['attention_mask'].append(np.zeros(LENGTH))
                batch['positive_embeddings'].append(corpus[ground_truth_indices])
                batch['negative_embeddings'].append(corpus[random_indices])
            
            
            batch['inputs_embeds'] = torch.tensor(batch['inputs_embeds']).float().unsqueeze(1).expand(-1, LENGTH, -1)  # (bsz, LENGTH, d)
            batch['attention_mask'] = torch.tensor(batch['attention_mask']).long()             # (bsz, LENGTH). Only the first token is 1, the rest are 0.
            batch['attention_mask'][:, 0] = 1
            batch['positive_embeddings'] = torch.tensor(batch['positive_embeddings']).float()  # (bsz, k, d), LENGTH > k
            batch['negative_embeddings'] = torch.tensor(batch['negative_embeddings']).float()  # (bsz, k, d)
            
            dataset_dicts = []
            for i in range(len(pairs)):
                positive = batch['positive_embeddings'][i]  # (k, d)
                negative = batch['negative_embeddings'][i]  # (k, d)

                dataset_dicts.append({"inputs_embeds": batch['inputs_embeds'][i].cpu().numpy(), 
                                    "attention_mask": batch['attention_mask'][i].cpu().numpy(), 
                                    "positive_embeddings": positive, 
                                    "negative_embeddings": negative})
            safe_from_list_and_save(dataset_dicts, out_dataset_path, batch_size=2000)

            print('actual data size: ', len(dataset_dicts))

        LENGTH = 8
        normalize = False            
        for split in ['train', 'test']:
            for data_type in ['linear', 'linear_multi_query', 'linear_ood', 'mlps', 'mlps_multi_query', 'mlps_ood']:
                data = load_synthetic_dataset(data_dir='gaussian/data/linear/', normalize=normalize)
                pairs = data['pairs_data'][split]
                out_data_path = f'../synthetic_datasets/synthetic_{data_type}_{split}' 
                
                create_synthetic_dataset(out_dataset_path=out_data_path, 
                                        pairs=pairs, queries=data['queries'], corpus=data['corpus'], LENGTH=LENGTH)
            
        