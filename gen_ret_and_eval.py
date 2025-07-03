import torch
import torch.distributed
import torch.optim as optim
import numpy as np
import argparse

from model import load_model
from tqdm import tqdm
import os
import json
from functools import partial
import glob
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from model import load_model
from prettytable import PrettyTable

from dataset import (
    load_embeddings_dataset,
    contrastive_eval_collator,
    mse_eval_collator,
    DataHandler
)
from utils import Config, set_seed, set_optim
import yaml
from pathlib import Path

from retrieval_utils import Indexer, add_passages, load_passages, index_encoded_data, add_passages_single_instance
import sys
from datasets import load_dataset

import structlog
logger = structlog.get_logger()


# import colorlog

# handler = colorlog.StreamHandler()
# handler.setFormatter(colorlog.ColoredFormatter(
# 	'%(log_color)s%(levelname)s:%(name)s:%(message)s'))
# logger = colorlog.getLogger(__name__)
# logger.addHandler(handler)
def write_jsonl(data, output_path):
    with open(output_path, 'w') as f:
        for inst in data:
            f.write(json.dumps(inst) + '\n')


def evaluate_loop(dataloader, model, device, max_new_tokens, use_gt_q_embed, use_eos, compute_loss = True):

    all_outputs = []
    all_losses = []
    all_labels = []
    all_lengths = []
    adaptive_max_new_tokens = (max_new_tokens is None)
        
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        if 'positive_embeddings' in batch and adaptive_max_new_tokens:
            max_new_tokens = batch['positive_embeddings'].size(1)
        

        output = model.generate(
            max_new_tokens=max_new_tokens,
            use_gt_q_embed=use_gt_q_embed,
            use_eos=use_eos,
            **batch
        )
        all_outputs.append(output.view(-1, output.size(-1)))

        if compute_loss:
            if 'labels' in batch:
                # compute the loss
                # print(output.size(), batch['labels'].size())
                assert output.size() == batch['labels'].size(), (output.size(), batch['labels'].size())
                loss = model.loss_fct(output.float(), batch['labels'].float())
                # print('loss', loss.item())
                all_lengths.append(batch['labels'].size(1))
                all_losses.append(loss.item())
                all_labels.append(batch['labels'].view(-1, batch['labels'].size(-1)))
            elif 'positive_embeddings' in batch:
                # print(output.size(), batch['positive_embeddings'].size(), batch['negative_embeddings'].size())
                loss = model.loss_fct(output.float(), batch['positive_embeddings'].float(), batch['negative_embeddings'].float())
                all_lengths.append(batch['positive_embeddings'].size(1))
                all_labels.append(batch['positive_embeddings'].view(-1, batch['positive_embeddings'].size(-1)))
                all_losses.append(loss.item())
    
    if compute_loss:
        return torch.cat(all_outputs, dim=0).cpu().numpy(), sum(all_losses) / len(all_losses), torch.cat(all_labels, dim=0).cpu().numpy(), all_lengths
    else:
        return torch.cat(all_outputs, dim=0).cpu().numpy(), None, None, all_lengths


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

def generate_input_data(loss_function, question_only, input_data_path, tokenizer):
        # Tokenize dataset
    def tokenize_function(examples):
        if 'question' in examples:
            question = examples['question']
        else:
            question = examples['question_text']
        examples['text'] = formulate_text(instruction, question)
        print(examples['text'][0])
        return tokenizer(examples['text'], padding=True, return_tensors='pt')
    
    def formulate_text(instruction, queries):
        return [instruction.replace('[QUERY]', query) for query in queries]

    def data_collator(features):
        """
        """
        batch = {'input_ids': [], 'attention_mask':[]}
        for inst in features:
            input_len = sum(inst['attention_mask'])
            for k in inst.keys():
                if k in ['input_ids', 'attention_mask']:
                    batch[k].append(torch.tensor(inst[k][:input_len]).unsqueeze(0))
        for k, v in batch.items():
            batch[k] = torch.cat(v, dim=0)
        return batch

    instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    instruction = f'{instruction_template}Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: [QUERY]\n{response_template}'
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(input_data_path))
    tokenizer.pad_token = tokenizer.eos_token
    seperator = tokenizer(response_template)[1:]
    
    # tokenize dataset => get the question embedding
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # # Load dataset
    # use_contrastive_data = 'Contrastive' in loss_function
    # logger.info('use_contrastive_data', more_than_strings=use_contrastive_data)
    # # Load dataset
    # if use_contrastive_data:
    #     collator = contrastive_eval_collator
    # else:
    #     collator = partial(mse_eval_collator, question_only=question_only)
    full_dataset = tokenized_datasets['train']
    data_handler = DataHandler(full_dataset, data_collator, 1, 'dev')
    dataloader = data_handler.get_full_dataloader()
    return dataloader


def eval_with_generation(input_data_path = 'autoregressive_wsd_train_dataset_1b',
         get_split = 'train-held-out',
         adapter_path = "results/test/toy/checkpoint_4", 
         base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct",
         linear_checkpoint_path = None,
         output_path = 'out_train_large.npy',
         output_lengths_path = 'out_train_large_lengths.npy',
         max_new_tokens = 2,
         embedding_model_dim = 1536,
         compute_loss = True,
         loss_function = 'Contrastive',
         question_only = False,
         batch_size_training = 1,
         use_gt_q_embed = False,
         use_eos = False):

    instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    instruction = f'{instruction_template}Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: [QUERY]\n{response_template}'

    # Define model and tokenizer
    model, tokenizer = load_model(train_lora=False,
                                  base_model_id=base_model_id, 
                                  adapter_path=adapter_path, 
                                  linear_checkpoint_path=linear_checkpoint_path,
                                  embedding_model_dim=embedding_model_dim, 
                                  loss_function=loss_function)
    
    # always use hungarian loss for evaluation
    # always not predict question for evaluation    
    if Path(input_data_path).is_dir():
        logger.info(f"Loading input data from {input_data_path}, using the question embedding")
        dataloader = load_input_data(loss_function, question_only, batch_size_training, get_split, input_data_path)
    else:
        logger.info(f"Generating input data from {input_data_path}, using the raw text")
        dataloader = generate_input_data(loss_function, question_only, input_data_path, tokenizer)


    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        outputs, loss, all_labels, all_lengths = evaluate_loop(dataloader, model, device, max_new_tokens=max_new_tokens, use_gt_q_embed=use_gt_q_embed, use_eos=use_eos, compute_loss=compute_loss)
        # outputs = outputs.reshape(-1, embedding_model_dim)
        print('len all lengths', len(all_lengths))
        np.save(output_path, outputs)
        if len(all_lengths) > 0:
            np.save(output_lengths_path, np.array(all_lengths))
        else:
            print('no lengths')
        return outputs, loss, all_labels
    
  
  
  

def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data
  


def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


    
  
# def aggregate_different_queries(top_ids_and_scores, MAX_LATENTS, top_k):
#     # aggregate top_ids_and_scores for different queries
#     assert len(top_ids_and_scores) % (MAX_LATENTS) == 0, (len(top_ids_and_scores), MAX_LATENTS)
#     aggregated_top_ids_and_scores = []
#     for i in range(len(top_ids_and_scores) // MAX_LATENTS):
#         aggregated_top_ids_and_scores_per_inst = []
#         ids_and_scores_to_aggregate = top_ids_and_scores[i*MAX_LATENTS:(i+1)*MAX_LATENTS]
#         # aggregate ids and scores to be a single list, and avoid duplicates
#         # take from each list in a round-robin manner until reaches top_k
#         # put them into aggregated_top_ids_and_scores, which follows the format of top_ids_and_scores
#         seen_ids = set()
    
#         # Find the maximum length among all lists
#         max_len = max(len(list(zip(*sublist))) for sublist in ids_and_scores_to_aggregate)
#         # Round-robin aggregation
#         for idx in range(max_len):
#             for query_results in ids_and_scores_to_aggregate:
#                 query_results = list(zip(*query_results))
#                 # Skip if we've processed all items from this query
#                 if idx >= len(query_results):
#                     continue
#                 # print(query_results[idx])
#                 current_id, current_score = query_results[idx]
                
#                 # Only add if we haven't seen this ID before
#                 if current_id not in seen_ids:
#                     aggregated_top_ids_and_scores_per_inst.append((current_id, current_score))
#                     seen_ids.add(current_id)
                
#                 if len(aggregated_top_ids_and_scores_per_inst) >= top_k:
#                     break
#             if len(aggregated_top_ids_and_scores_per_inst) >= top_k:
#                 break
#         aggregated_top_ids_and_scores_per_inst = list(zip(*aggregated_top_ids_and_scores_per_inst))
#         aggregated_top_ids_and_scores.append(aggregated_top_ids_and_scores_per_inst)
#     return aggregated_top_ids_and_scores


def aggregate_different_queries_by_length(top_ids_and_scores, lengths=None, MAX_LATENTS=None, top_k=100, aggregate_start_idx=0, aggregate_end_idx=None):
    # aggregate top_ids_and_scores for different queries
    if MAX_LATENTS is not None:
        assert len(top_ids_and_scores) % (MAX_LATENTS) == 0, (len(top_ids_and_scores), MAX_LATENTS)
        assert lengths is None, (lengths, MAX_LATENTS)
        lengths = [MAX_LATENTS] * (len(top_ids_and_scores) // MAX_LATENTS)
    else:
        assert len(top_ids_and_scores) == sum(lengths), (len(top_ids_and_scores), sum(lengths))
    
    aggregated_top_ids_and_scores = []
    start_idx = 0
    for i in range(len(lengths)):
        aggregated_top_ids_and_scores_per_inst = []
        ids_and_scores_to_aggregate = top_ids_and_scores[start_idx:start_idx+lengths[i]]
        if aggregate_end_idx is not None:
            ids_and_scores_to_aggregate = ids_and_scores_to_aggregate[aggregate_start_idx:aggregate_end_idx]
        else:
            ids_and_scores_to_aggregate = ids_and_scores_to_aggregate[aggregate_start_idx:]
        # print('lens', len(ids_and_scores_to_aggregate))
        start_idx += lengths[i]
        # aggregate ids and scores to be a single list, and avoid duplicates
        # take from each list in a round-robin manner until reaches top_k
        # put them into aggregated_top_ids_and_scores, which follows the format of top_ids_and_scores
        seen_ids = set()
    
        # Find the maximum length among all lists
        max_len = max(len(list(zip(*sublist))) for sublist in ids_and_scores_to_aggregate)
        # Round-robin aggregation
        for idx in range(max_len):
            for query_results in ids_and_scores_to_aggregate:
                query_results = list(zip(*query_results))
                # Skip if we've processed all items from this query
                if idx >= len(query_results):
                    continue
                # print(query_results[idx])
                current_id, current_score = query_results[idx]
                
                # Only add if we haven't seen this ID before
                if current_id not in seen_ids:
                    aggregated_top_ids_and_scores_per_inst.append((current_id, current_score))
                    seen_ids.add(current_id)
                
                if len(aggregated_top_ids_and_scores_per_inst) >= top_k:
                    break
            if len(aggregated_top_ids_and_scores_per_inst) >= top_k:
                break
        aggregated_top_ids_and_scores_per_inst = list(zip(*aggregated_top_ids_and_scores_per_inst))
        aggregated_top_ids_and_scores.append(aggregated_top_ids_and_scores_per_inst)
    
    assert start_idx == len(top_ids_and_scores), (start_idx, len(top_ids_and_scores))
    return aggregated_top_ids_and_scores


def load_index(embedding_size, passages_embeddings, save_or_load_index=False, use_gpu=False, shard_id=0, num_shards=1):
    logger.info("doing indexing...")
    index = Indexer(embedding_size, 0, 8, use_gpu=use_gpu)

    # index all passages
    input_paths = glob.glob(passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    
    if save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        logger.info(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, 100000, shard_id=shard_id, num_shards=num_shards)
        logger.info(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if save_or_load_index:
            index.serialize(embeddings_dir)
    return index
    
    
    
def main_test_google(passages_embeddings, passages_path, output_path, 
              raw_data_path = '/scratch/hc3337/projects/autoregressive/data/wsd/distinct/train.jsonl', 
              data_path = 'out.npy', lengths_path = "", embedding_size = 4096, top_k_per_query = 100, top_k = 100,
              start_idx = 0, end_idx = None, MAX_LATENTS = None, aggregate_start_idx = 0, aggregate_end_idx = None):
    
    # loading question embeddings
    logger.info('loading question embeddings and attempt to retrieve from %s', data_path)
    question_embeddings = np.load(data_path)
    logger.info('question embeddings shape: %s', question_embeddings.shape)
    
    # loading data
    logger.info('loading data from %s', raw_data_path)
    if end_idx is None:
        data = load_data(raw_data_path)[start_idx:]
    else:
        data = load_data(raw_data_path)[start_idx:end_idx]
    logger.info('length of the data to be retrieved: %s', len(data))

    # loading lengths; making sure the data and question embeddings are aligned
    if MAX_LATENTS is None:
        lengths = np.load(lengths_path)
        logger.info('loaded lengths from %s', lengths_path)
        assert len(data) == len(lengths), (len(data), len(lengths))
        assert question_embeddings.shape[0] == sum(lengths), (question_embeddings.shape[0], sum(lengths))
    else:
        lengths = None
    
    logger.info('google_api')
    if MAX_LATENTS is not None:
        assert lengths is None, (lengths, MAX_LATENTS)
        lengths = [MAX_LATENTS] * len(data)

    start = 0
    for i in range(len(data)):
        # load index and passages for each query
        index = load_index(embedding_size, (passages_embeddings) + '/' + str(i) + '/*')

        # load passages
        logger.info(f"Loading passages from {passages_path}/psgs_{i}.tsv")
        passages = load_passages(passages_path + f"/psgs_{i}.tsv")
        passage_id_map = {x["id"]: x for x in passages}
        
        start_time_retrieval = time.time()
        top_ids_and_scores_inst = index.search_knn(question_embeddings[start:start+lengths[i]].reshape(-1, embedding_size), top_k_per_query)
        logger.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
        top_ids_and_scores_inst = aggregate_different_queries_by_length(top_ids_and_scores_inst, [lengths[i]], None, top_k, aggregate_start_idx, aggregate_end_idx)
        assert len(top_ids_and_scores_inst) == 1, (len(top_ids_and_scores_inst))
        logger.info("top_ids_and_scores_inst[0][0]", lens=len(top_ids_and_scores_inst[0][0]))
        add_passages_single_instance(data[i], passage_id_map, top_ids_and_scores_inst[0])
        
        start += lengths[i]
        
    assert start == len(question_embeddings), (start, len(question_embeddings))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        for ex in data:
            json.dump(ex, fout, ensure_ascii=False)
            fout.write("\n")
    logger.info(f"Saved results to {output_path}")
    
      
def main_test(index, passage_id_map, output_path, 
              raw_data_path = '/scratch/hc3337/projects/autoregressive/data/wsd/distinct/train.jsonl', 
              data_path = 'out.npy', lengths_path = "", embedding_size = 4096, top_k_per_query = 100, top_k = 100,
              start_idx = 0, end_idx = None, MAX_LATENTS = None, aggregate_start_idx = 0, aggregate_end_idx = None):
    
    # loading question embeddings
    logger.info('loading question embeddings and attempt to retrieve from %s', data_path)
    question_embeddings = np.load(data_path)
    logger.info('question embeddings shape: %s', question_embeddings.shape)
    
    # loading data
    logger.info('loading data from %s', raw_data_path)
    if end_idx is None:
        data = load_data(raw_data_path)[start_idx:]
    else:
        data = load_data(raw_data_path)[start_idx:end_idx]
    logger.info('length of the data to be retrieved: %s', len(data))

    # loading lengths; making sure the data and question embeddings are aligned
    if MAX_LATENTS is None:
        lengths = np.load(lengths_path)
        logger.info('loaded lengths from %s', lengths_path)
        assert len(data) == len(lengths), (len(data), len(lengths))
        assert question_embeddings.shape[0] == sum(lengths), (question_embeddings.shape[0], sum(lengths))
    else:
        lengths = None

    
    # Start Search! Get top k results.
    start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(question_embeddings.reshape(-1, embedding_size), top_k_per_query)
    logger.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
    top_ids_and_scores = aggregate_different_queries_by_length(top_ids_and_scores, lengths, MAX_LATENTS, top_k, aggregate_start_idx, aggregate_end_idx)

    logger.info(f"length of the data to be retrieved: {len(data)}, length of the retrieved results: {len(top_ids_and_scores)}")
    add_passages(data, passage_id_map, top_ids_and_scores)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        for ex in data:
            json.dump(ex, fout, ensure_ascii=False)
            fout.write("\n")
    logger.info(f"Saved results to {output_path}")
    
    
def aggregate_sharded_results(output_path, num_shards):
    if num_shards == 1:
        return
    # read all the data
    query2docs = {}
    START = 0
    END = num_shards
    TOPK = 0
    for i in range(START, END):            
        data = read_jsonl(output_path.replace('.jsonl', f'_shard_{i}.jsonl'))

        if i == START:
            TOPK = len(data[0]['ctxs'])
            for d in data:
                query2docs[d['question']] = []

        for inst in data:
            query2docs[inst['question']].extend(inst['ctxs'])

    for query, docs in query2docs.items():
        sorted_docs = sorted(docs, key=lambda x: x['score'], reverse=True)[:TOPK]
        query2docs[query] = sorted_docs

    # reassign the sorted docs to the original data
    for inst in data:
        inst['ctxs'] = query2docs[inst['question']]

    write_jsonl(data, output_path)
    
    # remove the shard files
    for i in range(START, END):
        os.remove(output_path.replace('.jsonl', f'_shard_{i}.jsonl'))
    

    
def parse_args():
    parser = argparse.ArgumentParser(description='Generate embeddings and perform retrieval evaluation')
    # Data configuration
    parser.add_argument('--data_name', type=str, default='ambiguous_qe', 
                       choices=['nq', 'msmarco', 'qampari', 'ambiguous', 'ambiguous_qe', 'arguana_generated', 'kialo', 'opinionqa', 'wsd_distinct'],
                       help='Name of the dataset to evaluate on')
    parser.add_argument('--training_data_name', type=str, default='ambiguous_qe',
                       choices=['nq', 'msmarco', 'qampari', 'ambiguous', 'ambiguous_qe', 'wsd_distinct'],
                       help='Name of the dataset used for training')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'train', 'train-held-out'],
                       help='Data split to evaluate on')
    parser.add_argument('--suffix_list', nargs='+', default=["_contrastive_from_stage2_lr2e5_ep20_temp0.05_warmup0.05"],
                       help='List of model suffixes to evaluate')
    parser.add_argument('--retriever_list', nargs='+', default=['inf'], choices=['stella', 'inf', 'cont'],
                       help='List of retrievers to use')
    
    # Indexing configuration
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Whether to use GPU for indexing')
    parser.add_argument('--num_shards', type=int, default=1,
                       help='Number of shards for indexing')
    parser.add_argument('--save_or_load_index', action='store_true', default=False,
                       help='Whether to save/load index')
    
    # Checkpoint mapping
    parser.add_argument('--checkpoint_num', type=int, default=1501,
                       help='Checkpoint number to use (overrides suffix-based mapping)')
    parser.add_argument('--use_suffix_mapping', action='store_true', default=False,
                       help='Whether to use suffix-based checkpoint mapping')
    parser.add_argument('--use_best_model', action='store_true', default=False,
                       help='Whether to use best model')

    # Model configuration
    parser.add_argument('--base_model_id', type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                       help='Base model ID')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--compute_loss', action='store_true', default=False,
                       help='Whether to compute loss during evaluation')
    # Google API configuration
    parser.add_argument('--google_api', action='store_true', default=False,
                       help='Whether to use Google API')
    
    # Config parameters (previously from config file)
    parser.add_argument('--loss_function', type=str, default='Hungarian_Contrastive',
                       help='Loss function to use')
    parser.add_argument('--question_only', action='store_true', default=True,
                       help='Whether to use question only')
    parser.add_argument('--batch_size_training', type=int, default=1,
                       help='Batch size for training')
    parser.add_argument('--use_gt_q_embed', action='store_true', default=False,
                       help='Whether to use ground truth question embedding')
    parser.add_argument('--use_eos', action='store_true', default=False,
                       help='Whether to use end of sequence token')
    
    # Paths and directories
    parser.add_argument('--embeddings_root', type=str, default='/scratch/hc3337/embeddings/',
                       help='Root directory for embeddings')
    parser.add_argument('--root', type=str, default='/scratch/hc3337',
                       help='Root directory for data')
    
    # Retrieval configuration
    parser.add_argument('--top_k_per_query', type=int, default=100,
                       help='Number of top results per query')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of top results to aggregate')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for data processing')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='Ending index for data processing')
    parser.add_argument('--aggregate_start_idx', type=int, default=0,
                       help='Starting index for aggregation')
    parser.add_argument('--aggregate_end_idx', type=int, default=None,
                       help='Ending index for aggregation')   
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Checkpoint mapping dictionaries
    num_map = {
        "msmarco_stella": 10000, "msmarco_cont": 10000, "msmarco_inf": 10000,
        "nq_cont": 70000, "nq_inf": 70000, "nq_stella": 70000,
        "ambiguous_cont": 1000, "ambiguous_inf": 3501, "ambiguous_stella": 1000,
        "qampari_cont": 30000, "qampari_inf": 1, "qampari_stella": 30000,
        "qampari+ambiguous_inf": 10000, "nq+msmarco_inf": 280000, "ambiguous+qampari_inf": 20001,
        "ambiguous_qe_cont": 1000, "ambiguous_qe_inf": 1501, "ambiguous_qe_stella": 1000
    }
    
    num_map_inst = {
        "_contrastive_ordered_4_gpus_lr1e4_ep30_temp0.05_warmup0.05": 1601,
        "_contrastive_4_gpus_lr1e4_ep30_temp0.05_warmup0.05": 1601,
        "_contrastive_from_stage2_take_first_lr2e5_ep20_temp0.05_warmup0.05": 4001,
        "_contrastive_take_first_lr2e5_ep20_temp0.05_warmup0.05": 5501,
        "_contrastive_lr2e5_ep20_temp0.05_warmup0.05": 2501,
        "_contrastive_4_gpus_lr2e5_ep30_temp0.05_warmup0.05": 1,
        "_contrastive_from_stage1_lr2e5_ep20_temp0.05_warmup0.05_eos": 35001,
        "_contrastive_from_stage2_lr2e5_ep20_temp0.05_warmup0.05_eos_new": 5501,
        "_contrastive_ordered_from_stage2_lr2e5_ep20_temp0.05_warmup0.05_eos_new": 5501,
        "_contrastive_from_stage2_lr2e5_ep20_temp0.05_warmup0.05": 1501,
        "_contrastive": 70000
    }
    
    # Validate Google API usage
    if args.data_name in ['arguana_generated', 'kialo', 'opinionqa']:
        assert args.google_api, "Google API is required for these datasets"
    else:
        assert not args.google_api, "Google API is not allowed for these datasets"
    
    # Determine embeddings directory
    embeddings_dir = 'qampari_embeddings' if args.data_name in ['qampari', 'ambiguous_qe'] else args.data_name
    if args.data_name == 'ambiguous':
        embeddings_dir = 'nq'
    
    # Set up passage embeddings map
    if not args.google_api:
        passage_embeddings_map = {
            'stella': {"embedding_path": f"{args.embeddings_root}/stella_en_400M_v5/{embeddings_dir}/*", "embedding_dim": 1024},
            'inf': {"embedding_path": f"{args.embeddings_root}/inf/{embeddings_dir}/*", "embedding_dim": 1536},
            'cont': {"embedding_path": f"{args.embeddings_root}/Contriever/{embeddings_dir}/*", "embedding_dim": 768}
        }
    else:
        passage_embeddings_map = {
            'stella': {"embedding_path": f"{args.embeddings_root}/google_api/stella_embeddings/{args.data_name}", "embedding_dim": 1024},
            'inf': {"embedding_path": f"{args.embeddings_root}/google_api/inf_embeddings/{args.data_name}", "embedding_dim": 1536},
            'cont': {"embedding_path": f"{args.embeddings_root}/google_api/contriever_embeddings/{args.data_name}", "embedding_dim": 768}
        }
    
    # Set up passages path
    if not args.google_api:
        if args.data_name in ['qampari', 'ambiguous_qe']:
            passages_path = f'{args.root}/wikipedia_chunks/chunks_v5.tsv'
        elif args.data_name == 'ambiguous':
            passages_path = f'data/nq/corpus.tsv'
        elif args.data_name == 'wsd_distinct':
            passages_path = f'data/wsd/distinct/corpus.tsv'
        else:
            passages_path = f'data/{args.data_name}/corpus.tsv'
    else:
        passages_path = f'{args.root}/serpapi/contriever_psgs/{args.data_name}'
    
    # Set up dev data path
    if args.data_name == 'ambiguous' or args.data_name == 'ambiguous_qe':
        dev_data_path = f'data_creation/raw_data/{args.data_name}_{args.split}_question_only_2_to_5_ctxs.jsonl'
    elif args.data_name == 'qampari':
        dev_data_path = f'data_creation/raw_data/{args.data_name}_{args.split}_question_only_5_to_8_ctxs.jsonl'
    elif args.data_name in ['nq', 'msmarco', 'wsd_distinct']:
        dev_data_path = f'data_creation/raw_data/{args.data_name}_{args.split}_question_only.jsonl'
    else:
        dev_data_path = f"data_creation/raw_data/{args.data_name}_question_only.jsonl"
    
    # Main evaluation loop
    for suffix in args.suffix_list:
        for retriever in args.retriever_list:
            train_name = f"{args.training_data_name}_{retriever}"
            dataset_name = f"{args.data_name}_{retriever}"
            model_name = f"toy{suffix}"
            
            # Set up model paths
            Path(f"output_embeddings/{train_name}").mkdir(parents=True, exist_ok=True)
            
            # Determine checkpoint number
            if args.use_best_model:
                adapter_path = f"results/{train_name}/{model_name}/best_model"
                linear_checkpoint_path = f"results/{train_name}/{model_name}/best_model_linear.pt"
            else:
                if args.use_suffix_mapping and suffix in num_map_inst:
                    checkpoint_num = num_map_inst[suffix]
                else:
                    checkpoint_num = args.checkpoint_num
                
                adapter_path = f"results/{train_name}/{model_name}/checkpoint_{checkpoint_num}"
                linear_checkpoint_path = f"results/{train_name}/{model_name}/checkpoint_{checkpoint_num}_linear.pt"
            
            # Set up dataset path
            if args.data_name == 'ambiguous' or args.data_name == 'ambiguous_qe':
                dataset_path = f'training_datasets/{args.data_name}/{retriever}/autoregressive_{dataset_name}_dev_dataset_1b_contrastive_2_to_5_ctxs'
            elif args.data_name == 'qampari':
                dataset_path = f'training_datasets/{args.data_name}/{retriever}/autoregressive_{dataset_name}_dev_dataset_1b_contrastive_5_to_8_ctxs'
            elif args.data_name in ['nq', 'msmarco']:
                dataset_path = f'training_datasets/{args.data_name}/{retriever}/autoregressive_{dataset_name}_dev_dataset_1b_qemb'
            elif args.data_name in ['arguana_generated', 'kialo', 'opinionqa', 'wsd_distinct']:
                dataset_path = dev_data_path
            else:
                raise ValueError(f"Invalid data name: {args.data_name}")
            
            logger.info('loading embeddings dataset from %s', dataset_path)
            
            # Set up output paths
            generated_embeddings_path = f'output_embeddings/{train_name}/out_{dataset_name}_{suffix}_{args.split}.npy'
            generated_embeddings_lengths_path = f'output_embeddings/{train_name}/out_{dataset_name}_{suffix}_{args.split}_lengths.npy'
            
            # Generate embeddings
            outputs, loss_with_generation, all_labels_gen = eval_with_generation(
                adapter_path=adapter_path,
                base_model_id=args.base_model_id,
                linear_checkpoint_path=linear_checkpoint_path,
                output_path=generated_embeddings_path,
                output_lengths_path=generated_embeddings_lengths_path,
                input_data_path=dataset_path, 
                get_split=args.split,
                max_new_tokens=args.max_new_tokens,
                embedding_model_dim=passage_embeddings_map[retriever]["embedding_dim"],
                compute_loss=args.compute_loss,
                loss_function=args.loss_function,
                question_only=args.question_only,
                batch_size_training=args.batch_size_training,
                use_gt_q_embed=args.use_gt_q_embed,
                use_eos=args.use_eos
            )
            logger.info('writing to %s', generated_embeddings_path)
            outputs_gen = np.load(generated_embeddings_path)
            
            if not args.google_api:
                # Load passages
                logger.info(f"Loading passages from {passages_path}")
                passages = load_passages(passages_path)
                passage_id_map = {x["id"]: x for x in passages}
                
                for shard_id in range(args.num_shards):
                    # Load index
                    logger.info('passages_embeddings', passages_embeddings=passage_embeddings_map[retriever]["embedding_path"])
                    logger.info('passages_path', passages_path=passages_path, shard_id=shard_id, num_shards=args.num_shards)
                    index = load_index(
                        passage_embeddings_map[retriever]["embedding_dim"], 
                        passage_embeddings_map[retriever]["embedding_path"], 
                        save_or_load_index=args.save_or_load_index,
                        use_gpu=args.use_gpu,
                        shard_id=shard_id,
                        num_shards=args.num_shards
                    )
                    shard_string = f"_shard_{shard_id}" if args.num_shards > 1 else ""
                    
                    output_embedding_dir = f'output_embeddings/{args.training_data_name}_{retriever}/'
                    os.makedirs(output_embedding_dir, exist_ok=True)
                    
                    # Retrieve and evaluate
                    main_test(
                        index=index,
                        passage_id_map=passage_id_map,
                        raw_data_path=dev_data_path,
                        embedding_size=passage_embeddings_map[retriever]["embedding_dim"], 
                        lengths_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}_lengths.npy',
                        data_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}.npy',
                        output_path=f'results/{args.training_data_name}_{retriever}/toy{suffix}/retrieval_out_{args.split}_{args.data_name}{shard_string}.jsonl',
                        MAX_LATENTS=args.max_new_tokens,
                        top_k_per_query=args.top_k_per_query,
                        top_k=args.top_k,
                        start_idx=args.start_idx,
                        end_idx=args.end_idx,
                    )
                    
                    main_test(
                        index=index,
                        passage_id_map=passage_id_map,
                        raw_data_path=dev_data_path,
                        embedding_size=passage_embeddings_map[retriever]["embedding_dim"], 
                        lengths_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}_lengths.npy',
                        data_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}.npy',
                        output_path=f'results/{args.training_data_name}_{retriever}/toy{suffix}/retrieval_out_{args.split}_{args.data_name}_single{shard_string}.jsonl',
                        MAX_LATENTS=args.max_new_tokens,
                        top_k_per_query=args.top_k_per_query,
                        top_k=args.top_k,
                        start_idx=args.start_idx,
                        end_idx=args.end_idx,
                        aggregate_start_idx=0,
                        aggregate_end_idx=1
                    )
                    
                    main_test(
                        index=index,
                        passage_id_map=passage_id_map,
                        raw_data_path=dev_data_path,
                        embedding_size=passage_embeddings_map[retriever]["embedding_dim"], 
                        lengths_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}_lengths.npy',
                        data_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}.npy',
                        output_path=f'results/{args.training_data_name}_{retriever}/toy{suffix}/retrieval_out_{args.split}_{args.data_name}_from_2nd_to_3rd{shard_string}.jsonl',
                        MAX_LATENTS=args.max_new_tokens,
                        top_k_per_query=args.top_k_per_query,
                        top_k=args.top_k,
                        start_idx=args.start_idx,
                        end_idx=args.end_idx,
                        aggregate_start_idx=1,
                        aggregate_end_idx=2
                    )
                
                # Aggregate sharded results
                if args.num_shards > 1:
                    aggregate_sharded_results(f'results/{args.training_data_name}_{retriever}/toy{suffix}/retrieval_out_{args.split}_{args.data_name}.jsonl', args.num_shards)
                    aggregate_sharded_results(f'results/{args.training_data_name}_{retriever}/toy{suffix}/retrieval_out_{args.split}_{args.data_name}_single.jsonl', args.num_shards)
                    aggregate_sharded_results(f'results/{args.training_data_name}_{retriever}/toy{suffix}/retrieval_out_{args.split}_{args.data_name}_from_2nd_to_3rd.jsonl', args.num_shards)
            else:
                # Google API path
                main_test_google(
                    passages_embeddings=passage_embeddings_map[retriever]["embedding_path"], 
                    passages_path=passages_path, 
                    raw_data_path=dev_data_path,
                    embedding_size=passage_embeddings_map[retriever]["embedding_dim"], 
                    lengths_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}_lengths.npy',
                    data_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}.npy',
                    output_path=f'results/{args.training_data_name}_{retriever}/toy{suffix}/retrieval_out_{args.split}_{args.data_name}.jsonl',
                    MAX_LATENTS=args.max_new_tokens,
                    top_k_per_query=args.top_k_per_query,
                    top_k=args.top_k,
                    start_idx=args.start_idx,
                    end_idx=args.end_idx,
                    aggregate_start_idx=args.aggregate_start_idx,
                    aggregate_end_idx=args.aggregate_end_idx
                )
                
                main_test_google(
                    passages_embeddings=passage_embeddings_map[retriever]["embedding_path"], 
                    passages_path=passages_path, 
                    raw_data_path=dev_data_path,
                    embedding_size=passage_embeddings_map[retriever]["embedding_dim"], 
                    lengths_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}_lengths.npy',
                    data_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}.npy',
                    output_path=f'results/{args.training_data_name}_{retriever}/toy{suffix}/retrieval_out_{args.split}_{args.data_name}_single.jsonl',   
                    top_k_per_query=args.top_k_per_query,
                    top_k=args.top_k,
                    start_idx=args.start_idx,
                    end_idx=args.end_idx,
                    aggregate_start_idx=0,
                    aggregate_end_idx=1,
                    MAX_LATENTS=args.max_new_tokens
                )
                
                main_test_google(
                    passages_embeddings=passage_embeddings_map[retriever]["embedding_path"], 
                    passages_path=passages_path, 
                    raw_data_path=dev_data_path,
                    embedding_size=passage_embeddings_map[retriever]["embedding_dim"], 
                    lengths_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}_lengths.npy',
                    data_path=f'{output_embedding_dir}/out_{args.data_name}_{retriever}_{suffix}_{args.split}.npy',
                    output_path=f'results/{args.training_data_name}_{retriever}/toy{suffix}/retrieval_out_{args.split}_{args.data_name}_from_2nd_to_3rd.jsonl',   
                    top_k_per_query=args.top_k_per_query,
                    top_k=args.top_k,
                    start_idx=args.start_idx,
                    end_idx=args.end_idx,
                    aggregate_start_idx=1,
                    aggregate_end_idx=2,
                    MAX_LATENTS=args.max_new_tokens
                )


