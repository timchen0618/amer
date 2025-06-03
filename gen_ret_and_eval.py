import torch
import torch.distributed
import torch.optim as optim
import numpy as np

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

def evaluate_loop(dataloader, model, device, max_new_tokens, use_gt_q_embed, compute_loss = True):

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
         compute_loss = True):

    instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    instruction = f'{instruction_template}Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: [QUERY]\n{response_template}'
    
    with open('configs/eval.yaml') as f:
        config_dict = yaml.safe_load(f)

    print("Config:", config_dict)
    configs = Config(config_dict)

    # # Define model and tokenizer
    base_model_id = base_model_id  # Replace with your base model ID
    adapter_path = adapter_path  # Path to your saved adapter weights
    model, tokenizer = load_model(train_lora=False,
                                  base_model_id=base_model_id, 
                                  adapter_path=adapter_path, 
                                  linear_checkpoint_path=linear_checkpoint_path,
                                  embedding_model_dim=embedding_model_dim, 
                                  loss_function=configs.loss_function)
    
    # always use hungarian loss for evaluation
    # always not predict question for evaluation    
    if Path(input_data_path).is_dir():
        logger.info(f"Loading input data from {input_data_path}, using the question embedding")
        dataloader = load_input_data(configs.loss_function, configs.question_only, configs.batch_size_training, get_split, input_data_path)
    else:
        logger.info(f"Generating input data from {input_data_path}, using the raw text")
        dataloader = generate_input_data(configs.loss_function, configs.question_only, input_data_path, tokenizer)


    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        outputs, loss, all_labels, all_lengths = evaluate_loop(dataloader, model, device, max_new_tokens=max_new_tokens, use_gt_q_embed=configs.use_gt_q_embed, compute_loss=compute_loss)
        # outputs = outputs.reshape(-1, embedding_model_dim)
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


def load_index_and_passages(embedding_size, passages_embeddings, passages_path):
    logger.info("doing indexing...")
    index = Indexer(embedding_size, 0, 8)

    # index all passages
    input_paths = glob.glob(passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")

    logger.info(f"Indexing passages from files {input_paths}")
    start_time_indexing = time.time()
    index_encoded_data(index, input_paths, 100000)
    logger.info(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

    # load passages
    logger.info(f"Loading passages from {passages_path}")
    passages = load_passages(passages_path)
    passage_id_map = {x["id"]: x for x in passages}
    return index, passage_id_map, passages
        
def main_test(passages_embeddings, passages_path, output_path, 
              raw_data_path = '/scratch/hc3337/projects/autoregressive/data/wsd/distinct/train.jsonl', 
              data_path = 'out.npy', lengths_path = "", embedding_size = 4096, top_k_per_query = 100, top_k = 100,
              start_idx = 0, end_idx = None, MAX_LATENTS = None, aggregate_start_idx = 0, aggregate_end_idx = None, google_api = False):
    
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
    
    logger.info('google_api', google_api=google_api)
    if google_api:
        if MAX_LATENTS is not None:
            assert lengths is None, (lengths, MAX_LATENTS)
            lengths = [MAX_LATENTS] * len(data)

        start = 0
        for i in range(len(data)):
            # load index and passages for each query
            index, passage_id_map, _ = load_index_and_passages(embedding_size, 
                                                               (passages_embeddings) + '/' + str(i) + '/*', 
                                                               passages_path + f"/psgs_{i}.tsv")
            
            start_time_retrieval = time.time()
            top_ids_and_scores_inst = index.search_knn(question_embeddings[start:start+lengths[i]].reshape(-1, embedding_size), top_k_per_query)
            logger.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
            top_ids_and_scores_inst = aggregate_different_queries_by_length(top_ids_and_scores_inst, [lengths[i]], None, top_k, aggregate_start_idx, aggregate_end_idx)
            assert len(top_ids_and_scores_inst) == 1, (len(top_ids_and_scores_inst))
            logger.info("top_ids_and_scores_inst[0][0]", lens=len(top_ids_and_scores_inst[0][0]))
            add_passages_single_instance(data[i], passage_id_map, top_ids_and_scores_inst[0])
            
            start += lengths[i]
            
        assert start == len(question_embeddings), (start, len(question_embeddings))
    else:
        # load index and passages
        logger.info('passages_embeddings', passages_embeddings=passages_embeddings)
        logger.info('passages_path', passages_path=passages_path)
        index, passage_id_map, _ = load_index_and_passages(embedding_size, passages_embeddings, passages_path)
        
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
    
    

    
if __name__ == "__main__":    
    num_map = {
        "msmarco_stella": 10000, "msmarco_cont": 10000, "msmarco_inf": 10000,
        "nq_cont": 70000, "nq_inf": 70000, "nq_stella": 70000,
        "ambiguous_cont": 1000, "ambiguous_inf": 3501, "ambiguous_stella": 1000,
        "qampari_cont": 30000, "qampari_inf": 25001, "qampari_stella": 30000,
        "qampari+ambiguous_inf": 10000, "nq+msmarco_inf": 280000, "ambiguous+qampari_inf": 20001
    }
    # for data_name in ['nq', 'msmarco', 'qampari']:
    for data_name in ['ambiguous']:
        # for training_data_name in ['nq', 'msmarco', 'qampari']:
        for training_data_name in ['ambiguous']:
            # if data_name != training_data_name:
            #     continue
            suffix_list = ['_contrastive_ordered_from_stage2_lr2e5_ep20_temp0.05_warmup0.05']  # ["_contrastive_from_stage1_nq", "_contrastive_from_stage2_nq", "_contrastive_from_stage2_nq_lr1e3"] ["_contrastive_from_stage2_nq_lr1e5", "_contrastive_from_stage2_nq_lr1e5_ordered"]
            retriever_list = ['inf'] # ['stella', 'inf', 'cont']            
            
            split = 'dev'
            MAX_LATENTS = None
            compute_loss = True
            GOOGLE_API = False
            if data_name in ['arguana_generated', 'kialo', 'opinionqa']:
                assert GOOGLE_API, "Google API is required for these datasets"
            else:
                assert not GOOGLE_API, "Google API is not allowed for these datasets"
                
            
            embeddings_dir = 'qampari_embeddings' if data_name in ['qampari'] else data_name
            if data_name == 'ambiguous':
                embeddings_dir = 'nq'
            
            if not GOOGLE_API:
                passage_embeddings_map = {
                    'stella': {"embedding_path": f"/datastor1/hungting/stella_en_400M_v5/{embeddings_dir}/*", "embedding_dim": 1024},
                    'inf': {"embedding_path": f"/datastor1/hungting/inf/{embeddings_dir}/*", "embedding_dim": 1536},
                    'cont': {"embedding_path": f"/datastor1/hungting/Contriever/{embeddings_dir}/*", "embedding_dim": 768}
                }
            else:
                passage_embeddings_map = {
                    'stella': {"embedding_path": f"/datastor1/hungting/google_api/stella_embeddings/{data_name}", "embedding_dim": 1024},
                    'inf': {"embedding_path": f"/datastor1/hungting/google_api/inf_embeddings/{data_name}", "embedding_dim": 1536},
                    'cont': {"embedding_path": f"/datastor1/hungting/google_api/contriever_embeddings/{data_name}", "embedding_dim": 768}
                }
            
            ### passage embeddings ###
            if not GOOGLE_API:
                if data_name in ['qampari']:
                    passages_path = f'/datastor1/hungting/wikipedia_chunks/chunks_v5.tsv'
                elif data_name == 'ambiguous':
                    passages_path = f'/scratch/cluster/hungting/projects/autoregressive/data/nq/corpus.tsv'
                else:
                    passages_path = f'/scratch/cluster/hungting/projects/autoregressive/data/{data_name}/corpus.tsv'
            else:
                passages_path = f'/datastor1/hungting/serpapi/contriever_psgs/{data_name}'
            
            ### load dev data ###
            if data_name == 'ambiguous':
                dev_data_path = f'data_creation/raw_data/{data_name}_{split}_question_only_2_to_5_ctxs.jsonl'
            elif data_name == 'qampari':
                dev_data_path = f'data_creation/raw_data/{data_name}_{split}_question_only_5_to_8_ctxs.jsonl'
            elif data_name in ['nq', 'msmarco']:
                dev_data_path = f'data_creation/raw_data/{data_name}_{split}_question_only.jsonl'
            else:
                dev_data_path = f"data_creation/raw_data/{data_name}_question_only.jsonl"

            
            
            for suffix in suffix_list:
                ### retriever ###
                # for retriever in ['cont', 'stella', 'inf']:
                for retriever in retriever_list:
                    train_name = f"{training_data_name}_{retriever}"
                    dataset_name = f"{data_name}_{retriever}"
                    model_name = f"toy{suffix}"
                    
                    ### load model ###
                    Path(f"output_embeddings/{train_name}").mkdir(parents=True, exist_ok=True)
                    adapter_path = f"results/{train_name}/{model_name}/checkpoint_{num_map[train_name]}"
                    # adapter_path = None
                    base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
                    linear_checkpoint_path = f"results/{train_name}/{model_name}/checkpoint_{num_map[train_name]}_linear.pt"
                    # linear_checkpoint_path = None
                    # print('loading model from ', adapter_path)
                    
                    ### load embeddings dataset ###
                    if data_name == 'ambiguous':
                        dataset_path = f'training_datasets/autoregressive_{dataset_name}_dev_dataset_1b_contrastive_2_to_5_ctxs'
                    elif data_name == 'qampari':
                        dataset_path = f'training_datasets/autoregressive_{dataset_name}_dev_dataset_1b_contrastive_5_to_8_ctxs'
                    elif data_name in ['nq', 'msmarco']:
                        dataset_path = f'training_datasets/autoregressive_{dataset_name}_dev_dataset_1b_qemb'
                    elif data_name in ['arguana_generated', 'kialo', 'opinionqa']:
                        dataset_path = dev_data_path
                    else:
                        raise ValueError(f"Invalid data name: {data_name}")
                    
                    logger.info('loading embeddings dataset from %s', dataset_path)
                    
                    ### output embeddings path ###
                    generated_embeddings_path = f'output_embeddings/{train_name}/out_{dataset_name}_{suffix}_dev.npy'
                    generated_embeddings_lengths_path = f'output_embeddings/{train_name}/out_{dataset_name}_{suffix}_dev_lengths.npy'
                    get_split = 'dev'

                    ### generate embeddings ###
                    outputs, loss_with_generation, all_labels_gen = eval_with_generation(adapter_path = adapter_path,
                        base_model_id = base_model_id,
                        linear_checkpoint_path = linear_checkpoint_path,
                        output_path = generated_embeddings_path,
                        output_lengths_path = generated_embeddings_lengths_path,
                        input_data_path = dataset_path, 
                        get_split = get_split,
                        max_new_tokens = MAX_LATENTS,
                        embedding_model_dim = passage_embeddings_map[retriever]["embedding_dim"],
                        compute_loss = compute_loss)
                    logger.info('writing to %s', generated_embeddings_path)
                    outputs_gen = np.load(generated_embeddings_path)
            
                    ### retrieve and evaluate ###
                    main_test(
                            passages_embeddings = passage_embeddings_map[retriever]["embedding_path"], 
                            passages_path = passages_path, 
                            raw_data_path = dev_data_path,
                            embedding_size = passage_embeddings_map[retriever]["embedding_dim"], 
                            lengths_path = f'output_embeddings/{training_data_name}_{retriever}/out_{data_name}_{retriever}_{suffix}_{split}_lengths.npy',
                            data_path = f'output_embeddings/{training_data_name}_{retriever}/out_{data_name}_{retriever}_{suffix}_{split}.npy',
                            output_path = f'results/{training_data_name}_{retriever}/toy{suffix}/retrieval_out_{split}_{data_name}.jsonl',
                            MAX_LATENTS = MAX_LATENTS,
                            google_api = GOOGLE_API
                            )
                
                    main_test(
                            passages_embeddings = passage_embeddings_map[retriever]["embedding_path"], 
                            passages_path = passages_path, 
                            raw_data_path = dev_data_path,
                            embedding_size = passage_embeddings_map[retriever]["embedding_dim"], 
                            lengths_path = f'output_embeddings/{training_data_name}_{retriever}/out_{data_name}_{retriever}_{suffix}_{split}_lengths.npy',
                            data_path = f'output_embeddings/{training_data_name}_{retriever}/out_{data_name}_{retriever}_{suffix}_{split}.npy',
                            output_path = f'results/{training_data_name}_{retriever}/toy{suffix}/retrieval_out_{split}_{data_name}_single.jsonl',
                            MAX_LATENTS = MAX_LATENTS,
                            aggregate_start_idx = 0,
                            aggregate_end_idx = 1,
                            google_api = GOOGLE_API
                            )
                    
                    main_test(
                            passages_embeddings = passage_embeddings_map[retriever]["embedding_path"], 
                            passages_path = passages_path, 
                            raw_data_path = dev_data_path,
                            embedding_size = passage_embeddings_map[retriever]["embedding_dim"], 
                            lengths_path = f'output_embeddings/{training_data_name}_{retriever}/out_{data_name}_{retriever}_{suffix}_{split}_lengths.npy',
                            data_path = f'output_embeddings/{training_data_name}_{retriever}/out_{data_name}_{retriever}_{suffix}_{split}.npy',
                            output_path = f'results/{training_data_name}_{retriever}/toy{suffix}/retrieval_out_{split}_{data_name}_from_2nd_to_3rd.jsonl',
                            MAX_LATENTS = MAX_LATENTS,
                            aggregate_start_idx = 1,
                            aggregate_end_idx = 2,
                            google_api = GOOGLE_API
                            )
                    
        