import torch
import copy
import numpy as np
import argparse

from src.model import load_model
from tqdm import tqdm
import os
import json
import glob
import time
from tqdm import tqdm

from src.dataset import (
    DataHandler
)

from pathlib import Path
from src.retrieval_utils import Indexer, add_passages, load_passages, index_encoded_data, add_passages_single_instance
from datasets import load_dataset

import structlog
logger = structlog.get_logger()


def write_jsonl(data, output_path):
    with open(output_path, 'w') as f:
        for inst in data:
            f.write(json.dumps(inst) + '\n')


def evaluate_loop(dataloader, model, device, max_new_tokens, use_gt_q_embed, use_eos, compute_loss = True, pred_length = False):

    all_outputs = []
    all_losses = []
    all_labels = []
    all_lengths = []
    adaptive_max_new_tokens = (max_new_tokens is None and not pred_length)  # if doing pred length, we don't need to do adaptive max new tokens
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)
            
        # if doing adaptive max new tokens, just assign the max new tokens to the batch
        if 'positive_embeddings' in batch and adaptive_max_new_tokens:
            max_new_tokens = batch['positive_embeddings'].size(1)

        output = model.generate(
            max_new_tokens=max_new_tokens,
            use_gt_q_embed=use_gt_q_embed,
            use_eos=use_eos,
            **batch
        )
        instance_length = output.size(1)
        all_outputs.append(output.view(-1, output.size(-1)))
        

        if compute_loss:
            if 'labels' in batch:
                # compute the loss
                assert output.size() == batch['labels'].size(), (output.size(), batch['labels'].size())
                loss = model.loss_fct(output.float(), batch['labels'].float())
                all_lengths.append(batch['labels'].size(1))
                all_losses.append(loss.item())
                all_labels.append(batch['labels'].view(-1, batch['labels'].size(-1)))
            elif 'positive_embeddings' in batch:
                loss = model.loss_fct(output.float(), batch['positive_embeddings'].float(), batch['negative_embeddings'].float())
                all_lengths.append(batch['positive_embeddings'].size(1))
                all_labels.append(batch['positive_embeddings'].view(-1, batch['positive_embeddings'].size(-1)))
                all_losses.append(loss.item())
        else:
            if adaptive_max_new_tokens or pred_length:  # if doing pred length, we need to append the instance length.
                all_lengths.append(instance_length)
    
    if compute_loss:
        return torch.cat(all_outputs, dim=0).cpu().numpy(), sum(all_losses) / len(all_losses), torch.cat(all_labels, dim=0).cpu().numpy(), all_lengths
    else:
        return torch.cat(all_outputs, dim=0).cpu().numpy(), None, None, all_lengths


def get_instruction(base_model_type):
    if base_model_type == 'inf':
        instruction_template = "Instruct: "
        response_template = ""
    elif base_model_type == 'llama-1b' or base_model_type == 'llama-3b' or base_model_type == 'llama-8b':
        instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif base_model_type == 'qwen3-4b':
        instruction_template = "<|im_start|>user\n"
        response_template = "<|im_end|>\n<|im_start|>assistant\n"
    else:
        raise ValueError(f"Invalid base model type: {base_model_type}")
    instruction = (f'{instruction_template}Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: [QUERY]\n{response_template}').strip('\n')
    return instruction



def generate_input_data(input_data_path, tokenizer, base_model_type, batch_size_training, get_split):
        # Tokenize dataset
    def tokenize_function(examples):
        if 'question' in examples:
            question = examples['question']
        else:
            question = examples['question_text']
        examples['text'] = formulate_text(instruction, question)
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

    instruction = get_instruction(base_model_type)
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(input_data_path))
    tokenizer.pad_token = tokenizer.eos_token
    
    # tokenize dataset => get the question embedding
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # # Load dataset. Always batch size = 1
    full_dataset = tokenized_datasets['train']
    data_handler = DataHandler(full_dataset, data_collator, batch_size_training, get_split, 4)
    dataloader = data_handler.get_full_dataloader()
    return dataloader


def eval_with_generation(input_data_path = 'autoregressive_wsd_train_dataset_1b',
         get_split = 'train-held-out',
         adapter_path = "results/test/toy/checkpoint_4", 
         base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct",
         linear_checkpoint_path = None,
         base_model_type = 'inf',
         max_new_tokens = 2,
         embedding_model_dim = 1536,
         compute_loss = True,
         loss_function = 'Contrastive',
         batch_size_training = 1,
         use_gt_q_embed = False,
         use_eos = False,
         pred_length = False):
        
    # Define model and tokenizer
    model, tokenizer = load_model(train_lora=False,
                                  base_model_id=base_model_id, 
                                  adapter_path=adapter_path, 
                                  linear_checkpoint_path=linear_checkpoint_path,
                                  embedding_model_dim=embedding_model_dim, 
                                  model_type='EmbeddingModel', 
                                  loss_function=loss_function)
    
    logger.info(f"Generating input data from {input_data_path}, using the raw text")
    dataloader = generate_input_data(input_data_path, tokenizer, base_model_type, batch_size_training, get_split)


    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        outputs, loss, all_labels, all_lengths = evaluate_loop(dataloader, model, device, 
                                                               max_new_tokens=max_new_tokens, 
                                                               use_gt_q_embed=use_gt_q_embed, 
                                                               use_eos=use_eos, 
                                                               compute_loss=compute_loss, 
                                                               pred_length=pred_length)
        return outputs, loss, all_labels, all_lengths
    


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
  

def aggregate_different_queries_by_length(top_ids_and_scores, lengths=None, MAX_LATENTS=None, top_k=100, aggregate_start_idx=0, aggregate_end_idx=None, round_robin_percentage=1.0):
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
                current_id, current_score = query_results[idx]
                
                # Only add if we haven't seen this ID before
                if current_id not in seen_ids:
                    aggregated_top_ids_and_scores_per_inst.append((current_id, current_score))
                    seen_ids.add(current_id)
                
                if len(aggregated_top_ids_and_scores_per_inst) >= top_k * round_robin_percentage:
                    break
            if len(aggregated_top_ids_and_scores_per_inst) >= top_k * round_robin_percentage:
                break
        
        if round_robin_percentage < 1.0:
            for j in range(idx+1, max_len):
                aggregated_top_ids_and_scores_per_inst.append(ids_and_scores_to_aggregate[0][j])
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
    
     
      
def retrieve(num_shards, embeddings_path, embedding_model_dim, passage_id_map, output_path, 
              raw_data_path = '', 
              question_embeddings = None, lengths = None, embedding_size = 4096, top_k_per_query = 100, top_k = 100,
              start_idx = 0, end_idx = None, MAX_LATENTS = None, aggregate_start_idx = 0, aggregate_end_idx = None, 
              round_robin_percentage=1.0, save_before_aggregation=False):
    
    # loading question embeddings
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
        assert len(data) == len(lengths), (len(data), len(lengths))
        assert question_embeddings.shape[0] == sum(lengths), (question_embeddings.shape[0], sum(lengths))
    else:
        lengths = None
        
    data_before_aggregation = []
    if lengths is not None:
        for j, l in enumerate(lengths):
            for _ in range(l):
                data_before_aggregation.append(copy.deepcopy(data[j]))
    else:
        for j in range(len(data)):
            for _ in range(MAX_LATENTS):
                data_before_aggregation.append(copy.deepcopy(data[j]))
            
    # Start Retrieving!
    all_sharded_ids_and_scores = []
    for shard_id in range(num_shards):
        # Load index
        logger.info('passages_embeddings', passages_embeddings=embeddings_path)
        index = load_index(
            embedding_model_dim, 
            embeddings_path, 
            save_or_load_index=args.save_or_load_index,
            use_gpu=args.use_gpu,
            shard_id=shard_id,
            num_shards=args.num_shards
        )
    
        # Start Search! Get top k results.
        start_time_retrieval = time.time()
        sharded_ids_and_scores = index.search_knn(question_embeddings.reshape(-1, embedding_model_dim), top_k_per_query)
        logger.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
        all_sharded_ids_and_scores.append(sharded_ids_and_scores)
    
    top_ids_and_scores = aggregate_sharded_results(all_sharded_ids_and_scores, num_shards)
    logger.info(f"aggregated top_ids_and_scores for {num_shards} shards")
    add_passages(data_before_aggregation, passage_id_map, top_ids_and_scores)
    
    top_ids_and_scores = aggregate_different_queries_by_length(top_ids_and_scores, lengths, MAX_LATENTS, top_k, aggregate_start_idx, aggregate_end_idx, round_robin_percentage)
    logger.info(f"length of the data to be retrieved: {len(data)}, length of the retrieved results: {len(top_ids_and_scores)}")
    add_passages(data, passage_id_map, top_ids_and_scores)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        for ex in data:
            json.dump(ex, fout, ensure_ascii=False)
            fout.write("\n")
    logger.info(f"Saved results to {output_path}")
    
    if save_before_aggregation:
        with open(output_path.with_name(output_path.name.replace('.jsonl', '_before_agg.jsonl')), "w") as fout:
            for ex in data_before_aggregation:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        logger.info(f"Saved results to {output_path.with_name(output_path.name.replace('.jsonl', '_before_agg.jsonl'))}")
    
    
def aggregate_sharded_results(all_sharded_ids_and_scores, num_shards):
    if num_shards == 1:
        return all_sharded_ids_and_scores[0]

    # Aggregate results from all shards
    top_ids_and_scores = []
    for i in range(len(all_sharded_ids_and_scores[0])):
        top_ids_and_scores.append([])
        for _ in range(2):
            top_ids_and_scores[i].append([])
        for shard_id in range(num_shards):
            top_ids_and_scores[i][1] = np.append(top_ids_and_scores[i][1], all_sharded_ids_and_scores[shard_id][i][1])  # scores
            top_ids_and_scores[i][0].extend(all_sharded_ids_and_scores[shard_id][i][0])  # ids
            
        indices = np.argsort(top_ids_and_scores[i][1])[::-1]
        top_ids_and_scores[i][1] = top_ids_and_scores[i][1][indices]
        top_ids_and_scores[i][0] = [top_ids_and_scores[i][0][j] for j in indices]
            
    return top_ids_and_scores
    
    

    
def parse_args():
    parser = argparse.ArgumentParser(description='Generate embeddings and perform retrieval evaluation')
    # Data configuration
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'train', 'train-held-out'],
                       help='Data split to evaluate on')
    parser.add_argument('--dev_data_path', type=str, default='data_creation/raw_data/ambiguous_qe_dev_question_only.jsonl',
                       help='Path to the dev data')
    parser.add_argument('--corpus_path', type=str, default=None,
                       help='path to passages')
    
    # Embedding configuration
    parser.add_argument('--embedding_model_dim', type=int, default=1536,
                       help='Embedding model dimension')
    parser.add_argument('--embeddings_path', type=str, default=None,
                       help='path to embeddings')
    
    # Indexing configuration
    parser.add_argument('--use_gpu', action='store_true', default=False,
                       help='Whether to use GPU for indexing')
    parser.add_argument('--num_shards', type=int, default=1,
                       help='Number of shards for indexing')
    parser.add_argument('--save_or_load_index', action='store_true', default=False,
                       help='Whether to save/load index')

    # Model configuration
    parser.add_argument('--base_model_id', type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                       help='Base model ID')
    parser.add_argument('--adapter_path', type=str, default=None,
                       help='Adapter path')
    parser.add_argument('--linear_checkpoint_path', type=str, default=None,
                       help='Linear checkpoint path')
    parser.add_argument('--base_model_type', type=str, default='llama-1b',
                       help='Base model type')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--compute_loss', action='store_true', default=False,
                       help='Whether to compute loss during evaluation')
    parser.add_argument('--pred_length', action='store_true', default=False,
                       help='Whether to predict length')
    
    # Config parameters (previously from config file)
    parser.add_argument('--loss_function', type=str, default='Hungarian_Contrastive',
                       help='Loss function to use')
    parser.add_argument('--batch_size_training', type=int, default=1,
                       help='Batch size for training')
    parser.add_argument('--use_gt_q_embed', action='store_true', default=False,
                       help='Whether to use ground truth question embedding')
    parser.add_argument('--use_eos', action='store_true', default=False,
                       help='Whether to use end of sequence token')
    
    # Paths and directories
    parser.add_argument('--embeddings_root', type=str, default='/path/to/embeddings/',
                       help='Root directory for embeddings')
    parser.add_argument('--root', type=str, default='/path/to/data',
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
    parser.add_argument('--inference_modes', type=str, nargs='+', default='all',
                       choices=['first', 'second', 'all', 'average'],
                       help='Inference mode')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path')
    parser.add_argument('--round_robin_percentage', type=float, default=1.0,
                       help='Round robin percentage')
    parser.add_argument('--save_before_aggregation', action='store_true', default=False,
                       help='Whether to save before aggregation')
    

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info('using base model type: ', args.base_model_type)
    logger.info('adapter_path', args.adapter_path)
    if args.adapter_path == "None":
        args.adapter_path = None
    logger.info('base_model_id', args.base_model_id)
    logger.info('linear_checkpoint_path', args.linear_checkpoint_path)
    logger.info('args.max_new_tokens', args.max_new_tokens)
    
    # Generate Query embeddings
    outputs, _, _, lengths = eval_with_generation(
        adapter_path=args.adapter_path,
        base_model_id=args.base_model_id,
        linear_checkpoint_path=args.linear_checkpoint_path,
        base_model_type=args.base_model_type,
        input_data_path=args.dev_data_path, 
        get_split=args.split,
        max_new_tokens=args.max_new_tokens,
        embedding_model_dim=args.embedding_model_dim,
        compute_loss=args.compute_loss,
        loss_function=args.loss_function,
        batch_size_training=args.batch_size_training,
        use_gt_q_embed=args.use_gt_q_embed,
        use_eos=args.use_eos,
        pred_length=args.pred_length
    )
    assert len(lengths) > 0 or args.max_new_tokens is not None, "Lengths can only be empty if max_new_tokens is not None"
    
    # Run Retrieval
    for inference_mode in args.inference_modes:
        if inference_mode == 'first':
            inference_string = '_single'
            aggregate_start_idx = 0
            aggregate_end_idx = 1
        elif inference_mode == 'second':
            inference_string = '_from_2nd_to_3rd'
            aggregate_start_idx = 1
            aggregate_end_idx = 2
        elif inference_mode == 'all':
            inference_string = ''
            aggregate_start_idx = 0
            aggregate_end_idx = None
            if args.max_new_tokens is not None:
                inference_string = f'_max_new_tokens_{args.max_new_tokens}'
        elif inference_mode == 'average':
            inference_string = '_average'
            aggregate_start_idx = 0
            aggregate_end_idx = None
            if lengths is not None and len(lengths) > 0:
                new_outputs = []
                start_idx = 0
                for i in range(len(lengths)):
                    new_outputs.append(outputs[start_idx:start_idx+lengths[i]].mean(axis=0).reshape(1, -1))
                    start_idx += lengths[i]
                outputs = np.concatenate(new_outputs, axis=0)
            else:
                outputs = outputs.reshape(-1, args.max_new_tokens, outputs.shape[-1])
                outputs = outputs.mean(axis=1)
            args.max_new_tokens = 1
        else:
            raise ValueError(f"Invalid inference mode: {inference_mode}")
        
        # start retrieval
        # Load passages
        logger.info(f"Loading passages from {args.corpus_path}")
        passages = load_passages(args.corpus_path)
        passage_id_map = {x["id"]: x for x in passages}
        # Retrieve and evaluate
        retrieve(
            num_shards=args.num_shards, 
            embeddings_path=args.embeddings_path, 
            embedding_model_dim=args.embedding_model_dim, 
            passage_id_map=passage_id_map,
            raw_data_path=args.dev_data_path,
            lengths=lengths,
            question_embeddings=outputs,
            output_path=Path(args.output_path).parent / Path(args.output_path).name.replace('.jsonl', f'{inference_string}.jsonl'),
            MAX_LATENTS=args.max_new_tokens,
            top_k_per_query=args.top_k_per_query,
            top_k=args.top_k,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            aggregate_start_idx=aggregate_start_idx,
            aggregate_end_idx=aggregate_end_idx,
            round_robin_percentage=args.round_robin_percentage,
            save_before_aggregation=args.save_before_aggregation
        )
            