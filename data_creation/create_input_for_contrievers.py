import json
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
import torch
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path    
import csv
from nltk import word_tokenize
import regex
import string
import unicodedata
import os

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def _normalize(text):
    return unicodedata.normalize('NFD', text)

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False
        
        
def write_tsv(data, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(data)

def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def read_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def safe_from_list_and_save(dataset_dicts, out_dataset_path, batch_size=10000):
    datasets = []
    for i in range(0, len(dataset_dicts), batch_size):
        chunk = dataset_dicts[i:i + batch_size]
        chunk_dataset = Dataset.from_list(chunk)
        datasets.append(chunk_dataset)

    full_dataset = concatenate_datasets(datasets)
    full_dataset.save_to_disk(out_dataset_path)
    return full_dataset

def create_output_embeddings(train_data_path, out_data_path, out_lengths_path):
    data = read_jsonl(train_data_path)
    len_positives = []
    new_data = []

    _id = 0
    for inst in data:
        len_positives.append(len(inst['ground_truths']))
        if isinstance(inst['ground_truths'][0], list):
            ground_truths = [cluster[0] for cluster in inst['ground_truths']]
        else:
            ground_truths = inst['ground_truths']
        for gt in ground_truths:
            doc = gt['title'] + ' ' + gt['text']
            new_data.append({'question': doc, 'input': doc, 'id': str(_id), 'answers': [''], 'ctxs': [],})
            _id += 1
            
    print(len(new_data))
    with open(out_data_path, 'w') as f:
        json.dump(new_data, f, indent=4)
        
    import numpy as np
    np.save(out_lengths_path, np.array(len_positives))


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

def shift_to_right(t, padding=-100):
    shifted = torch.cat([t[:,1:,:], t[:,0:1,:]], dim=1)
    shifted[:,-1,:] = padding
    return shifted
    


@torch.no_grad()
def create_input_embeddings_for_query(model_name = "meta-llama/Llama-3.2-1B-Instruct", 
                                      input_data_path='', 
                                      batch_size=32, 
                                      len_outputs_path='', outputs_path='', 
                                      out_dataset_path='',
                                      max_length=-1):
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


    instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    instruction = f'{instruction_template}Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: [QUERY]\n{response_template}'
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(input_data_path))

    # # Define model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    seperator = tokenizer(response_template)[1:]
    
    # tokenize dataset => get the question embedding
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    # output size (# data, 128, hidden_size)
    dataloader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    # take care of outputs 
    output_lens = np.load(len_outputs_path)  # (bsz, )
    outputs = np.load(outputs_path)
    assert outputs.shape[0] == output_lens.sum(), (outputs.shape[0], output_lens.sum())

    import time
    start_time = time.time()
    i = 0
    actual_data_size = 0
    output_start = 0
    labels = []
    actual_data_indices = []
    
    dataset_dicts = []
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)
            
        # get the output length from numpy file
        output_len = output_lens[i]
        # get the output embeddings from numpy file
        labels = outputs[output_start:output_start+output_len]
        
        # truncate the labels if max_length is not -1
        if max_length != -1:
            labels = labels[:max_length]
        
        # if the labels are less than max_length, skip the current batch
        if (labels.shape[0] < max_length) and max_length != -1:
            i += 1
            output_start += output_len
            continue
        assert ((labels.shape[0] == max_length) or (max_length == -1)), (labels.shape[0], max_length)
        
        if max_length == -1:
            dataset_dicts.append({"input_ids": batch['input_ids'][0].cpu().numpy(), "labels": labels, "attention_mask": batch['attention_mask'][0].cpu().numpy(), "output_len": output_len.item()})
        else:
            dataset_dicts.append({"input_ids": batch['input_ids'][0].cpu().numpy(), "labels": labels, "attention_mask": batch['attention_mask'][0].cpu().numpy(), "output_len": max_length})
                        
                        
        
        output_start += output_len
        actual_data_size += 1
        actual_data_indices.append(i)
        i += 1
    
    safe_from_list_and_save(dataset_dicts, out_dataset_path, batch_size=2000)

    print('time elapsed: ', (time.time()-start_time)/60.0, 'min.')
    print('actual data size: ', actual_data_size)
    return actual_data_indices


@torch.no_grad()
def create_input_embeddings_for_contrastive(model_name = "meta-llama/Llama-3.2-1B-Instruct", 
                                      input_data_path='', 
                                      batch_size=32, 
                                      positive_embeddings_path='', 
                                      negative_embeddings_path='', 
                                      out_dataset_path='',
                                      pred_length_labels=False):
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
        
        if pred_length_labels:
            length_label = positive.shape[0]
            length_labels = tokenizer(str(length_label) + "<embed>", padding='max_length', truncation=True, max_length=257, return_tensors='pt')
            dataset_dicts[-1]['length_labels_input_ids'] = length_labels['input_ids'][:, 1:5].squeeze(0).cpu().numpy()
            dataset_dicts[-1]['length_labels_attention_mask'] = length_labels['attention_mask'][:, 1:5].squeeze(0).cpu().numpy()                        
                        
        
        actual_data_size += 1
        actual_data_indices.append(i)
        i += 1
        if i >= positive_embeddings.shape[0]:
            break

    safe_from_list_and_save(dataset_dicts, out_dataset_path, batch_size=2000)

    print('time elapsed: ', (time.time()-start_time)/60.0, 'min.')
    print('actual data size: ', actual_data_size)
    return actual_data_indices
    

def chunk_text(text, chunk_size=100):
    tokens = word_tokenize(text)
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [' '.join(chunk) for chunk in chunks]

def check_answer(inst):
    tokenizer = SimpleTokenizer()
    inst['positive_ctxs'] = []
    
    for j, answers in enumerate(inst['answers']):
        inst['positive_ctxs'].append([])
        for ctxs in inst['ctxs']:
            if has_answer(answers, ctxs['title'] + ' ' + ctxs['text'], tokenizer):
                inst['positive_ctxs'][j].append(ctxs)
                ctxs['has_answer'] = j

def check_evidences(new_data):
    no_evidence_cnt = 0
    evidence_cnt = 0
    evidence_data = []
    no_evidence_data = []
    for inst in tqdm(new_data):
        no_evidence = False
        for j in range(len(inst['positive_ctxs'])):
            inst['positive_ctxs'][j] = sorted(inst['positive_ctxs'][j], key=lambda x: x['has_answer'])
            if len(inst['positive_ctxs'][j]) == 0:
                no_evidence = True
        if no_evidence:
            no_evidence_cnt += 1
            no_evidence_data.append(inst)
        else:
            evidence_cnt += 1
            evidence_data.append(inst)
    return no_evidence_cnt, evidence_cnt, evidence_data, no_evidence_data

    
if __name__ == '__main__':
    rootdir = Path(__file__).parent
    print(rootdir)
    
    generate_split = 'contrastive'
    if generate_split in ['qampari_train', 'qampari_dev']:
        tag='qampari_org'
    elif generate_split in ['question_only', 'corpus', 'contrastive', 'contrastive_sequence', 'contrastive_sequence_check_answer', 'gaussian_synthetic']:
        tag=''
    else:
        raise ValueError(f'Invalid generate_split: {generate_split}')
       
    if generate_split == 'qampari_train':
        data_indices = create_input_embeddings_for_query(batch_size=1, 
                                        model_name="meta-llama/Llama-3.2-1B-Instruct",
                                        input_data_path=rootdir / f'raw_data/{tag}/qampari_train_only_question.jsonl', 
                                        len_outputs_path=rootdir / f'raw_data/{tag}/doc_lens_qampari_train.npy', 
                                        outputs_path=rootdir / f'raw_data/{tag}/doc_embeddings_qampari_train.npy',
                                        out_dataset_path='autoregressive_qampari_org_max5_train_dataset_1b_qemb',
                                        max_length=1)
        print(len(data_indices))
        ## for the input_data, only the question is used. So no need to worry about other fields.
    
    if generate_split == 'qampari_dev':
        data_indices = create_input_embeddings_for_query(batch_size=1, 
                                        model_name="meta-llama/Llama-3.2-1B-Instruct",
                                        input_data_path=rootdir / f'raw_data/{tag}/qampari_dev_only_question.jsonl', 
                                        len_outputs_path=rootdir / f'raw_data/{tag}/doc_lens_qampari_dev.npy', 
                                        outputs_path=rootdir / f'raw_data/{tag}/doc_embeddings_qampari_dev.npy',
                                        out_dataset_path='autoregressive_qampari_org_max5_dev_dataset_1b',
                                        max_length=5) 
        
        dev_data = read_jsonl('../../../projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl')
        dev_data = [dev_data[i] for i in data_indices]
        write_jsonl(dev_data, f'raw_data/{tag}/dev_data_gt_qampari_corpus_org_max5.jsonl')

    if generate_split == 'question_only':
        split='train'
        for model_name in ['inf', 'stella', 'cont']:
            for data_name in ['ambiguous_qe']:
            # for data_name in ['qampari', 'nq', 'msmarco']:
                rootdir = Path('../../autoregressive/data_creation/raw_data/') 
                data_indices = create_input_embeddings_for_query(batch_size=1, 
                                                model_name="meta-llama/Llama-3.2-1B-Instruct",
                                                input_data_path=rootdir / f'{data_name}_{split}_question_only.jsonl', 
                                                len_outputs_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_question_only_lens.npy', 
                                                outputs_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_question_only.npy',
                                                out_dataset_path=f'autoregressive_{data_name}_{model_name}_{split}_dataset_1b_qemb',
                                                max_length=1)
                print(len(data_indices))    
                
    if generate_split == 'contrastive':
        split='dev'
        base_model_name = 'meta-llama/Llama-3.2-1B-Instruct' # "Qwen/Qwen3-4B-Instruct-2507", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"
        
        pred_length_labels = False
        pred_length_labels_str = '_pred_length' if pred_length_labels else ''
        
        use_hard_negatives = False
        for split in ['train', 'dev']:
            for length in [5,6,7,8]:
            # for length in [2,3,4,5]:
                for model_name in ['inf']:
                    # for data_name in ['nq', 'msmarco']:
                    for data_name in ['qampari']:
                    # for data_name in ['ambiguous_qe']:
                        rootdir = Path('raw_data/')
                        if length == 1:
                            assert pred_length_labels == False, "pred_length_labels is not supported for length 1"
                            data_indices = create_input_embeddings_for_contrastive(batch_size=1, 
                                                            model_name=base_model_name,
                                                            input_data_path=rootdir / f'{data_name}_{split}_question_only.jsonl', 
                                                            positive_embeddings_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_question_only.npy', 
                                                            negative_embeddings_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_random_embeddings.npy' if not use_hard_negatives else rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_hard_negative_embeddings.npy',
                                                            out_dataset_path=f'autoregressive_{data_name}_{model_name}_{split}_dataset_1b_contrastive_query' if not use_hard_negatives else f'autoregressive_{data_name}_{model_name}_{split}_dataset_1b_contrastive_hard_negative_query',
                                                            pred_length_labels=pred_length_labels)
                        else:
                            data_indices = create_input_embeddings_for_contrastive(batch_size=1, 
                                                            model_name=base_model_name,
                                                            input_data_path=rootdir / f'{data_name}_{split}_question_only_{length}_ctxs.jsonl', 
                                                            positive_embeddings_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_positive_embeddings_{length}.npy', 
                                                            negative_embeddings_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_random_embeddings_{length}.npy' if not use_hard_negatives else rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_hard_negative_embeddings_{length}.npy',
                                                            out_dataset_path=f'autoregressive_{data_name}_{model_name}_{split}_dataset_1b_contrastive_{length}_ctxs{pred_length_labels_str}' if not use_hard_negatives else f'autoregressive_{data_name}_{model_name}_{split}_dataset_1b_contrastive_hard_negative_{length}_ctxs{pred_length_labels_str}',
                                                            pred_length_labels=pred_length_labels)
                        print(len(data_indices)) 
    
    if generate_split == 'gaussian_synthetic':
        import sys
        # model_name = sys.argv[1]  # 'inf', 'stella', 'cont'
        
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
        def create_synthetic_dataset(out_dataset_path, pairs, queries, corpus, LENGTH, hard_negatives=None, pred_length_labels=False, length_label=5, model_name='meta-llama/Llama-3.2-1B-Instruct'):
            if pred_length_labels:
                batch = {'inputs_embeds': [], 'attention_mask':[], 'positive_embeddings': [], 'negative_embeddings': [], 'length_labels_input_ids': [], 'length_labels_attention_mask': []}
            else:
                batch = {'inputs_embeds': [], 'attention_mask':[], 'positive_embeddings': [], 'negative_embeddings': []}
            use_hard_negatives = hard_negatives is not None
            if use_hard_negatives:
                assert len(hard_negatives) == len(pairs)
                assert len(hard_negatives[0]) == len(pairs[0]['ground_truth_indices'])
            if pred_length_labels:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(model_name)
                model.eval()
                model = model.cuda()
            for i in range(len(pairs)):
                query_vector = queries[pairs[i]['query_idx']]
                ground_truth_indices = pairs[i]['ground_truth_indices']
                if not use_hard_negatives:
                    # create random negative indices that are not in ground_truth_indices
                    random_indices = np.random.choice(len(corpus), size=len(ground_truth_indices), replace=False)
                    while np.any(np.isin(random_indices, ground_truth_indices)):
                        random_indices = np.random.choice(len(corpus), size=len(ground_truth_indices), replace=False)

                if pred_length_labels:
                    length_labels = tokenizer(str(length_label) + "<embed>", padding='max_length', truncation=True, max_length=257, return_tensors='pt')
                    # 5 -> 20,     27,  12529,     29,
                    # 2 -> 17,     27,  12529,     29,
                    # len(input_ids) = 4
                    batch['length_labels_input_ids'].append(length_labels['input_ids'][:, 1:5])
                    batch['length_labels_attention_mask'].append(length_labels['attention_mask'][:, 1:5])

                batch['inputs_embeds'].append(query_vector)
                batch['attention_mask'].append(np.zeros(LENGTH))
                batch['positive_embeddings'].append(corpus[ground_truth_indices])
                if use_hard_negatives:
                    batch['negative_embeddings'].append(corpus[hard_negatives[i]])
                else:
                    batch['negative_embeddings'].append(corpus[random_indices])
            
            if pred_length_labels:
                batch['length_labels_input_ids'] = torch.cat(batch['length_labels_input_ids'], dim=0).long()
                batch['length_labels_attention_mask'] = torch.cat(batch['length_labels_attention_mask'], dim=0)
            
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
                if pred_length_labels:
                    dataset_dicts[-1]['length_labels_input_ids'] = batch['length_labels_input_ids'][i].cpu().numpy()
                    dataset_dicts[-1]['length_labels_attention_mask'] = batch['length_labels_attention_mask'][i].cpu().numpy()
            safe_from_list_and_save(dataset_dicts, out_dataset_path, batch_size=2000)

            print('actual data size: ', len(dataset_dicts))
            
        
        def create_mse_dataset(out_dataset_path, mse_data, LENGTH, split):
            batch = {'inputs_embeds': [], 'attention_mask':[], 'labels': []}
            if split == 'test':
                mse_data = mse_data[:10000]
            else:
                mse_data = mse_data[10000:]
            for i in range(len(mse_data)):
                query_vector = mse_data[i]
                ground_truth_embeddings = query_vector
                batch['inputs_embeds'].append(query_vector)
                batch['attention_mask'].append(np.zeros(LENGTH))
                batch['labels'].append(ground_truth_embeddings)
            batch['inputs_embeds'] = torch.tensor(batch['inputs_embeds']).float().unsqueeze(1).expand(-1, LENGTH, -1)  # (bsz, LENGTH, d)
            batch['attention_mask'] = torch.tensor(batch['attention_mask']).long()             # (bsz, LENGTH). Only the first token is 1, the rest are 0.
            batch['attention_mask'][:, 0] = 1
            batch['labels'] = torch.tensor(batch['labels']).float()  # (bsz, k, d), LENGTH > k
            dataset_dicts = []
            for i in range(len(mse_data)):
                label = batch['labels'][i]  # (k, d)

                dataset_dicts.append({"inputs_embeds": batch['inputs_embeds'][i].cpu().numpy(), 
                                    "attention_mask": batch['attention_mask'][i].cpu().numpy(), 
                                    "labels": label.unsqueeze(0)})
                                
            safe_from_list_and_save(dataset_dicts, out_dataset_path, batch_size=2000)

            print('actual data size: ', len(dataset_dicts))
        
        split='test'  # ['train', 'test']
        LENGTH = 8
        normalize = False
        pred_length_labels = True
        length_label = 2
        hard_negatives = None
            
        for split in ['train', 'test']:
            data = load_synthetic_dataset(data_dir='./gaussian/data/new_mlps_rotation_large/', normalize=normalize)
            pairs = data['pairs_data'][split]
            
            pred_length_labels_str = '_pred_length' if pred_length_labels else ''
            normalized_str = '_normalized' if normalize else ''
            hard_negatives_str = '' if hard_negatives is None else '_hn'
            out_data_path = f'gaussian_new_mlps_rotation_{split}_dataset_1b_contrastive{normalized_str}{hard_negatives_str}{pred_length_labels_str}' 
            model_name = 'meta-llama/Llama-3.2-1B-Instruct'
            
            create_synthetic_dataset(out_dataset_path=out_data_path, 
                                    pairs=pairs, queries=data['queries'], corpus=data['corpus'], LENGTH=LENGTH, 
                                    hard_negatives=hard_negatives, pred_length_labels=pred_length_labels, length_label=length_label, model_name=model_name)
            
        