import json
from operator import ge
from transformers import Trainer, TrainingArguments, AutoModel, AutoTokenizer, AutoModelForCausalLM
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

# logger = logging.getLogger(__name__)

def _normalize(text):
    return unicodedata.normalize('NFD', text)

#Normalization and score functions from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
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
    # size = (bsz, len, dim)
    shifted = torch.cat([t[:,1:,:], t[:,0:1,:]], dim=1)
    shifted[:,-1,:] = padding
    return shifted
    


@torch.no_grad()
def create_input_embeddings_for_query(model_name = "meta-llama/Llama-3.2-1B-Instruct", 
                                      input_data_path='/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/dev_data.jsonl', 
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
    # model = LlamaModel.from_pretrained(model_name)
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
            
        # output = model(**batch, output_hidden_states=True, return_dict=True)
        # hidden_states = output['hidden_states'][0][0]  # (bsz, lengths, hidden_dim)

        # input_start_for_output = batch['attention_mask'].sum()
        
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
        
        # if max_length == -1:
        #     dataset_dicts.append({"hidden_states": hidden_states.cpu().numpy(), "labels": labels, "attention_mask": batch['attention_mask'][0].cpu().numpy(), "output_len": output_len.item()})
        # else:
        #     dataset_dicts.append({"hidden_states": hidden_states.cpu().numpy(), "labels": labels, "attention_mask": batch['attention_mask'][0].cpu().numpy(), "output_len": max_length})

        if max_length == -1:
            dataset_dicts.append({"input_ids": batch['input_ids'][0].cpu().numpy(), "labels": labels, "attention_mask": batch['attention_mask'][0].cpu().numpy(), "output_len": output_len.item()})
        else:
            dataset_dicts.append({"input_ids": batch['input_ids'][0].cpu().numpy(), "labels": labels, "attention_mask": batch['attention_mask'][0].cpu().numpy(), "output_len": max_length})
                        
                        
        
        output_start += output_len
        actual_data_size += 1
        actual_data_indices.append(i)
        i += 1

    # assert output_start == outputs.shape[0], (output_start, outputs.shape[0], i)

    # dataset = Dataset.from_list(dataset_dicts)
    # dataset.save_to_disk(out_dataset_path)
    
    safe_from_list_and_save(dataset_dicts, out_dataset_path, batch_size=2000)

    print('time elapsed: ', (time.time()-start_time)/60.0, 'min.')
    print('actual data size: ', actual_data_size)
    return actual_data_indices


@torch.no_grad()
def create_input_embeddings_for_contrastive(model_name = "meta-llama/Llama-3.2-1B-Instruct", 
                                      input_data_path='/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/dev_data.jsonl', 
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
    elif model_name == "meta-llama/Llama-3.2-1B-Instruct":
        print('Using meta-llama/Llama-3.2-1B-Instruct')
        instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    instruction = (f'{instruction_template}Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: [QUERY]\n{response_template}').strip('\n')
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(input_data_path))

    # # Define model and tokenizer
    # model = LlamaModel.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    seperator = tokenizer(response_template)[1:]
    
    # tokenize dataset => get the question embedding
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = model.to(device)
    # model.eval()
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
        # negative = negative_embeddings[i].reshape(1, -1)  # (bsz, k, d)
        negative = negative_embeddings[i]  # (bsz, k, d)
        # print('positive.shape, negative.shape', positive.shape, negative.shape)
        
        dataset_dicts.append({"input_ids": batch['input_ids'][0].cpu().numpy(), 
                              "attention_mask": batch['attention_mask'][0].cpu().numpy(), 
                              "positive_embeddings": positive, 
                              "negative_embeddings": negative})
                        
                        
        
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
    # rootdir = Path(__file__).parent.parent
    # meta-llama/Llama-3.1-8B-Instruct
    # meta-llama/Llama-3.2-1B-Instruct
    
    rootdir = Path(__file__).parent
    print(rootdir)
    
    generate_split = 'gaussian_synthetic'
    if generate_split in ['qampari_train', 'qampari_dev']:
        tag='qampari_org'
    elif generate_split in ['wsd_train', 'wsd_dev']:
        tag='wsd_distinct'
    elif generate_split in ['qampari_q_sm_500_train']:
        tag='qampari_q_sm_500'
    elif generate_split in ['question_only', 'corpus', 'contrastive', 'contrastive_sequence', 'contrastive_sequence_check_answer', 'gaussian_synthetic']:
        tag=''
    else:
        raise ValueError(f'Invalid generate_split: {generate_split}')
   
    if generate_split == 'wsd_train':
        create_input_embeddings_for_query(batch_size=1, 
                                        model_name="meta-llama/Llama-3.2-1B-Instruct",
                                        input_data_path=rootdir / f'../data/wsd/{tag}/train_large.jsonl', 
                                        len_outputs_path=rootdir / f'raw_data/{tag}/wsd_len_train_large_documents.npy', 
                                        outputs_path=rootdir / f'raw_data/{tag}/wsd_train_large_doc_embeds_inf.npy',
                                        out_dataset_path='autoregressive_wsd_train_dataset_1b')
    
    if generate_split == 'wsd_dev':
        create_input_embeddings_for_query(batch_size=1, 
                                        model_name="meta-llama/Llama-3.2-1B-Instruct",
                                        input_data_path=rootdir / f'raw_data/{tag}/wsd_dev_only_question.jsonl', 
                                        len_outputs_path=rootdir / f'raw_data/{tag}/wsd_len_dev_documents.npy', 
                                        outputs_path=rootdir / f'raw_data/{tag}/wsd_dev_doc_embeds_inf.npy',
                                        out_dataset_path='autoregressive_wsd_dev_dataset_1b',
                                        max_length=3)
    
    if generate_split == 'qampari_q_sm_500_train':
        create_input_embeddings_for_query(batch_size=1, 
                                        model_name="meta-llama/Llama-3.2-1B-Instruct",
                                        input_data_path=rootdir / f'/scratch/cluster/hungting/projects/Multi_Answer/mteb_retriever/outputs/inf/all_train_questions_q_sm.json', 
                                        len_outputs_path=rootdir / f'raw_data/{tag}/out_lens_qampari_q_sm_500.npy', 
                                        outputs_path=rootdir / f'raw_data/{tag}/out_centroids_qampari_q_sm_500.npy',
                                        out_dataset_path='autoregressive_qampari_q_sm_500_train_dataset_1b')
    
    if generate_split == 'qampari_train':
        data_indices = create_input_embeddings_for_query(batch_size=1, 
                                        model_name="meta-llama/Llama-3.2-1B-Instruct",
                                        input_data_path=rootdir / f'raw_data/{tag}/qampari_train_only_question.jsonl', 
                                        len_outputs_path=rootdir / f'raw_data/{tag}/doc_lens_qampari_train.npy', 
                                        outputs_path=rootdir / f'raw_data/{tag}/doc_embeddings_qampari_train.npy',
                                        out_dataset_path='autoregressive_qampari_org_max5_train_dataset_1b_qemb',
                                        max_length=1)
        print(len(data_indices))
        # train_data = read_jsonl('/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/train_data_gt_qampari_corpus.jsonl')
        # train_data = [train_data[i] for i in data_indices]
        # write_jsonl(train_data, f'raw_data/{tag}/train_data_gt_qampari_corpus_org_max5.jsonl')
        
        ## for the input_data, only the question is used. So no need to worry about other fields.
    
    if generate_split == 'qampari_dev':
        data_indices = create_input_embeddings_for_query(batch_size=1, 
                                        model_name="meta-llama/Llama-3.2-1B-Instruct",
                                        input_data_path=rootdir / f'raw_data/{tag}/qampari_dev_only_question.jsonl', 
                                        len_outputs_path=rootdir / f'raw_data/{tag}/doc_lens_qampari_dev.npy', 
                                        outputs_path=rootdir / f'raw_data/{tag}/doc_embeddings_qampari_dev.npy',
                                        out_dataset_path='autoregressive_qampari_org_max5_dev_dataset_1b',
                                        max_length=5) 
        
        dev_data = read_jsonl('/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl')
        dev_data = [dev_data[i] for i in data_indices]
        write_jsonl(dev_data, f'raw_data/{tag}/dev_data_gt_qampari_corpus_org_max5.jsonl')

    if generate_split == 'question_only':
        import sys
        # model_name = sys.argv[1]  # 'inf', 'stella', 'cont'
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
    
    if generate_split == 'corpus':
        # # id	text	title
        # nq_data = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/nq/corpus.jsonl')
        # msmarco_data = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/msmarco/corpus.jsonl')
        # nq_tsv_data = [[d['_id'], d['text'], d['title']] for d in (nq_data)]
        # msmarco_tsv_data = [[d['_id'], d['text'], d['title']] for d in (msmarco_data)]
        # write_tsv(nq_tsv_data, '/scratch/cluster/hungting/projects/autoregressive/data/nq/corpus.tsv')
        # write_tsv(msmarco_tsv_data, '/scratch/cluster/hungting/projects/autoregressive/data/msmarco/corpus.tsv')
        from beir.datasets.data_loader import GenericDataLoader
        from beir import util, LoggingHandler
        import logging
        import pathlib, os
        data_name = 'nq'

        #### Just some code to print debug information to stdout
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO,
                            handlers=[LoggingHandler()])
        #### /print debug information to stdout

        #### Download scifact.zip dataset and unzip the dataset
        # url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{data_name}.zip"
        # out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        # data_path = util.download_and_unzip(url, out_dir)
        data_path = f'/scratch/cluster/hungting/projects/autoregressive/data/{data_name}'
        split = 'train'
        #### Provide the data_path where scifact has been downloaded and unzipped
        actual_split = 'test' if split == 'dev' else split
        if data_name == 'nq' and actual_split == 'train':
            data_path = f'/scratch/cluster/hungting/projects/autoregressive/data/nq/nq-train'
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=actual_split) 
        print(len(corpus), len(queries), len(qrels))
        output_data = []
        for k, v in queries.items():
            output_data.append({'question': v, 'id': k, 'input': v, 'answers': [''], 'ctxs': []})
                 
        write_jsonl(output_data, f'/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/{data_name}_{split}_question_only.jsonl')
        if split == 'dev':
            write_json(output_data, f'/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/{data_name}_{split}_question_only.json')
        # nq_data = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/nq_dev_question_only.jsonl')
        # msmarco_data = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/msmarco_dev_question_only.jsonl')
        # new_nq_data = []
        # new_msmarco_data = []
        # for i, d in enumerate(nq_data):
        #     new_nq_data.append({'question': d['question'], 'id': str(i), 'input': d['question'], 'answers': [''], 'ctxs': []})
        # for i, d in enumerate(msmarco_data):
        #     new_msmarco_data.append({'question': d['question'], 'id': str(i), 'input': d['question'], 'answers': [''], 'ctxs': []})
        # write_json(new_nq_data, '/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/nq_dev_question_only_new.json')
        # write_json(new_msmarco_data, '/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/msmarco_dev_question_only_new.json')
        
        # /scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/nq_dev_question_only.jsonl 
        # /scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/msmarco_dev_question_only.jsonl 
    if generate_split == 'contrastive':
        import sys
        # model_name = sys.argv[1]  # 'inf', 'stella', 'cont'
        split='dev'
        length = 5  # [5,6,7,8] for qampari, 1 for the other ones.
        # model_name = 'meta-llama/Llama-3.2-1B-Instruct'
        base_model_name = 'infly/inf-retriever-v1-1.5b'
        
        use_hard_negatives = False
        for split in ['train', 'dev']:
            # for length in [5,6,7,8]:
            for length in [2,3,4,5]:
                # for model_name in ['inf']:
                for model_name in ['cont', 'stella', 'inf']:
                    # for data_name in ['nq', 'msmarco']:
                    # for data_name in ['qampari']:
                    for data_name in ['ambiguous_qe']:
                        rootdir = Path('raw_data/')
                        if length == 1:
                            data_indices = create_input_embeddings_for_contrastive(batch_size=1, 
                                                            model_name=base_model_name,
                                                            input_data_path=rootdir / f'{data_name}_{split}_question_only.jsonl', 
                                                            positive_embeddings_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_question_only.npy', 
                                                            negative_embeddings_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_random_embeddings.npy' if not use_hard_negatives else rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_hard_negative_embeddings.npy',
                                                            out_dataset_path=f'autoregressive_{data_name}_{model_name}_{split}_dataset_1b_contrastive_query' if not use_hard_negatives else f'autoregressive_{data_name}_{model_name}_{split}_dataset_1b_contrastive_hard_negative_query')
                        else:
                            data_indices = create_input_embeddings_for_contrastive(batch_size=1, 
                                                            model_name=base_model_name,
                                                            input_data_path=rootdir / f'{data_name}_{split}_question_only_{length}_ctxs.jsonl', 
                                                            positive_embeddings_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_positive_embeddings_{length}.npy', 
                                                            negative_embeddings_path=rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_random_embeddings_{length}.npy' if not use_hard_negatives else rootdir / f'{data_name}_{model_name}' / f'{data_name}_{split}_hard_negative_embeddings_{length}.npy',
                                                            out_dataset_path=f'autoregressive_{data_name}_{model_name}_{split}_dataset_1b_contrastive_{length}_ctxs' if not use_hard_negatives else f'autoregressive_{data_name}_{model_name}_{split}_dataset_1b_contrastive_hard_negative_{length}_ctxs')
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
                # # 'transformation_matrices': transformation_matrices,
                'pairs_data': pairs_data
            }
            
        @torch.no_grad()
        def create_synthetic_dataset(out_dataset_path, pairs, queries, corpus, LENGTH, hard_negatives=None, pred_length_labels=False, model_name='meta-llama/Llama-3.2-1B-Instruct'):
            instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
            response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
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
                    length_labels = tokenizer(str(5) + "<embed>", padding='max_length', truncation=True, max_length=257, return_tensors='pt')
                    # print('length_labels', length_labels)
                    # length_labels = {k: v.cuda() for k, v in length_labels.items()}
                    # outputs = model(**length_labels, output_hidden_states=True)
                    # hidden_states = outputs.hidden_states[0].clone().detach()
                    # print('hidden_states', hidden_states.shape)
                    # input_embeds = hidden_states[:, 1:5, :]
                    # print('input_embeds', input_embeds.shape) # (1, 4, 2048)
                    # print('query_vector', query_vector.shape) # (1024, )
                    # length_labels['length_labels_input_ids'] = length_labels['input_ids']
                    # length_labels['length_labels_attention_mask'] = length_labels['attention_mask']
                    # outputs = model(**length_labels)
                    # print('outputs', outputs)
                    # print('outputs.logits', outputs.logits.shape)
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
                # print('positive.shape, negative.shape', positive.shape, negative.shape)
                # print(batch['inputs_embeds'][0].cpu().numpy().shape, batch['attention_mask'][0].cpu().numpy().shape)

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
                # print('query_vector.shape', query_vector)
                ground_truth_embeddings = query_vector
                batch['inputs_embeds'].append(query_vector)
                batch['attention_mask'].append(np.zeros(LENGTH))
                batch['labels'].append(ground_truth_embeddings)
            # print('inputs_embeds', torch.tensor(batch['inputs_embeds'])[0])
            batch['inputs_embeds'] = torch.tensor(batch['inputs_embeds']).float().unsqueeze(1).expand(-1, LENGTH, -1)  # (bsz, LENGTH, d)
            # print('inputs_embeds', batch['inputs_embeds'][0])
            batch['attention_mask'] = torch.tensor(batch['attention_mask']).long()             # (bsz, LENGTH). Only the first token is 1, the rest are 0.
            batch['attention_mask'][:, 0] = 1
            # print('attention_mask', batch['attention_mask'][0])
            batch['labels'] = torch.tensor(batch['labels']).float()  # (bsz, k, d), LENGTH > k
            # print('labels', batch['labels'][0])
            dataset_dicts = []
            for i in range(len(mse_data)):
                label = batch['labels'][i]  # (k, d)
                # print('label.shape', label.unsqueeze(0).shape, 'inputembed', batch['inputs_embeds'][i])
                # print(batch['inputs_embeds'][0].cpu().numpy().shape, batch['attention_mask'][0].cpu().numpy().shape)

                dataset_dicts.append({"inputs_embeds": batch['inputs_embeds'][i].cpu().numpy(), 
                                    "attention_mask": batch['attention_mask'][i].cpu().numpy(), 
                                    "labels": label.unsqueeze(0)})
                                
            safe_from_list_and_save(dataset_dicts, out_dataset_path, batch_size=2000)

            print('actual data size: ', len(dataset_dicts))
        
        split='test'  # ['train', 'test']
        
        LENGTH = 8
        normalize = False
        pred_length_labels = True
        # hard_negatives = np.load('gaussian/data/opposing_pairs_data/contrastive_all_labels_ordered_hard_negatives.npy')
        hard_negatives = None
            
        for split in ['train', 'test']:
            data = load_synthetic_dataset(data_dir='./gaussian/data/new_mlps_rotation_large_2/', normalize=normalize)
            pairs = data['pairs_data'][split]
            
            pred_length_labels_str = '_pred_length' if pred_length_labels else ''
            normalized_str = '_normalized' if normalize else ''
            hard_negatives_str = '' if hard_negatives is None else '_hn'
            out_data_path = f'gaussian_new_mlps_rotation_2_{split}_dataset_1b_contrastive{normalized_str}{hard_negatives_str}{pred_length_labels_str}' 
            
            model_name = 'meta-llama/Llama-3.2-1B-Instruct'
            
            
            create_synthetic_dataset(out_dataset_path=out_data_path, 
                                    pairs=pairs, queries=data['queries'], corpus=data['corpus'], LENGTH=LENGTH, 
                                    hard_negatives=hard_negatives, pred_length_labels=pred_length_labels, model_name=model_name)
            
            # mse_data = np.load('gaussian/data/opposing_pairs_data/mse_labels.npy')
            # create_mse_dataset(out_dataset_path=f'gaussian_synthetic_{split}_dataset_1b_mse', 
            #                    mse_data=mse_data, LENGTH=LENGTH, split=split)
            
    if generate_split == 'contrastive_sequence':
        
        
        # rootdir = Path('../../autoregressive/data/ambiguous/nqopen')
        # # for split in ['train', 'dev', 'test']:
        # for split in ['dev', 'test']:
        #     data = read_json(rootdir / f'nqopen-{split}.json')
        #     new_data = []
        #     for inst in data:
        #         if len(inst['answer']) > 1:
        #             new_data.append(inst)
        #             new_data[-1]['answers'] = inst['answer']
        #             new_data[-1]['ctxs'] = []
        #             new_data[-1]['input'] = inst['question']
        #     write_json(new_data, rootdir / f'nqopen-{split}_multi_answer.json')
        #     print(len(new_data), len(data))
            
        
        rootdir = Path('../../autoregressive/data/ambiguous/ambignq')
        for split in ['dev', 'train']:
            data = read_json(rootdir / f'{split}_with_evidence_articles.json')
            new_data = []
            for inst in tqdm(data):
                multi_answer = False
                current_answers = []
                for ans in inst['annotations']:
                    if ans['type'] != 'singleAnswer':
                        multi_answer = True
                        qa_pairs = ans['qaPairs']
                        for pair in qa_pairs:
                            current_answers.append(pair['answer'])
                        break
                if not multi_answer:
                    continue
                
                raw_corpus = '\n'.join(inst['articles_plain_text'])
                documents = chunk_text(raw_corpus, chunk_size=100)
                documents = [{'title': '', 'text': doc} for doc in documents]
                
                new_data.append({
                    'question': inst['question'],
                    'id': inst['id'],
                    'answers': current_answers,
                    'ctxs': documents,
                    'input': inst['question']
                })
            
            # dict_keys(['question', 'id', 'annotations', 'articles_plain_text', 'articles_html_text'])
            for inst in tqdm(new_data):
                check_answer(inst)
                
            no_evidence_cnt, evidence_cnt, evidence_data, no_evidence_data = check_evidences(new_data)
            print(f'no evidence cnt: {no_evidence_cnt}')
            print(f'evidence cnt: {evidence_cnt}')
            write_json(evidence_data, rootdir / f'ambignq-{split}_multi_answer_evidence.json')
            write_json(no_evidence_data, rootdir / f'ambignq-{split}_multi_answer_no_evidence.json')
        
    if generate_split == 'contrastive_sequence_check_answer':
        rootdir = Path('/datastor1/hungting/retrieval_outputs/mteb_retriever/')
        for split in ['dev']:
            inf_data = read_jsonl(rootdir / f'inf/ambignq+nqopen_multi_answer_all_no_evidence.json')
            stella_data = read_jsonl(rootdir / f'stella-400M/ambignq+nqopen_multi_answer_all_no_evidence.json')
            contriever_data = read_jsonl('/scratch/cluster/hungting/projects/Multi_Answer/contriever/outputs/contriever_msmarco_nq/ambignq+nqopen-all_multi_answer_evidence_dev.json')
            raw_data = read_json('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/ambignq+nqopen_multi_answer_all_no_evidence.json')
            data = raw_data
            assert len(data) == len(stella_data)
            assert len(data) == len(inf_data)
            assert len(data) == len(contriever_data)
            
            
            for inst in data:
                assert len(inst['answers']) > 0, inst['question']
            
            print('combining stella and inf')
            for i in range(len(data)):
                if data[i]['question'] != stella_data[i]['question']:
                    print(data[i]['question'], stella_data[i]['question'])
                    break
                if data[i]['question'] != inf_data[i]['question']:
                    print(data[i]['question'], inf_data[i]['question'])
                    break
                if data[i]['question'] != contriever_data[i]['question']:
                    print(data[i]['question'], contriever_data[i]['question'])
                    break
                
                data[i]['ctxs'] = []
                data[i]['ctxs'] += stella_data[i]['ctxs']
                data[i]['ctxs'] += inf_data[i]['ctxs']
                data[i]['ctxs'] += contriever_data[i]['ctxs']
                assert len(data[i]['answers']) > 0, data[i]['question']
            
            print('finished combining stella and inf, checking answer')
            for inst in tqdm(data):
                assert len(inst['answers']) > 0, inst['question']
                if isinstance(inst['answers'][0], str):
                    inst['answers'] = [[l] for l in inst['answers']]
                assert len(inst['answers']) > 0, inst['question']
            for inst in tqdm(data):
                check_answer(inst)
    
            no_evidence_cnt, evidence_cnt, evidence_data, no_evidence_data = check_evidences(data)
            print(f'no evidence cnt: {no_evidence_cnt}')
            print(f'evidence cnt: {evidence_cnt}')
            savedir = Path('../../autoregressive/data/ambiguous/nqopen')
            write_json(evidence_data, savedir / f'ambignq+nqopen_multi_answer_all_no_evidence_evidence.json')
            write_json(no_evidence_data, savedir / f'ambignq+nqopen_multi_answer_all_no_evidence_no_evidence.json')
            
        # rootdir = Path('/datastor1/hungting/retrieval_outputs/mteb_retriever/')
        # for split in ['dev', 'train']:
        #     inf_data = read_jsonl(rootdir / f'inf/ambignq-{split}_multi_answer_no_evidence.json')
        #     stella_data = read_jsonl(rootdir / f'stella-400M/ambignq-{split}_multi_answer_no_evidence.json')
        #     data = inf_data
        #     assert len(data) == len(stella_data)
            
        #     print('combining stella and inf')
        #     for i in range(len(data)):
        #         if data[i]['question'] != stella_data[i]['question']:
        #             print(data[i]['question'], stella_data[i]['question'])
        #             break
        #         data[i]['ctxs'] += stella_data[i]['ctxs']
            
        #     print('finished combining stella and inf, checking answer')
        #     # for inst in tqdm(data):
        #     #     inst['answers'] = [[l] for l in inst['answers']]
        #     for inst in tqdm(data):
        #         check_answer(inst)
    
        #     no_evidence_cnt, evidence_cnt, evidence_data, no_evidence_data = check_evidences(data)
        #     print(f'no evidence cnt: {no_evidence_cnt}')
        #     print(f'evidence cnt: {evidence_cnt}')
        #     savedir = Path('../../autoregressive/data/ambiguous/ambignq')
        #     write_json(evidence_data, savedir / f'ambignq-{split}_multi_answer_no_evidence_evidence.json')
        #     write_json(no_evidence_data, savedir / f'ambignq-{split}_multi_answer_no_evidence_no_evidence.json')
  
        
        # collect all the data for contrastive sequence 
        # nqopen-dev / test (_evidence)
        # ambignq-dev / train (_evidence)
        # ambignq-dev / train (_no_evidence_evidence)
        
        
        # all_data = []
        # rootdir = Path('../../autoregressive/data/ambiguous/nqopen')
        # for split in ['dev', 'test']:
        #     data = read_json(rootdir / f'nqopen-{split}_multi_answer_evidence.json')
        #     no_evidence_cnt, evidence_cnt, _, _ = check_evidences(data)
        #     assert no_evidence_cnt == 0
        #     all_data.extend(data)
            
        # rootdir = Path('../../autoregressive/data/ambiguous/ambignq')
        # for split in ['dev', 'train']:
        #     data = read_json(rootdir / f'ambignq-{split}_multi_answer_evidence.json')
        #     no_evidence_cnt, evidence_cnt, _, _ = check_evidences(data)
        #     assert no_evidence_cnt == 0
        #     all_data.extend(data)
        
        # rootdir = Path('../../autoregressive/data/ambiguous/ambignq')
        # for split in ['dev', 'train']:
        #     data = read_json(rootdir / f'ambignq-{split}_multi_answer_no_evidence_evidence.json')
        #     no_evidence_cnt, evidence_cnt, _, _ = check_evidences(data)
        #     assert no_evidence_cnt == 0
        #     all_data.extend(data)
            
        # for inst in all_data:
        #     del inst['ctxs']
        
        # rootdir = Path('../../autoregressive/data/ambiguous')
        # write_jsonl(all_data, rootdir / f'ambignq+nqopen-all_multi_answer_evidence.jsonl')
        # write_json(all_data, rootdir / f'ambignq+nqopen-all_multi_answer_evidence.json')
        
        # all_data_2 = [l for l in all_data if len(l['positive_ctxs']) == 2]
        
        # data_name = 'ambiguous'
        # split = 'train'
        # for inst in all_data:
        #     keys = list(inst.keys())
        #     for k in keys:
        #         if k != 'question' and k != 'question_text':
        #             del inst[k]
                    
        # for inst in all_data_2:
        #     keys = list(inst.keys())
        #     for k in keys:
        #         if k != 'question' and k != 'question_text':
        #             del inst[k]
        
        
        # write_jsonl(all_data, rootdir / f'{data_name}_{split}_question_only.jsonl')
        # write_jsonl(all_data_2, rootdir / f'{data_name}_{split}_question_only_2_ctxs.jsonl')
                    
    
        
            

