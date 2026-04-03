# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import random
import json
import sys
import numpy as np
from src import normalize_text

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        add_input_negatives=False,
        training=False,
        training_mode='standard',
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
    ):
        self.training_mode = training_mode
        self.negative_ctxs = negative_ctxs
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.add_input_negatives = add_input_negatives
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)
        if not self.training:
        # if True:
            import random
            random.shuffle(self.data)
            self.data = self.data[:3000]
            print("Number of eval examples: ", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        if self.training_mode == 'standard':
            question = example["question"]
        elif self.training_mode in ['base', 'subtraction', 'gru', 'concat', 'subtraction_linear', 'linear_projection', 'sentence_transformer']:
            question = example["org_q"]
        elif self.training_mode == 'standard_org_q':
            question = example["question_text"]
        
        task = 'Given a query, retrieve relevant passages that answer the query'
        question = get_detailed_instruct(task, question)

        ## Key "hard_negatives_ctxs" is always there
        ## Key "input_negative_ctxs" is there if there are input negatives
        if self.training_mode in ['subtraction', 'gru', 'concat', 'subtraction_linear', 'linear_projection', 'sentence_transformer']:
            assert 'input_negative_ctxs' in example, (example, example.keys())
            
        if self.training:
            ### Training ###
            positive_string = 'positive_ctxs' if (self.training_mode != 'standard_org_q') else 'ground_truths'
            # if len(example[positive_string]) > self.max_positive_documents:
            #     gold = random.sample(example[positive_string], self.max_positive_documents)
            # else:
            #     gold = example[positive_string]
            gold = random.choice(example[positive_string])
                        

            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            
            # add input negatives to hard negatives
            if self.add_input_negatives:
                assert 'input_negative_ctxs' in example
                example['hard_negative_ctxs'] += example['input_negative_ctxs']
            
            negatives = []
            if n_random_negatives > 0:
                random_negatives = random.sample(example["negative_ctxs"], n_random_negatives)
                negatives += random_negatives
            if n_hard_negatives > 0:
                hard_negatives = random.sample(
                    example["hard_negative_ctxs"][self.negative_hard_min_idx :], n_hard_negatives
                )
                negatives += hard_negatives
        else:
            ### Evaluation ###
            gold = example["positive_ctxs"][0]
            nidx = 0
            if "negative_ctxs" in example:
                
                # add input negatives to hard negatives
                if self.add_input_negatives:
                    assert 'input_negative_ctxs' in example
                    example['hard_negative_ctxs'] += example['input_negative_ctxs']
                                
                if example['negative_ctxs']:
                    if 'hard_negative_ctxs' in example:  # only add in hard negatives if they exist
                        negatives = [example["negative_ctxs"][nidx]] + example['hard_negative_ctxs']
                    else:
                        negatives = example["negative_ctxs"]
                else:
                    negatives = example['hard_negative_ctxs']
            else:
                negatives = []
        
        # gold = [
        #     g["title"] + " " + g["text"] if ("title" in g and len(g["title"]) > 0) else g["text"] for g in gold
        # ]
        gold = gold["title"] + " " + gold["text"] if ("title" in gold and len(gold["title"]) > 0) else gold["text"]
        negatives = [
            n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives
        ]
        
        if self.training_mode in ['subtraction', 'gru', 'concat', 'subtraction_linear', 'linear_projection', 'sentence_transformer']:
            input_negatives = [n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in example['input_negative_ctxs']]
                
        example = {
            "query": self.normalize_fn(question),
            # "gold": [self.normalize_fn(g) for g in gold],
            "gold": self.normalize_fn(gold),
            "negatives": [self.normalize_fn(n) for n in negatives]
        }
        if self.training_mode in ['subtraction', 'gru', 'concat', 'subtraction_linear', 'linear_projection', 'sentence_transformer']:
            example['input_negatives'] = [self.normalize_fn(n) for n in input_negatives]
            
        return example

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

    def sample_n_hard_negatives(self, ex):

        if "hard_negative_ctxs" in ex:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0
        n_random_negatives = self.negative_ctxs - n_hard_negatives
        if "negative_ctxs" in ex:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        return n_hard_negatives, n_random_negatives

        
class SampleDataset(torch.utils.data.Dataset):
    """
        This dataset is used for training the model with different document lengths.
        The document lengths are specified in the doc_lengths argument.
        The default document lengths are [0,3,6].
        The first document length is 0, which means no document is used.
        The second document length is 3, which means 3 documents are used.
        
        Just use the train_gt_data.jsonl file for training.
    """
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        add_input_negatives=False,
        training=False,
        training_mode='standard',
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
        tokenizer=None,
        doc_lengths=[0,3,6]
    ):
        self.training_mode = training_mode
        self.negative_ctxs = negative_ctxs
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.add_input_negatives = add_input_negatives
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self.tokenizer = tokenizer
        self._load_data(datapaths, global_rank, world_size, maxload)
        self.doc_lengths = [int(x) for x in doc_lengths]
        if not self.training:
        # if True:
            import random
            random.shuffle(self.data)
            self.data = self.data[:3000]
            print("Number of eval examples: ", len(self.data))
        print(f"doc_lengths: {self.doc_lengths}", flush=True)

    def __len__(self):
        return len(self.data)

    def concat_docs(self, question, documents):
        rewritten_question = 'Question: [Question]\n\nDocuments: [Documents]'.replace('[Question]', question).replace('[Documents]', '\n'.join([ddd['title'] + ' ' + ddd['text'] for ddd in documents]))  
        # check the rewritten question does not exceed 512 tokens
        #print(len(self.tokenizer(rewritten_question)['input_ids']), flush=True)

        if len(self.tokenizer(rewritten_question)['input_ids']) > 8192:
            toks = 0
            print('=-===')
          #  print('Exceed 512 tokens', len(self.tokenizer(rewritten_question)['input_ids']))
            while len(self.tokenizer(rewritten_question)['input_ids']) > 8192:
                toks += 20
                rewritten_question = 'Question: [Question]\n\nDocuments: [Documents]'.replace('[Question]', question).replace('[Documents]', '\n'.join([ddd['title'] + ' ' + ddd['text'][:-toks] for ddd in documents]))  
           #     print('reduced by 60 chars', len(self.tokenizer(rewritten_question)['input_ids']))
        return rewritten_question
    
    
    def __getitem__(self, index):
        example = self.data[index]
        question = example["question_text"]
        task = 'Given a query, retrieve relevant passages that answer the query'
        question = get_detailed_instruct(task, question)
        
        ground_truths = []
        gt_ids = set()
        for gt in example['ground_truths']:
            if gt['id'] not in gt_ids:
                gt_ids.add(gt['id'])
                ground_truths.append(gt)
        
        _standard_mode = False
        if self.training_mode == 'standard':
            _standard_mode = True
        else:
            assert self.training_mode in ['subtraction', 'gru', 'concat', 'subtraction_linear', 'linear_projection', 'sentence_transformer', 'standard_org_q']
        
        if self.training_mode != "standard_org_q":
            # Check for force_base_retrieval flag
            if example.get('force_base_retrieval', False):
                num_input_documents = 0  # Force 0 docs
            else:
                # Original sampling logic
                valid_doc_lengths = [length for length in self.doc_lengths if length < len(ground_truths)]
                if not valid_doc_lengths:
                    num_input_documents = 0
                else:
                    num_input_documents = random.choice(valid_doc_lengths)
            
            if num_input_documents > 0:
                input_documents = random.sample(ground_truths, num_input_documents)
                if _standard_mode:
                    question = self.concat_docs(question, input_documents)
            else:
                input_documents = []
        else:
            input_documents = []
        
        # Target pool is merged docs, excluding inputs
        positive_ctxs = []
        input_doc_ids = set(doc['id'] for doc in input_documents)  # Track by ID
        for doc in ground_truths:  # Use merged pool
            if doc['id'] not in input_doc_ids:  # Exclude by ID
                positive_ctxs.append(doc)
        
        assert len(positive_ctxs) > 0, (len(ground_truths), len(input_documents), [c['id'] for c in ground_truths], [c['id'] for c in input_documents])
        
        # ... rest of the code stays the same
        
    # ... rest of the code stays the same
            
        if self.training:
            ### Training ###
            gold = random.choice(positive_ctxs)
            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            
            # add input negatives to hard negatives
            if self.add_input_negatives:
                example['hard_negative_ctxs'] += input_documents
            
            negatives = []
            if n_random_negatives > 0:
                random_negatives = random.sample(example["negative_ctxs"], n_random_negatives)
                negatives += random_negatives
            if n_hard_negatives > 0:
                hard_negatives = random.sample(
                    example["hard_negative_ctxs"][self.negative_hard_min_idx :], n_hard_negatives
                )
                negatives += hard_negatives
            
        else:
            ### Evaluation ###
            gold = positive_ctxs[0]
            nidx = 0
            if "negative_ctxs" in example:
                # add input negatives to hard negatives
                if self.add_input_negatives:
                    example['hard_negative_ctxs'] += input_documents
                                
                if example['negative_ctxs']:
                    if 'hard_negative_ctxs' in example:  # only add in hard negatives if they exist
                        negatives = [example["negative_ctxs"][nidx]] + example['hard_negative_ctxs']
                    else:
                        negatives = example["negative_ctxs"]
                else:
                    negatives = example['hard_negative_ctxs']
            else:
                negatives = []
        

        gold = gold["title"] + " " + gold["text"] if ("title" in gold and len(gold["title"]) > 0) else gold["text"]
        negatives = [
            n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives
        ]
        
        if self.training_mode in ['subtraction', 'gru', 'concat', 'subtraction_linear', 'linear_projection', 'sentence_transformer']:
            input_negatives = [n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in input_documents]
                
        example = {
            "query": self.normalize_fn(question),
            "gold": self.normalize_fn(gold),
            "negatives": [self.normalize_fn(n) for n in negatives]
        }
        if self.training_mode in ['subtraction', 'gru', 'concat', 'subtraction_linear', 'linear_projection', 'sentence_transformer']:
            example['input_negatives'] = [self.normalize_fn(n) for n in input_negatives]
            
        return example

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl_skip_malformed(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

    def _load_data_jsonl_skip_malformed(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        error_count = 0
        with open(path, "r") as fin:
            for line_num, line in enumerate(fin, 1):
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"Warning: Skipping malformed JSON at line {line_num} in {path}")
                    if error_count <= 5:  # Print first 5 problematic lines for debugging
                        print(f"Problematic line preview (first 200 chars): {line[:200]}...")
                    continue
            
                if maxload is not None and maxload > 0 and counter == maxload:
                    break
    
        if error_count > 0:
            print(f"Total malformed lines skipped: {error_count}", flush=True)
    
        return examples, counter

    def sample_n_hard_negatives(self, ex):

        if "hard_negative_ctxs" in ex:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0
        n_random_negatives = self.negative_ctxs - n_hard_negatives
        if "negative_ctxs" in ex:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        return n_hard_negatives, n_random_negatives




class SampleDataseOld(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        add_input_negatives=False,
        training=False,
        training_mode='standard',
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
        tokenizer=None,
        max_positive_documents=1,
    ):
        self.training_mode = training_mode
        self.negative_ctxs = negative_ctxs
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.add_input_negatives = add_input_negatives
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)
        self.tokenizer = tokenizer
        self.max_positive_documents = max_positive_documents
        if not self.training:
        # if True:
            import random
            random.shuffle(self.data)
            self.data = self.data[:3000]
            print("Number of eval examples: ", len(self.data))

    def __len__(self):
        return len(self.data)

    def concat_docs(self, question, documents):
        rewritten_question = 'Question: [Question]\n\nDocuments: [Documents]'.replace('[Question]', question).replace('[Documents]', '\n'.join([ddd['title'] + ' ' + ddd['text'] for ddd in documents]))  
        # check the rewritten question does not exceed 512 tokens
        if len(self.tokenizer(rewritten_question)['input_ids']) > 512:
            toks = 0
            print('=-===')
            print('Exceed 512 tokens', len(self.tokenizer(rewritten_question)['input_ids']))
            while len(self.tokenizer(rewritten_question)['input_ids']) > 512:
                toks += 20
                rewritten_question = 'Question: [Question]\n\nDocuments: [Documents]'.replace('[Question]', question).replace('[Documents]', '\n'.join([ddd['title'] + ' ' + ddd['text'][:-toks] for ddd in documents]))  
                print('reduced by 60 chars', len(self.tokenizer(rewritten_question)['input_ids']))
        return rewritten_question

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question_text"]
        ground_truths = []
        gt_ids = set()
        for gt in example['ground_truths']:
            if gt['id'] not in gt_ids:
                gt_ids.add(gt['id'])
                ground_truths.append(gt)
        
        _standard_mode = False
        if self.training_mode == 'standard':
            _standard_mode = True
        else:
            assert self.training_mode in ['subtraction', 'gru', 'concat', 'subtraction_linear', 'linear_projection', 'sentence_transformer']
        
        # sample input documents
        max_input = min(len(ground_truths)-1, 3) if _standard_mode else len(ground_truths)-1
        num_input_documents = random.randint(0, max_input)
        input_documents = random.sample(ground_truths, num_input_documents)
        if _standard_mode:
            question = self.concat_docs(question, input_documents)

        # get the positive documents
        positive_ctxs = []
        for doc in ground_truths:
            if doc not in input_documents:
                positive_ctxs.append(doc)
        assert len(positive_ctxs) > 0, (len(ground_truths), len(input_documents), [c['id'] for c in ground_truths], [c['id'] for c in input_documents], ground_truths)
                        
        if self.training:
            ### Training ###
            if len(positive_ctxs) > self.max_positive_documents:
                gold = random.sample(positive_ctxs, self.max_positive_documents)
            else:
                gold = positive_ctxs

            # handle negative ctxs
            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            
            # add input negatives to hard negatives
            if self.add_input_negatives:
                example['hard_negative_ctxs'] += input_documents
            
            negatives = []
            if n_random_negatives > 0:
                random_negatives = random.sample(example["negative_ctxs"], n_random_negatives)
                negatives += random_negatives
            if n_hard_negatives > 0:
                hard_negatives = random.sample(
                    example["hard_negative_ctxs"][self.negative_hard_min_idx :], n_hard_negatives
                )
                negatives += hard_negatives
        else:
            ### Evaluation ###
            gold = positive_ctxs
            nidx = 0
            
            # handle negative ctxs
            if "negative_ctxs" in example:
                # add input negatives to hard negatives
                if self.add_input_negatives:
                    assert 'input_negative_ctxs' in example
                    example['hard_negative_ctxs'] += input_documents
                                
                if example['negative_ctxs']:
                    if 'hard_negative_ctxs' in example:  # only add in hard negatives if they exist
                        negatives = [example["negative_ctxs"][nidx]] + example['hard_negative_ctxs']
                    else:
                        negatives = example["negative_ctxs"]
                else:
                    negatives = example['hard_negative_ctxs']
            else:
                negatives = []


        gold = [
            g["title"] + " " + g["text"] if ("title" in g and len(g["title"]) > 0) else g["text"] for g in gold
        ]
        negatives = [
            n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives
        ]
        
        if not _standard_mode:
            input_negatives = [n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in input_documents]
                
        example = {
            "query": self.normalize_fn(question),
            "gold": [self.normalize_fn(g) for g in gold],
            "negatives": [self.normalize_fn(n) for n in negatives]
        }
        if not _standard_mode:
            example['input_negatives'] = [self.normalize_fn(n) for n in input_negatives]
            
        return example

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

    def sample_n_hard_negatives(self, ex):

        if "hard_negative_ctxs" in ex:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0
        n_random_negatives = self.negative_ctxs - n_hard_negatives
        if "negative_ctxs" in ex:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        return n_hard_negatives, n_random_negatives


class Collator(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        labels = []
        # golds = []
        # for ex in batch:
        #     labels.append([(i + len(golds)) for i in range(len(ex["gold"]))])
        #     golds += (ex["gold"])
        golds = [ex["gold"] for ex in batch]
        negs = [item for ex in batch for item in ex["negatives"]]
        allpassages = golds + negs

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        # k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].to(torch.long)
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].to(torch.long)

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        n_tokens, n_mask = k_tokens[len(golds) :], k_mask[len(golds) :]

        
        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
        }
        batch['labels'] = None        
        if len(queries) == len(golds):  # if the same number of golds and queries, then only one positive each query (don't need labels)
            #print('same query lbael')
            return batch
        else:
            print('not same', len(queries), len(golds))
        
        batch['labels'] = labels
        return batch




class SubtractionCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):        
        queries = [ex["query"] for ex in batch]
        labels = []
        # golds = []
        # for ex in batch:
        #     labels.append([(i + len(golds)) for i in range(len(ex["gold"]))])
        #     golds += (ex["gold"])
        golds = [ex["gold"] for ex in batch]
        negs = [item for ex in batch for item in ex["negatives"]]
        
        input_negs = [item for ex in batch for item in ex["input_negatives"]]
        len_input_negs = [len(ex["input_negatives"]) for ex in batch]
        allpassages = golds + negs
        

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        if len(input_negs) == 0:
            inpnout = None
            inpn_tokens, inpn_mask = None, None
        else:
            inpnout = self.tokenizer.batch_encode_plus(
                input_negs,
                max_length=self.passage_maxlength,
                truncation=True,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        #     inpn_tokens, inpn_mask = inpnout["input_ids"], inpnout["attention_mask"].bool()
            
        # q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        # k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()
            inpn_tokens, inpn_mask = inpnout["input_ids"], inpnout["attention_mask"].to(torch.long)
            
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].to(torch.long)
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].to(torch.long)
        

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        n_tokens, n_mask = k_tokens[len(golds) :], k_mask[len(golds) :]
        
        # how to process the hard negatives sequentially?

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
            "inpn_tokens": inpn_tokens,
            "inpn_mask": inpn_mask,
            "len_input_negs": len_input_negs
        }
        batch['labels'] = None

        if len(queries) == len(golds):  # if the same number of golds and queries, then only one positive each query (don't need labels)
            return batch
        
        batch['labels'] = labels
        return batch




class SentenceTransformerCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        labels = []
        golds = []
        for ex in batch:
            labels.append([(i + len(golds)) for i in range(len(ex["gold"]))])
            golds += (ex["gold"])
        negs = [item for ex in batch for item in ex["negatives"]]
        
        input_negs = ['\n'.join(ex["input_negatives"]) for ex in batch]
        input_negs_inst_mask = [n != '' for n in input_negs]
        input_negs = [n for n in input_negs if n != '']
        
        allpassages = golds + negs

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        if len(input_negs) == 0:
            inpnout = None
            inpn_tokens, inpn_mask = None, None
        else:
            inpnout = self.tokenizer.batch_encode_plus(
                input_negs,
                max_length=self.passage_maxlength,
                truncation=True,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            inpn_tokens, inpn_mask = inpnout["input_ids"], inpnout["attention_mask"].bool()
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()
        # inpn_tokens, inpn_mask = inpnout["input_ids"], inpnout["attention_mask"].bool()

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        n_tokens, n_mask = k_tokens[len(golds) :], k_mask[len(golds) :]
        
        # how to process the hard negatives sequentially?

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
            "inpn_tokens": inpn_tokens,
            "inpn_mask": inpn_mask,
            "input_negs_inst_mask": input_negs_inst_mask
        }

        if len(queries) == len(golds):  # if the same number of golds and queries, then only one positive each query (don't need labels)
            return batch
        
        batch['labels'] = labels
        return batch
