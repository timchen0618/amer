# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import itertools
import random

from typing import Optional, Callable

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from dataclasses import dataclass
from data_creation.create_input_for_contrievers import has_answer, SimpleTokenizer
from collections import Counter

@dataclass
class MSETrainCollator:
    def __call__(self, features, shuffle=False, first_label_only=False, left_padding=False):
        batch = {}
        label_str = 'labels' if 'labels' in features[0] else 'positive_embeddings'
        lens_cnt = Counter(len(f[label_str]) for f in features)
        majority_len = max(lens_cnt, key=lens_cnt.get)
        
        keep_feature_indices = []
        for i in range(len(features)):
            if len(features[i][label_str]) == majority_len:
                keep_feature_indices.append(i)
        
        for k in features[0].keys():
            if shuffle:  # train data
                if k == 'labels' or k == 'positive_embeddings':
                    question_labels = [features[j][k][:1] for j in range(len(features)) if j in keep_feature_indices]
                    list_of_labels = [features[j][k][1:] for j in range(len(features)) if j in keep_feature_indices]
                    # randomly shuffle the remaining labels
                    for j in range(len(list_of_labels)):
                        random.shuffle(list_of_labels[j])
                    
                    question_labels = torch.tensor(question_labels)
                    list_of_labels = torch.tensor(list_of_labels)
                    if first_label_only:
                        batch[k] = question_labels
                    else:
                        batch[k] = torch.cat((question_labels, list_of_labels), dim=1)
                    
                else:
                    batch[k] = torch.tensor([features[j][k] for j in range(len(features)) if j in keep_feature_indices])
            else:    # eval data
                if (k == 'labels' or k == 'positive_embeddings') and first_label_only:
                    question_labels = [features[j][k][:1] for j in range(len(features)) if j in keep_feature_indices]
                    batch[k] = torch.tensor(question_labels)
                else:
                    batch[k] = torch.tensor([features[j][k] for j in range(len(features)) if j in keep_feature_indices])
        if 'negative_embeddings' in batch:
            del batch['negative_embeddings']
        if 'positive_embeddings' in batch:
            batch['labels'] = batch['positive_embeddings']
            del batch['positive_embeddings']
            
        if left_padding:
            max_length = int(batch['attention_mask'].sum(dim=1).max().item())
            for k in batch.keys():
                if k in ['input_ids', 'attention_mask', 'inputs_embeds']:
                    for i in range(len(batch[k])):
                        current_length = int(batch['attention_mask'][i].sum().item())
                        batch[k][i] = torch.roll(batch[k][i], shifts=(max_length-current_length), dims=0)
                        
            position_ids = batch['attention_mask'].cumsum(dim=1) - 1
            position_ids = position_ids.clamp(min=0)
            batch['position_ids'] = position_ids
        return batch
    

@dataclass
class ContrastiveTrainCollator:
    def __call__(self, features, shuffle=False, take_first=False, use_eos=False, left_padding=False):
        """
            Data Shape:
            positive_embeddings: (batch_size, length, embedding_dim)
            negative_embeddings: (batch_size, length, embedding_dim)
        """
        batch = {}
        
        lens_cnt = Counter(len(f['positive_embeddings']) for f in features)
        majority_len = max(lens_cnt, key=lens_cnt.get)
        if use_eos:
            eos_token = [0.5 for _ in range(len(features[0]['positive_embeddings'][0]))]
        
        # find out which features to keep
        keep_feature_indices = []
        for i in range(len(features)):
            if len(features[i]['positive_embeddings']) == majority_len:
                keep_feature_indices.append(i)
                if shuffle:  
                    # only shuffle the features with the majority length
                    # shuffle the positive and negative embeddings
                    shuffled_indices = torch.randperm(len(features[i]['positive_embeddings']))
                    features[i]['positive_embeddings'] = [features[i]['positive_embeddings'][j] for j in shuffled_indices]
                    features[i]['negative_embeddings'] = [features[i]['negative_embeddings'][j] for j in shuffled_indices]
                if use_eos:
                    features[i]['positive_embeddings'].append(eos_token)
                    features[i]['negative_embeddings'].append(random.choice(features[i]['negative_embeddings']))
                if take_first:
                    features[i]['positive_embeddings'] = features[i]['positive_embeddings'][:1]
                    features[i]['negative_embeddings'] = features[i]['negative_embeddings'][:1]
            
        for k in features[0].keys():  # loop through the keys => ['input_ids', 'attention_mask', 'positive_embeddings', 'negative_embeddings', 'length']
            if k != 'length':
                # only keep the features with the majority length
                batch_features = [f[k] for j, f in enumerate(features) if j in keep_feature_indices]
                batch[k] = torch.tensor(batch_features)
        
        if left_padding:
            max_length = int(batch['attention_mask'].sum(dim=1).max().item())
            for k in batch.keys():
                if k in ['input_ids', 'attention_mask', 'inputs_embeds']:
                    for i in range(len(batch[k])):
                        current_length = int(batch['attention_mask'][i].sum().item())
                        batch[k][i] = torch.roll(batch[k][i], shifts=(max_length-current_length), dims=0)
            
            position_ids = batch['attention_mask'].cumsum(dim=1) - 1
            position_ids = position_ids.clamp(min=0)
            batch['position_ids'] = position_ids
        
        return batch    
    


def contrastive_eval_collator(features):
    """
    """
    if 'input_ids' in features[0]:
        batch = {'input_ids': [], 'attention_mask':[], 'positive_embeddings': [], 'negative_embeddings': []}
    else:  # use inputs_embeds
        batch = {'inputs_embeds': [], 'attention_mask':[], 'positive_embeddings': [], 'negative_embeddings': []}
    
    for inst in features:
        input_len = sum(inst['attention_mask'])
        for k in inst.keys():
            if k in ['input_ids', 'attention_mask', 'inputs_embeds']:
                batch[k].append(torch.tensor(inst[k][:input_len]).unsqueeze(0))
            elif k in batch: # get the question embeddings
                batch[k].append(torch.tensor(inst[k]).unsqueeze(0))
        batch['attention_mask'][-1][:, -1] = 1
        
    
    for k, v in batch.items():
        try:
            batch[k] = torch.cat(v, dim=0)
        except:
            print(k, len(v))
            exit(0)
    return batch
    
def mse_eval_collator(features, first_label_only=False):
    """
    """
    batch = {'input_ids': [], 'attention_mask':[], 'question_embeddings': [], 'labels': []}
    
    for inst in features:
        input_len = sum(inst['attention_mask'])
        for k in inst.keys():
            if k in ['input_ids', 'attention_mask']:
                batch[k].append(torch.tensor(inst[k][:input_len]).unsqueeze(0))
            elif k in ['labels']: # get the question embeddings
                if first_label_only:
                    batch['question_embeddings'].append(torch.tensor(inst[k])[0].unsqueeze(0))
                    batch['labels'].append(torch.tensor(inst[k])[:1].unsqueeze(0))
                else:
                    batch['question_embeddings'].append(torch.tensor(inst[k])[0].unsqueeze(0))
                    batch['labels'].append(torch.tensor(inst[k])[1:].unsqueeze(0))
        batch['attention_mask'][-1][:, -1] = 1
        
    
    for k, v in batch.items():
        try:
            batch[k] = torch.cat(v, dim=0)
        except:
            print(k, len(v))
            exit(0)
    return batch

    
def load_embeddings_dataset(dataset_path='autoregressive_dev_dataset'):
    dataset = load_from_disk(dataset_path)
    return dataset


@dataclass
class DataHandler:
    dataset: Dataset
    collator: Callable
    batch_size: int
    split: str
    num_workers: int
    # a few scenarios 
    # 1. read the training dataset
    # 1-a. return only the train dataset
    # 1-b. return only the dev dataset
    # 2. read the validation dataset
    # 2-a. return the whole dataset. 
    
    """
        First of all, initialize to get the dataset. 
        Then, depending on the scenario, call the appropriate method to get the dataloader. 
        If it's easy to shuffle the data, just call a random sampler. 
        If we need to shuffle the data, we could try to shuffle the data and call a new dataloader. 
    """ 
    
    def __post_init__(self):
        if self.split == 'train':
            if 'positive_embeddings' in self.dataset[0]:
                # add length to the dataset
                lengths = [len(inst['positive_embeddings']) for inst in self.dataset]
                self.dataset = self.dataset.add_column('length', lengths)
            full_dataset = self.dataset.train_test_split(test_size=0.1, seed=42)
            self.train_dataset, self.dev_dataset = full_dataset['train'], full_dataset['test']
            self.group_data_by_length()
            
    def get_train_dev_dataloader(self, random_train_loader=False):
        if random_train_loader:
            return self.get_random_train_dataloader(), self.get_dev_dataloader()
        else:
            return self.get_sequential_train_dataloader(), self.get_dev_dataloader()
        
    def get_full_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            sampler=SequentialSampler(self.dataset),
        )
        
    def length_aware_shuffle(self):
        # only shuffle the train dataset
        # shuffle the data by length
        self.train_dataset = self.train_dataset.shuffle(seed=42)
        self.group_data_by_length()
        
    def group_data_by_length(self):
        # group the data by length  (length, dim)
        if 'length' in self.train_dataset[0]:
            print(self.train_dataset[0]['length'])
            self.train_dataset = self.train_dataset.sort('length')
            self.dev_dataset = self.dev_dataset.sort('length')


    def get_random_train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            sampler=RandomSampler(self.train_dataset),
        )
        return train_dataloader
    
    def get_sequential_train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            sampler=SequentialSampler(self.train_dataset),
        )
        return train_dataloader
    
    def get_dev_dataloader(self):
        valid_loss_dataloader = torch.utils.data.DataLoader(
            self.dev_dataset,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            sampler=SequentialSampler(self.dev_dataset),
        )
        return valid_loss_dataloader
    


def safe_from_list_and_save(dataset_dicts, out_dataset_path, batch_size=10000):
    datasets = []
    for i in range(0, len(dataset_dicts), batch_size):
        chunk = dataset_dicts[i:i + batch_size]
        chunk_dataset = Dataset.from_list(chunk)
        datasets.append(chunk_dataset)

    full_dataset = concatenate_datasets(datasets)
    full_dataset.save_to_disk(out_dataset_path)
    return full_dataset


def process_single_instance(args):
    inst, retrieved_inst, tokenizer = args
    if 'question' not in inst:
        assert (inst['question_text']) == (retrieved_inst['question'])
    else:
        assert (inst['question']) == (retrieved_inst['question'])
    num_cluster = len(inst['ground_truths'])
    if isinstance(inst['ground_truths'][0], list):
        gold_ids = [gt['id'] for cluster in inst['ground_truths'] for gt in cluster]
    elif isinstance(inst['ground_truths'][0], dict):
        gold_ids = [gt['id'] for gt in inst['ground_truths']]
    else:
        raise ValueError(f'Unknown ground truth type: {type(inst["ground_truths"])}')
    
    hard_negative_docs = []
    if len(inst['ground_truths']) <= 8:
        for j, ret_doc in enumerate(retrieved_inst['ctxs']):
            if ret_doc['id'] in gold_ids: # skip the gold documents
                continue
            
            contain_answer = False
            contain_answer_string = ""
            
            for cluster_answers in inst['answer_list']:
                if len(cluster_answers['aliases']) == 0:
                    continue
                if len(cluster_answers['aliases']) == 1 and cluster_answers['aliases'][0] == '':
                    continue
                
                if has_answer(cluster_answers['aliases'], ret_doc['text'], tokenizer): # -> has answer
                    contain_answer = True
                    contain_answer_string = cluster_answers['aliases']
                    break
            if not contain_answer:
                hard_negative_docs.append(ret_doc)
                if len(hard_negative_docs) >= num_cluster:
                    break
    
    while len(hard_negative_docs) < num_cluster and len(hard_negative_docs) > 0:
        hard_negative_docs = hard_negative_docs + hard_negative_docs[:num_cluster - len(hard_negative_docs)]
    
    if len(hard_negative_docs) > 0:
        assert len(hard_negative_docs) == num_cluster, (inst['question_text'], len(hard_negative_docs), num_cluster)
        
    inst['ctxs'] = hard_negative_docs
    if 'positive_ctxs' in inst:
        del inst['positive_ctxs']
    if 'ground_truths' in inst:
        del inst['ground_truths']
    return inst

