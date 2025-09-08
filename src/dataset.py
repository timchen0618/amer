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
            # len(features[0][k]) -> length
            # len(features) -> batch size
            # len(features[0][k][0]) -> embedding dim
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
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
                    # if inst['question_text'] == 'Which fictional character is present in the work Adventures of Superman?':
                    #     print(cluster_answers['aliases'], ret_doc['text'])
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


if __name__ == "__main__":    
    def read_jsonl(path):
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]
    
    def write_jsonl(data, path):
        with open(path, 'w') as f:
            for inst in data:
                f.write(json.dumps(inst) + '\n')
    
    def write_json(data, path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def read_json(path):
        with open(path, 'r') as f:
            return json.load(f)  
        
    
    command = 'combine_pred_length'
    
    if command == 'combine_datasets':
        # for retriever in ['cont', 'stella', 'inf']:
        use_hard_negatives = False
        for retriever in ['inf']:
            data_name = 'ambiguous_qe'
            split = 'dev'
            base_model_name = 'inf'
            for split in ['train', 'dev']:
                start_and_end_map = {"qampari": [5, 9], "ambiguous": [2, 6], "ambiguous_qe": [2, 6]}
                if use_hard_negatives:
                    dataset_paths = [f'training_datasets/{base_model_name}/{data_name}/{retriever}/autoregressive_{data_name}_{retriever}_{split}_dataset_1b_contrastive_hard_negative_{i}_ctxs' \
                    for i in range(start_and_end_map[data_name][0], start_and_end_map[data_name][1])]
                else:
                    dataset_paths = [f'training_datasets/{base_model_name}/{data_name}/{retriever}/autoregressive_{data_name}_{retriever}_{split}_dataset_1b_contrastive_{i}_ctxs' \
                    for i in range(start_and_end_map[data_name][0], start_and_end_map[data_name][1])]
                assert len(dataset_paths) == 4
                dataset_paths = [load_from_disk(path) for path in dataset_paths]
                # dataset_paths = [d.select([i for i in range(30)]) for d in dataset_paths]

                combined_dataset = concatenate_datasets(dataset_paths)
                if use_hard_negatives:
                    combined_dataset.save_to_disk(f'training_datasets/{base_model_name}/{data_name}/{retriever}/autoregressive_{data_name}_{retriever}_{split}_dataset_1b_contrastive_hard_negative_{start_and_end_map[data_name][0]}_to_{start_and_end_map[data_name][1]-1}_ctxs')
                else:
                    combined_dataset.save_to_disk(f'training_datasets/{base_model_name}/{data_name}/{retriever}/autoregressive_{data_name}_{retriever}_{split}_dataset_1b_contrastive_{start_and_end_map[data_name][0]}_to_{start_and_end_map[data_name][1]-1}_ctxs')
                
        # for split in ['train', 'dev']:
        #     start_and_end_map = {"qampari": [5, 9], "ambiguous": [2, 6], "ambiguous_qe": [2, 6]}
        #     data_name = 'ambiguous_qe'
        #     dataset_paths = [f'/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/{data_name}_{split}_question_only_{i}_ctxs.jsonl' \
        #         for i in range(start_and_end_map[data_name][0], start_and_end_map[data_name][1])]
        #     assert len(dataset_paths) == 4
        #     datas = [read_jsonl(path) for path in dataset_paths]
        #     # flatten the list
        #     combined_data = [inst for data in datas for inst in data]
        #     write_jsonl(combined_data, f'/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/{data_name}_{split}_question_only_{start_and_end_map[data_name][0]}_to_{start_and_end_map[data_name][1]-1}_ctxs.jsonl')
    if command == 'combine_pred_length':
        for split in ['train', 'test']:
            trans_5_data = load_from_disk(f'data_creation/gaussian_new_mlps_rotation_5_{split}_dataset_1b_contrastive_pred_length')
            trans_2_data = load_from_disk(f'data_creation/gaussian_new_mlps_rotation_2_{split}_dataset_1b_contrastive_pred_length')
            
            combined_dataset = concatenate_datasets([trans_5_data, trans_2_data])
            combined_dataset.save_to_disk(f'data_creation/gaussian_new_mlps_rotation_{split}_dataset_1b_contrastive_pred_length')

                
    if command == 'combine_ambiguous_and_qampari':
        for retriever in ['cont', 'stella', 'inf']:
            for split in ['train']:
                ambiguous_data = load_from_disk(f'training_datasets/autoregressive_ambiguous_{retriever}_{split}_dataset_1b_contrastive_2_to_5_ctxs')
                qampari_data = load_from_disk(f'training_datasets/autoregressive_qampari_{retriever}_{split}_dataset_1b_contrastive_5_to_8_ctxs')
                
                combined_dataset = concatenate_datasets([ambiguous_data, qampari_data])
                combined_dataset.save_to_disk(f'training_datasets/autoregressive_ambiguous_and_qampari_{retriever}_{split}_dataset_1b_contrastive_2_to_8_ctxs')
    
    if command == 'all_no_evidence':
        data_1 = read_json('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nqopen/nqopen-dev_multi_answer.json')
        data_2 = read_json('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nqopen/nqopen-test_multi_answer.json')
        data_3 = read_json('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/ambignq/ambignq-train_multi_answer_no_evidence.json')
        data_4 = read_json('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/ambignq/ambignq-dev_multi_answer_no_evidence.json')
        
        for inst in data_1:
            assert len(inst['answers']) > 0, (inst['question'], 'data_1')
        for inst in data_2:
            assert len(inst['answers']) > 0, (inst['question'], 'data_2')
        for inst in data_3:
            assert len(inst['answers']) > 0, (inst['question'], 'data_3')
        for inst in data_4:
            assert len(inst['answers']) > 0, (inst['question'], 'data_4')
            
        print(len(data_1), len(data_2), len(data_3), len(data_4))
        print(len(data_1) + len(data_2) + len(data_3) + len(data_4))
        
        pre_full_data = read_json('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/ambignq+nqopen_multi_answer_all_no_evidence.json')    
        print(len(pre_full_data))
        for inst in pre_full_data:
            assert len(inst['answers']) > 0, (inst['question'], 'pre_full')
    
    
    if command == 'create_evidence_data':
        full_data = read_json('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/ambignq+nqopen_multi_answer_all_no_evidence_evidence.json')    
        print(len(full_data))
        for inst in full_data:
            assert len(inst['answers']) > 0, (inst['question'], 'full')
        
        # additional data: AmbigQA
        ambig_train_data = read_json('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/ambignq/ambignq-train_multi_answer_evidence.json')
        ambig_dev_data = read_json('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/ambignq/ambignq-dev_multi_answer_evidence.json')
        
        import random
        random.seed(42)
        random.shuffle(full_data)
        # # split into train and dev
        train_data = full_data[:-300]
        dev_data = full_data[-300:]
        
        train_data = train_data + ambig_train_data + ambig_dev_data
        print(len(train_data), len(dev_data))
        
        
        # additional data: other data from evidence (qampari embeddings) 
        qs = [inst['question'] for inst in train_data]
        evidence_data = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence.jsonl')
        for inst in evidence_data:
            if inst['question'] not in qs:
                train_data.append(inst)
        
        print(len(train_data))
        
        # save the train and dev data
        write_jsonl(train_data, '/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_train.jsonl')
        write_jsonl(dev_data, '/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev.jsonl')
        write_json(train_data, '/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_train.json')
        write_json(dev_data, '/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev.json')
        
        # train_data = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/ambignq+nqopen-all_multi_answer_evidence_train.jsonl')
        # dev_data = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/ambignq+nqopen-all_multi_answer_evidence_dev.jsonl')
        
        for inst in dev_data:
            assert len(inst['answers']) > 0, inst['question']
        
        for inst in train_data:
            assert len(inst['answers']) > 0, (inst['question'], 'train')
    
    
    
    if command == 'create_hard_negative_data':
        # mine the hard negative data
        for split in ['dev']:
            for retriever in ['cont', 'inf']:
                raw_data = read_jsonl(f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_{split}.jsonl')
                if retriever in ['inf', 'stella-400M']:
                    retrieved_data = read_jsonl(f'/datastor1/hungting/retrieval_outputs/mteb_retriever/{retriever}/ambignq+nqopen-all_multi_answer_evidence_{split}.json')
                elif retriever == 'cont':
                    retrieved_data = read_jsonl(f'/scratch/cluster/hungting/projects/Multi_Answer/contriever/outputs/contriever_msmarco_nq/ambignq+nqopen-all_multi_answer_evidence_{split}.json')
                    
                assert len(raw_data) == len(retrieved_data), (len(raw_data), len(retrieved_data))
                
                # For each question, find the retrieved documents that do not contain any answers. 
                # make sure the length of positive and negative are the same  (# has_answer(answers, text, tokenizer) -> bool:)
                tokenizer = SimpleTokenizer()
                from tqdm import tqdm
                for inst, retrieved_inst in tqdm(zip(raw_data, retrieved_data)):
                    assert len(inst['question']) == len(inst['question'])
                    num_cluster = len(inst['positive_ctxs'])
                    hard_negative_docs = []
                    
                    for ret_doc in retrieved_inst['ctxs']:
                        contain_answer = False
                        for cluster_answers in inst['answers']:
                            if has_answer(cluster_answers, ret_doc['text'], tokenizer): # -> has answer
                                contain_answer = True
                                break
                        if not contain_answer:
                            hard_negative_docs.append(ret_doc)
                            if len(hard_negative_docs) >= num_cluster:
                                break
                    
                    assert len(hard_negative_docs) == num_cluster, (inst['question'], len(hard_negative_docs), num_cluster)
                    
                    inst['ctxs'] = hard_negative_docs
                    if 'positive_ctxs' in inst:
                        del inst['positive_ctxs']
                    if 'ground_truths' in inst:
                        del inst['ground_truths']
                
                
                write_jsonl(raw_data, f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/ambignq+nqopen-all_multi_answer_evidence_{split}_hard_negative_{retriever}.jsonl')
                
                
    if command == 'create_hard_negative_data_qampari':
        # mine the hard negative data
        for split in ['train']:
            for retriever in ['cont', 'stella-400M']:
                raw_data = read_jsonl(f'/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/{split}_data_gt_qampari_corpus.jsonl')
                # raw_data = read_jsonl('bbb')
                if retriever in ['inf', 'stella-400M']:
                    retrieved_data = read_jsonl(f'/datastor1/hungting/retrieval_outputs/mteb_retriever/{retriever}/{split}_data_gt_qampari_corpus.json')
                    # retrieved_data = read_jsonl('aaa')
                elif retriever == 'cont':
                    retrieved_data = read_jsonl(f'/scratch/cluster/hungting/projects/Multi_Answer/contriever/outputs/contriever_msmarco_qampari/{split}_data_gt_qampari_corpus.json')
                assert len(raw_data) == len(retrieved_data), (len(raw_data), len(retrieved_data))
            
                # For each question, find the retrieved documents that do not contain any answers. 
                # make sure the length of positive and negative are the same  (# has_answer(answers, text, tokenizer) -> bool:)
                tokenizer = SimpleTokenizer()
                from tqdm import tqdm
                import multiprocessing as mp
                from functools import partial

                # Prepare arguments for parallel processing
                process_args = [(inst, retrieved_inst, tokenizer) for inst, retrieved_inst in zip(raw_data, retrieved_data)]
                
                # Create a pool of workers
                num_workers = mp.cpu_count() - 1  # Leave one CPU free
                print(num_workers)
                with mp.Pool(processes=num_workers) as pool:
                    # Process instances in parallel with progress bar
                    processed_instances = list(tqdm(
                        pool.imap(process_single_instance, process_args),
                        total=len(process_args),
                        desc="Processing instances"
                    ))
                
                # Filter out None results and save
                processed_instances = [inst for inst in processed_instances if inst is not None]
                write_jsonl(processed_instances, f'/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/{split}_data_gt_qampari_corpus_hard_negative_{retriever}.jsonl')
                
                
    if command == 'test':
        # dataset = read_jsonl('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev.jsonl')
        # cnt = 0
        # for i, gold_inst in enumerate(dataset):
        #     for j in range(len(gold_inst['positive_ctxs'])):
        #         gold_inst['positive_ctxs'][j] = [doc for doc in gold_inst['positive_ctxs'][j] if 'id' in doc]
            
                     
        # for i, gold_inst in enumerate(dataset):
        #     for cluster in gold_inst['positive_ctxs']:
        #         if len(cluster) == 0:
        #             cnt += 1
        #             print(i)
        #     gold_inst['positive_ctxs'] = [cluster for cluster in gold_inst['positive_ctxs'] if len(cluster) > 0]
        
        # print(cnt)
        # write_jsonl(dataset, '/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_no_empty_clusters.jsonl')
        
        # dev_data = read_jsonl('/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl')        
        # train_data = read_jsonl('/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/train_data_gt_qampari_corpus.jsonl')
        
        # for inst in dev_data:
        #     inst['question'] = inst['question_text']
        #     inst['input'] = inst['question']
        #     inst['answers'] = ['']
        #     inst['ctxs'] = []
        
        # for inst in train_data:
        #     inst['question'] = inst['question_text']
        #     inst['input'] = inst['question']
        #     inst['answers'] = ['']
        #     inst['ctxs'] = []
        
        # write_json(dev_data, '/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.json')
        # write_json(train_data, '/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/train_data_gt_qampari_corpus.json')
        
        for data_type in ['arguana_generated', 'kialo', 'opinionqa']:
            print(f'doing /scratch/cluster/hungting/projects/Multi_Answer/Data/nqformat_data/{data_type}_1k.json')
            data = read_json(f'/scratch/cluster/hungting/projects/Multi_Answer/Data/nqformat_data/{data_type}_1k.json')
            del_keys = [k for k in data[0].keys() if k not in ['question', 'perspectives']]
            for inst in data:
                for k in del_keys:
                    del inst[k]
                    
            write_jsonl(data, f'/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/{data_type}_question_only.jsonl')
                
                
            
        
    if command == 'filter_by_context_length':
        data_name = 'ambiguous_qe'
        for split in ['dev']:
            filtered_question = []
            for length in [2,3,4,5]:
                filter_data = read_jsonl(f'/scratch/cluster/hungting/projects/autoregressive/data_creation/raw_data/{data_name}_{split}_question_only_{length}_ctxs.jsonl')
                filtered_question.extend([inst['question'] for inst in filter_data])
                
            data = read_jsonl(f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_{split}.jsonl')
            q2inst = {inst['question']: inst for inst in data}
            
            filtered_data = []
            for question in filtered_question:
                if question in q2inst:
                    filtered_data.append(q2inst[question])
                    
            write_jsonl(filtered_data, f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_{split}_2_to_5_ctxs.jsonl')
            write_json(filtered_data, f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_{split}_2_to_5_ctxs.json')
            
            
            
        

        