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
        training=False,
        training_mode='standard_org_q',
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
    ):
        self.training_mode = training_mode
        self.negative_ctxs = negative_ctxs
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)
        if not self.training:
            import random
            random.shuffle(self.data)
            self.data = self.data[:3000]
            print("Number of eval examples: ", len(self.data))
        self._compute_gold_counts()

    def _compute_gold_counts(self):
        """Pre-compute gold count per example for grouped batch sampling."""
        self.gold_counts = []
        if self.training_mode != 'multi':
            return
        for ex in self.data:
            if self.training:
                self.gold_counts.append(len(ex.get("ground_truths", [])))
            else:
                self.gold_counts.append(len(ex.get("positive_ctxs", [])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        # if self.training_mode == 'standard_org_q':
        question = example["question_text"]
        
        task = 'Given a query, retrieve relevant passages that answer the query'
        question = get_detailed_instruct(task, question)

        if self.training:
            ### Training ###
            positive_string = 'ground_truths'
            # if len(example[positive_string]) > self.max_positive_documents:
            #     gold = random.sample(example[positive_string], self.max_positive_documents)
            # else:
            #     gold = example[positive_string]
            if self.training_mode == 'standard_org_q':
                gold = random.choice(example[positive_string])
            elif self.training_mode == 'multi':
                # Copy before shuffling so we never mutate the cached example.
                # Shuffling each epoch regularises the AR teacher-forced input
                # sequence (step j sees a different gold context across epochs)
                # but is otherwise orthogonal to the Hungarian matching used in
                # the loss — Hungarian picks the optimal output<->gold pairing
                # regardless of order.
                gold = list(example[positive_string])
                random.shuffle(gold)

            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            if self.training_mode == 'multi':
                n_hard_negatives = n_hard_negatives * len(gold)
                n_random_negatives = n_random_negatives * len(gold)

            negatives = []
            if n_random_negatives > 0:
                pool = example["negative_ctxs"]
                random_negatives = random.choices(pool, k=n_random_negatives) if n_random_negatives > len(pool) else random.sample(pool, n_random_negatives)
                negatives += random_negatives
            if n_hard_negatives > 0:
                pool = example["hard_negative_ctxs"][self.negative_hard_min_idx :]
                hard_negatives = random.choices(pool, k=n_hard_negatives) if n_hard_negatives > len(pool) else random.sample(pool, n_hard_negatives)
                negatives += hard_negatives
        else:
            ### Evaluation ###
            if self.training_mode == 'multi':
                gold = example["positive_ctxs"]
            else:
                gold = example["positive_ctxs"][0]
            nidx = 0
            if "negative_ctxs" in example:
                if example['negative_ctxs']:
                    if 'hard_negative_ctxs' in example:
                        negatives = [example["negative_ctxs"][nidx]] + example['hard_negative_ctxs']
                    else:
                        negatives = example["negative_ctxs"]
                else:
                    negatives = example['hard_negative_ctxs']
            else:
                negatives = []
            if self.training_mode == 'multi':
                n_target = self.negative_ctxs * len(gold)
                if len(negatives) > n_target:
                    negatives = random.sample(negatives, n_target)
                elif len(negatives) < n_target and len(negatives) > 0:
                    negatives = negatives + [random.choice(negatives) for _ in range(n_target - len(negatives))]
        
        
        if self.training_mode == 'multi':
            gold = [
                g["title"] + " " + g["text"] if ("title" in g and len(g["title"]) > 0) else g["text"] for g in gold
            ]
        else:
            gold = gold["title"] + " " + gold["text"] if ("title" in gold and len(gold["title"]) > 0) else gold["text"]
        negatives = [
            n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives
        ]
        
        gold_instances = [self.normalize_fn(g) for g in gold] if self.training_mode == 'multi' else self.normalize_fn(gold) 
        example = {
            "query": self.normalize_fn(question),
            # "gold": [self.normalize_fn(g) for g in gold],
            "gold": gold_instances,
            "negatives": [self.normalize_fn(n) for n in negatives]
        }

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
        training=False,
        training_mode='standard_org_q',
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
        
        example = {
            "query": self.normalize_fn(question),
            "gold": self.normalize_fn(gold),
            "negatives": [self.normalize_fn(n) for n in negatives]
        }

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



class GoldLengthGroupedBatchSampler:
    """Batch sampler that groups dataset indices by gold count.

    Every batch contains only examples with the same number of gold documents,
    so the collator and model see a uniform nqe within each batch without
    any padding or truncation.
    """

    def __init__(self, gold_counts, batch_size, drop_last=False, shuffle=True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        from collections import defaultdict
        self.groups = defaultdict(list)
        for idx, count in enumerate(gold_counts):
            self.groups[count].append(idx)
        summary = {k: len(v) for k, v in sorted(self.groups.items())}
        print(f"GoldLengthGroupedBatchSampler: {summary}", flush=True)

    def __iter__(self):
        all_batches = []
        for _count, indices in self.groups.items():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            random.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        total = 0
        for _count, indices in self.groups.items():
            n_full = len(indices) // self.batch_size
            remainder = len(indices) % self.batch_size
            if self.drop_last:
                total += n_full
            else:
                total += n_full + (1 if remainder > 0 else 0)
        return total


class CollatorMulti(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        labels = []
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
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].to(torch.long)
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].to(torch.long)

        # Convert query tokens to left-padding
        bsz, seq_len = q_tokens.shape
        for i in range(bsz):
            num_pad = (q_mask[i] == 0).sum().item()
            if num_pad > 0:
                # Move real tokens to the right, padding to the left
                q_tokens[i] = torch.cat([q_tokens[i, -num_pad:].clone().fill_(self.tokenizer.pad_token_id), q_tokens[i, q_mask[i] == 1]])
                q_mask[i] = torch.cat([torch.zeros(num_pad, dtype=torch.long), torch.ones(seq_len - num_pad, dtype=torch.long)])

        # Compute position_ids for query tokens (left-padded)
        q_position_ids = q_mask.cumsum(dim=1) - 1
        q_position_ids = q_position_ids.clamp(min=0)

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        n_tokens, n_mask = k_tokens[len(golds) :], k_mask[len(golds) :]

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "q_position_ids": q_position_ids,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
        }
        batch['labels'] = None
        if len(queries) == len(golds):
            return batch

        batch['labels'] = labels
        return batch


class CollatorDocEncMultiQuery(object):
    """
    Builds batches for EmbeddingModelDocEncNoProj: documents are
    (batch_size, num_query_embeddings * 2, seq_len) with the first nqe slots
    positives and the second nqe negatives per example.

    Each example must provide ``gold`` (str) and ``negatives`` (list of str).
    Positives are the same gold repeated ``num_query_embeddings`` times;
    negatives are sampled (with replacement if needed) to length ``num_query_embeddings``.

    Also returns ``g_*`` / ``n_*`` as the first positive and first negative slot
    (each (batch_size, seq_len)) for eval code that expects 2D gold/neg tensors.
    """

    def __init__(self, tokenizer, passage_maxlength=200, num_query_embeddings=1):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        labels = []
        # each ex["gold"] is a list, flatten it
        # example: batch = [[1,2,3], [4,5,6]] -> golds = [1,2,3,4,5,6]
        golds = [item for ex in batch for item in ex["gold"]]
        bsz = len(batch)
        negs = [item for ex in batch for item in ex["negatives"]]
        assert len(golds) == len(negs), (len(golds), len(negs))
        flat_passages = golds + negs
        
        nqe = len(golds) // bsz
        assert nqe * bsz == len(golds), (len(golds), bsz, nqe)

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            flat_passages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].to(torch.long)
        k_tokens_flat = kout["input_ids"]
        k_mask_flat = kout["attention_mask"].to(torch.long)

        bsz_q, seq_len = q_tokens.shape
        assert bsz_q == bsz
        for i in range(bsz):
            num_pad = (q_mask[i] == 0).sum().item()
            if num_pad > 0:
                q_tokens[i] = torch.cat(
                    [
                        q_tokens[i, -num_pad:].clone().fill_(self.tokenizer.pad_token_id),
                        q_tokens[i, q_mask[i] == 1],
                    ]
                )
                q_mask[i] = torch.cat(
                    [
                        torch.zeros(num_pad, dtype=torch.long),
                        torch.ones(seq_len - num_pad, dtype=torch.long),
                    ]
                )

        q_position_ids = q_mask.cumsum(dim=1) - 1
        q_position_ids = q_position_ids.clamp(min=0)

        # gtokens in the first half, ntokens in the second half
        g_tokens = k_tokens_flat[: len(golds)]
        n_tokens = k_tokens_flat[len(golds) :]
        g_mask = k_mask_flat[: len(golds)]
        n_mask = k_mask_flat[len(golds) :]

        out = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "q_position_ids": q_position_ids,
            "k_tokens": k_tokens_flat,
            "k_mask": k_mask_flat,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
        }
        out["labels"] = None
        if len(queries) == len(golds):
            return out

        out["labels"] = labels
        return out


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




 