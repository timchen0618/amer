# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pdb
import os
import time
import sys
import torch
import logging
import time
import pickle
import json
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datetime import timedelta

from src.options import Options
from src import data, slurm, dist_utils, utils, qwen_retriever, finetuning_data, inbatch
import src.index
import src.data
import src.normalize_text
import src.qwen_retriever
import psutil
import gc
import wandb
from tqdm import tqdm

# add accelerator
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import LoggerType


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


##########
# For indexing the passages
##########

def embed_queries(per_gpu_batch_size, queries, model, tokenizer, device):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            # if args.lowercase:
            #     q = q.lower()
            # if args.normalize_text:
            #     q = src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=8192,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())

                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}", flush=True)
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids



def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]


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


def prepare_data(opt, tokenizer):
    if opt.training_mode in ["base", "standard", "standard_org_q"]:
        collator = finetuning_data.Collator(tokenizer, passage_maxlength=opt.chunk_length)
    elif opt.training_mode in ["subtraction", "gru", "linear_projection", "subtraction_linear"]:
        collator = finetuning_data.SubtractionCollator(tokenizer, passage_maxlength=opt.chunk_length)
    elif opt.training_mode in ["sentence_transformer"]:
        collator = finetuning_data.SentenceTransformerCollator(tokenizer, passage_maxlength=opt.chunk_length)
    else:
        raise NotImplementedError
    
    logger.info("Loading training data from %s", opt.train_data)
    
    if opt.sample_length:
        logger.info("Sampling document lengths: %s", ' '.join(str(x) for x in opt.doc_lengths))
        train_dataset = finetuning_data.SampleDataset(
            datapaths=opt.train_data,
            training_mode=opt.training_mode,
            negative_ctxs=opt.negative_ctxs,
            negative_hard_ratio=opt.negative_hard_ratio,
            negative_hard_min_idx=opt.negative_hard_min_idx,
            add_input_negatives=opt.add_input_negatives,
            normalize=opt.eval_normalize_text,
            global_rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            maxload=opt.maxload,
            training=True,
            tokenizer=tokenizer,
            doc_lengths=opt.doc_lengths
        )
        eval_dataset = finetuning_data.SampleDataset(
            datapaths=opt.eval_data,
            normalize=opt.eval_normalize_text,
            training_mode=opt.training_mode,
            global_rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            maxload=opt.maxload,
            training=False,
            tokenizer=tokenizer,
            doc_lengths=opt.doc_lengths
        )
    else:
        train_dataset = finetuning_data.Dataset(
            datapaths=opt.train_data,
            training_mode=opt.training_mode,
            negative_ctxs=opt.negative_ctxs,
            negative_hard_ratio=opt.negative_hard_ratio,
            negative_hard_min_idx=opt.negative_hard_min_idx,
            add_input_negatives=opt.add_input_negatives,
            normalize=opt.eval_normalize_text,
            global_rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            maxload=opt.maxload,
            training=True,
        )
        eval_dataset = finetuning_data.Dataset(
            datapaths=opt.eval_data,
            normalize=opt.eval_normalize_text,
            training_mode=opt.training_mode,
            global_rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            maxload=opt.maxload,
            training=False,
        )
    
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=0,                 # << was 4
        pin_memory=False,              # << was True
        persistent_workers=False,      # << was True when num_workers>0
        # prefetch_factor=2,           # only valid if num_workers>0
        collate_fn=collator,
    )
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_eval_batch_size,
        drop_last=False,
        num_workers=0,                 # << keep 0 for eval too
        pin_memory=False,              # << save host RAM
        persistent_workers=False,
        collate_fn=collator,
    )

    return train_dataloader, eval_dataloader

##########
##########


def evaluate_recall(opt, model, tokenizer, device):
    import glob
    model.eval()
    index = src.index.Indexer(1024, 0, 8)

    # index all passages
    input_paths = glob.glob("/scratch/cluster/hungting/models/Contriever/contriever_msmarcos/wikipedia_embeddings/*")
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    
    print(f"Indexing passages from files {input_paths}")
    start_time_indexing = time.time()
    index_encoded_data(index, input_paths, 1000000)
    print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

    # load passages
    passages = src.data.load_passages('/scratch/cluster/hungting/projects/ODQA/DPR/downloads/data/wikipedia_split/psgs_w100.tsv')
    passage_id_map = {x["id"]: x for x in passages}

    
    data = load_data('/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev.json')

    queries = [ex["question"] for ex in data]
    questions_embedding = embed_queries(64, queries, model, tokenizer, device)

    # get top k results
    start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(questions_embedding, 5)
    print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

    add_passages(data, passage_id_map, top_ids_and_scores)
    
    precision_score, mrecall_score, recall_score = eval_retrieve_docs(
        data,
        '../../data/qampari_data/dev_data.jsonl',
        has_gold_id=False,
        topk=5
    )
    return precision_score, mrecall_score, recall_score

##########
# Utils functions for evaluating the outputs
##########


import json
from sklearn.metrics import precision_score, recall_score, f1_score
import regex
import string
from tqdm import tqdm
import unicodedata

def score_recall(preds):
    # average 
    recalls = []
    for inst in preds:
        recalls.append(sum([any(preds_per_perspective) for preds_per_perspective in inst])/len(inst))

    return sum(recalls) / float(len(recalls))

def score_mrecall(preds):
    # average 
    mrecalls = []
    topk = len(preds[0][0])
    for inst in preds:  # compute mrecall for each instance
        if len(inst) > topk:
            mrecalls.append(int(sum([any(preds_per_perspective) for preds_per_perspective in inst])>=topk))
        else:
            mrecalls.append(int(all([any(preds_per_perspective) for preds_per_perspective in inst])))

    return sum(mrecalls) / float(len(mrecalls))

def score_precision(preds, topk):
    # average 
    precisions = []
    for inst in preds:
        assert len(inst[0]) >= topk, len(inst[0])
        # if len(inst[0]) < topk:
        #     topk = len(inst[0])
        num_perspective_containing_docs = 0
        for j in range(topk):
            contain_any_perspective = False
            for p in inst:
                if p[j]:
                    contain_any_perspective = True
                    break
            if contain_any_perspective:
                num_perspective_containing_docs += 1

        precisions.append(num_perspective_containing_docs / topk)

    return sum(precisions) / float(len(precisions))

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

def read_jsonl(file_path):
    import json
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


##########
##########

##########
# for evaluating the outputs
##########

def eval_retrieve_docs(retrieved_docs, data_path, has_gold_id=False, topk=100):
    dataset = read_jsonl(data_path)
    tok = SimpleTokenizer()
    # retrieved_docs = read_jsonl(retrieved_docs_path)
    len_docs = []
    # precisions = []
    preds = []
    
    for gold_inst, docs in tqdm(zip(dataset, retrieved_docs)):
        pred_inst = []
        # valid_answers = process_reference(gold_inst)
        # print(valid_answers)
        
        # for ans in valid_answers:
        if has_gold_id:
            gold_indices = [doc['id'] for doc in gold_inst['positive_ctxs']]
            for gold_index in gold_indices:
                pred_inst.append([])
                len_docs.append(len(docs['ctxs']))
                for doc in docs['ctxs'][:topk]:
                    # print(gold_index, doc['id'])
                    pred = gold_index == doc['id']
                    pred_inst[-1].append(pred)
        else:
            for answer in gold_inst['answer_list']:
                pred_inst.append([])
                len_docs.append(len(docs['ctxs']))
                for doc in docs['ctxs'][:topk]:
                    pred = has_answer(answer['aliases'], doc['text'], tok)
                    pred_inst[-1].append(pred)
        preds.append(pred_inst)    
        # precisions.append(sum([any(doc[pred_str]) for doc in docs[:topk]]) / topk)
        
    mrecall_score = score_mrecall(preds)
    recall_score = score_recall(preds)
    precision_score = score_precision(preds, topk)
    print(f'Precision: {100*precision_score:.2f}')
    print(f'Recall: {100*recall_score:.2f}')
    print(f'MRecall: {100*mrecall_score:.2f}')
    print('Average number of retrieved documents:', sum(len_docs) / len(len_docs)) 
    return precision_score, mrecall_score, recall_score

##########
##########


@torch.no_grad()
def evaluate(opt, state_dict, eval_loader, accelerator, step, device):
    # create a new model and use this to load the weights
    if opt.training_mode in ["base", "standard", "standard_org_q"]: 
        model = inbatch.InBatch(opt, None, None)
    elif opt.training_mode == "subtraction":
        model = inbatch.InBatchSubtraction(opt, None, None)
    elif opt.training_mode == "gru":
        model = inbatch.InBatchGRU(opt, None, None)
    elif opt.training_mode == "linear_projection":
        model = inbatch.InBatchLinearProjection(opt, None, None)
    elif opt.training_mode == "subtraction_linear":
        model = inbatch.InBatchSubtractionLinear(opt, None, None)
    elif opt.training_mode == 'sentence_transformer':
        model = inbatch.InBatchSentenceTransformer(opt, None, None)
    else:
        raise NotImplementedError
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print('finish getting model')
    # get the encoder
    encoder = model.encoder
    encoder = encoder.to(device)
    
    all_q, all_g, all_n = [], [], []
    
    
    for i, batch in tqdm(enumerate(eval_loader)):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        all_tokens = torch.cat([batch["g_tokens"], batch["n_tokens"]], dim=0)
        all_mask = torch.cat([batch["g_mask"], batch["n_mask"]], dim=0)
        # print(batch['q_tokens'].size())
        q_emb = encoder(input_ids=batch["q_tokens"], attention_mask=batch["q_mask"], normalize=opt.norm_query)
        all_emb = encoder(input_ids=all_tokens, attention_mask=all_mask, normalize=opt.norm_doc)

        g_emb, n_emb = torch.split(all_emb, [len(batch["g_tokens"]), len(batch["n_tokens"])])
        
        # Query Modification
        if opt.training_mode == "subtraction":
            if batch["inpn_tokens"] is not None:
                inpnemb = encoder(input_ids=batch["inpn_tokens"], attention_mask=batch["inpn_mask"], normalize=opt.norm_doc)
                assert len(batch["len_input_negs"]) == len(batch["q_tokens"])
                start = 0
                for i, _len in enumerate(batch["len_input_negs"]):
                    q_emb[i] = q_emb[i] - torch.sum(inpnemb[start:start+_len], dim=0)
                    start += _len
        elif opt.training_mode == 'gru':
            if batch["inpn_tokens"] is not None:
                inpnemb = encoder(input_ids=batch["inpn_tokens"], attention_mask=batch["inpn_mask"], normalize=opt.norm_doc)
                assert len(batch["len_input_negs"]) == len(batch["q_tokens"])
                start = 0
                gru = model.gru.to(device)
                for i, _len in enumerate(batch["len_input_negs"]):
                    if _len == 0:
                        continue
                    _, hn = gru(inpnemb[start:start+_len], q_emb[i].unsqueeze(0))
                    q_emb[i] = hn[-1]
                    start += _len
        elif opt.training_mode == 'linear_projection':
            if batch["inpn_tokens"] is not None:
                inpnemb = encoder(input_ids=batch["inpn_tokens"], attention_mask=batch["inpn_mask"], normalize=opt.norm_doc)
                assert len(batch["len_input_negs"]) == len(batch["q_tokens"])
                start = 0
                linear = model.linear.to(device)
                for i, _len in enumerate(batch["len_input_negs"]):
                    if _len == 0:
                        continue
                    # q_emb[i] = q_emb[i] - torch.sum(inpnemb[start:start+_len], dim=0)
                    doc_emb = inpnemb[start:start+_len].mean(dim=0)
                    q_doc_emb = torch.cat([q_emb[i].unsqueeze(0), doc_emb.unsqueeze(0)], dim=1)
                    q_emb[i] = linear(q_doc_emb).squeeze(0)
                    start += _len
        elif opt.training_mode == 'subtraction_linear':
            if batch["inpn_tokens"] is not None:
                weights = model.weights.to(device)
                bias = model.bias.to(device)
                sigmoid = model.sigmoid
                inpnemb = encoder(input_ids=batch["inpn_tokens"], attention_mask=batch["inpn_mask"], normalize=opt.norm_doc)
                assert len(batch["len_input_negs"]) == len(batch["q_tokens"])
                start = 0
                for i, _len in enumerate(batch["len_input_negs"]):
                    if _len == 0:
                        continue
                    # q_emb[i] = q_emb[i] - torch.sum(inpnemb[start:start+_len], dim=0)
                    doc_emb = sigmoid(weights * torch.t(inpnemb[start:start+_len]) + bias) # (768, _len)
                    q_emb[i] = q_emb[i] - torch.sum(doc_emb, dim=1)
                    start += _len
        elif opt.training_mode == 'sentence_transformer':
            if batch["inpn_tokens"] is not None:
                inpnemb = encoder(input_ids=batch["inpn_tokens"], attention_mask=batch["inpn_mask"], normalize=opt.norm_doc)
                linear = model.linear.to(device)
                q_emb = linear(torch.cat([q_emb, inpnemb], dim=1)) 
        

        all_q.append(q_emb)
        all_g.append(g_emb)
        all_n.append(n_emb)

    all_q = torch.cat(all_q, dim=0)
    all_g = torch.cat(all_g, dim=0)
    all_n = torch.cat(all_n, dim=0)

    labels = torch.arange(0, len(all_q), device=all_q.device, dtype=torch.long)


    scores_pos = torch.einsum("id, jd->ij", all_q, all_g)
    scores_neg = torch.einsum("id, jd->ij", all_q, all_n)
    scores = torch.cat([scores_pos, scores_neg], dim=-1)
    
    argmax_idx = torch.argmax(scores, dim=1)
    sorted_scores, indices = torch.sort(scores, descending=True)
    isrelevant = indices == labels[:, None]
    rs = [r.cpu().numpy().nonzero()[0] for r in isrelevant]
    mrr = np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])

    acc = (argmax_idx == labels).sum() / all_q.size(0)

    acc = acc.item()
    mrr = mrr.item()
    acc = 100 * acc
    
    message = [f"eval acc: {acc:.2f}%", f"eval mrr: {mrr:.3f}"]
    logger.info(" | ".join(message))

    accelerator.log({"eval_acc": acc, "eval_mrr": mrr}, step=step)
    return acc, mrr


def finetuning(opt, model, optimizer, scheduler, tokenizer, step):
    run_stats = utils.WeightedAvgStats()

    # prepare data
    
    train_dataloader, eval_loader = prepare_data(opt, tokenizer)    
    
    epoch = 1
    model.train()
    #########
    # accelerate: multi-gpu training
    #########
    accelerator = Accelerator(gradient_accumulation_steps=opt.accumulation_steps, 
                              log_with="wandb",
                              mixed_precision="bf16", 
                              kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
    device = accelerator.device
    
    wandb_config = {
        "entity": 'dq2024-new-york-university',
        "tags": ['finetuning', 'gru'],
        "name": opt.run_name
    }
    accelerator.init_trackers("diverse-retriever", config=opt, init_kwargs={"wandb": wandb_config})
    accelerator.log({"startup_check": 1}, step=0)
    
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    # get the state dict of the main model
    state_dict=accelerator.get_state_dict(model)
    
    # if accelerator.is_main_process:
    #     evaluate(opt, state_dict, eval_loader, accelerator, step, device)
    #     utils.save_state_dict(
    #         state_dict,
    #         optimizer,
    #         scheduler,
    #         step,
    #         opt,
    #         opt.output_dir + opt.run_name,
    #         f"step-{step}",
    #     )
                                  
    while step < opt.total_steps:
        logger.info(f"Start epoch {epoch}, number of batches: {len(train_dataloader)}")
        best_eval_metric = 0
        for i, batch in enumerate(train_dataloader):
            step += 1
            if step % 50 == 0:
                print(f"==> at global step {step}", flush=True)

            # if step % 25 == 0:  # Check every 25 steps since your log_freq is 25
            #     allocated = torch.cuda.memory_allocated() / 1e9
            #     max_allocated = torch.cuda.max_memory_allocated() / 1e9
            #     print(f"Step {step}, GPU memory: {allocated:.2f}GB / {max_allocated:.2f}GB", flush=True)
            #     torch.cuda.empty_cache()
            
            # if step % 100 == 0:
            #     process = psutil.Process()
            #     memory_gb = process.memory_info().rss / 1024**3
            #     print(f"System RAM usage: {memory_gb:.2f}GB")
            #     gc.collect()  # Force garbage collection

            with accelerator.accumulate(model):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and key not in ['g_tokens', 'g_mask', 'n_tokens', 'n_mask']:
                        batch[key] = value.to(device)
                        
                # batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

                train_loss, iter_stats = model(**batch, stats_prefix="train")
                accelerator.backward(train_loss)
                if opt.optim == "sam" or opt.optim == "asam":
                    optimizer.first_step(zero_grad=True)
                    sam_loss, _ = model(**batch, stats_prefix="train/sam_opt")
                    # sam_loss.backward()
                    accelerator.backward(sam_loss)
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                run_stats.update(iter_stats)

                if step % (opt.log_freq * opt.accumulation_steps) == 0:
                    log = f"{step} / {opt.total_steps}"
                    for k, v in sorted(run_stats.average_stats.items()):
                        log += f" | {k}: {v:.3f}"

                    log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
                    log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"

                    logger.info(log)
                    wandb_stats = {k: float(v) for k, v in run_stats.average_stats.items()}
                    accelerator.log(wandb_stats, step=step)
                   # accelerator.log(run_stats.average_stats, step=step)
                    run_stats.reset()

                if step % (opt.eval_freq * opt.accumulation_steps) == 0:
                    state_dict=accelerator.get_state_dict(model)
                    if accelerator.is_main_process:
                        _, mrr = evaluate(opt, state_dict, eval_loader, accelerator, step, device)
                        if opt.eval_recall:
                            precision_score, mrecall_score, recall_score = evaluate_recall(opt, model, tokenizer, device)
                            accelerator.log({"precision": precision_score, "mrecall": mrecall_score, "recall": recall_score}, step=step)    
                        
                        if mrr > best_eval_metric:  # save only when gets better performance
                            print('mrr', mrr, 'best eval metric', best_eval_metric)
                            best_eval_metric = mrr
                            if step % (opt.save_freq * opt.accumulation_steps) == 0 and dist_utils.get_rank() == 0:
                                if (not opt.not_save) and accelerator.is_main_process:
                                    utils.save_state_dict(
                                        state_dict,
                                        optimizer,
                                        scheduler,
                                        step,
                                        opt,
                                        opt.output_dir + opt.run_name,
                                        f"best_model",
                                    )

                    model.train()

                if step >= (opt.total_steps * opt.accumulation_steps):
                    break
        epoch += 1

    # end training
    accelerator.end_training()



def main():
    # parse arguments 
    logger.info("Start")
    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)

    # set up output directory
    directory_exists = os.path.isdir(opt.output_dir)        
    os.makedirs(opt.output_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    
    # set up logging
    utils.init_logger(opt)
    step = 0

    # load retriever
    retriever, tokenizer, retriever_model_id = qwen_retriever.load_retriever(opt.model_path, opt.pooling, opt.random_init)
    opt.retriever_model_id = retriever_model_id
    
    # load model to be finetuned
    if opt.training_mode in ["base", "standard", "standard_org_q"]:
        model = inbatch.InBatch(opt, retriever, tokenizer)
    elif opt.training_mode == "subtraction":
        model = inbatch.InBatchSubtraction(opt, retriever, tokenizer)
    elif opt.training_mode == "gru":
        model = inbatch.InBatchGRU(opt, retriever, tokenizer)
    elif opt.training_mode == "linear_projection":
        model = inbatch.InBatchLinearProjection(opt, retriever, tokenizer)
    elif opt.training_mode == "subtraction_linear":
        model = inbatch.InBatchSubtractionLinear(opt, retriever, tokenizer)
    elif opt.training_mode == "sentence_transformer":
        model = inbatch.InBatchSentenceTransformer(opt, retriever, tokenizer)
    else:
        raise NotImplementedError

    # set up optimizers
    optimizer, scheduler = utils.set_optim(opt, model)
    logger.info(utils.get_parameters(model))

    # take care of dropout
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = opt.dropout

    # Start Finetuning
    logger.info("Start training")
    finetuning(opt, model, optimizer, scheduler, tokenizer, step)

if __name__ == "__main__":
    main()
    # options = Options()
    # opt = options.parse()
    # dir_path = "/datastor1/hungting/models/diverse_response/checkpoint/org_random_steps20000_warmup500_lr0.00001_nhr0_nctxs2_bz16/checkpoint/step-20000"
    # model, tokenizer, retriever_model_id = contriever.load_retriever("facebook/contriever-msmarco")
    
    # checkpoint_path = os.path.join(dir_path, "checkpoint.pth")
    # checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # opt_checkpoint = checkpoint["opt"]
    # state_dict = checkpoint["model"]
    
    # model.load_state_dict(state_dict, strict=True)
    # # model, optimizer, scheduler, opt_checkpoint, step = utils.load(src.contriever.Contriever, dir_path, opt, reset_params=False)
    # evaluate_recall(opt, model.cuda(), tokenizer)
