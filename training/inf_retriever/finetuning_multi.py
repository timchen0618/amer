# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pdb
import os
import time
import sys
import torch
import torch.nn.functional as F
import logging
import time
import pickle
import json
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datetime import timedelta

from src.options import Options
from src import data, slurm, dist_utils, utils, inf_retriever, finetuning_data, inbatch
import src.index
import src.data
import src.normalize_text
import src.inf_retriever

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
            # Format with instruction for INF-Retriever
           # q_with_instruction = get_detailed_instruct(task, q)
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
    if opt.training_mode == 'standard_org_q':
        collator = finetuning_data.CollatorMulti(tokenizer, passage_maxlength=opt.chunk_length)
    elif opt.training_mode == 'multi':
        collator = finetuning_data.CollatorDocEncMultiQuery(tokenizer, passage_maxlength=opt.chunk_length)
    else:
        raise NotImplementedError
    
    logger.info("Loading training data from %s", opt.train_data)
    print(opt.sample_length, flush=True)
    
    if opt.sample_length:
        print("Sampling document lengths: %s", ' '.join(str(x) for x in opt.doc_lengths), flush=True)
        train_dataset = finetuning_data.SampleDataset(
            datapaths=opt.train_data,
            training_mode=opt.training_mode,
            negative_ctxs=opt.negative_ctxs,
            negative_hard_ratio=opt.negative_hard_ratio,
            negative_hard_min_idx=opt.negative_hard_min_idx,

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
        print("sample didn't work", flush=True)
        train_dataset = finetuning_data.Dataset(
            datapaths=opt.train_data,
            training_mode=opt.training_mode,
            negative_ctxs=opt.negative_ctxs,
            negative_hard_ratio=opt.negative_hard_ratio,
            negative_hard_min_idx=opt.negative_hard_min_idx,

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
    
    
    if opt.training_mode == 'multi' and hasattr(train_dataset, 'gold_counts') and train_dataset.gold_counts:
        train_batch_sampler = finetuning_data.GoldLengthGroupedBatchSampler(
            train_dataset.gold_counts, opt.per_gpu_batch_size, drop_last=True, shuffle=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=collator,
        )
    else:
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=opt.per_gpu_batch_size,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=collator,
        )

    eval_collator = finetuning_data.CollatorDocEncMultiQuery(tokenizer, passage_maxlength=opt.chunk_length) if opt.training_mode == 'multi' else collator
    if opt.training_mode == 'multi' and hasattr(eval_dataset, 'gold_counts') and eval_dataset.gold_counts:
        eval_batch_sampler = finetuning_data.GoldLengthGroupedBatchSampler(
            eval_dataset.gold_counts, opt.per_gpu_eval_batch_size, drop_last=False, shuffle=False,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_sampler=eval_batch_sampler,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=eval_collator,
        )
    else:
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=opt.per_gpu_eval_batch_size,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=eval_collator,
        )

    return train_dataloader, eval_dataloader

##########
##########


def evaluate_recall(opt, model, tokenizer, device):
    import glob
    model.eval()
    index = src.index.Indexer(1536, 0, 8)

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
    if opt.training_mode == 'standard_org_q':
        model = inbatch.EmbeddingModelDocEncNoProjSingleQuery(opt, None, None)
    elif opt.training_mode == 'multi':
        model = inbatch.EmbeddingModelDocEncNoProj(opt, None, None)
    else:
        raise NotImplementedError
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    print('finish getting model')

    is_multi = opt.training_mode == 'multi'
    total_correct = 0
    total_mrr = 0.0
    total_queries = 0

    # Multi-embedding diagnostics
    multi_step_correct = {}   # step_j -> total correct across all batches
    multi_step_total = 0
    total_pairwise_sim = 0.0
    total_pairwise_count = 0

    for i, batch in tqdm(enumerate(eval_loader)):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        all_tokens = torch.cat([batch["g_tokens"], batch["n_tokens"]], dim=0)  # (2 * batch_size, seq_len)
        all_mask = torch.cat([batch["g_mask"], batch["n_mask"]], dim=0)  # (2 * batch_size, seq_len)

        q_out = model.encoder(input_ids=batch["q_tokens"], attention_mask=batch["q_mask"], position_ids=batch["q_position_ids"])
        q_emb = model.last_token_pool(q_out.last_hidden_state, batch["q_mask"])
        if opt.norm_query:
            q_emb = torch.nn.functional.normalize(q_emb, dim=-1)

        g_emb, n_emb = model.encode_documents(input_document_ids=all_tokens, attention_mask_document=all_mask)  # (batch_size, embedding_dim)
        if opt.norm_doc:
            g_emb = torch.nn.functional.normalize(g_emb, dim=-1)
            n_emb = torch.nn.functional.normalize(n_emb, dim=-1)

        bsz = q_emb.size(0)
        all_docs = torch.cat([g_emb, n_emb], dim=0)
        scores = torch.einsum("id, jd->ij", q_emb, all_docs)

        if is_multi:
            nqe = g_emb.size(0) // bsz
            for qi in range(bsz):
                pos_start = qi * nqe
                pos_end = pos_start + nqe
                pos_indices = set(range(pos_start, pos_end))
                sorted_indices = scores[qi].argsort(descending=True)
                total_correct += int(sorted_indices[0].item() in pos_indices)
                for rank, idx in enumerate(sorted_indices):
                    if idx.item() in pos_indices:
                        total_mrr += 1.0 / (rank + 1)
                        break
                total_queries += 1

            # --- Multi-embedding diagnostics ---
            # Autoregressively generate k embeddings (no teacher forcing)
            multi_embs = model.generate(
                batch["q_tokens"], batch["q_mask"], batch["q_position_ids"],
                max_new_tokens=nqe,
            )  # (bsz, nqe, hidden_dim)
            if opt.norm_query:
                multi_embs = F.normalize(multi_embs, dim=-1)

            # Pairwise cosine similarity within each query's k embeddings
            if nqe > 1:
                emb_norm = F.normalize(multi_embs.float(), dim=-1)  # (bsz, nqe, d)
                sim_mat = torch.bmm(emb_norm, emb_norm.transpose(1, 2))  # (bsz, nqe, nqe)
                eye_mask = torch.eye(nqe, dtype=torch.bool, device=sim_mat.device).unsqueeze(0)
                total_pairwise_sim += sim_mat.masked_fill(eye_mask, 0.0).sum().item()
                total_pairwise_count += bsz * nqe * (nqe - 1)

            # Per-step accuracy: for step j, is top-1 retrieved doc any gold of that query?
            for step_j in range(nqe):
                if step_j not in multi_step_correct:
                    multi_step_correct[step_j] = 0
                step_emb = multi_embs[:, step_j, :]  # (bsz, d)
                step_scores = torch.einsum("id,jd->ij", step_emb, all_docs)  # (bsz, total_docs)
                top1 = step_scores.argmax(dim=-1)  # (bsz,)
                for qi in range(bsz):
                    pos_start = qi * nqe
                    pos_end = pos_start + nqe
                    multi_step_correct[step_j] += int(top1[qi].item() in range(pos_start, pos_end))
            multi_step_total += bsz

        else:
            labels = torch.arange(0, bsz, device=q_emb.device, dtype=torch.long)
            argmax_idx = torch.argmax(scores, dim=1)
            total_correct += (argmax_idx == labels).sum().item()
            sorted_indices = torch.argsort(scores, dim=1, descending=True)
            for qi in range(bsz):
                rank = (sorted_indices[qi] == labels[qi]).nonzero(as_tuple=True)[0]
                total_mrr += 1.0 / (rank[0].item() + 1) if rank.numel() > 0 else 0.0
            total_queries += bsz

    acc = 100 * total_correct / max(total_queries, 1)
    mrr = total_mrr / max(total_queries, 1)

    message = [f"eval acc: {acc:.2f}%", f"eval mrr: {mrr:.3f}"]
    log_dict = {"eval_acc": acc, "eval_mrr": mrr}

    if is_multi and multi_step_total > 0:
        if total_pairwise_count > 0:
            avg_pairwise_sim = total_pairwise_sim / total_pairwise_count
            log_dict["eval_pairwise_cos_sim"] = avg_pairwise_sim
            message.append(f"pairwise_cos_sim: {avg_pairwise_sim:.4f}")

        for j in sorted(multi_step_correct):
            step_acc = 100.0 * multi_step_correct[j] / multi_step_total
            log_dict[f"eval_step_{j}_acc"] = step_acc
            message.append(f"step_{j}_acc: {step_acc:.2f}%")

    logger.info(" | ".join(message))
    accelerator.log(log_dict, step=step)
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
    accelerator = Accelerator(gradient_accumulation_steps=opt.accumulation_steps, mixed_precision="bf16", log_with="wandb", kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
    device = accelerator.device
    
    wandb_config = {
        "tags": ['finetuning', opt.training_mode],
        "name": opt.run_name
    }
    accelerator.init_trackers(project_name="amer", config=opt, init_kwargs={"wandb": {**wandb_config, "settings": {"init_timeout": 300}}})
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
            with accelerator.accumulate(model):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and key not in ['g_tokens', 'g_mask', 'n_tokens', 'n_mask']:
                        batch[key] = value.to(device)
                        
                # batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

                train_loss, iter_stats = model(**batch, stats_prefix="train", sampling_rate=step/opt.total_steps)
                accelerator.backward(train_loss)
                if opt.optim == "sam" or opt.optim == "asam":
                    optimizer.first_step(zero_grad=True)
                    sam_loss, _ = model(**batch, stats_prefix="train/sam_opt", sampling_rate=step/opt.total_steps)
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
    # retriever, tokenizer, retriever_model_id = inf_retriever.load_retriever(opt.model_path, opt.pooling, opt.random_init)
    # opt.retriever_model_id = retriever_model_id
    opt.retriever_model_id = "infly/inf-retriever-v1-1.5b"
    
    # # load model to be finetuned
    # if opt.training_mode == 'standard_org_q':
    #     model = inbatch.InBatch(opt, retriever, tokenizer)
    # else:
    #     raise NotImplementedError

    model = inbatch.EmbeddingModelDocEncNoProj(opt, None, None)
    tokenizer = model.tokenizer

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
