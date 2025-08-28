import json
from sklearn.metrics import precision_score, recall_score, f1_score
import regex
import string
from tqdm import tqdm
import unicodedata
from nltk import word_tokenize

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
    
    
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data




import pytrec_eval
import json

def eval_retrieve_docs(retrieved_docs_path, data_path, has_gold_id=False, topk=100, selected_indices=None):
    dataset = read_jsonl(data_path)
    tok = SimpleTokenizer()
    retrieved_docs = read_jsonl(retrieved_docs_path)
    len_docs = []
    # precisions = []
    preds = []
    qrel = {}
    run = {}
    mrrs = []
    
    assert len(dataset) == len(retrieved_docs), f'Length of dataset and retrieved_docs do not match: {len(dataset)} vs {len(retrieved_docs)}'
    if selected_indices is not None:
        dataset = [dataset[i] for i in selected_indices]
        retrieved_docs = [retrieved_docs[i] for i in selected_indices]
    
    qid = 0
    for gold_inst, docs in tqdm(zip(dataset, retrieved_docs)):
        pred_inst = []
        mrr_inst = {}
        gold_question = gold_inst['question_text'] if 'question_text' in gold_inst else gold_inst['question']
        docs_question = docs['question_text'] if 'question_text' in docs else docs['question']
        if "Relevant Keywords:" in docs_question:
            docs_question = docs_question.split("Relevant Keywords:")[0].split("Question: ")[1].strip('\n').strip()
            gold_question = gold_question.strip('\n').strip()
        assert gold_question == docs_question, (f'Questions do not match: {gold_question} vs {docs_question}', len(gold_question), len(docs_question))
            
        # valid_answers = process_reference(gold_inst)
        # print(valid_answers)
        
        # for ans in valid_answers:
        if has_gold_id:
            qrel[str(qid)] = {}
            if 'positive_ctxs' in gold_inst:
                gold_clusters = [doc for doc in gold_inst['positive_ctxs']]
            elif 'ground_truths' in gold_inst:
                gold_clusters = [doc for doc in gold_inst['ground_truths']]
            else:
                raise ValueError('No positive_ctxs or ground_truths')
            for cluster in gold_clusters:
                gold_ids_per_cluster = [doc['id'] for doc in cluster]
                pred_inst.append([])
                len_docs.append(len(docs['ctxs']))
                for rank, doc in enumerate(docs['ctxs'][:topk]):
                    pred = (doc['id'] in set(gold_ids_per_cluster))
                    pred_inst[-1].append(pred)
                    
                    # handle MRR if prediction correct
                    for gold_id in gold_ids_per_cluster:
                        if doc['id'] == gold_id:
                            mrr_inst[gold_id] = 1 / (rank + 1)
                    
                for gold_id in gold_ids_per_cluster:
                    qrel[str(qid)][gold_id] = 1
                    if gold_id not in mrr_inst: # if not found
                        mrr_inst[gold_id] = 0
                        
            run[str(qid)] = {}
            for rank, doc in enumerate(docs['ctxs'][:topk]):
                # run[str(qid)][doc['id']] = 1 / (rank + 1)
                run[str(qid)][doc['id']] = float(doc['score'])
        else:
            if 'answer_list' in gold_inst:
                answer_list_string = 'answer_list'
            else:
                answer_list_string = 'answers'
            for answer in gold_inst[answer_list_string]:
                pred_inst.append([])
                len_docs.append(len(docs['ctxs']))
                for doc in docs['ctxs'][:topk]:
                    if answer_list_string == 'answer_list':
                        pred = has_answer(answer['aliases'], doc['text'], tok)
                    else:
                        pred = has_answer(answer, doc['text'], tok)
                    pred_inst[-1].append(pred)
        preds.append(pred_inst)  
        qid += 1  
        # compute average MRR for this instance
        _mrr = 0
        for _, value in mrr_inst.items():
            _mrr += value
        if len(mrr_inst) > 0:
            _mrr /= len(mrr_inst)
        mrrs.append(_mrr)
        
        # precisions.append(sum([any(doc[pred_str]) for doc in docs[:topk]]) / topk)
    if has_gold_id:
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
        scores_dict = evaluator.evaluate(run)
        ndcgs = []
        maps = []
        for qid in scores_dict:
            ndcgs.append(scores_dict[qid]['ndcg'])
            maps.append(scores_dict[qid]['map'])
        nDCG = sum(ndcgs) / len(ndcgs)
        mAP = sum(maps) / len(maps)
        
    MRR = sum(mrrs) / len(mrrs)
    mrecall_score = score_mrecall(preds)
    recall_score = score_recall(preds)
    precision_score = score_precision(preds, topk)
    if has_gold_id:
        print(f'MRecall: {100*mrecall_score:.2f} | Recall: {100*recall_score:.2f} | Precision: {100*precision_score:.2f} | mAP: {mAP:.4f} | nDCG: {nDCG:.4f} | MRR: {MRR:.4f}')
        return '%2.2f'%(100*mrecall_score), '%2.2f'%(100*recall_score), '%2.2f'%(100*precision_score), '%2.4f'%(mAP), '%2.4f'%(nDCG), '%2.4f'%(MRR), qrel, run
    else:
        print(f'MRecall: {100*mrecall_score:.2f} | Recall: {100*recall_score:.2f} | Precision: {100*precision_score:.2f} | MRR: {MRR:.4f}')
        return '%2.2f'%(100*mrecall_score), '%2.2f'%(100*recall_score), '%2.2f'%(100*precision_score), '%2.4f'%(MRR), qrel, run
    # print(f'MRecall: {100*mrecall_score:.2f} | Recall: {100*recall_score:.2f} | Precision: {100*precision_score:.2f} | nDCG: {nDCG:.4f} | mAP: {mAP:.4f}')
    print('Average number of retrieved documents:', sum(len_docs) / len(len_docs)) 
    
    

import pytrec_eval
import json

def eval_retrieve_docs_id(retrieved_docs_path, data_path, has_gold_id=False, topk=100):
    dataset = read_jsonl(data_path)
    tok = SimpleTokenizer()
    retrieved_docs = read_jsonl(retrieved_docs_path)
    len_docs = []
    # precisions = []
    preds = []
    qrel = {}
    run = {}
    mrrs = []
    q2inst = {}
    for inst in dataset:
        q2inst[inst['question_text']] = inst
    
    qid = 0
    # for gold_inst, docs in zip(dataset, retrieved_docs):
    for docs in retrieved_docs:
        gold_inst = q2inst[docs['question']]
        pred_inst = []
        mrr_inst = {}        
        
        if has_gold_id:
            qrel[str(qid)] = {}
            if 'positive_ctxs' in gold_inst:
                gold_clusters = [doc for doc in gold_inst['positive_ctxs']]
            elif 'ground_truths' in gold_inst:
                gold_clusters = [doc for doc in gold_inst['ground_truths']]
            else:
                raise ValueError('No positive_ctxs or ground_truths')
            for cluster in gold_clusters:
                gold_ids_per_cluster = [doc['id'] for doc in cluster]
                pred_inst.append([])
                len_docs.append(len(docs['ctxs']))
                for rank, doc in enumerate(docs['ctxs'][:topk]):
                    pred = (doc['id'] in set(gold_ids_per_cluster))
                    pred_inst[-1].append(pred)
                    
                    # handle MRR if prediction correct
                    for gold_id in gold_ids_per_cluster:
                        if doc['id'] == gold_id:
                            mrr_inst[gold_id] = 1 / (rank + 1)
                    
                for gold_id in gold_ids_per_cluster:
                    qrel[str(qid)][gold_id] = 1
                    if gold_id not in mrr_inst: # if not found
                        mrr_inst[gold_id] = 0
                        
            run[str(qid)] = {}
            for rank, doc in enumerate(docs['ctxs'][:topk]):
                run[str(qid)][doc['id']] = 1 / (rank + 1)
        else:
            for answer in gold_inst['answer_list']:
                pred_inst.append([])
                len_docs.append(len(docs['ctxs']))
                for doc in docs['ctxs'][:topk]:
                    pred = has_answer(answer['aliases'], doc['text'], tok)
                    pred_inst[-1].append(pred)
        preds.append(pred_inst)  
        qid += 1  
        # compute average MRR for this instance
        _mrr = 0
        for _, value in mrr_inst.items():
            _mrr += value
        if len(mrr_inst) > 0:
            _mrr /= len(mrr_inst)
        mrrs.append(_mrr)
        
        # precisions.append(sum([any(doc[pred_str]) for doc in docs[:topk]]) / topk)
    if has_gold_id:
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
        scores_dict = evaluator.evaluate(run)
        ndcgs = []
        maps = []
        for qid in scores_dict:
            ndcgs.append(scores_dict[qid]['ndcg'])
            maps.append(scores_dict[qid]['map'])
        nDCG = sum(ndcgs) / len(ndcgs)
        mAP = sum(maps) / len(maps)
        
    MRR = sum(mrrs) / len(mrrs)
    mrecall_score = score_mrecall(preds)
    recall_score = score_recall(preds)
    precision_score = score_precision(preds, topk)
    if has_gold_id:
        print(f'MRecall: {100*mrecall_score:.2f} | Recall: {100*recall_score:.2f} | Precision: {100*precision_score:.2f} | mAP: {mAP:.4f} | nDCG: {nDCG:.4f} | MRR: {MRR:.4f}')
        return '%2.2f'%(100*mrecall_score), '%2.2f'%(100*recall_score), '%2.2f'%(100*precision_score), '%2.4f'%(mAP), '%2.4f'%(nDCG), '%2.4f'%(MRR)
    else:
        print(f'MRecall: {100*mrecall_score:.2f} | Recall: {100*recall_score:.2f} | Precision: {100*precision_score:.2f} | MRR: {MRR:.4f}')
        return '%2.2f'%(100*mrecall_score), '%2.2f'%(100*recall_score), '%2.2f'%(100*precision_score), '%2.4f'%(MRR)
    # print(f'MRecall: {100*mrecall_score:.2f} | Recall: {100*recall_score:.2f} | Precision: {100*precision_score:.2f} | nDCG: {nDCG:.4f} | mAP: {mAP:.4f}')
    print('Average number of retrieved documents:', sum(len_docs) / len(len_docs)) 
    
    
### Evaluating whether repeat input documents
def eval_retrieve_docs_for_repeats(retrieved_docs_path, data_path, topk=100):
    dataset = read_jsonl(data_path)
    tok = SimpleTokenizer()
    retrieved_docs = read_jsonl(retrieved_docs_path)
    len_docs = []
    preds = []
    
    for gold_inst, docs in zip(dataset, retrieved_docs):
        pred_inst = []
        # valid_answers = process_reference(gold_inst)
        # print(valid_answers)
        # for ans in valid_answers:
        gold_indices = [doc['id'] for doc in gold_inst['input_negative_ctxs']]

        for gold_index in gold_indices:
            pred_inst.append([])
            len_docs.append(len(docs['ctxs']))
            for doc in docs['ctxs'][:topk]:
                # print(gold_index, doc['id'])
                pred = gold_index == doc['id']
                pred_inst[-1].append(pred)
        
        preds.append(pred_inst)    

    mrecall_score = score_mrecall(preds)
    recall_score = score_recall(preds)
    precision_score = score_precision(preds, topk)
    print(f'MRecall: {100*mrecall_score:.2f} | Recall: {100*recall_score:.2f} | Precision: {100*precision_score:.2f}')
    print('Average number of retrieved documents:', sum(len_docs) / len(len_docs)) 
    return '%2.2f'%(100*mrecall_score), '%2.2f'%(100*recall_score), '%2.2f'%(100*precision_score)



from beir import util, LoggingHandler
import logging
import pathlib, os
import pytrec_eval
from typing import Optional, List, Dict, Tuple


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def mrr(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"] / len(qrels), 5)
        logging.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    return MRR


def evaluate(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
    ignore_identical_ids: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    if ignore_identical_ids:
        logger.info(
            "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
        )
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

    for eval in [ndcg, _map, recall, precision]:
        logger.info("\n")
        for k in eval.keys():
            logger.info(f"{k}: {eval[k]:.4f}")

    return ndcg, _map, recall, precision

