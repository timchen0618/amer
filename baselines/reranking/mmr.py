import logging
import argparse
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def write_jsonl(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            fout.write(json.dumps(d) + '\n')
            
def read_jsonl(filename):
    with open(filename, 'r') as fin:
        return [json.loads(line) for line in fin]
    
class Reranker:
    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device 
        self.args = args
        self.similarity_matrix = None
        self._lambda = args._lambda
        self.max_score = 1

    def rerank(self, retrieval_results):
        assert 'ctxs' in retrieval_results[0]
        
        # compute max score
        self.max_scores = []
        for inst in retrieval_results:
            max_score = 0
            for doc in inst['ctxs']:
                max_score = max(max_score, float(doc['score']))
            self.max_scores.append(max_score)
        print('max_scores:', self.max_scores)
        
        data_idx = 0
        for inst in tqdm(retrieval_results):
            documents = inst['ctxs'][:self.args.num_docs]
            new_document_ids_n_scores = [(0, float(documents[0]['score']))]
            for _ in range(len(documents) - 1):
                new_document_id, mmr_score = self.add_one_document(documents, new_document_ids_n_scores, data_idx)
                new_document_ids_n_scores.append((new_document_id, mmr_score))
            inst['ctxs'] = []
            
            for id_n_score in new_document_ids_n_scores:
                doc = documents[id_n_score[0]]
                doc['score'] = id_n_score[1]
                inst['ctxs'].append(doc)
                
            data_idx += 1
        
        return retrieval_results
    
    def load_similarity_matrix(self, similarity_matrix_path):
        self.similarity_matrix = np.load(similarity_matrix_path)

    def add_one_document(self, documents, retrieved_document_ids_n_scores, data_idx):
        mmr = -100
        new_document_id = 0
        retrieved_document_ids = [i for i, _ in retrieved_document_ids_n_scores]
        for i in range(len(documents)):
            if i not in retrieved_document_ids:
                max_score = self.max_scores[data_idx] if self.max_scores[data_idx] != 0 else self.max_score
                mmr_i = self._lambda * float(documents[i]['score']) / max_score
                
                # compute maximum similarity 
                max_sim_between_docs = -100
                for j in retrieved_document_ids:
                    assert i != j
                    sim_i_j = self.similarity_matrix[data_idx][i][j]
                    if sim_i_j > max_sim_between_docs:
                        max_sim_between_docs = sim_i_j
                 
                mmr_i -= (1 - self._lambda) * max_sim_between_docs
        
                if mmr_i > mmr:
                    mmr = mmr_i
                    new_document_id = i
        return new_document_id, mmr

    def similarity(self, i, j):
        assert i != j
        return self.similarity_matrix[i, j]
    
    
def compute_max_score(retrieval_results):
    max_score = 0
    avg_scores = []
    for inst in retrieval_results:
        for doc in inst['ctxs']:
            max_score = max(max_score, float(doc['score']))
            avg_scores.append(float(doc['score']))
    return max_score, np.mean(np.array(avg_scores))

def main(args):
    logger.info(f"Reranking results for lambda={args._lambda}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = None, None
    
    reranker = Reranker(tokenizer=tokenizer, model=model, device=device, args=args)

    project_dir = '/path/to/project'
    if args.base_retriever == 'inf':
        # base
        rootdir = f'{project_dir}/results/base_retrievers/inf/'
        data_types = ['ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs', 'dev_data_gt_qampari_corpus_5_to_8_ctxs']
    elif args.base_retriever == 'qampari_stage1':
        # stage 1 qampari
        rootdir = f'{project_dir}/results/llama-1b/qampari_inf/toy_qemb_from_nq/'
        data_types = ['retrieval_out_dev_qampari_5_to_8_max_new_tokens_1']
    elif args.base_retriever == 'nq_stage2':
        # stage 2 nq
        rootdir = f'{project_dir}/results/llama-1b/nq_inf/toy_contrastive/'
        data_types = ['retrieval_out_dev_ambiguous_qe_max_new_tokens_1']
    elif args.base_retriever in ['llama-1b_qampari', 'llama-3b_qampari', 'llama-8b_qampari', 'qwen3-4b_qampari']:
        base_model = args.base_retriever.split('_')[0]
        rootdir = f'{project_dir}/results/{base_model}/ambiguous_qe_inf/single'
        data_types = ['retrieval_out_dev_qampari_5_to_8_max_new_tokens_1']
    elif args.base_retriever in ['llama-1b_ambignq', 'llama-3b_ambignq', 'llama-8b_ambignq', 'qwen3-4b_ambignq']:
        base_model = args.base_retriever.split('_')[0]
        rootdir = f'{project_dir}/results/{base_model}/ambiguous_qe_inf/single'
        data_types = ['retrieval_out_dev_ambiguous_qe_max_new_tokens_1']
    elif args.base_retriever in ['llama-1b_qampari_multi', 'llama-3b_qampari_multi', 'llama-8b_qampari_multi', 'qwen3-4b_qampari_multi']:
        base_model = args.base_retriever.split('_')[0]
        rootdir = f'{project_dir}/results/{base_model}/qampari_inf/multi'
        data_types = ['retrieval_out_dev_qampari_5_to_8_max_new_tokens_5']
    elif args.base_retriever in ['llama-1b_ambignq_multi', 'llama-3b_ambignq_multi', 'llama-8b_ambignq_multi', 'qwen3-4b_ambignq_multi']:
        base_model = args.base_retriever.split('_')[0]
        rootdir = f'{project_dir}/results/{base_model}/ambiguous_qe_inf/multi'
        data_types = ['retrieval_out_dev_ambiguous_qe_max_new_tokens_2']
    logger.info('collecting retrieval results')
    
    for data_type in data_types:
        retrieval_results = read_jsonl(f'{rootdir}/{data_type}.jsonl')
        reranker.load_similarity_matrix(f'{rootdir}/{data_type}_similarities.npy')
        reranked_results = reranker.rerank(retrieval_results)
        write_jsonl(f'{rootdir}/{data_type}_reranked_l{args._lambda}.jsonl', reranked_results)
                              
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--_lambda", type=float, default=0.9)
    parser.add_argument("--num_docs", type=int, default=500)
    parser.add_argument("--base_retriever", type=str, default='inf', choices=['inf', 'qampari_stage1', 'nq_stage2', 'qampari_hungarian_contrastive', 'qampari_contrastive_all_labels_shuffled', 'qampari_contrastive_all_labels_ordered', 'ambiguous_qe_hungarian_contrastive', 'ambiguous_qe_contrastive_all_labels_shuffled', 'ambiguous_qe_contrastive_all_labels_ordered', 'llama-1b_qampari', 'llama-3b_qampari', 'llama-8b_qampari', 'qwen3-4b_qampari', 'llama-1b_ambignq', 'llama-3b_ambignq', 'llama-8b_ambignq', 'qwen3-4b_ambignq', 'llama-1b_qampari_multi', 'llama-3b_qampari_multi', 'llama-8b_qampari_multi', 'qwen3-4b_qampari_multi', 'llama-1b_ambignq_multi', 'llama-3b_ambignq_multi', 'llama-8b_ambignq_multi', 'qwen3-4b_ambignq_multi'])
    args = parser.parse_args()
    
    main(args)