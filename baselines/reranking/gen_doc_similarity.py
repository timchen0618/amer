import logging
import argparse
import json
from tqdm import tqdm

import numpy as np
import torch

from sentence_transformers import SentenceTransformer, util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def read_jsonl(filename):
    with open(filename, 'r') as fin:
        return [json.loads(line) for line in fin]
    
class DocSimilarity:
    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.saved_similarities = [] # list of document similarities (100*100 matrices)
        self.num_docs = args.num_docs

    @torch.no_grad()
    def similarity(self, documents):
        assert len(documents) == self.num_docs, len(documents)
        if len(documents) != self.num_docs:
            print('!=self.num_docs', len(documents))

        self.saved_similarities.append(np.zeros((self.num_docs, self.num_docs)))
        # all_vecs = self.model.encode(documents, to_numpy=False)
        
        query_embeddings = self.model.encode(documents)
        doc_embeddings = self.model.encode(documents)
        # (2, 1024) (2, 1024)

        similarities = util.cos_sim(query_embeddings, doc_embeddings)
        print(similarities.shape)
        for j in range(similarities.shape[1]-1):
            for k in range(j+1, similarities.shape[1]):
                assert similarities[j, k] - similarities[k, j] < 1e-6, (j, k, similarities[j, k], similarities[k, j])
                    
        self.saved_similarities[-1] = similarities.cpu().numpy()
            
    def clear_similarity(self):
        self.saved_similarities = []

def main(args):

    tokenizer = None
    model = SentenceTransformer(args.model_name, trust_remote_code=True).cuda()
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    if args.base_retriever == 'inf':
        # base
        rootdir = '/scratch/hc3337/projects/autoregressive/results/base_retrievers/inf/'
        data_types = ['ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs', 'dev_data_gt_qampari_corpus_5_to_8_ctxs']
    elif args.base_retriever == 'qampari_stage1':
        # stage 1 qampari
        rootdir = '/scratch/hc3337/projects/autoregressive/results/llama-1b/qampari_inf/toy_qemb_from_nq/'
        data_types = ['retrieval_out_dev_qampari_5_to_8_max_new_tokens_1']
    elif args.base_retriever == 'nq_stage2':
        # stage 2 nq
        rootdir = '/scratch/hc3337/projects/autoregressive/results/llama-1b/nq_inf/toy_contrastive/'
        data_types = ['retrieval_out_dev_ambiguous_qe_max_new_tokens_1']
    
    
    doc_sim = DocSimilarity(model, tokenizer, device, args)

    for data_type in data_types:
        retrieval_results = read_jsonl(f'{rootdir}/{data_type}.jsonl')
        for ret_inst in retrieval_results:
            doc_sim.similarity([doc['text'] + ' ' + doc['title'] if 'title' in doc else doc['text'] for doc in ret_inst['ctxs'][:args.num_docs]])
        
        doc_sim.saved_similarities = [sim.reshape(-1, sim.shape[1], sim.shape[1]) for sim in doc_sim.saved_similarities]
        similarities = np.concatenate(doc_sim.saved_similarities, axis=0)
        print(similarities.shape)
        
        # for i in range(similarities.shape[0]):
        #     for j in range(similarities.shape[1]-1):
        #         for k in range(j+1, similarities.shape[1]):
                    # assert similarities[i, j, k] == similarities[i, k, j]
                    # _sim = (similarities[i, k, j] + similarities[i, j, k]) / 2
                    # similarities[i, j, k] = _sim
                    # similarities[i, k, j] = _sim
                    
        
        np.save(f'{rootdir}/{data_type}_similarities.npy', similarities)
        doc_sim.clear_similarity()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="infly/inf-retriever-v1-1.5b")
    parser.add_argument("--num_docs", type=int, default=500)
    parser.add_argument("--base_retriever", type=str, default='inf', choices=['inf', 'qampari_stage1', 'nq_stage2'])
    args = parser.parse_args()
    
    main(args)
    
    #  python gen_doc_similarity.py --compute --corpus sphere --retriever bm25
    # CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python reranking/gen_doc_similarity.py --compute --retriever bm25