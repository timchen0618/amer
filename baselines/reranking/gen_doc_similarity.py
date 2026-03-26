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
        
        query_embeddings = self.model.encode(documents, batch_size=4)
        doc_embeddings = self.model.encode(documents, batch_size=4)
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
    
    project_dir = '/scratch/hc3337/projects/autoregressive'
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
    # elif args.base_retriever == 'qampari_hungarian_contrastive':
    #     rootdir = f'{project_dir}/results/llama-1b/qampari_inf/normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep60_warmup0.05_srm10'
    #     data_types = ['retrieval_out_dev_qampari_max_new_tokens_5']
    # elif args.base_retriever == 'qampari_contrastive_all_labels_shuffled':
    #     rootdir = f'{project_dir}/results/llama-1b/qampari_inf/normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10'
    #     data_types = ['retrieval_out_dev_qampari_max_new_tokens_5']
    # elif args.base_retriever == 'qampari_contrastive_all_labels_ordered':
    #     rootdir = f'{project_dir}/results/llama-1b/qampari_inf/normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep60_warmup0.05_srm10'
    #     data_types = ['retrieval_out_dev_qampari_max_new_tokens_5']
    # elif args.base_retriever == 'ambiguous_qe_hungarian_contrastive':
    #     rootdir = f'{project_dir}/results/llama-1b/ambiguous_qe_inf/normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr1e-4_temp0.05_batch32_ep120_warmup0.05_srm1'
    #     data_types = ['retrieval_out_dev_ambiguous_qe_max_new_tokens_2']
    # elif args.base_retriever == 'ambiguous_qe_contrastive_all_labels_shuffled':
    #     rootdir = f'{project_dir}/results/llama-1b/ambiguous_qe_inf/normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1'
    #     data_types = ['retrieval_out_dev_ambiguous_qe_max_new_tokens_2']
    # elif args.base_retriever == 'ambiguous_qe_contrastive_all_labels_ordered':
    #     rootdir = f'{project_dir}/results/llama-1b/ambiguous_qe_inf/normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1'
    #     data_types = ['retrieval_out_dev_ambiguous_qe_max_new_tokens_2']
    elif args.base_retriever in ['llama-1b_qampari_0', 'llama-3b_qampari_0', 'llama-8b_qampari_0', 'qwen3-4b_qampari_0', 
                                 'llama-1b_qampari_1', 'llama-3b_qampari_1', 'llama-8b_qampari_1', 'qwen3-4b_qampari_1', 
                                 'llama-1b_qampari_2', 'llama-3b_qampari_2', 'llama-8b_qampari_2', 'qwen3-4b_qampari_2']:
        base_model = args.base_retriever.split('_')[0]
        _num = args.base_retriever.split('_')[-1]
        rootdir = f'{project_dir}/results/{base_model}/qampari_inf/{base_model}_single_{_num}'
        data_types = ['retrieval_out_dev_qampari_5_to_8_max_new_tokens_1']
    elif args.base_retriever in ['llama-1b_ambignq_0', 'llama-3b_ambignq_0', 'llama-8b_ambignq_0', 'qwen3-4b_ambignq_0', 
                                 'llama-1b_ambignq_1', 'llama-3b_ambignq_1', 'llama-8b_ambignq_1', 'qwen3-4b_ambignq_1', 
                                 'llama-1b_ambignq_2', 'llama-3b_ambignq_2', 'llama-8b_ambignq_2', 'qwen3-4b_ambignq_2']:
        base_model = args.base_retriever.split('_')[0]
        _num = args.base_retriever.split('_')[-1]
        rootdir = f'{project_dir}/results/{base_model}/ambiguous_qe_inf/{base_model}_single_{_num}'
        data_types = ['retrieval_out_dev_ambiguous_qe_max_new_tokens_1']
    # SS (Scheduled Sampling)
    elif args.base_retriever in ['llama-1b_qampari_multi_SS_0', 'llama-3b_qampari_multi_SS_0', 'llama-8b_qampari_multi_SS_0', 'qwen3-4b_qampari_multi_SS_0', 
                                 'llama-1b_qampari_multi_SS_1', 'llama-3b_qampari_multi_SS_1', 'llama-8b_qampari_multi_SS_1', 'qwen3-4b_qampari_multi_SS_1', 
                                 'llama-1b_qampari_multi_SS_2', 'llama-3b_qampari_multi_SS_2', 'llama-8b_qampari_multi_SS_2', 'qwen3-4b_qampari_multi_SS_2']:
        base_model = args.base_retriever.split('_')[0]
        _num = args.base_retriever.split('_')[-1]
        rootdir = f'{project_dir}/results/{base_model}/qampari_inf/{base_model}_multi_SS_{_num}'
        data_types = ['retrieval_out_dev_qampari_5_to_8_max_new_tokens_5']
    elif args.base_retriever in ['llama-1b_ambignq_multi_SS_0', 'llama-3b_ambignq_multi_SS_0', 'llama-8b_ambignq_multi_SS_0', 'qwen3-4b_ambignq_multi_SS_0', 
                                 'llama-1b_ambignq_multi_SS_1', 'llama-3b_ambignq_multi_SS_1', 'llama-8b_ambignq_multi_SS_1', 'qwen3-4b_ambignq_multi_SS_1', 
                                 'llama-1b_ambignq_multi_SS_2', 'llama-3b_ambignq_multi_SS_2', 'llama-8b_ambignq_multi_SS_2', 'qwen3-4b_ambignq_multi_SS_2']:
        base_model = args.base_retriever.split('_')[0]
        _num = args.base_retriever.split('_')[-1]
        rootdir = f'{project_dir}/results/{base_model}/ambiguous_qe_inf/{base_model}_multi_SS_{_num}'
        data_types = ['retrieval_out_dev_ambiguous_qe_max_new_tokens_2']
    # Sampling
    elif args.base_retriever in ['llama-1b_qampari_multi_sampling_0', 'llama-3b_qampari_multi_sampling_0', 'llama-8b_qampari_multi_sampling_0', 'qwen3-4b_qampari_multi_sampling_0', 
                                 'llama-1b_qampari_multi_sampling_1', 'llama-3b_qampari_multi_sampling_1', 'llama-8b_qampari_multi_sampling_1', 'qwen3-4b_qampari_multi_sampling_1', 
                                 'llama-1b_qampari_multi_sampling_2', 'llama-3b_qampari_multi_sampling_2', 'llama-8b_qampari_multi_sampling_2', 'qwen3-4b_qampari_multi_sampling_2']:
        base_model = args.base_retriever.split('_')[0]
        _num = args.base_retriever.split('_')[-1]
        rootdir = f'{project_dir}/results/{base_model}/qampari_inf/{base_model}_multi_sampling_{_num}'
        data_types = ['retrieval_out_dev_qampari_5_to_8_max_new_tokens_5']
    elif args.base_retriever in ['llama-1b_ambignq_multi_sampling_0', 'llama-3b_ambignq_multi_sampling_0', 'llama-8b_ambignq_multi_sampling_0', 'qwen3-4b_ambignq_multi_sampling_0', 
                                 'llama-1b_ambignq_multi_sampling_1', 'llama-3b_ambignq_multi_sampling_1', 'llama-8b_ambignq_multi_sampling_1', 'qwen3-4b_ambignq_multi_sampling_1', 
                                 'llama-1b_ambignq_multi_sampling_2', 'llama-3b_ambignq_multi_sampling_2', 'llama-8b_ambignq_multi_sampling_2', 'qwen3-4b_ambignq_multi_sampling_2']:
        base_model = args.base_retriever.split('_')[0]
        _num = args.base_retriever.split('_')[-1]
        rootdir = f'{project_dir}/results/{base_model}/ambiguous_qe_inf/{base_model}_multi_sampling_{_num}'
        data_types = ['retrieval_out_dev_ambiguous_qe_max_new_tokens_2']
    else:
        raise ValueError(f"Invalid base retriever: {args.base_retriever}")
    
    doc_sim = DocSimilarity(model, tokenizer, device, args)

    for data_type in data_types:
        retrieval_results = read_jsonl(f'{rootdir}/{data_type}.jsonl')
        for ret_inst in tqdm(retrieval_results):
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
    parser.add_argument("--base_retriever", type=str, default='inf')
    args = parser.parse_args()
    
    main(args)
    
    #  python gen_doc_similarity.py --compute --corpus sphere --retriever bm25
    # CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python reranking/gen_doc_similarity.py --compute --retriever bm25
