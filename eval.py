from src.eval_utils import read_jsonl, eval_retrieve_docs, eval_retrieve_docs_for_repeats, evaluate, mrr

# root = '/scratch/cluster/hungting/projects/Multi_Answer/contriever/outputs/contriever_msmarco_nq/'
# root = '/scratch/cluster/hungting/projects/Multi_Answer/mteb_retriever/outputs/'

# root = '/scratch/cluster/hungting/projects/autoregressive/results/ambiguous_qe_inf/'
root = '/scratch/hc3337/projects/autoregressive/results/ambiguous_qe_inf/'
# root = '/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/nq_embeddings_data/'

reranking_root = '/scratch/cluster/hungting/projects/diverse_response/retrieval_outputs/qampari_2nd_stage/'
project_root = '/scratch/hc3337/projects/'
TOPK = 5
num_input = 2
all_scores = []

for num_input in [3]:
    reranking = False
    second_stage = False
    data_type = 'ambiguous_qe' # ['qampari', 'ambiguous']
    if data_type == 'qampari':
        split = 'dev_5_to_8' # ['dev', 'train', 'dev_2_to_5', 'dev_5_to_8]
    elif data_type == 'ambiguous' or data_type == 'ambiguous_qe':
        split = 'dev_2_to_5'
        
    # selected_indices = [int(l.strip('\n')) for l in open('/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/qampari_embeddings_data/large_distance_indices.txt', 'r')]
    selected_indices = None
    
    # file_list = ['inf/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.json']
    # file_list = ['toy_contrastive_from_stage2_lr2e5_ep20_temp0.05_warmup0.05/retrieval_out_dev_ambiguous_qe.jsonl']
    file_list = ['toy_contrastive_from_stage2_lr2e5_ep20_temp0.05_warmup0.05/retrieval_out_dev_ambiguous_qe_single.jsonl', 
                 'toy_contrastive_from_stage2_lr2e5_ep20_temp0.05_warmup0.05/retrieval_out_dev_ambiguous_qe.jsonl', 
                 'toy_contrastive_from_stage2_lr2e5_ep20_temp0.05_warmup0.05/retrieval_out_dev_ambiguous_qe_from_2nd_to_3rd.jsonl']


    data_mapping = {
        "qampari": {
            'dev': f'{project_root}/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl',
            'dev_5_to_8': f'{project_root}/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus_5_to_8_ctxs.jsonl',
            'train': f'{project_root}/diverse_response/data/qampari_data/train_data_gt_qampari_corpus.jsonl',
            'second_stage': f'{project_root}/diverse_response/data/qampari_data/2nd_stage_test_data/dev_data_qampari_corpus_inp{num_input}.jsonl'
        },
        "ambiguous": {
            'dev': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_no_empty_clusters.jsonl',
            'dev_2_to_5': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.jsonl',
            'train': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data',
        },
        "ambiguous_qe": {
            'dev': f'{project_root}/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev.jsonl',
            'dev_2_to_5': f'{project_root}/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.jsonl',
            'train': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data',
        },
    }
    data_path = data_mapping[data_type][split]
        
    if reranking:
        root = reranking_root

    
    for _file in file_list:
        scores_per_file = []
        mrecalls = []
        recalls = []
        for TOPK in [100, 10]:
            scores_per_file += [0]
            _input = root + _file
            print('--'*20)
            print(_file)
            scores = eval_retrieve_docs(
                _input,
                data_path,
                has_gold_id=False,
                topk=TOPK,
                selected_indices=selected_indices
            )
            qrels = scores[-2]
            runs = scores[-1]
            scores = scores[:-2]
            if second_stage:
                repeat_scores = eval_retrieve_docs_for_repeats(
                    _input,
                    data_path,
                    topk=TOPK
                )
                scores = list(scores) + list(repeat_scores)
            else:
                scores = list(scores)
            scores_per_file += scores
            print(len(scores))
            mrecalls.append(scores[0])
            recalls.append(scores[1])
            
        # all_scores.append(scores_per_file[1:])
            
        all_scores.append([scores_per_file[1], scores_per_file[2], scores_per_file[4], scores_per_file[6], scores_per_file[7], scores_per_file[9]])
        
        
        # ndcg, _, _, _ = evaluate(
        #     qrels,
        #     runs,
        #     k_values=[1, 3, 5, 10, 100],
        #     ignore_identical_ids=False
        # )
        # mrr(
        #     qrels,
        #     runs,
        #     k_values=[1, 3, 5, 10, 100],
        # )
        # all_scores.append([mrecalls[1], recalls[1], round(ndcg[f"NDCG@100"], 4), mrecalls[0], recalls[0], round(ndcg[f"NDCG@10"], 4)])

import pandas as pd
if second_stage:
    df = pd.DataFrame(all_scores, columns=['MRecall', 'Recall', 'Precision', 'mAP', 'nDCG', 'MRR', 'MRecall-Repeat', 'Recall-Repeat', 'Precision-Repeat', '-', 'MRecall', 'Recall', 'Precision', 'mAP', 'nDCG', 'MRR', 'MRecall-Repeat', 'Recall-Repeat', 'Precision-Repeat'])
else:
    df = pd.DataFrame(all_scores, columns=['MRecall@100', 'Recall@100', 'nDCG@100', 'MRecall@10', 'Recall@10', 'nDCG@10'])
df.to_csv('qampari_dev_2nd_stage_scores.csv', index=False)