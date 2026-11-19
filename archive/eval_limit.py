from src.eval_utils import evaluate, mrr, read_jsonl
from beir.datasets.data_loader import GenericDataLoader

data_name = 'limit'

for data_name in ['limit', 'limit-small']:
    if data_name == 'limit':
        data_path = f'/scratch/hc3337/projects/autoregressive/data/limit/data/limit'
    elif data_name == 'limit-small':
        data_path = f'/scratch/hc3337/projects/autoregressive/data/limit/data/limit-small'

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split='test') 

    k_values = [2, 10, 20, 100]

    rootdir = '/scratch/hc3337/projects/autoregressive/results/base_retrievers/inf/'
    raw_outputs = read_jsonl(f'{rootdir}/{data_name}_retrieved.jsonl')

    # rootdir = 'results/llama-1b/nq_inf/toy_contrastive/checkpoint_70000'
    # raw_outputs = read_jsonl(f'{rootdir}/retrieval_out_dev_{data_name}_max_new_tokens_1.jsonl')


    results = {}
    q2docs = {}
    for i, output in enumerate(raw_outputs):
        q2docs[output['question']] = output['ctxs']
        
    for qid, question in queries.items():
        results[qid] = {}
        for doc in q2docs[question]:
            results[qid][doc['id']] = float(doc['score'])

    # print(results.keys())
    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
    ndcg, _map, recall, precision = evaluate(qrels, results, k_values)
    # _mrr = mrr(qrels, results, k_values)