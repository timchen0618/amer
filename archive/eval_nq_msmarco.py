from src.eval_utils import evaluate, mrr, read_jsonl
from beir.datasets.data_loader import GenericDataLoader

data_name = 'nq'

data_path = f'/scratch/cluster/hungting/projects/autoregressive/data/{data_name}'
split = 'dev'
#### Provide the data_path where scifact has been downloaded and unzipped
actual_split = 'test' if split == 'dev' else split
if data_name == 'nq' and actual_split == 'train':
    data_path = f'/scratch/cluster/hungting/projects/autoregressive/data/nq/nq-train'
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=actual_split) 

k_values = [1, 3, 5, 10, 100, 1000]
# rootdir = '/datastor1/hungting/retrieval_outputs/mteb_retriever/'
# rootdir = '/scratch/cluster/hungting/projects/Multi_Answer/contriever/outputs/'
rootdir = '/scratch/cluster/hungting/projects/autoregressive/results'


for train_name in ['nq_inf', 'nq_cont', 'nq_stella']:
    for suffix in ['toy_contrastive_from_nq']:
        print('-' * 20)
        print('train_name:', train_name, 'suffix:', suffix)
        raw_outputs = read_jsonl(f'{rootdir}/{train_name}/{suffix}/retrieval_out_dev_{data_name}.jsonl')

        results = {}
        q2docs = {}
        for i, output in enumerate(raw_outputs):
            q2docs[output['question']] = output['ctxs']
            
        for qid, question in queries.items():
            results[qid] = {}
            for doc in q2docs[question]:
                results[qid][doc['id']] = float(doc['score'])

        print(results.keys())
        #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
        ndcg, _map, recall, precision = evaluate(qrels, results, k_values)
        _mrr = mrr(qrels, results, k_values)