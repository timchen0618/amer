from src.utils import collect_retrieval_results, read_jsonl, write_jsonl
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_DOCS=100
corpus = 'sphere' # ['wiki', 'sphere', 'arguana']
retriever = 'tart' # ['contriever', 'google', 'dpr', 'bm25', 'tart']
data_types = ['arguana_generated', 'kialo', 'opinionqa']

raw_results, rootdir = collect_retrieval_results(corpus, retriever, data_types, query_exp=False, rerank=False)
qe_results, _ = collect_retrieval_results(corpus, retriever, data_types, query_exp=True, rerank=False)

assert len(raw_results) == len(qe_results), (len(raw_results), len(qe_results))
assert len(raw_results) == len(data_types), (len(raw_results), len(data_types))
# for i in range(len(raw_results)):
#     assert len(raw_results[i]) == len(qe_results[i])
    
for data_type, data, qe_data in zip(data_types, raw_results, qe_results):
    logger.info(f'Processing {data_type}')
    query2doc_sets = {}  # query mapped to a set of docs, where each set come from a different query expansion 
    query2docs = {}      # the actual container for the docs, combining the sets from query2doc_sets (from each qe results)
    for inst in qe_data: # group docs based on question
        assert 'ctxs' in inst
        if inst['org_q'].replace('\"', '') not in query2doc_sets:
            query2doc_sets[inst['org_q'].replace('\"', '')] = []
            # print(inst['org_q'])
        query2doc_sets[inst['org_q'].replace('\"', '')].append(inst['ctxs'])
        
    for k, v in query2doc_sets.items():  # take turns taking top documents from each qe set
        query2docs[k] = []
        for i in range(len(v[0])):
            for docs in v:
                query2docs[k].append(docs[i])
            if len(query2docs[k]) >= NUM_DOCS:
                query2docs[k] = query2docs[k][:NUM_DOCS]
                break
            
        assert len(query2docs[k]) == NUM_DOCS

    for inst in data:
        inst['ctxs'] = query2docs[inst['question'].replace('\"', '')]
    
    write_jsonl(f'{rootdir}/{data_type}_1k_query_exp_processed.jsonl', data)
    logger.info(f'Finished processing {data_type}')