from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def compute_similarity(embeddings1, embeddings2):
    return np.diag(cosine_similarity(embeddings1, embeddings2)).mean()


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


if __name__ == "__main__":
    # rootdir = Path('output_embeddings/nq_inf')
    # embeddings = np.load(rootdir / 'out_ambiguous_inf__contrastive_dev_double.npy')
    # emb_1 = []
    # emb_2 = []
    # for i in range(embeddings.shape[0]):
    #     if i % 2 == 0:
    #         emb_1.append(embeddings[i])
    #     else:
    #         emb_2.append(embeddings[i])
    # emb_1 = np.array(emb_1)
    # emb_2 = np.array(emb_2)
    # print(compute_similarity(emb_1, emb_2))
    
    rootdir = Path('results/ambiguous_inf/toy_contrastive_dev_ep30_temp0.2')
    ret_docs_1 = read_jsonl(rootdir / 'retrieval_out_dev_ambiguous_single.jsonl')
    ret_docs_2 = read_jsonl(rootdir / 'retrieval_out_dev_ambiguous_from_2nd_to_3rd.jsonl')
    
    js_list = []
    for i in range(len(ret_docs_1)):
        js = jaccard_similarity([doc['id'] for doc in ret_docs_1[i]['ctxs']], [doc['id'] for doc in ret_docs_2[i]['ctxs']])
        js_list.append(js)
    print(np.mean(js_list))
        
