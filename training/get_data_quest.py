from pathlib import Path
from src.utils.file_utils import read_jsonl, read_tsv, write_jsonl

import csv
import sys
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

        
data_type='quest'

split='dev'
train_str = '_train' if split == 'train' else ''

# get top N retrieved documents
TOP_N = 5
rootdir = Path('/scratch/cluster/hungting/projects/diverse_response/')
doc_path = rootdir / f'retrieval_outputs/{data_type}/{data_type}{train_str}_contriever.jsonl'
top_docs = read_jsonl(doc_path)

# get bm25 retrieved documents
bm25_doc_path = rootdir / f'retrieval_outputs/{data_type}/{data_type}{train_str}_bm25.jsonl'
bm25_top_docs = read_jsonl(bm25_doc_path)

# get ground truth documents
data_path = rootdir / f'data/{data_type}_data/{split}_data.jsonl'
data = read_jsonl(data_path)

# get the title-text mapping
title_text_map = {}
corpus = read_tsv(rootdir / f'data/{data_type}_data/corpus/quest_documents.tsv')
for row in corpus:
    # 2 - title, 1 - text
    title_text_map[row[2]] = row[1]


# compute the difference 
# identify the documents that are in the ground truth but not in the retrieved documents
# these are the new ground truth
assert len(top_docs) == len(data), f'{len(top_docs)} != {len(data)}'
new_data = []
new_eval_data = []
i = 0
for top_doc, bm25_top_doc, d in zip(top_docs, bm25_top_docs, data):
    question_str = 'query' if data_type == 'quest' else 'question'
    retrieved_documents = [doc['title'] for doc in top_doc['ctxs'][:TOP_N]]
    retrieved_documents_wtexts = [{'text': title_text_map[doc['title']], 'title': doc['title']} for doc in top_doc['ctxs'][:TOP_N]]
    new_ground_truth_titles = [gt for gt in d['docs'] if gt not in retrieved_documents]
    if not new_ground_truth_titles:
        print('No new ground truth')
        continue
    if new_ground_truth_titles == d['docs']:
        print('All ground truth documents are not retrieved')
        continue

    rewritten_question = 'Question: [Question]\n\nDocuments: [Documents]'.replace('[Question]', d[question_str]).replace('[Documents]', '\n'.join([ddd['title'] + ' ' + ddd['text'] for ddd in retrieved_documents_wtexts]))  
    new_data.append({'id': i, 'question': rewritten_question, 'doc_titles': new_ground_truth_titles, 'org_q': d[question_str]})
    new_ground_truth = [{'text': title_text_map[title], 'title': title} for title in new_ground_truth_titles]
    
    # if split == 'dev':
    #     new_data[-1]['positive_ctxs'] = [new_ground_truth[0]]
    #     for i in range(1, len(new_ground_truth)):
    #         new_data.append({'id': i, 'question': question_str, 'positive_ctxs': [new_ground_truth[i]]}) 
    # else:
    new_data[-1]['positive_ctxs'] = new_ground_truth
    
    # getting hard negatives
    hard_negatives = []
    # 1. getting hard negatives from already retrieved documents
    for ret in retrieved_documents:
        if ret not in new_ground_truth_titles:
            hard_negatives.append(ret)
    
    # 2. getting hard negatives from the bm25 retrieved documents
    for ret in bm25_top_doc['ctxs'][:TOP_N]:
        if ret['title'] not in new_ground_truth_titles:
            hard_negatives.append(ret['title'])
    
    new_data[-1]['hard_negative_ctxs'] = [{'text': title_text_map[title], 'title': title} for title in hard_negatives]
    # getting normal negative ctxs
    if split == 'dev':
        new_data[-1]['negative_ctxs'] = [{'text': title_text_map[title], 'title': title} for title in hard_negatives]
        new_data[-1]['docs'] = new_ground_truth_titles
    
    
    # add eval data
    new_eval_data.append(d)
    new_eval_data[-1]['docs'] = new_ground_truth_titles

    
    i += 1
print(len(new_data))

# write the new ground truth to a jsonl file
write_jsonl(new_data, rootdir / f'data/{data_type}_data/2nd_stage/{split}_data.jsonl')
write_jsonl(new_eval_data, rootdir / f'data/{data_type}_data/2nd_stage/{split}_eval_data.jsonl')