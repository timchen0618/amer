from pathlib import Path
import json
from tqdm import tqdm
from tqdm import trange
"""
The script is used to mess with the retrieved documents, by aggregating the context chunks in different ways.
None of them work well.
"""

def mess_with_ctxs(retrieved_docs):
    new_data = []
    ctxs = []
    js = []
    topk=100
    rr_percentage=0.0
    for i in trange(len(retrieved_docs)):
        ctxs.append(retrieved_docs[i]['ctxs'][:500])

        if i % 5 == 4:
            # process the ctxs
            seen_ids = set()
            aggregated_ctxs = []
            max_len = max(len(ctx) for ctx in ctxs)
            for j in range(max_len):
                for ctx in ctxs:
                    if j < len(ctx):
                        if ctx[j]['id'] not in seen_ids:
                            seen_ids.add(ctx[j]['id'])
                            aggregated_ctxs.append(ctx[j])
                    if len(aggregated_ctxs) >= topk*rr_percentage:
                        break
                if len(aggregated_ctxs) >= topk*rr_percentage:
                    js.append(j)
                    break  
            ctx = ctxs[0]
            for j in range(js[-1], len(ctx)):
                aggregated_ctxs.append(ctx[j])
                if len(aggregated_ctxs) >= topk:
                    break
            question_str = 'question' if 'question' in retrieved_docs[i] else 'question_text'
            new_data.append({
                'ctxs': aggregated_ctxs,
                'question': retrieved_docs[i][question_str],
            })
            ctxs = []

    print('avg js ', sum(js) / len(js))
    return new_data

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]




suffix_list = ["normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep60_warmup0.05_srm10",
    "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm10",
    "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPad_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep60_warmup0.05_srm10"
]
rootdir = '/scratch/hc3337/projects/autoregressive/results/llama-1b/qampari_inf/'
for suffix in suffix_list:
    file_path = Path(rootdir) / suffix / 'retrieval_out_dev_qampari_max_new_tokens_5_before_agg.jsonl'
    retrieved_docs = read_jsonl(file_path)
    
    retrieved_docs = mess_with_ctxs(retrieved_docs)
    print('length of retrieved_docs', len(retrieved_docs))
    with open(Path(rootdir) / suffix / 'retrieval_out_dev_qampari_max_new_tokens_5_after_agg.jsonl', 'w') as file:
        for doc in retrieved_docs:
            file.write(json.dumps(doc) + '\n')