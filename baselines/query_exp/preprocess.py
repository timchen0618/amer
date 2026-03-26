import json

def read_jsonl(filename):  
    data = []
    with open(filename, 'r') as fin:
            for line in fin:
                data.append(json.loads(line))
    return data

def write_jsonl(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            fout.write(json.dumps(d) + '\n')

def write_json(filename, data):
    with open(filename, 'w') as fout:
        json.dump(data, fout, indent=4)


rootdir = '/scratch/hc3337/projects/autoregressive/baselines/query_exp/'
outdir = '/scratch/hc3337/projects/diverse_response/data/qampari_data/nqformat_data/'
data_types = ['ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs', 'dev_data_gt_qampari_corpus_5_to_8_ctxs']

for data_type in data_types:
    data = read_jsonl(f'{rootdir}/{data_type}_query_exp.jsonl')
    print(data[0].keys())
    for inst in data:
        inst['ctxs'] = []
        inst['answers'] = ['']
    write_json(f'{outdir}/{data_type}_query_exp.json', data)