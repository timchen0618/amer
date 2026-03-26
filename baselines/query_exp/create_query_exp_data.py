import json
from tqdm import trange, tqdm

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
        fout.write(json.dumps(data, indent=4))

project_dir = '/path/to/project'
rootdir = f'{project_dir}/data/generated_data'
nq_rootdir = f'{project_dir}/data/nqformat_data'
pers_rootdir = f'{project_dir}/data/gen_perspectives'
suffix = '_perspectives_wostance.jsonl'

# for data_type in ['arguana_generated', 'kialo', 'opinionqa']:
#     new_data = []
#     data = read_jsonl(f'{rootdir}/{data_type}_1k.jsonl')
    # perspectives = read_jsonl(f'{pers_rootdir}/{data_type}_1k{suffix}')
    # assert len(data) == len(perspectives)
    # _id = 0
    # for i in range(len(data)):
    #     assert data[i]['question'] == perspectives[i]['question']
    #     for p in perspectives[i]['perspectives']:
    #         # print(p['name'], p['text'])
    #         if type(p['text']) == dict:
    #             p['text'] = list(p['text'].values())[0]
    #         new_data.append({
    #             'org_id': data[i]['id'],
    #             'id': _id,
    #             'org_q': data[i]['question'],
    #             'perspective': p['name'],
    #             'text': p['text'],
    #             'question': data[i]['question'] + ' ' + p['text'],
    #             'input': data[i]['question'] + ' ' + p['text']
    #         })
    #         _id += 1
            
    # write_jsonl(f'{rootdir}/{data_type}_1k_query_exp.jsonl', new_data)
    
    # for inst in new_data:
    #     inst['ctxs'] = []
    #     inst['answers'] = []
    
    # write_json(f'{nq_rootdir}/{data_type}_1k_query_exp.json', new_data)
    
    
for data_type in ['arguana_generated', 'kialo', 'opinionqa']:
    new_data = []
    data = read_jsonl(f'{rootdir}/{data_type}_1k.jsonl')
    # assert len(data) == len(perspectives)
    _id = 0
    for i in trange(len(data)):
        if len(data[i]['perspectives']) > 2:
            continue
        for p in data[i]['perspectives']:
            new_data.append({
                'org_id': data[i]['id'],
                'id': _id,
                'org_q': data[i]['question'],
                'perspective': p,
                'question': p,
                'input': p
            })
            _id += 1
            
    write_jsonl(f'{rootdir}/{data_type}_1k_syco.jsonl', new_data)
    
    for inst in new_data:
        inst['ctxs'] = []
        inst['answers'] = []
    
    write_json(f'{nq_rootdir}/{data_type}_1k_syco.json', new_data)