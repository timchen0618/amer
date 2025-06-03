import json

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def find_overlap(train_data, dev_data):
    train_ids = [x['question'] for x in train_data]
    dev_ids = [x['question'] for x in dev_data]
    return set(train_ids) & set(dev_ids)


train_path = '/scratch/cluster/hungting/projects/autoregressive/data/wsd/distinct/train_large.jsonl'
dev_path = '/scratch/cluster/hungting/projects/autoregressive/data/wsd/distinct/dev.jsonl'

train_data = read_jsonl(train_path)
dev_data = read_jsonl(dev_path)

overlap = find_overlap(train_data, dev_data)
print(len(overlap))










