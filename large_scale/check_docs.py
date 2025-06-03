from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

root_dir = Path("/datastor1/hungting/retrieval_outputs/bm25")

num_ctxs = []
for data_name in ['msmarco_document']:
    data = read_jsonl(root_dir / f"{data_name}_retrieved_train.jsonl")
    for inst in tqdm(data):
        num_ctxs.append(len(inst['ctxs']))

print(np.mean(np.array(num_ctxs)))
print(np.max(np.array(num_ctxs)))
print(np.min(np.array(num_ctxs)))
