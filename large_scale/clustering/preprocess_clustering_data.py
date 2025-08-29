from pathlib import Path
from tqdm import tqdm

import json
import random
import csv
import sys
csv.field_size_limit(sys.maxsize)

def read_jsonl(file_path):
    return [json.loads(l) for l in open(file_path, "r")]

def write_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for l in data:
            f.write(json.dumps(l) + "\n")

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_corpus(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row[0] == "id":
                continue
            data.append(row)
    return data

def read_data(file_path):
    return [l for l in open(file_path, "r")]

def write_data(data, file_path):
    with open(file_path, "w") as f:
        for l in data:
            f.write(l)

def check_contexts(file_path, num_contexts=500):
    # read JSONL
    with open(file_path, "r") as f:
        for l in tqdm(f):
            data = json.loads(l)
            contexts = data["ctxs"]
            if len(contexts) != num_contexts:
                print(f"Contexts length is not {num_contexts}: {len(contexts)}")
                return False
    return True


def split_results(data, num_shards=50):
    # split data into num_shards shards, following order
    # each shard contains 1/num_shards of the data
    # get a list of indices that are evenly distributed
    indices = list(range(len(data)))
    shard_size = len(indices) // num_shards
    shards = [indices[i:i+shard_size] for i in range(0, len(indices), shard_size)]
    out_data = []
    for shard in shards:
        out_data.append([data[i] for i in shard])
    return out_data

def merge_results(inf_data, cont_data, bm25_data):
    # merge data_list into one list
    for i in range(len(inf_data)):
        inf_data[i]["ctxs"] += cont_data[i]["ctxs"]
        inf_data[i]["ctxs"] += bm25_data[i]["ctxs"]
    return inf_data

################################################################################
# # merge and split
################################################################################

def merge_and_split():
    question_type = "eli5"

    cont_root_dir = f"/scratch/hc3337/MassiveDS-140B/contriever_datastore/retrieved_results/{question_type}"
    inf_root_dir = f"/scratch/hc3337/MassiveDS-140B/inf_datastore/retrieved_results/{question_type}"
    bm25_root_dir = f"/scratch/hc3337/MassiveDS-140B/bm25_datastore/retrieved_results/{question_type}"

    # check if the data contains top 500 documents
    # check_contexts(f"{inf_root_dir}/{question_type}_merged_top500_8shards.jsonl")
    # check_contexts(f"{cont_root_dir}/{question_type}_merged_top500_8shards.jsonl")
    check_contexts(f"{bm25_root_dir}/{question_type}_merged_top500_8shards.jsonl")
    
    # merge the data
    inf_data = read_jsonl(f"{inf_root_dir}/{question_type}_merged_top500_8shards.jsonl")
    cont_data = read_jsonl(f"{cont_root_dir}/{question_type}_merged_top500_8shards.jsonl")
    bm25_data = read_jsonl(f"{bm25_root_dir}/{question_type}_merged_top500_8shards.jsonl")
    print('done loading data, merging...')
    merged_data = merge_results(inf_data, cont_data, bm25_data)
    write_jsonl(merged_data, f"/scratch/hc3337/projects/autoregressive/large_scale/data/{question_type}/{question_type}_inf+contriever+bm25_top500.jsonl")
    
    # check the context length for the merged data (should be 1000)
    print('finished merging, checking context length...')
    check_contexts(f"/scratch/hc3337/projects/autoregressive/large_scale/data/{question_type}/{question_type}_inf+contriever+bm25_top500.jsonl", num_contexts=1500)
    
    # split the data into 50 shards
    print('splitting data...')
    data = read_data(f"/scratch/hc3337/projects/autoregressive/large_scale/data/{question_type}/{question_type}_inf+contriever+bm25_top500.jsonl")
    out_data_list = split_results(data)
    for i, out_data in enumerate(out_data_list):
        write_data(out_data, f"/scratch/hc3337/projects/autoregressive/large_scale/data/{question_type}/{question_type}_inf+contriever+bm25_top500_split_{i}.jsonl")
    
################################################################################
# # create negatives
################################################################################

def create_negatives():
    corpus = read_corpus('/scratch/hc3337/MassiveDS-140B/massive_ds_140b.tsv')
    print('finished reading corpus')
    for question_type in ["eli5", "researchy_questions", "ner_retrieve"]:
        data = read_json(f'/scratch/hc3337/projects/Multi_Answer/mteb_retriever/data/{question_type}.json')
        write_jsonl(data, f'/scratch/hc3337/projects/autoregressive/data_creation/raw_data/{question_type}_train_question_only.jsonl')
        
        len_corpus = len(corpus)
        out_data = []
        for _ in range(len(data)):
            random_idx = random.randint(0, len_corpus - 1)
            
            out_data.append({"id": corpus[random_idx][0], "text": corpus[random_idx][1], "title": corpus[random_idx][2] if len(corpus[random_idx]) > 2 else ""})
        
        write_jsonl(out_data, f'/scratch/hc3337/projects/autoregressive/data_creation/raw_data/{question_type}_train_random_negatives.jsonl')
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default="create_negatives")
    args = parser.parse_args()
    command = args.command
    if command == "create_negatives":
        create_negatives()
    elif command == "merge_and_split":
        merge_and_split()