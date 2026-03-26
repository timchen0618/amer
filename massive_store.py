from pathlib import Path
from datasets import load_dataset
import csv
from tqdm import tqdm

def write_tsv(data, path):
    with open(path, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(data)

def read_tsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return [row for row in reader]

dataset_name = "rulins/MassiveDS-140B"

# dataset = load_dataset(dataset_name, split="train")
dataset = load_dataset(dataset_name, split="train", streaming=True)
tsv_data = [['id', 'text', 'title']]
write_tsv(tsv_data, '/datastor1/hungting/MassiveDS-140B/massive_ds_140b.tsv')
for i, example in tqdm(enumerate(dataset)):
    if i <= 2495327:
        continue
    tsv_data.append([str(i), example['text'], ''])
    write_tsv([tsv_data[-1]], '/datastor1/hungting/MassiveDS-140B/massive_ds_140b.tsv')


# data = read_tsv('/datastor1/hungting/MassiveDS-140B/massive_ds_140b.tsv')
# print(len(data))




