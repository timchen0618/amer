from pathlib import Path
import csv
import json
import sys
from tqdm import tqdm
import os

path_to_chunks = sys.argv[1]

rootdir = (Path(path_to_chunks) / "wikipedia_chunks") / "chunks_v5"

def append_tsv(data, file_name):
    with open(file_name, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(data)

if os.path.exists(Path(path_to_chunks) / "chunks_v5.tsv"):
    os.remove(Path(path_to_chunks) / "chunks_v5.tsv")
append_tsv([["id", "text", "title"]], Path(path_to_chunks) / "chunks_v5.tsv")
for file in tqdm(rootdir.glob("*.jsonl")):
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            append_tsv([[data["id"], data["contents"], data["meta"]["title"]]], Path(path_to_chunks) / "chunks_v5.tsv")
            