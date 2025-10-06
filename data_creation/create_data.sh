#!/bin/bash
python process_chunks.py /scratch/hc3337/wikipedia_chunks/subset
python generate_embeddings.py --corpus_file /scratch/hc3337/wikipedia_chunks/subset/chunks_v5.tsv
python create_input_dataset.py contrastive
python combine_datasets.py
