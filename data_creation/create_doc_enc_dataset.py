"""
Create training datasets with raw document token IDs for the doc-encoder-trained model.

Each example in the output dataset contains:
  - input_ids:              (query_seq_len,)          tokenized query
  - attention_mask:         (query_seq_len,)
  - input_document_ids:     (num_pos + num_neg, doc_seq_len)   first half = positives, second half = negatives
  - attention_mask_document:(num_pos + num_neg, doc_seq_len)
  - num_positives:          int   number of positive docs (= num_neg; needed when merging lengths)

This format is consumed by ContrastiveTrainCollatorDocEncTrained in src/dataset.py.

Usage:
        # Single length
        python data_creation/create_doc_enc_dataset.py \
            --data_name qampari \
            --split train \
            --length 3 \
            --model_id infly/inf-retriever-v1-1.5b \
            --corpus_tsv /scratch/hc3337/wikipedia_chunks/chunks_v5.tsv \
            --out_dir training_datasets/doc_enc_trained

        # One or more lengths (merged into one dataset)
        python data_creation/create_doc_enc_dataset.py \
            --data_name ambiguous_qe \
            --split train \
            --lengths 2 3 4 5 \
            --model_id infly/inf-retriever-v1-1.5b \
            --corpus_tsv /scratch/hc3337/wikipedia_chunks/chunks_v5.tsv \
            --out_dir training_datasets/doc_enc_trained
"""

import argparse
import json
import random
import csv
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def read_tsv(path, max_rows=3_000_000):
    data = []
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader, desc=f"reading {Path(path).name}"):
            if row[0] == "id":
                continue
            data.append(row)
            if len(data) >= max_rows:
                break
    return data


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_queries(data, tokenizer, model_id, max_length):
    """Tokenize questions using the same prompt template as create_input_dataset.py."""
    if 'inf-retriever' in model_id:
        instruction_template = "Instruct: "
        response_template = ""
    elif 'Llama-3' in model_id or 'Llama-3.1' in model_id or 'Llama-3.2' in model_id:
        instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif 'Qwen' in model_id:
        instruction_template = "<|im_start|>user\n"
        response_template = "<|im_end|>\n<|im_start|>assistant\n"
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    prompt = f"{instruction_template} Given a web search query, retrieve relevant passages that answer the query \nQuery: [QUERY]{response_template}".strip('\n')

    results = []
    for inst in tqdm(data, desc="tokenizing queries"):
        question = inst.get('question') or inst.get('question_text')
        text = prompt.replace('[QUERY]', question)
        enc = tokenizer(text, padding='max_length', truncation=True,
                        max_length=max_length, return_tensors='pt')
        results.append({
            'input_ids': enc['input_ids'][0].tolist(),
            'attention_mask': enc['attention_mask'][0].tolist(),
        })
    return results


def tokenize_documents(docs, tokenizer, max_length):
    """Tokenize a list of {'title': ..., 'text': ...} dicts."""
    results = []
    texts = [d.get('title', '') + ' ' + d['text'] for d in docs]
    for i in tqdm(range(0, len(texts), 256), desc="tokenizing documents"):
        batch = texts[i:i + 256]
        enc = tokenizer(batch, padding='max_length', truncation=True,
                        max_length=max_length, return_tensors='pt')
        for j in range(len(batch)):
            results.append({
                'input_ids': enc['input_ids'][j].tolist(),
                'attention_mask': enc['attention_mask'][j].tolist(),
            })
    return results


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def is_same_doc(ctx, candidate):
    """Return True if candidate overlaps with a known positive context."""
    if isinstance(ctx, list):
        return any(is_same_doc(c, candidate) for c in ctx)
    if isinstance(ctx, dict):
        return (ctx.get('title') and ctx.get('title') == candidate.get('title')) \
               or ctx.get('text') == candidate.get('text')
    return False

def sample_negatives(inst, corpus, length):
    """Sample `length` random negatives that don't overlap with positives."""
    positives = inst.get('positive_ctxs') or inst.get('ground_truths') or inst.get('ctxs')
    negatives = []
    while len(negatives) < length:
        candidate = random.choice(corpus)
        if not is_same_doc(positives, candidate):
            negatives.append(candidate)
    return negatives


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', required=True,
                        choices=['ambiguous_qe', 'qampari'],
                        help='Dataset name')
    parser.add_argument('--split', required=True, choices=['train', 'dev'])
    parser.add_argument('--lengths', type=int, nargs='+', required=True,
                        help='One or more lengths to include/merge (e.g. --lengths 3  or  --lengths 2 3 4 5)')
    parser.add_argument('--model_id', default='infly/inf-retriever-v1-1.5b',
                        help='HuggingFace model ID (used for tokenization)')
    parser.add_argument('--corpus_tsv',
                        default='/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv',
                        help='Path to corpus TSV (id, text, title)')
    parser.add_argument('--data_dir',
                        default='/scratch/hc3337/projects/autoregressive/data_creation/raw_data',
                        help='Directory containing the raw JSONL data files')
    parser.add_argument('--out_dir',
                        default='training_datasets/doc_enc_trained',
                        help='Output directory for HuggingFace datasets')
    parser.add_argument('--query_max_length', type=int, default=257,
                        help='Max token length for queries')
    parser.add_argument('--doc_max_length', type=int, default=256,
                        help='Max token length for documents')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # -----------------------------------------------------------------------
    # Locate input file
    # -----------------------------------------------------------------------
    data_file_map = {
        'ambiguous_qe': {
            'train': '/scratch/hc3337/projects/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_train.jsonl',
            'dev':   '/scratch/hc3337/projects/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev.jsonl',
        },
        'qampari': {
            'train': '/scratch/hc3337/projects/diverse_response/data/qampari_data/train_data_gt_qampari_corpus.jsonl',
            'dev':   '/scratch/hc3337/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl',
        },
    }
    data_file = data_file_map[args.data_name][args.split]
    print(f"Loading data from {data_file}")
    all_data = read_jsonl(data_file)

    # -----------------------------------------------------------------------
    # Load corpus for negative sampling
    # -----------------------------------------------------------------------
    print("Loading corpus for negative sampling...")
    corpus_rows = read_tsv(args.corpus_tsv)
    corpus = [{'title': row[2], 'text': row[1]} for row in corpus_rows]

    # -----------------------------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------------------------
    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # Process each length group and collect examples
    # -----------------------------------------------------------------------
    dataset_dicts = []

    for length in args.lengths:
        # Filter to examples with exactly `length` positives
        if args.data_name in ('ambiguous_qe',):
            data = [d for d in all_data if len(d.get('positive_ctxs', [])) == length]
        elif args.data_name == 'qampari':
            data = [d for d in all_data if len(d.get('ground_truths', [])) == length]
        print(f"Examples with {length} positives: {len(data)}")

        query_encs = tokenize_queries(data, tokenizer, args.model_id, args.query_max_length)

        print(f"Collecting positive contexts and sampling negatives (length={length})...")
        all_positives = []
        all_negatives = []

        for inst in tqdm(data, desc=f"collecting docs (length={length})"):
            positives = inst.get('positive_ctxs') or inst.get('ground_truths') or []
            # positive_ctxs: list of lists of passage dicts — take first passage per answer
            if positives and isinstance(positives[0], list):
                positives = [p[0] for p in positives]
            # ground_truths in qampari are dicts with a 'passages' key
            elif positives and isinstance(positives[0], dict) and 'passages' in positives[0]:
                positives = [p['passages'][0] for p in positives]
            assert len(positives) == length, \
                f"Expected {length} positives, got {len(positives)}"
            all_positives.extend(positives)
            all_negatives.extend(sample_negatives(inst, corpus, length))

        print(f"Tokenizing {len(all_positives)} positive docs...")
        pos_encs = tokenize_documents(all_positives, tokenizer, args.doc_max_length)

        print(f"Tokenizing {len(all_negatives)} negative docs...")
        neg_encs = tokenize_documents(all_negatives, tokenizer, args.doc_max_length)

        for i, (q_enc, inst) in enumerate(zip(query_encs, data)):
            pos_start = i * length
            pos_docs = pos_encs[pos_start:pos_start + length]
            neg_docs = neg_encs[pos_start:pos_start + length]

            # input_document_ids: [pos_0, pos_1, ..., neg_0, neg_1, ...]
            # first half = positives, second half = negatives (required by collator)
            input_document_ids = [d['input_ids'] for d in pos_docs] + \
                                 [d['input_ids'] for d in neg_docs]
            attention_mask_document = [d['attention_mask'] for d in pos_docs] + \
                                      [d['attention_mask'] for d in neg_docs]

            dataset_dicts.append({
                'input_ids': q_enc['input_ids'],
                'attention_mask': q_enc['attention_mask'],
                'input_document_ids': input_document_ids,
                'attention_mask_document': attention_mask_document,
                'num_positives': length,
            })

    # -----------------------------------------------------------------------
    # Save as HuggingFace dataset
    # -----------------------------------------------------------------------
    lengths_str = '_'.join(str(l) for l in sorted(args.lengths))
    out_name = f"autoregressive_{args.data_name}_inf_{args.split}_dataset_contrastive_{lengths_str}_ctxs"
    out_path = Path(args.out_dir) / out_name
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(dataset_dicts)} examples to {out_path}")
    # Save in chunks to avoid OOM on large datasets
    chunk_size = 5000
    chunks = []
    for i in range(0, len(dataset_dicts), chunk_size):
        chunks.append(Dataset.from_list(dataset_dicts[i:i + chunk_size]))
    full_dataset = concatenate_datasets(chunks)
    full_dataset.save_to_disk(str(out_path))

    print(f"Done. Dataset saved to {out_path}")
    print(f"  Examples:         {len(full_dataset)}")
    print(f"  Lengths included: {sorted(args.lengths)}")
    print(f"  Query seq length: {args.query_max_length}")
    print(f"  Doc seq length:   {args.doc_max_length}")


if __name__ == '__main__':
    main()
