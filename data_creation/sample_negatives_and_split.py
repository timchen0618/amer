"""
Sample random negative passages from a Wikipedia corpus for training data,
filter by gold passage count, and split into train/dev sets.

Negative candidates are rejected if:
  1. Their title matches any gold passage title (case-insensitive)
  2. Their unigram overlap with any gold passage exceeds a threshold (default 30%)

Memory-efficient: stores only byte offsets of the corpus file, reads passages on demand.
"""

import argparse
import array
import json
import os
import random
import re
import sys


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def compute_overlap(candidate_tokens: set[str], gold_tokens: set[str]) -> float:
    if not candidate_tokens:
        return 0.0
    return len(candidate_tokens & gold_tokens) / len(candidate_tokens)


def build_offset_index(corpus_path: str) -> array.array:
    """First pass: collect byte offsets for each data line (skip header).
    Uses array.array('Q') for compact storage (~8 bytes/offset vs ~28 for Python ints).
    """
    print(f"Indexing corpus at {corpus_path} ...")
    offsets = array.array("Q")  # unsigned 64-bit
    with open(corpus_path, "rb") as f:
        f.readline()  # skip header
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(offset)
            if len(offsets) % 1_000_000 == 0:
                print(f"  Indexed {len(offsets):,} lines ...", flush=True)
    print(f"  Total: {len(offsets):,} corpus lines ({len(offsets)*8/1e6:.0f} MB index)")
    return offsets


def read_passage(f, offset: int) -> tuple[str, str, str] | None:
    """Read a single passage from the corpus file at the given byte offset."""
    f.seek(offset)
    line = f.readline().decode("utf-8", errors="replace").strip()
    parts = line.split("\t")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None


def sample_negatives(
    corpus_fh,
    offsets: list[int],
    gold_passages: list[dict],
    num_negatives: int,
    overlap_threshold: float,
    rng: random.Random,
    max_total_attempts: int = 500,
) -> list[dict]:
    gold_titles = {gt["title"].strip().lower() for gt in gold_passages}
    gold_token_sets = [tokenize(gt["text"]) for gt in gold_passages]

    negatives = []
    corpus_size = len(offsets)
    total_attempts = 0

    while len(negatives) < num_negatives and total_attempts < max_total_attempts:
        idx = rng.randint(0, corpus_size - 1)
        total_attempts += 1

        passage = read_passage(corpus_fh, offsets[idx])
        if passage is None:
            continue

        cid, ctext, ctitle = passage

        if ctitle.strip().lower() in gold_titles:
            continue

        candidate_tokens = tokenize(ctext)
        rejected = False
        for gold_tokens in gold_token_sets:
            if compute_overlap(candidate_tokens, gold_tokens) > overlap_threshold:
                rejected = True
                break

        if rejected:
            continue

        negatives.append({"id": cid, "text": ctext, "title": ctitle})

    return negatives


def main():
    parser = argparse.ArgumentParser(
        description="Sample negatives, filter by gold count, and split train/dev"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/scratch/hc3337/projects/diverse_response/data/qampari_data/train_data_gt_qampari_corpus.jsonl",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/hc3337/projects/autoregressive/data/training/filtered",
    )
    parser.add_argument("--num_negatives", type=int, default=25)
    parser.add_argument("--overlap_threshold", type=float, default=0.3)
    parser.add_argument("--min_gold", type=int, default=5)
    parser.add_argument("--max_gold", type=int, default=8)
    parser.add_argument("--dev_size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    offsets = build_offset_index(args.corpus_path)

    print(f"\nReading input from {args.input_path} ...")
    with open(args.input_path, "r") as f:
        lines = f.readlines()
    total_instances = len(lines)
    print(f"  {total_instances:,} total instances")

    print(f"\nFiltering to {args.min_gold}-{args.max_gold} gold passages and sampling {args.num_negatives} negatives each")
    print(f"  Overlap threshold: {args.overlap_threshold}")

    processed = []
    skipped_gold_count = 0
    skipped_neg_count = 0

    corpus_fh = open(args.corpus_path, "rb")

    for i, line in enumerate(lines):
        if i % 1000 == 0:
            print(
                f"\r  Processing: {i:,}/{total_instances:,} "
                f"({100*i/total_instances:.1f}%) | kept={len(processed):,}",
                end="",
                flush=True,
            )

        instance = json.loads(line)
        gold_passages = instance["ground_truths"]
        num_gold = len(gold_passages)

        if not (args.min_gold <= num_gold <= args.max_gold):
            skipped_gold_count += 1
            continue

        negatives = sample_negatives(
            corpus_fh,
            offsets,
            gold_passages,
            args.num_negatives,
            args.overlap_threshold,
            rng,
        )

        if len(negatives) < args.num_negatives:
            skipped_neg_count += 1
            continue

        instance["positive_ctxs"] = gold_passages
        instance["negative_ctxs"] = negatives
        instance["hard_negative_ctxs"] = []
        processed.append(instance)

    corpus_fh.close()

    print(f"\r  Processing: {total_instances:,}/{total_instances:,} (100.0%) | kept={len(processed):,}  ")
    print(f"\nKept {len(processed):,} instances")
    print(f"Skipped {skipped_gold_count:,} (gold count outside {args.min_gold}-{args.max_gold})")
    print(f"Skipped {skipped_neg_count:,} (could not sample enough negatives)")

    rng.shuffle(processed)

    dev_size = min(args.dev_size, len(processed))
    dev_data = processed[:dev_size]
    train_data = processed[dev_size:]

    print(f"\nDev set:   {len(dev_data):,} instances")
    print(f"Train set: {len(train_data):,} instances")

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train_data.jsonl")
    dev_path = os.path.join(args.output_dir, "dev_data.jsonl")

    for path, data in [(train_path, train_data), (dev_path, dev_data)]:
        with open(path, "w") as f:
            for instance in data:
                f.write(json.dumps(instance) + "\n")
        print(f"Wrote {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
