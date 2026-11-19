"""Clean v3: v2 with answer-containing random negatives replaced (2026-10-04).

v2 (data_creation/CLEAN_V2.md) samples random negatives with a title filter and a unigram-overlap
filter but no answer filter, so 1.26% (QAMPARI) / 0.88% (AmbigQA) of them contain an answer string
and act as false negatives. v3 changes only those negatives:

  for every example, every random negative (negative_ctxs) whose text contains any answer string,
  by the same matcher as eval.py and the hard-negative miner (src.eval_utils.has_answer on the
  answer aliases), is replaced by a new random corpus passage that passes the original sampler's
  filters (title not equal to a gold title, unigram overlap with every gold <= 30%) AND the answer
  filter, and is not already among the example's negatives. Replacements are drawn from the
  CSV-parsed corpus with a random.Random seeded from (dataset, example id), so the plain and the
  hard-negative file of a dataset get identical replacements.

Everything else is copied unchanged: questions, golds, hard negatives (already answer-filtered by
the miner), example order, dev sets. Inputs: data/training/clean_v2{,_hn}/<ds>/{train,dev}_data.jsonl
(AmbigQA's v2 directories link to v1). Outputs: data/training/clean_v3{,_hn}/<ds>/ and
data/training/clean_v3/<ds>/stats_v3.json.

Usage (repo root, compute node, ~16 GB RAM):
    python data_creation/build_clean_v3.py
"""
import array
import csv
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys

sys.path.insert(0, os.getcwd())
from src.eval_utils import SimpleTokenizer, _normalize, has_answer  # noqa: E402

CORPUS = "/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv"
SRC = "data/training/clean_v2"
DST = "data/training/clean_v3"
OVERLAP = 0.3
csv.field_size_limit(10**9)


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def overlap(cand, gold):
    return len(cand & gold) / len(cand) if cand else 0.0


def answers_of(ex, ds):
    if ds == "qampari":
        return [a["aliases"] for a in ex["answer_list"]]
    return [a if isinstance(a, list) else [a] for a in ex["answers"]]


def golds_of(ex):
    out = []
    for g in ex.get("positive_ctxs") or ex.get("ground_truths") or []:
        out += g if isinstance(g, list) else [g]
    return [g for g in out if isinstance(g, dict)]


def ex_key(ex):
    return str(ex.get("qid") or ex.get("id") or ex.get("question_text") or ex.get("question"))


def index_corpus():
    offsets = array.array("Q")
    with open(CORPUS, "rb") as f:
        f.readline()
        while True:
            off = f.tell()
            if not f.readline():
                break
            offsets.append(off)
    return offsets


def read_row(fh, off):
    fh.seek(off)
    line = fh.readline().decode("utf-8", errors="replace")
    row = next(csv.reader([line], delimiter="\t"))
    return row if len(row) == 3 else None  # id, text, title


# Fast equivalent of any(has_answer(alts, text, tok) for alts in answers): the same normalization
# and tokenizer, but the text is tokenized once and every answer alias once per example, and the
# token-sequence match is a set lookup of the text's n-grams. One deliberate difference: an alias
# that tokenizes to nothing (e.g. punctuation only) makes has_answer match every text; such aliases
# are skipped here (counted in stats as empty_aliases), otherwise no replacement could ever pass.
TOK = SimpleTokenizer()


def answer_tuples(answers):
    tups, empty = set(), 0
    for alts in answers:
        for a in alts:
            t = tuple(TOK.tokenize(_normalize(a), uncased=True))
            if t:
                tups.add(t)
            else:
                empty += 1
    return tups, empty


def contains_answer_fast(tups, text):
    if not tups:
        return False
    toks = TOK.tokenize(_normalize(text), uncased=True)
    for n in {len(t) for t in tups}:
        grams = {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}
        if any(t in grams for t in tups if len(t) == n):
            return True
    return False


def detect(args):
    """Indices of an example's random negatives that contain an answer (worker process)."""
    line, ds = args
    ex = json.loads(line)
    tups, empty = answer_tuples(answers_of(ex, ds))
    bad = [i for i, n in enumerate(ex.get("negative_ctxs") or []) if contains_answer_fast(tups, n.get("text", ""))]
    return bad, empty, len(ex.get("negative_ctxs") or [])


def main():
    pool = mp.Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    print("indexing corpus ...", flush=True)
    offsets = index_corpus()
    print(f"  {len(offsets):,} passages", flush=True)
    fh = open(CORPUS, "rb")
    for ds in ("qampari", "ambigqa"):
        stats = {}
        replaced = {}  # (split, key) -> new negative_ctxs, shared by the plain and _hn files
        for split in ("train", "dev"):
            src = f"{SRC}/{ds}/{split}_data.jsonl"
            n_ex = n_neg = n_bad = n_ex_bad = draws = n_empty_alias = 0
            rows = []
            lines = open(src).readlines()
            detected = pool.map(detect, [(l, ds) for l in lines], chunksize=64)
            print(f"  {ds} {split}: detection done", flush=True)
            for line, (bad, empty, nn) in zip(lines, detected):
                ex = json.loads(line)
                negs = ex.get("negative_ctxs") or []
                n_ex += 1; n_neg += len(negs); n_bad += len(bad); n_ex_bad += bool(bad); n_empty_alias += empty
                if bad:
                    tups, _ = answer_tuples(answers_of(ex, ds))
                    golds = golds_of(ex)
                    gold_titles = {g.get("title", "").strip().lower() for g in golds}
                    gold_toks = [tokens(g.get("text", "")) for g in golds]
                    have = {n.get("id") for n in negs}
                    seed = int(hashlib.sha256(f"{ds}|{split}|{ex_key(ex)}".encode()).hexdigest()[:16], 16)
                    rng = random.Random(seed)
                    new = list(negs)
                    for i in bad:
                        while True:
                            draws += 1
                            row = read_row(fh, offsets[rng.randrange(len(offsets))])
                            if row is None:
                                continue
                            cid, text, title = row
                            if cid in have or title.strip().lower() in gold_titles:
                                continue
                            ct = tokens(text)
                            if any(overlap(ct, gt) > OVERLAP for gt in gold_toks):
                                continue
                            if contains_answer_fast(tups, text):
                                continue
                            new[i] = {"id": cid, "text": text, "title": title}
                            have.add(cid)
                            break
                    replaced[(split, ex_key(ex))] = new
                    ex["negative_ctxs"] = new
                rows.append(ex)
            os.makedirs(f"{DST}/{ds}", exist_ok=True)
            with open(f"{DST}/{ds}/{split}_data.jsonl", "w") as f:
                for ex in rows:
                    f.write(json.dumps(ex) + "\n")
            stats[split] = {"examples": n_ex, "random_negatives": n_neg,
                            "answer_containing_replaced": n_bad, "examples_affected": n_ex_bad,
                            "rate": round(n_bad / max(n_neg, 1), 5), "candidate_draws": draws,
                            "empty_aliases_skipped": n_empty_alias}
            print(ds, split, stats[split], flush=True)
            # hard-negative twin: same examples and random negatives plus hard_negative_ctxs
            hsrc = f"{SRC}_hn/{ds}/{split}_data.jsonl"
            if os.path.exists(hsrc):
                os.makedirs(f"{DST}_hn/{ds}", exist_ok=True)
                n_h = n_sub = 0
                with open(hsrc) as fin, open(f"{DST}_hn/{ds}/{split}_data.jsonl", "w") as fout:
                    for line in fin:
                        ex = json.loads(line); n_h += 1
                        k = (split, ex_key(ex))
                        if k in replaced:
                            ex["negative_ctxs"] = replaced[k]; n_sub += 1
                        fout.write(json.dumps(ex) + "\n")
                stats[split]["hn_file"] = {"examples": n_h, "examples_with_replacements": n_sub}
                print(ds, split, "hn twin", stats[split]["hn_file"], flush=True)
        json.dump({"source": SRC, "stats": stats}, open(f"{DST}/{ds}/stats_v3.json", "w"), indent=2)


if __name__ == "__main__":
    main()
