"""Slim retrieval output files (results/phaseC/full/<ds>/<run>_s<step>/{dev,test}/<ds>.jsonl) and restore them.

A retrieval output line is the eval-data record plus "ctxs": 500 retrieved passages with id, title, text and score
(src/retrieval_utils.py:add_passages). The title and text are copies of corpus rows, so they can be dropped and put
back from the corpus by id. A slim line keeps only the query identity and the ranked ids and scores:

    {"id"/"qid": ..., "question"/"question_text": ..., "_slim": 1, "ctxs": [{"id": ..., "score": ...}, ...]}

What reads a slim file as is:
  - eval.py on QAMPARI (gold-id scoring: needs question_text + ctxs ids/scores only);
  - tools/fusion_combo.py (load_single reads ctxs ids and scores only).
What needs `restore` first:
  - eval.py on AmbigQA (--no-gold-id: answer-string matching on ctx["text"]).

    python slim_retrieval.py slim    IN OUT
    python slim_retrieval.py verify  ORIG SLIM                      # same rows, query identity, ids and scores
    python slim_retrieval.py restore SLIM OUT [--data EVAL.jsonl]   # re-attach title/text from the corpus;
                                                                    # --data also re-adds the other eval-data fields
    python slim_retrieval.py restore-check ORIG SLIM [SLIM ...]     # (in bulk) every ctx title/text of ORIG equals
                                                                    # the corpus row of its id, i.e. restore is exact
Default corpus: /scratch/hc3337/wikipedia_chunks/chunks_v5.tsv (columns id, text, title), the --passages of
tools/retrieve_ckpt.sbatch. Restoring reads the whole corpus (15 GB) once: run it on a compute node.
"""
import argparse, csv, json, os, sys

CORPUS = "/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv"
ID_KEYS = ("qid", "id")
Q_KEYS = ("question_text", "question")


def read(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def is_slim(path):
    with open(path) as f:
        first = f.readline()
    return bool(first) and json.loads(first).get("_slim") == 1


def slim_row(r):
    out = {k: r[k] for k in ID_KEYS + Q_KEYS if k in r}
    out["_slim"] = 1
    out["ctxs"] = [{"id": c["id"], "score": c["score"]} for c in r["ctxs"]]
    return out


def slim(inp, out):
    n = 0
    with open(out, "w") as f:
        for r in read(inp):
            f.write(json.dumps(slim_row(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def verify(orig, sl):
    """Raise unless SLIM has the same rows, query identity, ids and scores (in order) as ORIG."""
    n = 0
    with open(orig) as fo, open(sl) as fs:
        for lo, ls in zip(fo, fs):
            o, s = json.loads(lo), json.loads(ls)
            assert s.get("_slim") == 1, f"row {n}: not slim"
            for k in ID_KEYS + Q_KEYS:
                assert o.get(k) == s.get(k), f"row {n}: {k} differs"
            assert [(c["id"], c["score"]) for c in o["ctxs"]] == [(c["id"], c["score"]) for c in s["ctxs"]], \
                f"row {n}: ctx ids/scores differ"
            n += 1
        assert fo.readline() == "" and fs.readline() == "", "row counts differ"
    return n


def load_corpus(ids, path=CORPUS, strict=True):
    """{id: (title, text)} for the wanted ids, read the way src/retrieval_utils.py:load_passages reads the TSV."""
    csv.field_size_limit(sys.maxsize)
    out = {}
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            if row[0] in ids:
                out[row[0]] = (row[2], row[1])
    missing = set(ids) - set(out)
    assert not (strict and missing), f"{len(missing)} ids not in corpus, e.g. {sorted(missing)[:3]}"
    return out


def ctx_ids(path):
    return {c["id"] for r in read(path) for c in r["ctxs"]}


def restore(sl, out, data=None, corpus=CORPUS):
    passages = load_corpus(ctx_ids(sl), corpus)
    rows_data = read(data) if data else None
    n = 0
    with open(out, "w") as f:
        for r in read(sl):
            ctxs = [{"id": c["id"], "title": passages[c["id"]][0], "text": passages[c["id"]][1], "score": c["score"]}
                    for c in r["ctxs"]]
            if rows_data is not None:  # retrieval_inf.py writes the eval-data record with "ctxs" (re)set last
                d = next(rows_data)
                for k in ID_KEYS + Q_KEYS:
                    assert k not in r or d.get(k) == r[k], f"row {n}: {k} differs from --data"
                d["ctxs"] = ctxs
            else:
                d = {k: v for k, v in r.items() if k not in ("_slim", "ctxs")}
                d["ctxs"] = ctxs
            f.write(json.dumps(d, ensure_ascii=False) + "\n")  # as retrieval_inf.py writes it
            n += 1
    return n


def restore_check(orig, passages):
    """Raise unless every ctx title/text in ORIG equals the corpus row of its id."""
    n = 0
    for i, r in enumerate(read(orig)):
        for c in r["ctxs"]:
            assert passages[c["id"]] == (c["title"], c["text"]), f"row {i}: ctx {c['id']} title/text != corpus"
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("slim"); p.add_argument("inp"); p.add_argument("out")
    p = sub.add_parser("verify"); p.add_argument("orig"); p.add_argument("slim")
    p = sub.add_parser("restore"); p.add_argument("slim"); p.add_argument("out")
    p.add_argument("--data"); p.add_argument("--corpus", default=CORPUS)
    p = sub.add_parser("restore-check"); p.add_argument("orig", nargs="+"); p.add_argument("--corpus", default=CORPUS)
    a = ap.parse_args()
    if a.cmd == "slim":
        print(f"slimmed {slim(a.inp, a.out)} rows: {os.path.getsize(a.inp)} -> {os.path.getsize(a.out)} bytes")
    elif a.cmd == "verify":
        print(f"OK {verify(a.orig, a.slim)} rows")
    elif a.cmd == "restore":
        print(f"restored {restore(a.slim, a.out, a.data, a.corpus)} rows")
    else:
        ids = set().union(*(ctx_ids(o) for o in a.orig))
        passages = load_corpus(ids, a.corpus)
        for o in a.orig:
            print(f"OK {restore_check(o, passages)} ctxs {o}")


if __name__ == "__main__":
    main()
