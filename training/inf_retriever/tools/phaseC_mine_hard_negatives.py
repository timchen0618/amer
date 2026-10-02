"""Phase C knob: training data with base-retriever hard negatives.

For every training question, take the base retriever's top-100 passages
(results/phaseC/mining/<ds>_train_base_top100.jsonl, from retrieval_base.py), drop
likely false negatives, and store the survivors (in rank order) as `hard_negative_ctxs`:
  - any gold passage (QAMPARI: by id),
  - any passage whose text contains an answer string (QAMPARI answer aliases; AmbigQA
    answers), since QAMPARI list questions and ambiguous AmbigQA questions have
    valid passages beyond the labelled golds,
  - the first --skip_top candidates (the most likely unlabelled positives),
  - any passage whose title matches a gold passage's title (another chunk of a gold article)
    or an answer entity (QAMPARI: answer_text / aliases / original_answer; AmbigQA: the
    answers), case-insensitive. Added after the 2026-09-27 audit: 3.6% (QAMPARI) and 10%
    (AmbigQA) of hard negatives were other chunks of a gold article.
Everything else in each example is kept unchanged, so training with
--negative_hard_ratio 0 reproduces the original data. Output:
<dst>/{train_data,dev_data}.jsonl (dev is copied unchanged).

Usage (repo root, compute node):
    python training/inf_retriever/tools/phaseC_mine_hard_negatives.py --ds qampari
    python training/inf_retriever/tools/phaseC_mine_hard_negatives.py --ds qampari \
        --src data/training/clean/qampari --dst data/training/clean_hn/qampari \
        --mined results/phaseC/mining/qampari_cleantrain_base_top100.jsonl
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.getcwd())
from src.eval_utils import SimpleTokenizer, has_answer  # noqa: E402  (answer matching used by eval.py)


def answers_of(ex, ds):
    if ds == "qampari":
        return [a["aliases"] for a in ex["answer_list"]]
    return ex["answers"]


def norm_title(t):
    return " ".join((t or "").lower().split())


def blocked_titles(ex, ds):
    """Titles a hard negative may not have: gold titles and answer entities."""
    titles = {norm_title(g.get("title")) for g in ex.get("positive_ctxs", []) if isinstance(g, dict)}
    if ds == "qampari":
        for a in ex["answer_list"]:
            titles.update(norm_title(x) for x in [a.get("answer_text"), a.get("original_answer")] + a.get("aliases", []))
    else:
        for alts in ex["answers"]:
            titles.update(norm_title(x) for x in (alts if isinstance(alts, list) else [alts]))
    titles.discard("")
    return titles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["qampari", "ambigqa"])
    ap.add_argument("--skip_top", type=int, default=3)
    ap.add_argument("--max_hard", type=int, default=30)
    ap.add_argument("--src", default=None, help="default: data/training/filtered/<ds>")
    ap.add_argument("--dst", default=None, help="default: data/training/filtered_hn/<ds>")
    ap.add_argument("--mined", default=None, help="default: results/phaseC/mining/<ds>_train_base_top100.jsonl")
    args = ap.parse_args()
    tok = SimpleTokenizer()
    src = args.src or f"data/training/filtered/{args.ds}"
    dst = args.dst or f"data/training/filtered_hn/{args.ds}"
    os.makedirs(dst, exist_ok=True)
    mined = open(args.mined or f"results/phaseC/mining/{args.ds}_train_base_top100.jsonl")
    n = kept_total = dropped_gold = dropped_ans = dropped_title = empty = 0
    with open(f"{src}/train_data.jsonl") as fin, open(f"{dst}/train_data.jsonl", "w") as fout:
        for line, mline in zip(fin, mined):
            ex, m = json.loads(line), json.loads(mline)
            q = ex.get("question_text") or ex.get("question")
            assert (m.get("question_text") or m.get("question")) == q, "mining output out of order"
            gold_ids = {g.get("id") for g in ex.get("positive_ctxs", []) if isinstance(g, dict) and g.get("id")}
            answers = answers_of(ex, args.ds)
            blocked = blocked_titles(ex, args.ds)
            hard = []
            for rank, c in enumerate(m["ctxs"]):
                if rank < args.skip_top:
                    continue
                if c["id"] in gold_ids:
                    dropped_gold += 1; continue
                if norm_title(c.get("title")) in blocked:
                    dropped_title += 1; continue
                if any(has_answer(a, c["text"], tok) for a in answers):
                    dropped_ans += 1; continue
                hard.append({"id": c["id"], "title": c.get("title", ""), "text": c["text"]})
                if len(hard) >= args.max_hard:
                    break
            ex["hard_negative_ctxs"] = hard
            kept_total += len(hard); n += 1; empty += not hard
            fout.write(json.dumps(ex) + "\n")
        assert mined.readline() == "", "mining file has more lines than the training file"
    assert n == sum(1 for _ in open(f"{src}/train_data.jsonl")), "mining file has fewer lines than the training file"
    shutil.copy(f"{src}/dev_data.jsonl", f"{dst}/dev_data.jsonl")
    print(f"{args.ds}: {n} examples, {kept_total / n:.1f} hard negatives each on average; "
          f"dropped {dropped_gold} gold, {dropped_title} same-title/answer-entity and {dropped_ans} "
          f"answer-containing candidates; {empty} examples got none")


if __name__ == "__main__":
    main()
