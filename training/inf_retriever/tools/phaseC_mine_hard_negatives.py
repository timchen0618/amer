"""Phase C knob: training data with base-retriever hard negatives.

For every training question, take the base retriever's top-100 passages
(results/phaseC/mining/<ds>_train_base_top100.jsonl, from retrieval_base.py), drop
likely false negatives, and store the survivors (in rank order) as `hard_negative_ctxs`:
  - any gold passage (QAMPARI: by id),
  - any passage whose text contains an answer string (QAMPARI answer aliases; AmbigQA
    answers), since QAMPARI list questions and ambiguous AmbigQA questions have
    valid passages beyond the labelled golds,
  - the first --skip_top candidates (the most likely unlabelled positives).
Everything else in each example is kept unchanged, so training with
--negative_hard_ratio 0 reproduces the original data. Output:
data/training/filtered_hn/<ds>/{train_data,dev_data}.jsonl (dev is copied unchanged).

Usage (repo root, compute node):
    python training/inf_retriever/tools/phaseC_mine_hard_negatives.py --ds qampari
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["qampari", "ambigqa"])
    ap.add_argument("--skip_top", type=int, default=3)
    ap.add_argument("--max_hard", type=int, default=30)
    args = ap.parse_args()
    tok = SimpleTokenizer()
    src = f"data/training/filtered/{args.ds}"
    dst = f"data/training/filtered_hn/{args.ds}"
    os.makedirs(dst, exist_ok=True)
    mined = open(f"results/phaseC/mining/{args.ds}_train_base_top100.jsonl")
    n = kept_total = dropped_gold = dropped_ans = 0
    with open(f"{src}/train_data.jsonl") as fin, open(f"{dst}/train_data.jsonl", "w") as fout:
        for line, mline in zip(fin, mined):
            ex, m = json.loads(line), json.loads(mline)
            q = ex.get("question_text") or ex.get("question")
            assert (m.get("question_text") or m.get("question")) == q, "mining output out of order"
            gold_ids = {g.get("id") for g in ex.get("positive_ctxs", []) if isinstance(g, dict) and g.get("id")}
            answers = answers_of(ex, args.ds)
            hard = []
            for rank, c in enumerate(m["ctxs"]):
                if rank < args.skip_top:
                    continue
                if c["id"] in gold_ids:
                    dropped_gold += 1; continue
                if any(has_answer(a, c["text"], tok) for a in answers):
                    dropped_ans += 1; continue
                hard.append({"id": c["id"], "title": c.get("title", ""), "text": c["text"]})
                if len(hard) >= args.max_hard:
                    break
            ex["hard_negative_ctxs"] = hard
            kept_total += len(hard); n += 1
            fout.write(json.dumps(ex) + "\n")
    shutil.copy(f"{src}/dev_data.jsonl", f"{dst}/dev_data.jsonl")
    print(f"{args.ds}: {n} examples, {kept_total / n:.1f} hard negatives each on average; "
          f"dropped {dropped_gold} gold and {dropped_ans} answer-containing candidates")


if __name__ == "__main__":
    main()
