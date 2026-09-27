"""Phase A: build held-out validation query sets from the training dev splits.

Outputs files in the same format as the test sets in data/amer_data/eval_data/, so
retrieval_inf.py / retrieval_base.py / eval.py treat them identically:
  - QAMPARI: `n_qampari` questions sampled (seeded) from dev_data.jsonl.
    Gold passages go in `ground_truths` grouped one group per answer ([[g1], [g2], ...]),
    and `positive_ctxs` is dropped (eval.py would treat its flat list as groups).
  - AmbigQA: all dev questions; scored by answer strings (eval.py --no-gold-id), so
    only question/id/answers/positive_ctxs/input are kept.
No dev question appears in the test sets or the training data (checked 2026-09-25).

Usage (repo root, compute node):
    python training/inf_retriever/tools/phaseA_make_dev_sets.py --out_dir data/phaseA
"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/phaseA")
    ap.add_argument("--n_qampari", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    q = [json.loads(l) for l in open("data/training/filtered/qampari/dev_data.jsonl")]
    random.Random(args.seed).shuffle(q)
    q = q[: args.n_qampari]
    with open(os.path.join(args.out_dir, f"qampari_dev{args.n_qampari}.jsonl"), "w") as f:
        for ex in q:
            golds = ex["ground_truths"]
            assert all(isinstance(g, dict) and "id" in g for g in golds)
            f.write(json.dumps({
                "qid": ex["qid"],
                "question_text": ex["question_text"],
                "entities": ex.get("entities", []),
                "answer_list": ex["answer_list"],
                "ground_truths": [[g] for g in golds],
            }) + "\n")
    print(f"qampari: {len(q)} questions, golds per question "
          f"{min(len(e['ground_truths']) for e in q)}-{max(len(e['ground_truths']) for e in q)}")

    a = [json.loads(l) for l in open("data/training/filtered/ambigqa/dev_data.jsonl")]
    with open(os.path.join(args.out_dir, f"ambigqa_dev{len(a)}.jsonl"), "w") as f:
        for ex in a:
            f.write(json.dumps({k: ex[k] for k in ("question", "id", "answers", "positive_ctxs", "input")}) + "\n")
    print(f"ambigqa: {len(a)} questions")


if __name__ == "__main__":
    main()
