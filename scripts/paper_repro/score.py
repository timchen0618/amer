"""Paper-repro eval step 3: aggregate per-vector top-500 lists and score with the PAPER's scorer.

Round-robin = fe0bd78 gen_ret_and_eval.aggregate_different_queries_by_length (dedup, list order); RRF = sum 1/(rank+1+60)
(as retrieval_inf.aggregate_rrf). The top-100 aggregated list is written in the paper's retrieval-output format
({question, ctxs:[{id,title,text,score}]}, passage text from chunks_v5.tsv) and scored with fe0bd78
src/eval_utils.eval_retrieve_docs (QAMPARI: gold ids per cluster; AmbigQA: has_answer on passage text), at top 100 and 10,
on all questions and on the paper's low-similarity subset.
usage: python score.py <dataset> <out_json> <top500.pkl> [...]
"""
import csv, json, os, pickle, sys
sys.path.insert(0, os.environ.get('PAPER_CODE', '.'))
from src.eval_utils import eval_retrieve_docs

R = '/scratch/hc3337/projects/autoregressive'
CFG = {
    'qampari': dict(q=f'{R}/data/questions/qampari_dev_question_only_5_to_8_ctxs.jsonl',
                    gold='/scratch/hc3337/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus_5_to_8_ctxs.jsonl',
                    low=f'{R}/data/qampari_5_to_8/large_distance_indices_inf.txt', gold_id=True),
    'ambiguous_qe': dict(q=f'{R}/data/questions/ambiguous_qe_dev_question_only_2_to_5_ctxs.jsonl',
                         gold=f'{R}/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.jsonl',
                         low=f'{R}/data/ambiguous/qampari_embeddings_data/large_distance_indices_inf.txt', gold_id=False),
}
ds, out_json, pkls = sys.argv[1], sys.argv[2], sys.argv[3:]
c = CFG[ds]
questions = [json.loads(l) for l in open(c['q'])]
low = [int(l) for l in open(c['low']) if l.strip()]
nq = len(questions)

def rr(lists, k=100):
    out, seen = [], set()
    for i in range(max(map(len, lists))):
        for L in lists:
            if i < len(L) and L[i] not in seen:
                seen.add(L[i]); out.append(L[i])
                if len(out) >= k: return out
    return out

def rrf(lists, k=100, c=60):
    s = {}
    for L in lists:
        for r, d in enumerate(L): s[d] = s.get(d, 0.0) + 1.0 / (r + 1 + c)
    return sorted(s, key=lambda d: -s[d])[:k]

ranked = {}
for p in pkls:
    for run, (ids, _) in pickle.load(open(p, 'rb')).items():
        n = len(ids) // nq; assert n * nq == len(ids), (run, len(ids), nq)
        per_q = [ids[i * n:(i + 1) * n] for i in range(nq)]
        for agg, f in (('rr', rr), ('rrf', rrf)):
            ranked[(os.path.basename(p), run, agg)] = [f(l) for l in per_q]
need = set(d for v in ranked.values() for l in v for d in l)
text = {}
with open('/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv') as f:
    for row in csv.reader(f, delimiter='\t'):
        if row[0] in need: text[row[0]] = (row[2], row[1])
missing = len(need - set(text)); assert missing == 0, f'{missing} ids not in chunks_v5'
tmp = out_json + '.tmp.jsonl'; res = {}
for key, lists in sorted(ranked.items()):
    with open(tmp, 'w') as f:
        for q, l in zip(questions, lists):
            f.write(json.dumps({**q, 'ctxs': [{'id': d, 'title': text[d][0], 'text': text[d][1], 'score': str(100 - r)} for r, d in enumerate(l)]}) + '\n')
    m = {}
    for k in (100, 10):
        for sub, idx in (('all', None), ('lowsim', low)):
            o = eval_retrieve_docs(tmp, c['gold'], has_gold_id=c['gold_id'], topk=k, selected_indices=idx)
            m[f'mrecall@{k}_{sub}'] = float(o[0]); m[f'recall@{k}_{sub}'] = float(o[1])
            if k == 100 and sub == 'all': m['per_q_mrecall@100'] = [int(x) for x in o[-2]]
    res['|'.join(key)] = m
    print(key, {k: v for k, v in m.items() if not k.startswith('per_q')}, flush=True)
os.remove(tmp)
json.dump(res, open(out_json, 'w'))
