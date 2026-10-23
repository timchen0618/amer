"""Paper-repro data build (branch paper-repro, plan: reports/paper_repro_2026-10-23/PLAN.md in fsdp-clean-recipe).

Rebuilds the paper's frozen-encoder training data (gen_embed_for_auto_training.py `gen_contrastive` +
create_input_for_contrievers.py `contrastive` + src/dataset.py `combine_datasets`) with these fixes:
  FIX-1  strip gold title/text before embedding (QAMPARI golds have ' '-padded title/text)
  FIX-2  drop QAMPARI train questions whose gold-id set has Jaccard >= 0.8 with a test question's gold set
  FIX-3  drop AmbigQA train questions whose normalized question equals a test question
  FIX-4  seeded random negatives (paper: unseeded random.choice)
Everything else follows the paper code: inf-retriever via SentenceTransformer (max_seq_length 8192, no passage
prompt, batch 128, L2-normalized), one random negative per gold drawn from the first 3M rows of chunks_v5.tsv
(read_tsv stops there), check_context_same rejection, per-gold-count .npy files, HF datasets built with the
paper's own create_input_embeddings_for_contrastive, gold counts concatenated in increasing order.

Stages:
  filter  (CPU)  -> <out>/raw_data/{ds}_train_question_only_{L}_ctxs.jsonl, {ds}_train_{L}.jsonl, filter_report.json
  embed   (GPU)  -> <out>/raw_data/{ds}_inf/{ds}_train_{positive,random}_embeddings_{L}.npy
  hf      (CPU)  -> <out>/training_datasets/<lm>/{ds}/inf/autoregressive_{ds}_inf_train_dataset_1b_contrastive_{lo}_to_{hi}_ctxs
"""
import argparse, csv, json, os, random, re, string, sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO = '/scratch/hc3337/projects/autoregressive'
SRC = {
    'qampari': {
        'train': '/scratch/hc3337/projects/diverse_response/data/qampari_data/train_data_gt_qampari_corpus.jsonl',
        'test_gold': '/scratch/hc3337/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus_5_to_8_ctxs.jsonl',
        'lengths': [5, 6, 7, 8], 'gold_key': 'ground_truths', 'q_key': 'question_text',
        'paper_counts': {5: 13001, 6: 8223, 7: 5692, 8: 5107},
    },
    'ambiguous_qe': {
        'train': f'{REPO}/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_train.jsonl',
        'test_q': f'{REPO}/data/questions/ambiguous_qe_dev_question_only_2_to_5_ctxs.jsonl',
        'lengths': [2, 3, 4, 5], 'gold_key': 'positive_ctxs', 'q_key': 'question',
        'paper_counts': None,  # paper total 5044 (checked below)
    },
}
LM_IDS = {'llama-1b': 'meta-llama/Llama-3.2-1B-Instruct', 'llama-3b': 'meta-llama/Llama-3.2-3B-Instruct',
          'llama-8b': 'meta-llama/Llama-3.1-8B-Instruct', 'qwen3-4b': 'Qwen/Qwen3-4B-Instruct-2507'}
CORPUS_TSV = '/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv'


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def write_jsonl(data, p):
    with open(p, 'w') as f:
        for x in data:
            f.write(json.dumps(x) + '\n')


def golds_of(inst, ds):
    g = inst[SRC[ds]['gold_key']]
    return [grp[0] for grp in g] if isinstance(g[0], list) else g  # paper: AmbigQA takes grp[0] per answer group


def gold_ids(gt):  # flatten (QAMPARI test golds can be per-answer clusters)
    return set(str(x['id']) for g in gt for x in (g if isinstance(g, list) else [g]))


def norm_q(q):
    q = q.lower().strip()
    q = ''.join(ch for ch in q if ch not in set(string.punctuation))
    return ' '.join(q.split())


# ---------------------------------------------------------------- filter
def stage_filter(args):
    out = Path(args.out) / 'raw_data'; out.mkdir(parents=True, exist_ok=True)
    report = {}
    for ds in args.datasets:
        cfg = SRC[ds]
        data = read_jsonl(cfg['train'])
        if ds == 'qampari':
            test = [gold_ids(d['ground_truths']) for d in read_jsonl(cfg['test_gold'])]
            inv = {}
            for i, s in enumerate(test):
                for x in s:
                    inv.setdefault(x, set()).add(i)
        else:
            test_q = set(norm_q(json.loads(l)['question']) for l in open(cfg['test_q']))
        rep = {'before': {}, 'removed': {}, 'after': {}}
        for L in cfg['lengths']:
            sub = [d for d in data if len(d[cfg['gold_key']]) == L]  # paper filter (gen_contrastive)
            rep['before'][L] = len(sub)
            keep = []
            for d in sub:
                if ds == 'qampari':  # FIX-2
                    s = gold_ids(d['ground_truths'])
                    cand = set().union(*[inv.get(x, set()) for x in s]) if s else set()
                    leak = any(len(s & test[i]) / len(s | test[i]) >= 0.8 for i in cand)
                else:  # FIX-3
                    leak = norm_q(d['question']) in test_q
                if not leak:
                    keep.append(d)
            rep['removed'][L] = len(sub) - len(keep); rep['after'][L] = len(keep)
            write_jsonl([{cfg['q_key']: d[cfg['q_key']]} for d in keep], out / f'{ds}_train_question_only_{L}_ctxs.jsonl')
            write_jsonl([{cfg['q_key']: d[cfg['q_key']], 'golds': golds_of(d, ds)} for d in keep], out / f'{ds}_train_{L}.jsonl')
        if cfg['paper_counts']:
            assert rep['before'] == cfg['paper_counts'], ('pre-filter counts differ from the paper', rep['before'])
        else:
            assert sum(rep['before'].values()) == 5044, ('AmbigQA pre-filter total != paper 5044', rep['before'])
        report[ds] = rep
        print(ds, json.dumps(rep), flush=True)
    json.dump(report, open(out / 'filter_report.json', 'w'), indent=1)


# ---------------------------------------------------------------- embed
def read_tsv_first3m(path):  # paper read_tsv: stops once > 3,000,000 rows
    data = []
    with open(path) as f:
        for row in tqdm(csv.reader(f, delimiter='\t')):
            if not row[0] == 'id':
                data.append(row)
            if len(data) > 3000000:
                break
    return data


def check_context_same(positive_ctxs, ctx2):  # paper code, unchanged
    if ctx2 == '':
        return True
    for ctx in positive_ctxs:
        if (ctx['title'] == ctx2['title'] and ctx['title'] != '') or ctx['text'] == ctx2['text']:
            return True
    return False


def stage_embed(args):
    import torch
    from sentence_transformers import SentenceTransformer
    rng = random.Random(args.seed)  # FIX-4
    model = SentenceTransformer('infly/inf-retriever-v1-1.5b', trust_remote_code=True)
    model.max_seq_length = 8192  # paper load_model
    model = model.to('cuda')

    def embed(passages):  # paper embed_passages_stella: title + ' ' + text, batch 128, normalize
        texts = [str(p['title']) + ' ' + str(p['text']) for p in passages]
        out = []
        for i in tqdm(range(0, len(texts), 128)):
            out.append(model.encode(texts[i:i + 128]))
        e = np.concatenate(out, 0)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)

    corpus = read_tsv_first3m(CORPUS_TSV)
    corpus = list({c[0]: {'title': c[2], 'text': c[1]} for c in corpus}.values())  # paper cid2corpus values
    raw = Path(args.out) / 'raw_data'
    for ds in args.datasets:
        od = raw / f'{ds}_inf'; od.mkdir(parents=True, exist_ok=True)
        stats = {}
        for L in SRC[ds]['lengths']:
            data = read_jsonl(raw / f'{ds}_train_{L}.jsonl')
            golds, negs, n_ws = [], [], 0
            for d in data:
                gs = []
                for g in d['golds']:
                    n_ws += (g['title'] != g['title'].strip()) or (g['text'] != g['text'].strip())
                    gs.append({'title': g['title'].strip(), 'text': g['text'].strip()})  # FIX-1
                golds.extend(gs)
                for _ in range(L):  # paper get_random_embeddings: one negative per gold
                    n = rng.choice(corpus)
                    while check_context_same(gs, n):
                        n = rng.choice(corpus)
                    negs.append(n)
            assert all(len(d['golds']) == L for d in data)
            pe = embed(golds).reshape(len(data), L, -1).astype(np.float32)
            ne = embed(negs).reshape(len(data), L, -1).astype(np.float32)
            np.save(od / f'{ds}_train_positive_embeddings_{L}.npy', pe)
            np.save(od / f'{ds}_train_random_embeddings_{L}.npy', ne)
            write_jsonl([{'negatives': [{'title': n['title'], 'text': n['text'][:80]} for n in negs[i * L:(i + 1) * L]]}
                         for i in range(len(data))], od / f'{ds}_train_negatives_preview_{L}.jsonl')
            stats[L] = {'n': len(data), 'golds_with_whitespace_before_strip': n_ws, 'shape': list(pe.shape)}
            print(ds, L, stats[L], flush=True)
        json.dump(stats, open(od / 'embed_report.json', 'w'), indent=1)


# ---------------------------------------------------------------- hf
def stage_hf(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from create_input_for_contrievers import create_input_embeddings_for_contrastive  # paper code, unchanged
    from datasets import load_from_disk, concatenate_datasets
    raw = Path(args.out) / 'raw_data'
    for lm in args.lms:
        for ds in args.datasets:
            lens = SRC[ds]['lengths']
            od = Path(args.out) / 'training_datasets' / lm / ds / 'inf'; od.mkdir(parents=True, exist_ok=True)
            parts = []
            for L in lens:
                p = od / f'autoregressive_{ds}_inf_train_dataset_1b_contrastive_{L}_ctxs'
                create_input_embeddings_for_contrastive(
                    batch_size=1, model_name=LM_IDS[lm],
                    input_data_path=raw / f'{ds}_train_question_only_{L}_ctxs.jsonl',
                    positive_embeddings_path=raw / f'{ds}_inf' / f'{ds}_train_positive_embeddings_{L}.npy',
                    negative_embeddings_path=raw / f'{ds}_inf' / f'{ds}_train_random_embeddings_{L}.npy',
                    out_dataset_path=str(p))
                parts.append(load_from_disk(str(p)))
            comb = concatenate_datasets(parts)  # paper combine_datasets (order: increasing gold count)
            comb.save_to_disk(str(od / f'autoregressive_{ds}_inf_train_dataset_1b_contrastive_{lens[0]}_to_{lens[-1]}_ctxs'))
            print(lm, ds, 'rows', len(comb), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['filter', 'embed', 'hf'])
    ap.add_argument('--out', default=f'{REPO}/paper_repro_data')
    ap.add_argument('--datasets', nargs='+', default=['qampari', 'ambiguous_qe'])
    ap.add_argument('--lms', nargs='+', default=list(LM_IDS))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    {'filter': stage_filter, 'embed': stage_embed, 'hf': stage_hf}[a.stage](a)
