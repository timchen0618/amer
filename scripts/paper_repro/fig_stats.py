# Distance statistics for the Fig. 5-style plot: mean pairwise cosine distance and L2 distance (unit-normalized)
# between the k generated query embeddings of each AMER run, and between the gold targets of each training example.
# usage: python fig_stats.py <out.json>
import glob, json, os, sys
import numpy as np
from datasets import load_from_disk

D = '/scratch/hc3337/projects/autoregressive/paper_repro_data'
LMS = ['llama-1b', 'llama-3b', 'qwen3-4b', 'llama-8b']
RANGE = {'ambiguous_qe': '2_to_5', 'qampari': '5_to_8'}


def pairwise(x):
    """x: (n, k, d) -> (mean cosine distance, mean L2 distance) over all within-row pairs, averaged per row then over rows."""
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    S = np.einsum('nid,njd->nij', x, x)
    iu = np.triu_indices(x.shape[1], 1)
    cos = S[:, iu[0], iu[1]]
    l2 = np.sqrt(np.clip(2 - 2 * cos, 0, None))
    return float((1 - cos).mean()), float(l2.mean())


out = {'runs': {}, 'targets': {}}
for ds in RANGE:
    for lm in LMS:
        q = f'{D}/eval/{lm}/{ds}/qemb'
        nq = np.load(f'{q}/{lm}_single_0.npy', mmap_mode='r').shape[0]
        for f in sorted(glob.glob(f'{q}/{lm}_multi_SS*_0.npy')):
            run = os.path.basename(f)[len(lm) + 1:-6]  # multi_SS or multi_SS_srm10
            x = np.load(f).astype(np.float32)
            cd, l2 = pairwise(x.reshape(nq, -1, x.shape[-1]))
            out['runs'][f'{ds}|{lm}|{run}'] = {'cos_dist': cd, 'l2': l2, 'cos_sim': 1 - cd}
            print(ds, lm, run, round(1 - cd, 3), round(l2, 3), flush=True)
    # Gold targets are retriever embeddings, identical across LMs: use the 1B training set.
    tr = load_from_disk(f'{D}/training_datasets/llama-1b/{ds}/inf/autoregressive_{ds}_inf_train_dataset_1b_contrastive_{RANGE[ds]}_ctxs')
    if hasattr(tr, 'keys'):
        tr = tr['train']
    cds, l2s = [], []
    for ex in tr.select_columns(['positive_embeddings']).with_format('numpy'):
        p = np.asarray(ex['positive_embeddings'], dtype=np.float32).reshape(1, -1, 1536)
        if p.shape[1] < 2:
            continue
        cd, l2 = pairwise(p)
        cds.append(cd); l2s.append(l2)
    out['targets'][ds] = {'cos_dist': float(np.mean(cds)), 'l2': float(np.mean(l2s)), 'cos_sim': 1 - float(np.mean(cds)), 'n': len(cds)}
    print(ds, 'targets', out['targets'][ds], flush=True)

json.dump(out, open(sys.argv[1], 'w'), indent=1)
