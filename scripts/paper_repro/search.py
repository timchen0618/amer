"""Paper-repro eval step 2. Search all saved paper-8B query embeddings against one corpus using the paper's own Indexer (fe0bd78 src/retrieval_utils).
Same procedure as fe0bd78 gen_ret_and_eval.retrieve(): 16 file-shards, IndexFlatIP on GPU, top_k_per_query per shard, merge by score.
Keeps the top 500 per query vector. Output: pickle {run: (ids [nvec][500], scores)}"""
import sys, glob, os, pickle, numpy as np, time
sys.path.insert(0, os.environ['PAPER_CODE'])
from src.retrieval_utils import Indexer, index_encoded_data
corpus_glob, qdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
K, NSH = 500, 16
runs = sorted(os.path.basename(p)[:-4] for p in glob.glob(qdir + "/*.npy"))
Q = {r: np.load(f'{qdir}/{r}.npy').astype('float32') for r in runs}
for r in runs: print(r, Q[r].shape, flush=True)
allq = np.concatenate([Q[r] for r in runs]); offs = np.cumsum([0] + [len(Q[r]) for r in runs])
files = sorted(glob.glob(corpus_glob))
ids_parts, sc_parts = [], []
for sh in range(NSH):
    t = time.time()
    index = Indexer(1536, 0, 8, use_gpu=True)
    index_encoded_data(index, files, 100000, shard_id=sh, num_shards=NSH)
    res = index.search_knn(allq, K)
    ids_parts.append(np.array([r[0] for r in res], dtype=object)); sc_parts.append(np.array([r[1] for r in res], dtype=np.float32))
    print(f'shard {sh} n={len(index.index_id_to_db_id)} {time.time()-t:.0f}s', flush=True)
    del index
ids = np.concatenate(ids_parts, axis=1); sc = np.concatenate(sc_parts, axis=1)
order = np.argsort(-sc, axis=1, kind='stable')[:, :K]
ids = np.take_along_axis(ids, order, 1); sc = np.take_along_axis(sc, order, 1)
pickle.dump({r: (ids[offs[i]:offs[i+1]].tolist(), sc[offs[i]:offs[i+1]]) for i, r in enumerate(runs)}, open(out, 'wb'))
print('saved', out, 'total passages indexed across shards checked above')
