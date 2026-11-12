# Fig. 5 style with 3 seeds: L2 and cosine distance between the generated query embeddings of each AMER run,
# mean over seeds 0-2 with ±1 sd error bars; paper values as black ticks. Target-gold distances come from fig_stats.json
# (they are a property of the training data, not of a seed).
# AmbigQA AMER = multi_SS (multiplier 1); QAMPARI AMER = multi_SS_srm10 (multiplier 10).
# usage: python plot_seed_div.py <fig_stats.json> <out_dir>
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

E = '/scratch/hc3337/projects/autoregressive/paper_repro_data/eval'
LMS = ['llama-1b', 'llama-3b', 'qwen3-4b', 'llama-8b']
LM_LABELS = ['Llama-1B', 'Llama-3B', 'Qwen3-4B', 'Llama-8B']
DATASETS = [('ambiguous_qe', 'AmbigQA'), ('qampari', 'QAMPARI')]
AMER_RUN = {'ambiguous_qe': 'multi_SS', 'qampari': 'multi_SS_srm10'}
SEEDS = [0, 1, 2]
# Paper values hard-coded in plot_output_similarity.py @fe0bd78 (cosine stored as similarity).
PAPER = {'ambiguous_qe': {'l2': [0.42, 0.322, 0.457, 0.462, 0.558], 'cos_sim': [0.90, 0.948, 0.894, 0.891, 0.843]},
         'qampari': {'l2': [0.53, 0.564, 0.808, 0.799, 0.847], 'cos_sim': [0.86, 0.828, 0.661, 0.671, 0.63]}}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8, 'axes.edgecolor': '#cccccc', 'axes.grid': True, 'grid.color': '#dddddd',
    'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 0, 'ytick.major.size': 0,
})
SANS = {'fontfamily': ['DejaVu Sans']}


def pairwise(x):
    """x: (n, k, d) -> (mean cosine distance, mean L2 distance of unit vectors) over within-question pairs."""
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    S = np.einsum('nid,njd->nij', x, x)
    iu = np.triu_indices(x.shape[1], 1)
    cos = S[:, iu[0], iu[1]]
    return float((1 - cos).mean()), float(np.sqrt(np.clip(2 - 2 * cos, 0, None)).mean())


stats = json.load(open(sys.argv[1])); out = sys.argv[2]
os.makedirs(out, exist_ok=True)
seed = {}  # (ds, lm) -> {'cos_dist': [..], 'l2': [..]}
for ds, _ in DATASETS:
    for lm in LMS:
        q = f'{E}/{lm}/{ds}/qemb'
        nq = np.load(f'{q}/{lm}_single_0.npy', mmap_mode='r').shape[0]
        r = {'cos_dist': [], 'l2': []}
        for s in SEEDS:
            x = np.load(f'{q}/{lm}_{AMER_RUN[ds]}_{s}.npy').astype(np.float32)
            cd, l2 = pairwise(x.reshape(nq, -1, x.shape[-1]))
            r['cos_dist'].append(cd); r['l2'].append(l2)
        seed[(ds, lm)] = r
        print(ds, lm, 'cos_sim', [round(1 - v, 3) for v in r['cos_dist']], 'l2', [round(v, 3) for v in r['l2']], flush=True)

settings = ['Between Targets'] + LM_LABELS
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x_pos = np.arange(len(DATASETS)) * 1.2
bw = 0.18
for ax, metric, title in [(axes[0], 'l2', 'L2 Distance'), (axes[1], 'cos_dist', 'Cosine Distance')]:
    for i, setting in enumerate(settings):
        mu, sd, ref = [], [], []
        for ds, _ in DATASETS:
            if i == 0:
                mu.append(stats['targets'][ds][metric]); sd.append(0)
            else:
                v = seed[(ds, LMS[i - 1])][metric]; mu.append(np.mean(v)); sd.append(np.std(v, ddof=1))
            p = PAPER[ds]['l2' if metric == 'l2' else 'cos_sim'][i]
            ref.append(p if metric == 'l2' else 1 - p)
        xs = x_pos + i * bw
        ax.bar(xs, mu, bw, yerr=[e if e else np.nan for e in sd], label=setting, color=colors[i], alpha=0.8,
               edgecolor='white', linewidth=1, error_kw=dict(ecolor='#333333', elinewidth=1, capsize=3))
        ax.scatter(xs, ref, marker='_', s=260, color='black', linewidths=2.2, zorder=3,
                   label='Paper (Fig. 5)' if i == 0 else None)
    ax.set_title(title, fontsize=23, fontweight='bold', pad=15)
    ax.set_xlabel('Dataset', fontsize=23, fontweight='medium', **SANS)
    ax.set_xticks(x_pos + bw * 2)
    ax.set_xticklabels([d for _, d in DATASETS], fontsize=18, **SANS)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_axisbelow(True)
    ax.set_facecolor('#fafafa')
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=16, frameon=True)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{out}/fig_output_similarity_3seed.{ext}', dpi=200, bbox_inches='tight')
