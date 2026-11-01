# Paper-style figures for the reproduction, following plot_output_similarity.py (Fig. 5) and
# plot_single_vs_multi.py (Fig. 4) at fe0bd78. Full 25.86M index, RRF; QAMPARI AMER = sampling multiplier 10.
# usage: python plot_figs.py <fig_stats.json> <out_dir>
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

# seaborn "whitegrid" / "paper" look of the original scripts, without seaborn
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8, 'axes.edgecolor': '#cccccc', 'axes.grid': True, 'grid.color': '#dddddd',
    'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 0, 'ytick.major.size': 0,
})
SANS = {'fontfamily': ['Arial', 'DejaVu Sans']}

stats_path, out = sys.argv[1], sys.argv[2]
stats = json.load(open(stats_path))
os.makedirs(out, exist_ok=True)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}/{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------- Fig. 5 style: diversity of the generated query embeddings ----------
# Paper values hard-coded in plot_output_similarity.py @fe0bd78 (cosine stored as similarity).
PAPER = {'ambiguous_qe': {'l2': [0.42, 0.322, 0.457, 0.462, 0.558], 'cos_sim': [0.90, 0.948, 0.894, 0.891, 0.843]},
         'qampari': {'l2': [0.53, 0.564, 0.808, 0.799, 0.847], 'cos_sim': [0.86, 0.828, 0.661, 0.671, 0.63]}}
settings = ['Between Targets'] + LM_LABELS
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x_pos = np.arange(len(DATASETS)) * 1.2
bw = 0.18
for ax, metric, title in [(axes[0], 'l2', 'L2 Distance'), (axes[1], 'cos_dist', 'Cosine Distance')]:
    for i, setting in enumerate(settings):
        vals, ref = [], []
        for ds, _ in DATASETS:
            v = stats['targets'][ds] if i == 0 else stats['runs'][f'{ds}|{LMS[i - 1]}|{AMER_RUN[ds]}']
            vals.append(v[metric])
            p = PAPER[ds]['l2' if metric == 'l2' else 'cos_sim'][i]
            ref.append(p if metric == 'l2' else 1 - p)
        xs = x_pos + i * bw
        ax.bar(xs, vals, bw, label=setting, color=colors[i], alpha=0.8, edgecolor='white', linewidth=1)
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
save(fig, 'fig_output_similarity')

# ---------- Fig. 4 style: single-query vs AMER, whole and low-similarity sets ----------
scores = {(ds, lm): json.load(open(f'{E}/{lm}/{ds}/scores.json')) for ds, _ in DATASETS for lm in LMS}


def get(ds, lm, run, fusion, key):
    d = scores[(ds, lm)]
    return d[next(k for k in d if k.endswith(f'|{lm}_{run}_0|{fusion}'))][key]


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
xp = np.arange(len(LMS))
bw = 0.35
methods = [('single', '#1f77b4', 'Single'), ('amer', '#d62728', 'AMER')]
for r, (ds, dname) in enumerate(DATASETS):
    for c, (key, sname) in enumerate([('mrecall@100_all', 'Whole Set'), ('mrecall@100_lowsim', 'Low Similarity Set')]):
        ax = axes[r, c]
        for j, (m, color, label) in enumerate(methods):
            # the single-query run produces one embedding, so its RR and RRF lists are identical
            vals = [get(ds, lm, 'single', 'rr', key) if m == 'single' else get(ds, lm, AMER_RUN[ds], 'rrf', key) for lm in LMS]
            bars = ax.bar(xp + (j - 0.5) * bw, vals, bw, color=color, label=label, alpha=0.8, edgecolor='white', linewidth=0.8)
            ax.bar_label(bars, fmt='%.1f', fontsize=9, padding=2)
        ax.set_title(f'{dname}, {sname}', fontsize=18, fontweight='bold', pad=15)
        if r == 1:
            ax.set_xlabel('Base Models', fontsize=16, fontweight='medium', **SANS)
        if c == 0:
            ax.set_ylabel('MRecall@100', fontsize=16, fontweight='medium', **SANS)
        ax.set_xticks(xp)
        ax.set_xticklabels(LM_LABELS, fontsize=12, **SANS)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        ax.set_axisbelow(True)
        ax.set_facecolor('#fafafa')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=2, fontsize=14, frameon=True)
fig.tight_layout()
save(fig, 'fig_performance')
print('saved to', out)
