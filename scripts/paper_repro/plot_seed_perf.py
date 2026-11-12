# Fig. 4 style with 3 seeds: single-query vs AMER (RRF), mean over seeds 0-2 with ±1 sd error bars.
# Full 25.86M index. AmbigQA AMER = multi_SS (multiplier 1); QAMPARI AMER = multi_SS_srm10 (multiplier 10).
# usage: python plot_seed_perf.py <out_dir>
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

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8, 'axes.edgecolor': '#cccccc', 'axes.grid': True, 'grid.color': '#dddddd',
    'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 0, 'ytick.major.size': 0,
})
SANS = {'fontfamily': ['Arial', 'DejaVu Sans']}
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
scores = {(ds, lm): json.load(open(f'{E}/{lm}/{ds}/scores.json')) for ds, _ in DATASETS for lm in LMS}


def vals(ds, lm, run, fusion, key):
    d = scores[(ds, lm)]
    return [d[next(k for k in d if k.endswith(f'|{lm}_{run}_{s}|{fusion}'))][key] for s in SEEDS]


fig, axes = plt.subplots(2, 2, figsize=(10, 8.4))
xp = np.arange(len(LMS))
bw = 0.35
methods = [('single', '#1f77b4', 'Single-query'), ('amer', '#d62728', 'AMER (RRF)')]
table = {}
for r, (ds, dname) in enumerate(DATASETS):
    for c, (key, sname) in enumerate([('mrecall@100_all', 'Whole Set'), ('mrecall@100_lowsim', 'Low Similarity Set')]):
        ax = axes[r, c]
        for j, (m, color, label) in enumerate(methods):
            # the single-query run produces one embedding, so its RR and RRF lists are identical
            v = np.array([vals(ds, lm, 'single', 'rr', key) if m == 'single' else vals(ds, lm, AMER_RUN[ds], 'rrf', key) for lm in LMS])
            mu, sd = v.mean(1), v.std(1, ddof=1)
            table[(ds, key, m)] = list(zip(mu.round(2), sd.round(2)))
            bars = ax.bar(xp + (j - 0.5) * bw, mu, bw, yerr=sd, color=color, label=label, alpha=0.8,
                          edgecolor='white', linewidth=0.8, error_kw=dict(ecolor='#333333', elinewidth=1, capsize=3))
            for x, top, m_ in zip(xp + (j - 0.5) * bw, mu + sd, mu):  # value label above the error bar
                ax.annotate(f'{m_:.1f}', (x, top), xytext=(0, 2), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9)
        ax.set_title(f'{dname}, {sname}', fontsize=18, fontweight='bold', pad=15)
        if r == 1:
            ax.set_xlabel('Base Models', fontsize=16, fontweight='medium', **SANS)
        if c == 0:
            ax.set_ylabel('MRecall@100', fontsize=16, fontweight='medium', **SANS)
        ax.set_xticks(xp)
        ax.set_xticklabels(LM_LABELS, fontsize=12, **SANS)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.set_axisbelow(True)
        ax.set_facecolor('#fafafa')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=2, fontsize=14, frameon=True)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{out}/fig_performance_3seed.{ext}', dpi=200, bbox_inches='tight')
for k, v in table.items():
    print(k, v)
