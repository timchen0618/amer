import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up professional styling for research paper
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 3.0,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
})

# Actual data from the file
# Each subset has a 4x4 array (4 base models × 4 model types: [single, query exp, re-ranking, multi])
data_subsets = {
    'AmbigQA, Whole Set': np.array([
        [53.08,57.44,53.45,55.14], 
        [56.23,56.71,56.35,60.82], 
        [60.82,61.19,61.31,63.60], 
        [57.92,59.01,58.40,63.36] 
    ]),
    'AmbigQA, Low Similarity Set': np.array([
        [42.18,44.36,43.27,42.91],
        [44.36,42.91,44.73,49.45],
        [45.82,47.27,46.91,50.55],
        [45.45,45.45,45.45,49.45]
    ]),
    'QAMPARI, Whole Set': np.array([
        [7.53,5.65,6.78,8.85],
        [6.03,5.46,4.52,7.72],
        [5.08,5.08,5.27,5.84],
        [9.60,7.72,9.42,9.98]
    ]),
    'QAMPARI, Low Similarity Set': np.array([
        [2.26,2.26,2.26,5.08],
        [1.13,1.69,1.69,3.95],
        [1.13,1.13,1.69,4.52],
        [5.65,3.39,5.65,9.04]
    ])
}







# Create figure with 2×2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Define professional colors for 4 model types
colors = {
    'single': '#1f77b4',      # Professional blue
    'query_exp': '#ff7f0e',   # Professional orange
    'reranking': '#2ca02c',   # Professional green
    'multi': '#d62728'        # Professional red
}

# Base model labels (customize based on your actual models)
base_models = ['Llama-1B', 'Llama-3B', 'Qwen3-4B', 'Llama-8B']
x_positions = np.arange(len(base_models))

# Bar width and positioning for grouped bars
bar_width = 0.2
bar_positions = {
    'single': x_positions - 1.5 * bar_width,
    'query_exp': x_positions - 0.5 * bar_width,
    'reranking': x_positions + 0.5 * bar_width,
    'multi': x_positions + 1.5 * bar_width
}

# Plot each subset
subset_names = list(data_subsets.keys())
for i, (subset_name, score_array) in enumerate(data_subsets.items()):
    # Convert linear index to 2D coordinates for 2x2 layout
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    # Extract scores for all 4 model types
    single_scores = score_array[:, 0]
    query_exp_scores = score_array[:, 1]
    reranking_scores = score_array[:, 2]
    multi_scores = score_array[:, 3]
    
    # Model types and their corresponding data
    model_types = ['single', 'query_exp', 'reranking', 'multi']
    model_labels = ['Single', 'Query Exp', 'Re-ranking', 'AMER']
    scores_list = [single_scores, query_exp_scores, reranking_scores, multi_scores]
    
    # Plot bars for all 4 model types
    for j, (model_type, scores, label) in enumerate(zip(model_types, scores_list, model_labels)):
        ax.bar(bar_positions[model_type], scores,
               width=bar_width,
               color=colors[model_type],
               label=label,
               alpha=0.8,
               edgecolor='white',
               linewidth=0.8)
    
    # Professional styling for each subplot
    if row == 1:  # Only add x-label to bottom row
        ax.set_xlabel('Base Models', fontsize=16, fontweight='medium', fontfamily='Arial')
    if col == 0:  # Only add y-label to left column
        ax.set_ylabel('Performance Score', fontsize=16, fontweight='medium', fontfamily='Arial')
    ax.set_title(f'{subset_name}', fontsize=18, fontweight='bold', pad=15)
    
    # Customize grid
    ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.4)
    ax.set_axisbelow(True)
    
    # Set axis limits and ticks for bar graph
    ax.set_xlim(-0.5, len(base_models) - 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(base_models, fontsize=15, fontfamily='Arial')
    
    # Set y-axis limits based on data range (start from 0 for bar charts)
    if 'AmbigQA' in subset_name:
        y_min = 0
        y_max = 80
    elif 'QAMPARI' in subset_name:
        y_min = 0
        y_max = 12
    ax.set_ylim(y_min, y_max)
    
    # Set y-tick label font
    ax.tick_params(axis='y', labelsize=15)
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Add subtle background
    ax.set_facecolor('#fafafa')

# Adjust layout to make room for legend
plt.tight_layout(pad=1.2)

# Add single legend for the entire figure outside the plots
# Get legend handles and labels from the first subplot
handles, labels = axes[0, 0].get_legend_handles_labels()

# Create a single legend positioned outside the figure
fig.legend(handles, labels,
          loc='lower center',
          bbox_to_anchor=(0.5, -0.02),  # Position below the plots
          ncol=4,  # 4 columns for all model types
          fontsize=18,
          frameon=True,
          fancybox=True,
          shadow=True,
          borderpad=0.2,
          columnspacing=1.5,
          handlelength=2.0,
          handletextpad=0.6)

# Adjust bottom margin to accommodate the legend
plt.subplots_adjust(bottom=0.11)

# Save with high quality for research paper
plt.savefig('figures/single_vs_multi_comparison.pdf',
           dpi=300,
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none',
           format='pdf')

plt.savefig('figures/single_vs_multi_comparison.png',
           dpi=300,
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none',
           format='png')

plt.show()

print("Single vs Multi comparison plots saved as:")
print("  - figures/single_vs_multi_comparison.pdf (for paper)")
print("  - figures/single_vs_multi_comparison.png (for preview)")

# Print summary statistics
print("\n" + "="*60)
print("PERFORMANCE COMPARISON SUMMARY")
print("="*60)

for subset_name, score_array in data_subsets.items():
    single_scores = score_array[:, 0]
    multi_scores = score_array[:, 1]
    improvement = ((multi_scores - single_scores) / single_scores * 100)
    
    print(f"\n{subset_name}:")
    print(f"  Single Model - Mean: {single_scores.mean():.3f}, Std: {single_scores.std():.3f}")
    print(f"  Multi Model  - Mean: {multi_scores.mean():.3f}, Std: {multi_scores.std():.3f}")
    print(f"  Average Improvement: {improvement.mean():.1f}%")
