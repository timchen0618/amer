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
# Each subset has a 4x2 array (4 base models × 2 model types: [single, multi])
data_subsets = {
    'AmbigQA, Whole Set': np.array([
        [63.48, 64.09],
        [56.23, 60.82],
        [60.82, 63.60],
        [57.92, 63.36]
    ]),
    'AmbigQA, Low Similarity Set': np.array([
        [48.73, 51.27],
        [44.36, 49.45],
        [45.82, 50.55],
        [45.45, 49.45]
    ]),
    'QAMPARI, Whole Set': np.array([
        [6.59, 8.66],
        [6.03, 7.72],
        [5.08, 5.84],
        [9.60, 9.98]
    ]),
    'QAMPARI, Low Similarity Set': np.array([
        [1.13, 3.95],
        [1.13, 3.95],
        [1.13, 4.52],
        [5.65, 9.04]
    ])
}







# Create figure with 2×2 subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 7))

# Define professional colors
colors = {
    'single': '#1f77b4',  # Professional blue
    'multi': '#d62728'    # Professional red
}

# Base model labels (customize based on your actual models)
base_models = ['Llama-1b', 'Llama-3b', 'Qwen3-4b', 'Llama-8b']
x_positions = np.arange(len(base_models))

# Plot each subset
subset_names = list(data_subsets.keys())
for i, (subset_name, score_array) in enumerate(data_subsets.items()):
    # Convert linear index to 2D coordinates
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    # Extract single and multi scores
    single_scores = score_array[:, 0]
    multi_scores = score_array[:, 1]
    
    # Plot lines for single and multi models
    ax.plot(x_positions, single_scores,
           color=colors['single'],
           marker='o',
           linestyle='-',
           linewidth=3.0,
           markersize=10,
           markerfacecolor=colors['single'],
           markeredgecolor='white',
           markeredgewidth=1.5,
           label='Single',
           alpha=0.9)
    
    ax.plot(x_positions, multi_scores,
           color=colors['multi'],
           marker='s',  # Square marker for distinction
           linestyle='-',
           linewidth=3.0,
           markersize=10,
           markerfacecolor=colors['multi'],
           markeredgecolor='white',
           markeredgewidth=1.5,
           label='Multi',
           alpha=0.9)
    
    # Professional styling for each subplot
    if row == 1:  # Only add x-label to bottom row
        ax.set_xlabel('Base Models', fontsize=18, fontweight='medium', fontfamily='Arial')
    if col == 0:  # Only add y-label to left column
        ax.set_ylabel('Performance Score', fontsize=16, fontweight='medium', fontfamily='Arial')
    ax.set_title(f'{subset_name}', fontsize=18, fontweight='bold', pad=15)
    
    # Customize grid
    ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.4)
    ax.set_axisbelow(True)
    
    # Set axis limits and ticks
    ax.set_xlim(-0.2, len(base_models) - 0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(base_models, fontsize=14, fontfamily='Arial')
    
    # Set y-axis limits based on data range
    # y_min = min(single_scores.min(), multi_scores.min()) - 0.02
    # y_max = max(single_scores.max(), multi_scores.max()) + 0.02
    if 'AmbigQA' in subset_name:
        y_min = 40
        y_max = 80
    elif 'QAMPARI' in subset_name:
        y_min = 0
        y_max = 20
    ax.set_ylim(y_min, y_max)
    
    # Set y-tick label font
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Add subtle background
    ax.set_facecolor('#fafafa')
    
    # Remove individual subplot legends - we'll add a single figure legend instead
    if i == 1:
        legend = ax.legend(loc='upper right',
                          frameon=True,
                          fancybox=True,
                          shadow=True,
                          borderpad=1,
                          columnspacing=1.5,
                          handlelength=2.5,
                          handletextpad=0.8,
                          fontsize=16)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(0.5)

# # Tight layout with minimal padding
plt.tight_layout(pad=0.5)

# # Add single legend for the entire figure
# # Get legend handles and labels from the first subplot
# handles, labels = axes[0, 0].get_legend_handles_labels()

# # Create a single legend positioned within the figure bounds
# fig.legend(handles, labels,
#           loc='upper center',
#           bbox_to_anchor=(0.5, 0.95),  # Position at top within figure
#           ncol=2,  # 2 columns for Single and Multi
#           fontsize=14,
#           frameon=True,
#           fancybox=True,
#           shadow=True,
#           borderpad=0.5,
#           columnspacing=1.5,
#           handlelength=2.0,
#           handletextpad=0.6)

# No need for bottom adjustment since legend is at top
# plt.subplots_adjust(bottom=0.1)

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
