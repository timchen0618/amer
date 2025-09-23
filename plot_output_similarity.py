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

# Data from the file
data = {
    'AmbigQA': {
        'L2': [0.42, ],
        'Cosine': [0.90, ]
    },
    'QAMPARI': {
        'L2': [0.53,],
        'Cosine': [0.86, ]
    },
    'MLP Data': {
        'L2': [1.448, ],
        'Cosine': [ -0.051,]
    },
    'Linear Data': {
        'L2': [1.531, 1.398],
        'Cosine': [-0.2, -0.13]
    }
}


## Positive Correlation Data
data_points = [(0.42, 1), (0.56, 5.2), (0.53, 31.4), (0.61, 249.6), (1.448, 363)]

# Setting labels (removed "Between Queries")
settings = ['Between\nTargets (Same Ex)', 'Between\nOutputs (Same Ex)']

# Create figure with two subplots (L2 and Cosine)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Define colors for each dataset
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
datasets = list(data.keys())

# X positions for bars - group by dataset
x_pos = np.arange(len(datasets))
bar_width = 0.25
positions = [x_pos + i * bar_width for i in range(len(settings))]

# Plot L2 Distance (skip index 0 which is "Between Queries")
for i, setting in enumerate(settings):
    l2_values = [data[dataset]['L2'][i] for dataset in datasets]  # i+1 to skip first index
    ax1.bar(positions[i], l2_values, bar_width, 
           label=setting, color=colors[i], alpha=0.8, edgecolor='white', linewidth=1)

ax1.set_xlabel('Dataset', fontsize=20, fontweight='medium', fontfamily='Arial')
ax1.set_ylabel('L2 Distance', fontsize=18, fontweight='medium', fontfamily='Arial')
ax1.set_title('L2 Distance', fontsize=23, fontweight='bold', pad=15)
ax1.set_xticks(x_pos + bar_width/2)  # Adjust centering for 2 bars instead of 3
ax1.set_xticklabels(datasets, fontsize=16, fontfamily='Arial')
ax1.tick_params(axis='y', labelsize=16)  # Show y-tick labels again
for label in ax1.get_yticklabels():
    label.set_fontfamily('Arial')

# Customize grid for L2
ax1.grid(True, alpha=0.6, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)
ax1.set_facecolor('#fafafa')

# Add legend for L2
legend1 = ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                    borderpad=1, columnspacing=1.5, handlelength=2.5, handletextpad=0.8,
                    fontsize=16)
legend1.get_frame().set_facecolor('white')
legend1.get_frame().set_alpha(0.9)
legend1.get_frame().set_edgecolor('gray')
legend1.get_frame().set_linewidth(0.5)

# Plot Cosine Distance (1 - Cosine Similarity, skip index 0 which is "Between Queries")
for i, setting in enumerate(settings):
    cosine_distance_values = [1 - data[dataset]['Cosine'][i+1] for dataset in datasets]  # i+1 to skip first index
    ax2.bar(positions[i], cosine_distance_values, bar_width, 
           label=setting, color=colors[i], alpha=0.8, edgecolor='white', linewidth=1)

ax2.set_xlabel('Dataset', fontsize=20, fontweight='medium', fontfamily='Arial')
ax2.set_ylabel('Cosine Distance', fontsize=18, fontweight='medium', fontfamily='Arial')
ax2.set_title('Cosine Distance', fontsize=23, fontweight='bold', pad=15)
ax2.set_xticks(x_pos + bar_width/2)  # Adjust centering for 2 bars instead of 3
ax2.set_xticklabels(datasets, fontsize=16, fontfamily='Arial')
ax2.tick_params(axis='y', labelsize=16)  # Show y-tick labels again
for label in ax2.get_yticklabels():
    label.set_fontfamily('Arial')

# Customize grid for Cosine
ax2.grid(True, alpha=0.6, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)
ax2.set_facecolor('#fafafa')

# Add legend for Cosine
legend2 = ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                    borderpad=1, columnspacing=1.5, handlelength=2.5, handletextpad=0.8,
                    fontsize=16)
legend2.get_frame().set_facecolor('white')
legend2.get_frame().set_alpha(0.9)
legend2.get_frame().set_edgecolor('gray')
legend2.get_frame().set_linewidth(0.5)

# Adjust layout
plt.tight_layout(pad=2.0)

# Save the plots
plt.savefig('figures/data_similarity_analysis.pdf', 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none',
           format='pdf')

plt.savefig('figures/data_similarity_analysis.png', 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none',
           format='png')

plt.show()

print("Data similarity analysis plots saved as:")
print("  - figures/data_similarity_analysis.pdf (for paper)")
print("  - figures/data_similarity_analysis.png (for preview)")

# Print summary statistics
print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)

for dataset in datasets:
    print(f"\n{dataset}:")
    print(f"  L2 Distance:")
    print(f"    Target-Target (Same): {data[dataset]['L2'][1]:.3f}")
    print(f"    Target-Target (Diff): {data[dataset]['L2'][2]:.3f}")
    print(f"  Cosine Distance:")
    print(f"    Target-Target (Same): {1 - data[dataset]['Cosine'][1]:.3f}")
    print(f"    Target-Target (Diff): {1 - data[dataset]['Cosine'][2]:.3f}")
