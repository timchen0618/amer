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
        'L2': [0.42, 0.322, 0.457, 0.462, 0.558],
        'Cosine': [0.90, 0.948, 0.894, 0.891, 0.843],
    },
    'QAMPARI': {
        'L2': [0.53, 0.564, 0.808, 0.799, 0.847],
        'Cosine': [0.86, 0.828, 0.661, 0.671, 0.63],
    }
}


# Setting labels with single-line text for better legend display
settings = ['Between Targets', 'Llama-1b', 'Llama-3b', 'Qwen3-4b', 'Llama-8b']

# Create figure with two subplots (L2 and Cosine)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Define colors for each setting
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
datasets = list(data.keys())

# X positions for bars - group by dataset with balanced spacing
x_pos = np.arange(len(datasets)) * 1.2  # Moderate spacing between datasets
bar_width = 0.18  # Slightly wider bars for better visibility
positions = [x_pos + i * bar_width for i in range(len(settings))]

# Plot L2 Distance
for i, setting in enumerate(settings):
    l2_values = [data[dataset]['L2'][i] for dataset in datasets]
    ax1.bar(positions[i], l2_values, bar_width, 
           label=setting, color=colors[i], alpha=0.8, edgecolor='white', linewidth=1)

ax1.set_xlabel('Dataset', fontsize=23, fontweight='medium', fontfamily='Arial')
# ax1.set_ylabel('L2 Distance', fontsize=18, fontweight='medium', fontfamily='Arial')
ax1.set_title('L2 Distance', fontsize=23, fontweight='bold', pad=15)
ax1.set_xticks(x_pos + bar_width * 2)  # Center the tick marks among the 5 bars
ax1.set_xticklabels(datasets, fontsize=18, fontfamily='Arial')
ax1.tick_params(axis='y', labelsize=18)  # Show y-tick labels again
for label in ax1.get_yticklabels():
    label.set_fontfamily('Arial')

# Customize grid for L2
ax1.grid(True, alpha=0.6, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)
ax1.set_facecolor('#fafafa')

# Plot Cosine Distance (1 - Cosine Similarity)
for i, setting in enumerate(settings):
    cosine_distance_values = [1 - data[dataset]['Cosine'][i] for dataset in datasets]
    ax2.bar(positions[i], cosine_distance_values, bar_width, 
           label=setting, color=colors[i], alpha=0.8, edgecolor='white', linewidth=1)

ax2.set_xlabel('Dataset', fontsize=23, fontweight='medium', fontfamily='Arial')
# ax2.set_ylabel('Cosine Distance', fontsize=18, fontweight='medium', fontfamily='Arial')
ax2.set_title('Cosine Distance', fontsize=23, fontweight='bold', pad=15)
ax2.set_xticks(x_pos + bar_width * 2)  # Center the tick marks among the 5 bars
ax2.set_xticklabels(datasets, fontsize=18, fontfamily='Arial')
ax2.tick_params(axis='y', labelsize=18)  # Show y-tick labels again
for label in ax2.get_yticklabels():
    label.set_fontfamily('Arial')

# Customize grid for Cosine
ax2.grid(True, alpha=0.6, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)
ax2.set_facecolor('#fafafa')

# Add legend for Cosine
legend2 = ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                    borderpad=0.5, columnspacing=1.1, handlelength=1.0, handletextpad=0.4,
                    fontsize=18)
legend2.get_frame().set_facecolor('white')
legend2.get_frame().set_alpha(0.9)
legend2.get_frame().set_edgecolor('gray')
legend2.get_frame().set_linewidth(0.5)

# Adjust layout
plt.tight_layout(pad=2.0)

# Save the plots
plt.savefig('figures/output_similarity_analysis.pdf', 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none',
           format='pdf')

plt.savefig('figures/output_similarity_analysis.png', 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none',
           format='png')

plt.show()

print("Output similarity analysis plots saved as:")
print("  - figures/output_similarity_analysis.pdf (for paper)")
print("  - figures/output_similarity_analysis.png (for preview)")

# Print summary statistics
print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)

for dataset in datasets:
    print(f"\n{dataset}:")
    print(f"  L2 Distance:")
    for i, setting in enumerate(settings):
        print(f"    {setting.replace(chr(10), ' ')}: {data[dataset]['L2'][i]:.3f}")
    print(f"  Cosine Distance:")
    for i, setting in enumerate(settings):
        print(f"    {setting.replace(chr(10), ' ')}: {1 - data[dataset]['Cosine'][i]:.3f}")
