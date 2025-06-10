import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Load the data
df = pd.read_csv(r"C:\Users\sajan.muhammad\Documents\To publish\To send\8, .0001\Defoult_8.csv")
df['iteration'] = df['iteration'] * 16

# Set global font properties for readability
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'DejaVu Sans',
    'axes.labelweight': 'bold',
    'axes.titlesize': 20,
    'axes.labelsize': 22,
    'legend.fontsize': 20,
    'legend.title_fontsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})

# Set figure size
plt.figure(figsize=(12, 8))

# Plotting with high-contrast colors
plt.plot(df['iteration'], df['enn'], label='ENN-GFN', color='blue', linestyle='-', linewidth=2.5)
plt.plot(df['iteration'], df['enn_imp'], label='ENN-GFN-Enhanced', color='red', linestyle='-', linewidth=2.5)
plt.plot(df['iteration'], df['tsgfn'], label='TS-GFN', color='green', linestyle='-', linewidth=2.5)
plt.plot(df['iteration'], df['defoult'], label='Default-GFN', color='black', linestyle='-', linewidth=2.5)

# Axis ticks
plt.xticks([0, 25000, 50000, 75000, 100000])
plt.yticks()

# Y-axis formatting
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

# Axis labels
plt.xlabel('Trajectories Sampled', fontsize=24, fontweight='bold')
plt.ylabel('L1 Distance', fontsize=24, fontweight='bold')

# Legend with clarity
plt.legend(loc='best', frameon=False)

# Improve layout
plt.tight_layout()
plt.grid(False)

# Save the plot with better rendering
plt.savefig("L1distance.pdf", format='pdf', bbox_inches='tight')




