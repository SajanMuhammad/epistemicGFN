import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Load the data
df = pd.read_csv(r"C:\Users\sajan.muhammad\Documents\To publish\To send\4D,16,.001\enn_gfn-enh.csv")

# Multiply the 'iteration' column by 16
df['iteration'] = df['iteration'] * 16

# Set figure size
plt.figure(figsize=(12, 8))  # Slightly larger for clarity

# Plotting
plt.plot(df['iteration'], df['enn'], label='ENN-GFN', color='blue', linestyle='-', linewidth=2)
plt.plot(df['iteration'], df['enn_enh'], label='ENN-GFN-Enhanced', color='red', linestyle='-', linewidth=2)
plt.plot(df['iteration'], df['ts-gfn'], label='TS-GFN', color='green', linestyle='-', linewidth=2)
plt.plot(df['iteration'], df['defoult'], label='Default-GFN', color='black', linestyle='-', linewidth=2)

# Axis ticks
#plt.xticks([0, 25000, 50000, 75000, 100000], fontsize=22, fontweight='bold')
plt.xticks(fontsize=22, fontweight='bold')
plt.yticks(fontsize=22, fontweight='bold')

# Y-axis formatting
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))

# Axis labels
plt.xlabel('Trajectories Sampled', fontsize=30, fontweight='bold')
plt.ylabel('L1 Distance', fontsize=30, fontweight='bold')

# Legend (extra large and bold)
plt.legend(prop={'weight': 'bold', 'size': 30}, loc='best')

# Layout and display
plt.tight_layout()
plt.grid(False)

# Save the plot as a PDF
plt.savefig("L1distance.pdf", format='pdf')
