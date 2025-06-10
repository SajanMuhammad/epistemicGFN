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
plt.plot(df['iteration'], df['enn_m'], label='ENN_GFN', color='blue', linestyle='-', linewidth=2)
plt.plot(df['iteration'], df['enn_en_m'], label='ENN_GFN_Enhanced', color='red', linestyle='-', linewidth=2)
plt.plot(df['iteration'], df['ts-gfn_m'], label='T_GFN', color='green', linestyle='-', linewidth=2)
plt.plot(df['iteration'], df['defoult_m'], label='Default_GFN', color='black', linestyle='-', linewidth=2)

# Axis ticks
plt.xticks([0, 25000, 50000, 75000, 100000], fontsize=14, fontweight='bold')
plt.yticks([0, 500, 1000, 1500,2000, 2500, 3000, 3500],fontsize=14, fontweight='bold')

# Y-axis formatting
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))

# Axis labels
plt.xlabel('Trajectories Sampled', fontsize=18, fontweight='bold')
plt.ylabel('Modes Found', fontsize=18, fontweight='bold')

# Legend (extra large and bold)
plt.legend(prop={'weight': 'bold', 'size': 24}, loc='best')

# Layout and display
plt.tight_layout()
plt.grid(True)
plt.show()