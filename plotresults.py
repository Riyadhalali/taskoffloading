import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Recreate the DataFrame from the output
data = [
    ['EdgeServer', 0.00, 2.250000, 3, 3, 1, 'Local', 2.250000],
    ['EdgeServer', 2.25, 8.500000, 5, 5, 2, 'Local', 6.250000],
    ['EdgeServer', 8.50, 16.666667, 7, 7, 3, 'Offloaded', 8.166667],
    ['EdgeServer', 8.50, 16.666667, 7, 7, 3, 'Cloud', 8.166667]
]

df = pd.DataFrame(data, columns=['Device', 'Start Time', 'End Time', 'Duration', 'Complexity', 'Priority', 'Type',
                                 'Processing Time'])

plt.figure(figsize=(12, 8))

colors = {'Local': 'blue', 'Offloaded': 'red', 'Cloud': 'green'}

for task_type in df['Type'].unique():
    data = df[df['Type'] == task_type]
    plt.scatter(data['Complexity'], data['Processing Time'], label=task_type, color=colors[task_type], s=100)

    for i, row in data.iterrows():
        plt.annotate(f"({row['Complexity']}, {row['Processing Time']:.2f})",
                     (row['Complexity'], row['Processing Time']),
                     xytext=(5, 5), textcoords='offset points')

# Add trend line for local processing
local_data = df[df['Type'] == 'Local']
z = np.polyfit(local_data['Complexity'], local_data['Processing Time'], 1)
p = np.poly1d(z)
plt.plot(local_data['Complexity'], p(local_data['Complexity']), "b--", label="Local trend")

plt.xlabel('Task Complexity', fontsize=12)
plt.ylabel('Processing Time', fontsize=12)
plt.title('Processing Time vs Task Complexity', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.ylim(bottom=0)  # Start y-axis from 0

plt.tight_layout()
plt.savefig('improved_processing_time_vs_complexity.png', dpi=300)
plt.close()

print("Improved plot has been saved as 'improved_processing_time_vs_complexity.png'")