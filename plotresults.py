import matplotlib.pyplot as plt
import numpy as np

# Data
servers = ['EdgeServer_0', 'EdgeServer_1', 'EdgeServer_2', 'Cloud']
avg_times = [9.72, 5.65, 10.00, 0.40]

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(servers, avg_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Customize the chart
ax.set_ylabel('Average Processing Time')
ax.set_title('Comparison of Average Processing Times')
ax.set_ylim(0, max(avg_times) * 1.1)  # Set y-axis limit with some headroom

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom')

# Add a horizontal line for the overall average
overall_avg = np.mean(avg_times)
ax.axhline(y=overall_avg, color='r', linestyle='--', label=f'Overall Average: {overall_avg:.2f}')

# Add legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()