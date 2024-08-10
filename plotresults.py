# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data
# servers = ['EdgeServer_0', 'EdgeServer_1', 'EdgeServer_2', 'Cloud']
# avg_times = [9.72, 5.65, 10.00, 0.40]
#
# # Create bar chart
# fig, ax = plt.subplots(figsize=(10, 6))
# bars = ax.bar(servers, avg_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
#
# # Customize the chart
# ax.set_ylabel('Average Processing Time')
# ax.set_title('Comparison of Average Processing Times')
# ax.set_ylim(0, max(avg_times) * 1.1)  # Set y-axis limit with some headroom
#
# # Add value labels on top of each bar
# for bar in bars:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height,
#             f'{height:.2f}',
#             ha='center', va='bottom')
#
# # Add a horizontal line for the overall average
# overall_avg = np.mean(avg_times)
# ax.axhline(y=overall_avg, color='r', linestyle='--', label=f'Overall Average: {overall_avg:.2f}')
#
# # Add legend
# ax.legend()
#
# # Show the plot
# plt.tight_layout()
# plt.show()


# v6.1
import matplotlib.pyplot as plt
import numpy as np

# Data for Proposed Algorithm
proposed_local_times = [6.944444444444445, 4.0, 7.0, 1.3636363636363635]
proposed_cloud_times = [0.0789473684210526, 0.17763157894736836]
proposed_edge_servers = ['EdgeServer1', 'EdgeServer2', 'EdgeServer2', 'EdgeServer3']

# Data for Edge Server offload tasks to cloud Algorithm
offload_local_times = [9.722222222222221, 4.444444444444445, 6.944444444444445, 10.0]
offload_cloud_times = [0.4005374507959224, 0.534813766585396]
offload_edge_servers = ['EdgeServer_1', 'EdgeServer_2', 'EdgeServer_2', 'EdgeServer_3']

# Plotting
plt.figure(figsize=(14, 7))

# Plot local task processing times
plt.bar(np.arange(len(proposed_local_times)), proposed_local_times, width=0.4, label='Proposed Algorithm - Local', align='center')
plt.bar(np.arange(len(offload_local_times)) + 0.4, offload_local_times, width=0.4, label='Offload to Cloud Algorithm - Local', align='center')

# Plot cloud task processing times
plt.bar(np.arange(len(proposed_cloud_times)) + 4.8, proposed_cloud_times, width=0.4, color='skyblue', label='Proposed Algorithm - Cloud', align='center')
plt.bar(np.arange(len(offload_cloud_times)) + 5.2, offload_cloud_times, width=0.4, color='orange', label='Offload to Cloud Algorithm - Cloud', align='center')

# Adding labels and title
plt.xlabel('Task Type and Processing Location')
plt.ylabel('Processing Time (s)')
plt.title('Comparison of Task Processing Times Between Algorithms')
plt.xticks(np.arange(len(proposed_local_times) + len(proposed_cloud_times)),
           proposed_edge_servers + offload_edge_servers, rotation=45)
plt.legend()

# Save the plot as a file
plt.savefig('/mnt/data/comparison_plot.png')

# Display the plot
plt.show()


