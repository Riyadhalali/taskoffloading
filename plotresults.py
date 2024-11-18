# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data
# servers = ['EdgeServer_1', 'EdgeServer_2', 'EdgeServer_3', 'Cloud']
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
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data for Proposed Algorithm
# proposed_local_times = [6.944444444444445, 4.0, 7.0, 1.3636363636363635]
# proposed_cloud_times = [0.0789473684210526, 0.17763157894736836]
# proposed_edge_servers = ['EdgeServer1', 'EdgeServer2', 'EdgeServer2', 'EdgeServer3']
#
# # Data for Edge Server offload tasks to cloud Algorithm
# offload_local_times = [9.722222222222221, 4.444444444444445, 6.944444444444445, 10.0]
# offload_cloud_times = [0.4005374507959224, 0.534813766585396]
# offload_edge_servers = ['EdgeServer_1', 'EdgeServer_2', 'EdgeServer_2', 'EdgeServer_3']
#
# # Combining server labels for local and cloud tasks
# combined_edge_servers = proposed_edge_servers + offload_edge_servers
#
# # Plotting
# plt.figure(figsize=(14, 7))
#
# # Plot local task processing times
# bar_positions_local = np.arange(len(proposed_local_times)) * 2
# plt.bar(bar_positions_local, proposed_local_times, width=0.4, label='Proposed Algorithm - Local', align='center')
# plt.bar(bar_positions_local + 0.4, offload_local_times, width=0.4, label='Offload to Cloud Algorithm - Local', align='center')
#
# # Plot cloud task processing times
# bar_positions_cloud = np.arange(len(proposed_cloud_times)) * 2 + len(proposed_local_times) + 1
# plt.bar(bar_positions_cloud, proposed_cloud_times, width=0.4, color='skyblue', label='Proposed Algorithm - Cloud', align='center')
# plt.bar(bar_positions_cloud + 0.4, offload_cloud_times, width=0.4, color='orange', label='Offload to Cloud Algorithm - Cloud', align='center')
#
# # Adding labels and title
# plt.xlabel('Task Type and Processing Location')
# plt.ylabel('Processing Time (s)')
# plt.title('Comparison of Task Processing Times Between Algorithms')
#
# # Set x-ticks correctly by concatenating local and cloud server names
# x_ticks = list(bar_positions_local) + list(bar_positions_cloud)
# plt.xticks(x_ticks, combined_edge_servers, rotation=45)
#
# plt.legend()
#
# # Display the plot
# plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data into a DataFrame
data = pd.DataFrame({
    'Edge Server': ['EdgeServer1', 'EdgeServer2', 'EdgeServer3', 'EdgeServer3', 'EdgeServer1', 'EdgeServer3'],
    'Local Start Time': [0.689836, 1.003402, 1.217896, 1.489836, 0.432444, 0.566938],
    'Local End Time': [3.189836, 4.203402, 6.520927, 6.944381, 0.531128, 0.690293],
    'Cloud Start Time': [0.689836, 1.003402, 1.217896, 1.489836, 0.432444, 0.566938],
    'Cloud End Time': [0.787704, 1.137435, 1.784834, 1.784834, 0.530827, 0.690086],
    'Task Type': ['Local', 'Local', 'Local', 'Local', 'Cloud', 'Cloud'],
    'Processing Time': [2.5, 3.2, 5.303030, 5.454545, 0.098684, 0.123355]
})

# Calculate the processing times for local and cloud tasks
local_times = data[data['Task Type'] == 'Local']['Processing Time']
cloud_times = data[data['Task Type'] == 'Cloud']['Processing Time']

# Get the unique edge server names
edge_servers = data['Edge Server'].unique()

# Plotting
plt.figure(figsize=(12, 6))

# Plot local task processing times
bar_positions_local = np.arange(len(local_times))
plt.bar(bar_positions_local, local_times, width=0.4, label='Local Task', align='center')

# Plot cloud task processing times
bar_positions_cloud = np.arange(len(cloud_times)) + len(local_times) + 1
plt.bar(bar_positions_cloud, cloud_times, width=0.4, label='Cloud Task', align='center')

# Adding labels and title
plt.xlabel('Edge Server')
plt.ylabel('Processing Time (s)')
plt.title('Comparison of Task Processing Times Between Algorithms')

# Set x-ticks and labels
plt.xticks(list(bar_positions_local) + list(bar_positions_cloud), edge_servers, rotation=45)

plt.legend()

# Display the plot
plt.show()