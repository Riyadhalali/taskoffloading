# Install necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data for RL output
data_rl = {
    "Type": ["Local", "Local", "Local", "Local", "Local", "Offloaded", "Offloaded", "Cloud", "Cloud"],
    "Priority": [1, 2, 2, 3, 1, 1, 2, 1, 2],
    "Complexity": [3, 4, 4, 5, 6, 3, 5, 3, 5],
    "Processing Time": [2.5, 4.444444, 5.555556, 9.722222, 10.0, 2.500973, 4.758116, 2.096151, 4.153293]
}

# Sample data for original output
data_original = {
    "Type": ["Local", "Local", "Local", "Local", "Local", "Offloaded", "Offloaded", "Cloud", "Cloud"],
    "Priority": [1, 2, 2, 3, 1, 1, 2, 1, 2],
    "Complexity": [3, 4, 4, 5, 6, 3, 5, 3, 5],
    "Processing Time": [2.5, 4.444444, 5.555556, 9.722222, 10.0, 1.979999, 2.488570, 1.649035, 2.117606]
}

# Convert to DataFrames
df_rl = pd.DataFrame(data_rl)
df_original = pd.DataFrame(data_original)

# Filter out offloaded tasks
df_rl_local = df_rl[df_rl["Type"] == "Local"]
df_original_local = df_original[df_original["Type"] == "Local"]

# Plotting
plt.figure(figsize=(12, 6))

# Plot for RL output
plt.subplot(1, 2, 1)
sns.barplot(x='Priority', y='Processing Time', data=df_rl_local, hue='Complexity', palette='viridis')
plt.title('RL Output - Local Tasks')
plt.xlabel('Priority')
plt.ylabel('Processing Time')

# Plot for Original output
plt.subplot(1, 2, 2)
sns.barplot(x='Priority', y='Processing Time', data=df_original_local, hue='Complexity', palette='viridis')
plt.title('Original Output - Local Tasks')
plt.xlabel('Priority')
plt.ylabel('Processing Time')

plt.tight_layout()
plt.show()
