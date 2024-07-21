import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

class Task:
    def __init__(self, duration, complexity, data_size):
        self.duration = duration
        self.complexity = complexity  # e.g., 1-10
        self.data_size = data_size  # in MB

def generate_task():
    return Task(
        duration=random.randint(1, 10),
        complexity=random.randint(1, 10),
        data_size=random.uniform(0.1, 10)  # 0.1 to 10 MB
    )

def calculate_network_delay(data_size):
    # Assume 10 MB/s transfer rate
    return data_size / 10

class EdgeServer:
    def __init__(self, env, name, cloud_env):
        self.env = env
        self.name = name
        self.cloud_env = cloud_env
        self.local_tasks = []
        self.offloaded_tasks = []
        self.action = env.process(self.run())

    def run(self):
        while True:
            task = generate_task()
            offload_decision = self.decide_offload(task)

            if offload_decision:
                start_time = self.env.now
                network_delay = calculate_network_delay(task.data_size)
                yield self.env.timeout(network_delay)
                yield self.env.process(self.cloud_env.process_task(task, self.name))
                end_time = self.env.now
                processing_time = end_time - start_time
                self.offloaded_tasks.append((self.name, start_time, end_time, task.duration, task.complexity, task.data_size, 'Offloaded', processing_time))
            else:
                start_time = self.env.now
                local_processing_time = self.local_processing_time(task)
                yield self.env.timeout(local_processing_time)
                end_time = self.env.now
                self.local_tasks.append((self.name, start_time, end_time, task.duration, task.complexity, task.data_size, 'Local', local_processing_time))

            yield self.env.timeout(random.randint(1, 5))

    def decide_offload(self, task):
        if task.complexity > 7 or task.data_size > 5 or len(self.local_tasks) > 3:
            return True
        return False

    def local_processing_time(self, task):
        return task.duration * (1 + task.complexity / 5)

class CloudEnvironment:
    def __init__(self, env, num_servers):
        self.env = env
        self.servers = simpy.Resource(env, num_servers)
        self.processed_tasks = []

    def process_task(self, task, device_name):
        with self.servers.request() as request:
            yield request
            start_time = self.env.now
            process_time = task.duration * (1 + task.complexity / 10)
            yield self.env.timeout(process_time)
            end_time = self.env.now
            self.processed_tasks.append((device_name, start_time, end_time, task.duration, task.complexity, task.data_size, 'Cloud', process_time))

# Setup and run simulation
env = simpy.Environment()
cloud_env = CloudEnvironment(env, num_servers=2)
edge_servers = [EdgeServer(env, f'EdgeServer {i}', cloud_env) for i in range(3)]
env.run(until=1000)  # Run for 1000 time units

# Collect data
columns = ['Device', 'Start Time', 'End Time', 'Duration', 'Complexity', 'Data Size', 'Type', 'Processing Time']
df = pd.DataFrame(columns=columns)

for server in edge_servers:
    df = pd.concat([df, pd.DataFrame(server.local_tasks, columns=columns)], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(server.offloaded_tasks, columns=columns)], ignore_index=True)

df = pd.concat([df, pd.DataFrame(cloud_env.processed_tasks, columns=columns)], ignore_index=True)

# Analysis and Comparison
print("\nComparison between Local and Cloud Processing:")
print("-----------------------------------------------")

local_df = df[df['Type'] == 'Local']
cloud_df = df[df['Type'] == 'Cloud']
offloaded_df = df[df['Type'] == 'Offloaded']

# Table 1: Task Count
task_count_table = [
    ["Total tasks processed", len(df)],
    ["Tasks processed locally", len(local_df)],
    ["Tasks processed in cloud", len(cloud_df)]
]
print("\nTask Count:")
print(tabulate(task_count_table, headers=["Metric", "Count"], tablefmt="grid"))

# Table 2: Average Processing Times
avg_times_table = [
    ["Local", local_df['Processing Time'].mean()],
    ["Cloud (including network delay)", offloaded_df['Processing Time'].mean()],
    ["Cloud (excluding network delay)", cloud_df['Processing Time'].mean()]
]
print("\nAverage Processing Times:")
print(tabulate(avg_times_table, headers=["Type", "Average Time"], tablefmt="grid", floatfmt=".2f"))

# Table 3: Median Processing Times
median_times_table = [
    ["Local", local_df['Processing Time'].median()],
    ["Cloud (including network delay)", offloaded_df['Processing Time'].median()],
    ["Cloud (excluding network delay)", cloud_df['Processing Time'].median()]
]
print("\nMedian Processing Times:")
print(tabulate(median_times_table, headers=["Type", "Median Time"], tablefmt="grid", floatfmt=".2f"))

# Table 4: Processing Time by Task Complexity
complexity_groups = df.groupby(['Type', 'Complexity'])['Processing Time'].mean().unstack(level=0)
print("\nProcessing Time by Task Complexity:")
print(tabulate(complexity_groups, headers="keys", tablefmt="grid", floatfmt=".2f"))

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(complexity_groups.index, complexity_groups['Local'], label='Local')
plt.plot(complexity_groups.index, complexity_groups['Cloud'], label='Cloud')
plt.plot(complexity_groups.index, complexity_groups['Offloaded'], label='Offloaded (incl. network delay)')
plt.xlabel('Task Complexity')
plt.ylabel('Average Processing Time')
plt.title('Processing Time vs Task Complexity')
plt.legend()
plt.grid(True)
plt.savefig('processing_time_comparison.png')
plt.close()

print("\nVisualization saved as 'processing_time_comparison.png'")

# Save detailed results to CSV
df.to_csv('simulation_results.csv', index=False)
print("Detailed results saved to 'simulation_results.csv'")