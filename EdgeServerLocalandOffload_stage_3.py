import simpy
import pandas as pd
import heapq
import matplotlib.pyplot as plt
import random


# EXAMPLE FOR TASK OFFLOADING WITH NETWORK LATENCY
class Network:
    def __init__(self, env):
        self.env = env
        self.latency = 50  # Initial latency in ms
        self.bandwidth = 100  # Mbps
        self.env.process(self.fluctuate_latency())

    def fluctuate_latency(self):
        while True:
            self.latency = random.uniform(10, 100)  # ms
            yield self.env.timeout(60)  # Update every minute

    def transfer_time(self, data_size):
        # data_size in MB
        return (data_size * 8 / self.bandwidth) + (self.latency / 1000)

class Task:
    def __init__(self, duration, complexity, priority):
        self.duration = duration
        self.complexity = complexity
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

class EdgeServer:
    def __init__(self, env, name, cloud_env, cpu_power, memory):
        self.env = env
        self.name = name
        self.cloud_env = cloud_env
        self.cpu_power = cpu_power
        self.memory = memory
        self.task_queue = []
        self.local_tasks = []
        self.offloaded_tasks = []
        self.action = env.process(self.run())
        self.network = Network(env)

    def add_task(self, task):
        heapq.heappush(self.task_queue, task)

    def run(self):
        while True:
            if self.task_queue:
                task = heapq.heappop(self.task_queue)

                if task.complexity > 5:
                    # Offload to cloud
                    start_time = self.env.now
                    print(f'{self.name} offloading task (priority {task.priority}, complexity {task.complexity}) at {self.env.now}')
                    transfer_time = self.network.transfer_time(task.duration)
                    yield self.env.timeout(transfer_time)
                    yield self.env.process(self.cloud_env.process_task(task, self.name))
                    end_time = self.env.now
                    self.offloaded_tasks.append(
                        (start_time, end_time, task.duration, task.complexity, task.priority, 'Offloaded'))
                else:
                    # Process locally
                    start_time = self.env.now
                    print(f'{self.name} processing task (priority {task.priority}, complexity {task.complexity}) locally at {self.env.now}')
                    processing_time = self.calculate_local_processing_time(task)
                    yield self.env.timeout(processing_time)
                    end_time = self.env.now
                    self.local_tasks.append((start_time, end_time, task.duration, task.complexity, task.priority, 'Local'))
            else:
                yield self.env.timeout(1)

    def calculate_local_processing_time(self, task):
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 8))

class CloudEnvironment:
    def __init__(self, env, num_servers, cpu_power, memory):
        self.env = env
        self.servers = simpy.Resource(env, num_servers)
        self.cpu_power = cpu_power
        self.memory = memory
        self.processed_tasks = []
        self.network = Network(env)

    def process_task(self, task, device_name):
        with self.servers.request() as request:
            yield request
            start_time = self.env.now
            processing_time = self.calculate_cloud_processing_time(task)
            yield self.env.timeout(processing_time)
            transfer_time = self.network.transfer_time(task.duration)
            yield self.env.timeout(transfer_time)
            end_time = self.env.now
            self.processed_tasks.append(
                (device_name, start_time, end_time, task.duration, task.complexity, task.priority, 'Cloud'))
            print(f'Cloud processed task (priority {task.priority}, complexity {task.complexity}) from {device_name} at {self.env.now}')

    def calculate_cloud_processing_time(self, task):
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 16))

# Setup and start the simulation
env = simpy.Environment()

# Create cloud environment with specified capabilities
cloud_env = CloudEnvironment(env, num_servers=1, cpu_power=32.0, memory=64)

# Create edge server with specified capabilities
edge_server = EdgeServer(env, 'EdgeServer', cloud_env, cpu_power=2.0, memory=8)

# Add tasks with different priorities and complexities
edge_server.add_task(Task(duration=5, complexity=5, priority=2))
edge_server.add_task(Task(duration=3, complexity=3, priority=1))
edge_server.add_task(Task(duration=7, complexity=7, priority=3))

# Run the simulation
env.run(until=100)

# Collect and process data
all_data = (
        [(edge_server.name, *task) for task in edge_server.local_tasks] +
        #[(edge_server.name, *task) for task in edge_server.offloaded_tasks] +
        [task for task in cloud_env.processed_tasks]
)

# Create a DataFrame
df = pd.DataFrame(all_data, columns=['Device', 'Start Time', 'End Time', 'Duration', 'Complexity', 'Priority', 'Type'])

# Calculate processing time
df['Processing Time'] = df['End Time'] - df['Start Time']

def plot_processing_times(df):
    plt.figure(figsize=(10, 6))
    for task_type in df['Type'].unique():
        data = df[df['Type'] == task_type]
        plt.scatter(data['Complexity'], data['Processing Time'], label=task_type)

    plt.xlabel('Task Complexity')
    plt.ylabel('Processing Time')
    plt.title('Processing Time vs Task Complexity')
    plt.legend()
    plt.grid(True)
    plt.savefig('processing_time_vs_complexity.png')
    plt.close()

def plot_priority_processing_times(df):
    plt.figure(figsize=(10, 6))
    for task_type in df['Type'].unique():
        data = df[df['Type'] == task_type]
        plt.scatter(data['Priority'], data['Processing Time'], label=task_type)

    plt.xlabel('Task Priority')
    plt.ylabel('Processing Time')
    plt.title('Processing Time vs Task Priority')
    plt.legend()
    plt.grid(True)
    plt.savefig('processing_time_vs_priority.png')
    plt.close()

def plot_task_timeline(df):
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, task in df.iterrows():
        ax.barh(task['Type'],
                task['End Time'] - task['Start Time'],
                left=task['Start Time'],
                height=0.5,
                align='center',
                alpha=0.8,
                label=f"Priority {task['Priority']}, Complexity {task['Complexity']}")

    ax.set_xlabel('Simulation Time')
    ax.set_ylabel('Task Type')
    ax.set_title('Task Processing Timeline')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('task_timeline.png')
    plt.close()

plot_processing_times(df)
plot_priority_processing_times(df)
plot_task_timeline(df)

# Print the DataFrame
print(df)

# Print comparison
print("\nComparison of processing times:")
for _, row in df.iterrows():
    print(f"Type: {row['Type']}, Priority: {row['Priority']}, Complexity: {row['Complexity']}, Processing Time: {row['Processing Time']}")

# Optionally, save to a CSV file
df.to_csv('simulation_results.csv', index=False)