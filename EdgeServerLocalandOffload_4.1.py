import simpy
import pandas as pd
import heapq
import random


# TASK OFFLOADING BASED CURRENT LOAD OF EDGE SERVER AND IF EDGESERVER REJECT TASK TO CAN BE OFFLOADED
class Network:
    def __init__(self, env):
        self.env = env
        self.latency = 30
        self.bandwidth = 200
        self.env.process(self.fluctuate_latency())

    def fluctuate_latency(self):
        while True:
            self.latency = random.uniform(5, 50)
            yield self.env.timeout(60)

    def transfer_time(self, data_size):
        return (data_size * 8 / self.bandwidth) + (self.latency / 1000)

class Task:
    def __init__(self, duration, complexity, priority, data_size):
        self.duration = duration
        self.complexity = complexity
        self.priority = priority
        self.data_size = data_size

    def __lt__(self, other):
        return self.priority < other.priority

class EdgeServer:
    def __init__(self, env, name, cloud_env, cpu_power, memory, max_concurrent_tasks):
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
        self.current_load = 0
        self.max_load = cpu_power * 2
        self.max_concurrent_tasks = max_concurrent_tasks
        self.current_tasks = 0

    def add_task(self, task):
        if self.current_tasks < self.max_concurrent_tasks:
            heapq.heappush(self.task_queue, task)
            self.current_tasks += 1
            print(f"{self.name} accepted task (priority {task.priority}, complexity {task.complexity})")
        else:
            print(f"{self.name} offloading task (priority {task.priority}, complexity {task.complexity}) to cloud due to capacity")
            self.env.process(self.offload_to_cloud(task))

    def run(self):
        while True:
            if self.task_queue:
                task = heapq.heappop(self.task_queue)
                self.current_tasks -= 1

                if self.should_offload(task):
                    self.env.process(self.offload_to_cloud(task))
                else:
                    self.process_locally(task)
            else:
                yield self.env.timeout(1)

    def offload_to_cloud(self, task):
        start_time = self.env.now
        print(f'{self.name} offloading task (priority {task.priority}, complexity {task.complexity}) at {self.env.now}')
        transfer_time = self.network.transfer_time(task.data_size)
        yield self.env.timeout(transfer_time)
        yield self.env.process(self.cloud_env.process_task(task, self.name))
        end_time = self.env.now
        self.offloaded_tasks.append(
            (start_time, end_time, task.duration, task.complexity, task.priority, 'Offloaded'))

    def process_locally(self, task):
        start_time = self.env.now
        print(f'{self.name} processing task (priority {task.priority}, complexity {task.complexity}) locally at {self.env.now}')
        processing_time = self.calculate_local_processing_time(task)
        self.current_load += task.complexity
        yield self.env.timeout(processing_time)
        self.current_load -= task.complexity
        end_time = self.env.now
        self.local_tasks.append((start_time, end_time, task.duration, task.complexity, task.priority, 'Local'))

    def calculate_local_processing_time(self, task):
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 10))

    def should_offload(self, task):
        local_time = self.calculate_local_processing_time(task)
        cloud_time = self.cloud_env.calculate_cloud_processing_time(task)
        transfer_time = self.network.transfer_time(task.data_size) * 2
        load_factor = self.current_load / self.max_load
        time_factor = (cloud_time + transfer_time) / local_time
        complexity_factor = task.complexity / 10
        offload_score = 0.3 * time_factor + 0.4 * complexity_factor + 0.3 * load_factor
        return offload_score > 0.4

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
            transfer_time = self.network.transfer_time(task.data_size)
            yield self.env.timeout(transfer_time)
            end_time = self.env.now
            self.processed_tasks.append(
                (device_name, start_time, end_time, task.duration, task.complexity, task.priority, 'Cloud'))
            print(f'Cloud processed task (priority {task.priority}, complexity {task.complexity}) from {device_name} at {self.env.now}')

    def calculate_cloud_processing_time(self, task):
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 12))

# Setup and start the simulation
env = simpy.Environment()

# Create cloud environment with specified capabilities
cloud_env = CloudEnvironment(env, num_servers=2, cpu_power=4.0, memory=32)

# Create edge server with specified capabilities and max concurrent tasks
edge_server = EdgeServer(env, 'EdgeServer', cloud_env, cpu_power=1.5, memory=4, max_concurrent_tasks=3)

# Add tasks with different priorities, complexities, and data sizes
tasks = [
    Task(duration=5, complexity=8, priority=2, data_size=20),
    Task(duration=3, complexity=6, priority=1, data_size=15),
    Task(duration=7, complexity=9, priority=3, data_size=25),
    Task(duration=4, complexity=7, priority=2, data_size=18),
    Task(duration=6, complexity=8, priority=1, data_size=22),
    Task(duration=5, complexity=7, priority=2, data_size=19),
    Task(duration=4, complexity=5, priority=1, data_size=14)
]

for task in tasks:
    edge_server.add_task(task)

# Run the simulation
env.run(until=200)

# Collect and process data
all_data = (
        [(edge_server.name, *task) for task in edge_server.local_tasks] +
        [(edge_server.name, *task) for task in edge_server.offloaded_tasks] +
        [task for task in cloud_env.processed_tasks]
)

# Create a DataFrame
df = pd.DataFrame(all_data, columns=['Device', 'Start Time', 'End Time', 'Duration', 'Complexity', 'Priority', 'Type'])

# Calculate processing time
df['Processing Time'] = df['End Time'] - df['Start Time']

# Print the DataFrame
print(df)

# Print comparison
print("\nComparison of processing times:")
for _, row in df.iterrows():
    print(f"Type: {row['Type']}, Priority: {row['Priority']}, Complexity: {row['Complexity']}, Processing Time: {row['Processing Time']}")

# Optionally, save to a CSV file
df.to_csv('simulation_results.csv', index=False)