import simpy
import pandas as pd
import heapq
import random

class Task:
    def __init__(self, duration, complexity, priority, data_size, deadline, energy_requirement, io_intensity, memory_requirement, task_type):
        self.duration = duration
        self.complexity = complexity
        self.priority = priority
        self.data_size = data_size  # in MB
        self.deadline = deadline
        self.energy_requirement = energy_requirement  # in Joules
        self.io_intensity = io_intensity  # scale of 1-10
        self.memory_requirement = memory_requirement  # in MB
        self.task_type = task_type  # e.g., 'computation', 'data', 'io'

    def __lt__(self, other):
        # Compare tasks based on priority (lower number means higher priority)
        return self.priority < other.priority
class EdgeServer:
    def __init__(self, env, name, cloud_env, cpu_power, memory, energy_capacity, network_bandwidth):
        self.env = env
        self.name = name
        self.cloud_env = cloud_env
        self.cpu_power = cpu_power
        self.memory = memory
        self.energy_capacity = energy_capacity
        self.network_bandwidth = network_bandwidth
        self.task_queue = []
        self.local_tasks = []
        self.offloaded_tasks = []
        self.action = env.process(self.run())

    def add_task(self, task):
        heapq.heappush(self.task_queue, (task.priority, task))

    def run(self):
        while True:
            if self.task_queue:
                _, task = heapq.heappop(self.task_queue)

                # Decide whether to process locally or offload
                if self.should_offload(task):
                    # Offload to cloud
                    start_time = self.env.now
                    print(f'{self.name} offloading task (priority {task.priority}) at {self.env.now}')
                    transfer_time = task.data_size / self.network_bandwidth
                    yield self.env.timeout(transfer_time)
                    yield self.env.process(self.cloud_env.process_task(task, self.name))
                    end_time = self.env.now
                    self.offloaded_tasks.append(
                        (start_time, end_time, task.duration, task.complexity, task.priority, 'Offloaded', task.data_size, task.deadline, task.energy_requirement, task.io_intensity, task.memory_requirement, task.task_type))
                else:
                    # Process locally
                    start_time = self.env.now
                    print(f'{self.name} processing task (priority {task.priority}) locally at {self.env.now}')
                    processing_time = self.calculate_local_processing_time(task)
                    yield self.env.timeout(processing_time)
                    end_time = self.env.now
                    self.local_tasks.append(
                        (start_time, end_time, task.duration, task.complexity, task.priority, 'Local', task.data_size, task.deadline, task.energy_requirement, task.io_intensity, task.memory_requirement, task.task_type))
            else:
                yield self.env.timeout(1)

    def should_offload(self, task):
        # Simple decision logic: offload if task requires more than 50% of available resources
        return (task.memory_requirement > self.memory / 2 or
                task.energy_requirement > self.energy_capacity / 2 or
                task.io_intensity > 7)

    def calculate_local_processing_time(self, task):
        base_time = (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 8))
        io_factor = 1 + (task.io_intensity / 10)
        return base_time * io_factor

class CloudEnvironment:
    def __init__(self, env, num_servers, cpu_power, memory, energy_capacity):
        self.env = env
        self.servers = simpy.Resource(env, num_servers)
        self.cpu_power = cpu_power
        self.memory = memory
        self.energy_capacity = energy_capacity
        self.processed_tasks = []

    def process_task(self, task, device_name):
        with self.servers.request() as request:
            yield request
            start_time = self.env.now
            processing_time = self.calculate_cloud_processing_time(task)
            yield self.env.timeout(processing_time)
            end_time = self.env.now
            self.processed_tasks.append(
                (device_name, start_time, end_time, task.duration, task.complexity, task.priority, 'Cloud', task.data_size, task.deadline, task.energy_requirement, task.io_intensity, task.memory_requirement, task.task_type))
            print(f'Cloud processed task (priority {task.priority}) from {device_name} at {self.env.now}')

    def calculate_cloud_processing_time(self, task):
        base_time = (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 16))
        io_factor = 1 + (task.io_intensity / 20)
        return base_time * io_factor

# Setup and start the simulation
env = simpy.Environment()

# Create cloud environment with specified capabilities
cloud_env = CloudEnvironment(env, num_servers=1, cpu_power=4.0, memory=32, energy_capacity=1000)

# Create edge server with specified capabilities
edge_server = EdgeServer(env, 'EdgeServer', cloud_env, cpu_power=2.0, memory=8, energy_capacity=500, network_bandwidth=10)

# Add tasks with different characteristics
for i in range(10):
    task = Task(
        duration=random.uniform(1, 10),
        complexity=random.uniform(1, 10),
        priority=random.randint(1, 5),
        data_size=random.uniform(10, 1000),
        deadline=random.uniform(50, 200),
        energy_requirement=random.uniform(10, 100),
        io_intensity=random.randint(1, 10),
        memory_requirement=random.uniform(100, 1000),
        task_type=random.choice(['computation', 'data', 'io'])
    )
    edge_server.add_task(task)

# Run the simulation
env.run(until=500)

# Collect and process data
all_data = (
        [(edge_server.name, *task) for task in edge_server.local_tasks] +
        [(edge_server.name, *task) for task in edge_server.offloaded_tasks] +
        [task for task in cloud_env.processed_tasks]
)

# Create a DataFrame
df = pd.DataFrame(all_data, columns=['Device', 'Start Time', 'End Time', 'Duration', 'Complexity', 'Priority', 'Type', 'Data Size', 'Deadline', 'Energy Requirement', 'IO Intensity', 'Memory Requirement', 'Task Type'])

# Calculate processing time
df['Processing Time'] = df['End Time'] - df['Start Time']

# Print the DataFrame
print(df)

# Print comparison
print("\nComparison of processing times:")
for _, row in df.iterrows():
    print(f"Type: {row['Type']}, Priority: {row['Priority']}, Processing Time: {row['Processing Time']}, Task Type: {row['Task Type']}")

# Optionally, save to a CSV file
df.to_csv('simulation_results_enhanced.csv', index=False)