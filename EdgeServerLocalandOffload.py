import simpy
import pandas as pd


class Task:
    def __init__(self, duration, complexity):
        self.duration = duration
        self.complexity = complexity


class EdgeServer:
    def __init__(self, env, name, cloud_env, cpu_power, memory):
        self.env = env
        self.name = name
        self.cloud_env = cloud_env
        self.cpu_power = cpu_power  # in GHz
        self.memory = memory  # in GB
        self.local_tasks = []
        self.offloaded_tasks = []
        self.action = env.process(self.run())

    def run(self):
        task = Task(duration=5, complexity=5)

        # Process locally
        start_time = self.env.now
        print(f'{self.name} processing task locally at {self.env.now}')
        processing_time = self.calculate_local_processing_time(task)
        yield self.env.timeout(processing_time)
        end_time = self.env.now
        self.local_tasks.append((start_time, end_time, task.duration, task.complexity, 'Local'))

        # Offload to cloud
        start_time = self.env.now
        print(f'{self.name} offloading task at {self.env.now}')
        yield self.env.process(self.cloud_env.process_task(task, self.name))
        end_time = self.env.now
        self.offloaded_tasks.append((start_time, end_time, task.duration, task.complexity, 'Offloaded'))

    def calculate_local_processing_time(self, task):
        # Calculate processing time based on CPU power and task complexity
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 8))


class CloudEnvironment:
    def __init__(self, env, num_servers, cpu_power, memory):
        self.env = env
        self.servers = simpy.Resource(env, num_servers)
        self.cpu_power = cpu_power  # in GHz
        self.memory = memory  # in GB
        self.processed_tasks = []

    def process_task(self, task, device_name):
        with self.servers.request() as request:
            yield request
            start_time = self.env.now
            processing_time = self.calculate_cloud_processing_time(task)
            yield self.env.timeout(processing_time)
            end_time = self.env.now
            self.processed_tasks.append((device_name, start_time, end_time, task.duration, task.complexity, 'Cloud'))
            print(f'Cloud processed task from {device_name} at {self.env.now}')

    def calculate_cloud_processing_time(self, task):
        # Calculate processing time based on CPU power and task complexity
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 16))


# Setup and start the simulation
env = simpy.Environment()

# Create cloud environment with specified capabilities
cloud_env = CloudEnvironment(env, num_servers=1, cpu_power=3.0, memory=16)

# Create edge server with specified capabilities
edge_server = EdgeServer(env, 'EdgeServer', cloud_env, cpu_power=2.0, memory=8)

# Run the simulation
env.run(until=100)

# Collect and process data
all_data = (
        [(edge_server.name, *task) for task in edge_server.local_tasks] +
        [(edge_server.name, *task) for task in edge_server.offloaded_tasks] +
        [task for task in cloud_env.processed_tasks]
)

# Create a DataFrame
df = pd.DataFrame(all_data, columns=['Device', 'Start Time', 'End Time', 'Duration', 'Complexity', 'Type'])

# Calculate processing time
df['Processing Time'] = df['End Time'] - df['Start Time']

# Print the DataFrame
print(df)

# Print comparison
print("\nComparison of processing times:")
for _, row in df.iterrows():
    print(f"Type: {row['Type']}, Processing Time: {row['Processing Time']}")

# Optionally, save to a CSV file
df.to_csv('simulation_results.csv', index=False)