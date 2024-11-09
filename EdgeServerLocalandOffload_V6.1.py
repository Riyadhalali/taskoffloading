import simpy
import pandas as pd    # for dataframes
import random
import numpy as np     # numeric library in  python
from tabulate import tabulate


class Network:
    def __init__(self, env):
        self.env = env
        self.latency = 30
        self.bandwidth = 200
        self.env.process(self.fluctuate_latency())

    # change every 60 time unit to simulate network fluctuate
    def fluctuate_latency(self):
        while True:
            self.latency = random.uniform(5, 50)
            yield self.env.timeout(60)
# transfer time = transmission time + latency
    def transfer_time(self, data_size):
        return (data_size * 8 / self.bandwidth) + (self.latency / 1000)


class Task:
    def __init__(self, duration, complexity, priority, data_size):
        self.duration = duration
        self.complexity = complexity
        self.priority = priority
        self.data_size = data_size


# sorting task using its prority
    def __lt__(self, other):
        return self.priority < other.priority


class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0,
                 exploration_decay=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.q_table = np.random.uniform(low=-1, high=1, size=(state_size, action_size))
        self.total_reward = 0


# if the action is lower the exploration rate so explor and return the action or pick the best value for the know state
    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)  # return exploration
        return np.argmax(self.q_table[state])  # return best state known for action

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q
        print(f"Updated Q-value for state {state}, action {action}: {new_q:.2f}")

# Reduce the exploration among time
    def decay_exploration(self):
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)


class EdgeServer:
    def __init__(self, env, name, cloud_env, cpu_power, memory, max_concurrent_tasks):
        self.env = env
        self.name = name
        self.cloud_env = cloud_env
        self.cpu_power = cpu_power
        self.memory = memory
        self.task_queue = []
        self.local_tasks = []
        self.network = Network(env)
        self.current_load = 0
        self.max_load = cpu_power * 2
        self.max_concurrent_tasks = max_concurrent_tasks
        self.current_tasks = 0

    def add_task(self, task):
        self.env.process(self.send_to_cloud(task))

    def send_to_cloud(self, task):
        transfer_time = self.network.transfer_time(task.data_size)
        yield self.env.timeout(transfer_time)
        self.cloud_env.receive_task(task, self)

    def process_locally(self, task):
        start_time = self.env.now
        processing_time = self.calculate_local_processing_time(task)
        self.current_load += task.complexity
        yield self.env.timeout(processing_time)
        self.current_load -= task.complexity
        end_time = self.env.now
        self.local_tasks.append((start_time, end_time, task.duration, task.complexity, task.priority, 'Local'))
        print(
            f'{self.name} completed local task (priority {task.priority}, complexity {task.complexity}) at {self.env.now}. Processing time: {end_time - start_time:.2f}')

    def calculate_local_processing_time(self, task):
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 10))


class CloudEnvironment:
    def __init__(self, env, num_servers, cpu_power, memory, num_edge_servers):
        self.env = env
        self.servers = simpy.Resource(env, num_servers)
        self.cpu_power = cpu_power
        self.memory = memory
        self.processed_tasks = []
        self.network = Network(env)
        self.rl_agent = RLAgent(state_size=10000, action_size=num_edge_servers + 1)  # +1 for cloud processing
        self.edge_servers = []
        self.task_data = []  # New attribute to store task data

    def add_edge_server(self, edge_server):
        self.edge_servers.append(edge_server)

    def receive_task(self, task, edge_server):
        state = self.get_state(task)
        load_states = [min(int(server.current_load / server.max_load * 10), 9) for server in self.edge_servers]
        network_state = min(int(self.network.latency / 5), 9)
        complexity_state = min(task.complexity - 1, 9)
        priority_state = min(task.priority - 1, 9)

        task_info = {
            "Time": self.env.now,
            "Edge Server": edge_server.name,
            "Task Duration": task.duration,
           # "Task Complexity": task.complexity,
            "Task Priority": task.priority,
        #    "Task Data Size": task.data_size,
            "Edge Server Loads": ','.join(map(str, load_states)),
        #    "Network State": network_state,
         #   "Task Complexity State": complexity_state,
            "Task Priority State": priority_state,
            "Computed State Value": state
        }
        self.task_data.append(task_info)
        self.env.process(self.decide_and_process(task, edge_server))

    def decide_and_process(self, task, edge_server):
        state = self.get_state(task)
        action = self.rl_agent.get_action(state)
        # if return 3 so process on cloud else if returns number of servers 0,1,2 it will not process on cloud it is a way to check for process on cloud
        if action == len(self.edge_servers):  # Last action corresponds to cloud processing
            action_str = "Process on cloud"
        else:  # Process on one of the edge servers
            action_str = f"Send to {self.edge_servers[action].name}"

        self.task_data[-1]["Action"] = action_str

        if action == len(self.edge_servers):
            yield self.env.process(self.process_on_cloud(task, edge_server))
        else:
            yield self.env.process(self.send_to_edge(task, self.edge_servers[action]))

    def get_state(self, task):
        load_states = [min(int(server.current_load / server.max_load * 10), 9) for server in self.edge_servers]
        network_state = min(int(self.network.latency / 5), 9)
        complexity_state = min(task.complexity - 1, 9)
        priority_state = min(task.priority - 1, 9)
        state = 0
        for i, load_state in enumerate(load_states):
            state += load_state * (10 ** (4 + i))
        state += network_state * 1000 + complexity_state * 10 + priority_state
        return state

    def send_to_edge(self, task, edge_server):
        transfer_time = self.network.transfer_time(task.data_size)
        yield self.env.timeout(transfer_time)
        yield self.env.process(edge_server.process_locally(task))

    def process_on_cloud(self, task, edge_server):
        # wait until resource on cloud as available to be handled
        with self.servers.request() as request:
            yield request
            start_time = self.env.now
            processing_time = self.calculate_cloud_processing_time(task)
            yield self.env.timeout(processing_time)
            end_time = self.env.now
            self.processed_tasks.append(
                (edge_server.name, start_time, end_time, task.duration, task.complexity, task.priority, 'Cloud'))
            print(
                f'Cloud processed task (priority {task.priority}, complexity {task.complexity}) from {edge_server.name} at {self.env.now}')


# processing_time= (task duration *  task complexity) / (cpu power * (1 + (memory / 12))
# Dividing by 12 makes the memory impact more gradual. For instance, if self.memory is 12 GB,
    # then (1 + self.memory / 12) becomes 2, doubling the effect of self.cpu_power.
    def calculate_cloud_processing_time(self, task):
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 12))


# Setup and start the simulation
env = simpy.Environment()

# Create cloud environment with specified capabilities
cloud_env = CloudEnvironment(env, num_servers=2, cpu_power=32.0, memory=64, num_edge_servers=3)

# Create 3 edge servers with specified capabilities and max concurrent tasks
edge_servers = [
    EdgeServer(env, 'EdgeServer1', cloud_env, cpu_power=2.0, memory=8, max_concurrent_tasks=5),
    EdgeServer(env, 'EdgeServer2', cloud_env, cpu_power=2.5, memory=10, max_concurrent_tasks=5),
    EdgeServer(env, 'EdgeServer3', cloud_env, cpu_power=3.0, memory=12, max_concurrent_tasks=5),
]

# Register edge servers in the cloud environment
for edge_server in edge_servers:
    cloud_env.add_edge_server(edge_server)

# Add tasks with different priorities, complexities, and data sizes
tasks = [
    Task(duration=5, complexity=4, priority=2, data_size=10),
    Task(duration=3, complexity=3, priority=1, data_size=8),
    Task(duration=7, complexity=5, priority=3, data_size=15),
    Task(duration=4, complexity=4, priority=2, data_size=12),
    Task(duration=6, complexity=6, priority=1, data_size=18),
    Task(duration=5, complexity=5, priority=2, data_size=14),
]

# Distribute tasks among all edge servers
for i, task in enumerate(tasks):
    edge_servers[i % len(edge_servers)].add_task(task)


# Run the simulation
env.run(until=20000)

# Print task data as a table
print("\nTask Data:")
headers = ["Time", "Edge Server", "Task Duration", "Task Priority",
          "Task Priority State", "Computed State Value",
           "Action"]
task_table = [[task.get(key, "N/A") for key in headers] for task in cloud_env.task_data]
print(tabulate(task_table, headers=headers, tablefmt="grid"))

# Print summary of locally processed tasks for each edge server and calculate average processing time
for edge_server in edge_servers:
    print(f"\nSummary of locally processed tasks on {edge_server.name}:")
    total_processing_time = 0
    task_count = len(edge_server.local_tasks)

    for task in edge_server.local_tasks:
        start_time, end_time, duration, complexity, priority, _ = task
        processing_time = end_time - start_time
        total_processing_time += processing_time
        print(f"Priority: {priority}, Complexity: {complexity}, Processing Time: {processing_time:.2f}")

    if task_count > 0:
        avg_processing_time = total_processing_time / task_count
        print(f"Average processing time on {edge_server.name}: {avg_processing_time:.2f}")
    else:
        print(f"No tasks processed on {edge_server.name}.")

# Print summary of cloud processed tasks
print("\nSummary of cloud processed tasks:")
for task in cloud_env.processed_tasks:
    device_name, start_time, end_time, duration, complexity, priority, _ = task
    print(
        f"Device: {device_name}, Priority: {priority}, Complexity: {complexity}, Processing Time: {end_time - start_time:.2f}")

# Collect and process data
all_data = (
        [(edge_server.name, *task) for edge_server in edge_servers for task in edge_server.local_tasks] +   # for extracting all tasks processed in edge servers
        [task for task in cloud_env.processed_tasks]
)

# Create a DataFrame
df = pd.DataFrame(all_data, columns=['Device', 'Start Time', 'End Time', 'Duration', 'Complexity', 'Priority', 'Type'])

# Calculate processing time
df['Processing Time'] = df['End Time'] - df['Start Time']

# Print the DataFrame
print("\nAll tasks:")
print(df)

# Print comparison
print("\nComparison of processing times:")
for _, row in df.iterrows():   # for looping in rows (_)
    print(
        f"Type: {row['Type']}, Priority: {row['Priority']}, Complexity: {row['Complexity']}, Processing Time: {row['Processing Time']}")

# Calculate and print average processing time for local tasks
local_df = df[df['Type'] == 'Local']
avg_local_processing_time = local_df['Processing Time'].mean()   # mean used for average in row calculate in panadas
print(f"\nAverage processing time for local tasks: {avg_local_processing_time:.2f}")

# Calculate and print average processing time for cloud tasks
cloud_df = df[df['Type'] == 'Cloud']
avg_cloud_processing_time = cloud_df['Processing Time'].mean()   # mean used for average in row calculate in pandas
print(f"\nAverage processing time for cloud tasks: {avg_cloud_processing_time:.2f}")

# Optionally, save to a CSV file
df.to_csv('simulation_results.csv', index=False)

# Print the Q-table
print("\nQ-table:")
non_zero = np.count_nonzero(cloud_env.rl_agent.q_table)
print(f"Number of non-zero elements in Q-table: {non_zero}")
print("Sample of Q-table (first 10 rows, all columns):")
print(cloud_env.rl_agent.q_table[:10])

# Print total reward for RL agent
print(f"\nTotal reward for RL agent: {cloud_env.rl_agent.total_reward}")