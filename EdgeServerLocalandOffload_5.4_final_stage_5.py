import simpy
import pandas as pd
import heapq
import random
import numpy as np

# create multiple edge servers with robin
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
        self.q_table = np.random.uniform(low=-1, high=1,
                                         size=(state_size, action_size))  # Initialize with random values
        self.total_reward = 0

    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q
        print(f"Updated Q-value for state {state}, action {action}: {new_q:.2f}")

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
        self.offloaded_tasks = []
        self.action = env.process(self.run())
        self.network = Network(env)
        self.current_load = 0
        self.max_load = cpu_power * 2
        self.max_concurrent_tasks = max_concurrent_tasks
        self.current_tasks = 0
        self.rl_agent = RLAgent(state_size=10000, action_size=2)  # Reduced state size for simplicity


    def add_task(self, task):
        if self.current_tasks < self.max_concurrent_tasks:
            heapq.heappush(self.task_queue, task)
            self.current_tasks += 1
            print(f"{self.name} accepted task (priority {task.priority}, complexity {task.complexity})")
        else:
            print(
                f"{self.name} offloading task (priority {task.priority}, complexity {task.complexity}) to cloud due to capacity")
            self.env.process(self.offload_to_cloud(task))

    def run(self):
        while True:
            if self.task_queue:
                task = heapq.heappop(self.task_queue)
                self.current_tasks -= 1

                if self.current_load >= self.max_load or self.should_offload(task):
                    self.env.process(self.offload_to_cloud(task))
                else:
                    self.env.process(self.process_locally(task))
            else:
                yield self.env.timeout(1)

    def get_state(self, task):
        load_state = min(int(self.current_load / self.max_load * 10), 9)
        network_state = min(int(self.network.latency / 5), 9)
        complexity_state = min(task.complexity - 1, 9)  # Assuming complexity starts from 1
        priority_state = min(task.priority - 1, 9)  # Assuming priority starts from 1
        return load_state * 1000 + network_state * 100 + complexity_state * 10 + priority_state

    def should_offload(self, task):
        state = self.get_state(task)
        action = self.rl_agent.get_action(state)
        print(
            f"Deciding for task (priority {task.priority}, complexity {task.complexity}): {'Offload' if action == 1 else 'Local'}")
        return action == 1  # 1 for offload, 0 for local processing

    def update_rl_agent(self, initial_state, action, reward, next_state):
        self.rl_agent.update_q_table(initial_state, action, reward, next_state)
        self.rl_agent.decay_exploration()

    def offload_to_cloud(self, task):
        initial_state = self.get_state(task)
        start_time = self.env.now
        print(f'{self.name} offloading task (priority {task.priority}, complexity {task.complexity}) at {self.env.now}')
        transfer_time = self.network.transfer_time(task.data_size)
        yield self.env.timeout(transfer_time)
        yield self.env.process(self.cloud_env.process_task(task, self.name))
        end_time = self.env.now
        processing_time = end_time - start_time

        # Calculate reward with additional factors
        priority_bonus = task.priority  # Reward for higher priority tasks
        load_penalty = 0  # No penalty for offloading, adjust if needed
        reward = -processing_time + priority_bonus - load_penalty

        self.rl_agent.total_reward += reward  # Update total reward
        next_state = self.get_state(task)
        self.update_rl_agent(initial_state, 1, reward, next_state)
        self.offloaded_tasks.append(
            (start_time, end_time, task.duration, task.complexity, task.priority, 'Offloaded'))

    def process_locally(self, task):
        initial_state = self.get_state(task)
        start_time = self.env.now
        print(
            f'{self.name} processing task (priority {task.priority}, complexity {task.complexity}) locally at {self.env.now}')
        processing_time = self.calculate_local_processing_time(task)
        self.current_load += task.complexity
        yield self.env.timeout(processing_time)
        self.current_load -= task.complexity
        end_time = self.env.now

        # Calculate reward with additional factors
        priority_bonus = task.priority  # Reward for higher priority tasks
        load_penalty = 0.1 * task.complexity  # Penalty for increasing server load, adjust factor as needed
        reward = -processing_time + priority_bonus - load_penalty

        self.rl_agent.total_reward += reward  # Update total reward
        next_state = self.get_state(task)
        self.update_rl_agent(initial_state, 0, reward, next_state)
        self.local_tasks.append((start_time, end_time, task.duration, task.complexity, task.priority, 'Local'))
        print(
            f'{self.name} completed local task (priority {task.priority}, complexity {task.complexity}) at {self.env.now}. Processing time: {end_time - start_time:.2f}')

    def calculate_local_processing_time(self, task):
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 10))


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
            print(
                f'Cloud processed task (priority {task.priority}, complexity {task.complexity}) from {device_name} at {self.env.now}')

    def calculate_cloud_processing_time(self, task):
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 12))


# Setup and start the simulation
env = simpy.Environment()

# Create cloud environment with specified capabilities
cloud_env = CloudEnvironment(env, num_servers=2, cpu_power=32.0, memory=64)


def assign_task_priority_based(task, edge_servers):
    # Sort servers by current load (ascending)
    sorted_servers = sorted(edge_servers, key=lambda server: server.current_load)

    # Assign high priority tasks to less loaded servers
    if task.priority == 3:  # Assuming 3 is the highest priority
        sorted_servers[0].add_task(task)
    elif task.priority == 2:
        # Assign to the middle server in terms of load
        sorted_servers[len(sorted_servers) // 2].add_task(task)
    else:  # Low priority tasks
        # Assign to the most loaded server
        sorted_servers[-1].add_task(task)


# Modify the task distribution part of the code:
# Create multiple edge servers
num_edge_servers = 3  # You can change this number as needed
edge_servers = []
for i in range(num_edge_servers):
    edge_server = EdgeServer(env, f'EdgeServer_{i}', cloud_env, cpu_power=2.0, memory=8, max_concurrent_tasks=5)
    edge_servers.append(edge_server)

# Add tasks with different priorities, complexities, and data sizes
tasks = [
    Task(duration=5, complexity=4, priority=2, data_size=10),
    Task(duration=3, complexity=3, priority=1, data_size=8),
    Task(duration=7, complexity=5, priority=3, data_size=15),
    Task(duration=4, complexity=4, priority=2, data_size=12),
    Task(duration=6, complexity=6, priority=1, data_size=18),
    Task(duration=5, complexity=5, priority=2, data_size=14),
]

# Distribute tasks among edge servers based on priority
for task in tasks:
    assign_task_priority_based(task, edge_servers)

# Run the simulation
env.run(until=20000)

# Print summary of locally processed tasks for each edge server
for edge_server in edge_servers:
    print(f"\nSummary of locally processed tasks for {edge_server.name}:")
    for task in edge_server.local_tasks:
        start_time, end_time, duration, complexity, priority, _ = task
        print(f"Priority: {priority}, Complexity: {complexity}, Processing Time: {end_time - start_time:.2f}")

# Print summary of cloud processed tasks
print("\nSummary of cloud processed tasks:")
for task in cloud_env.processed_tasks:
    device_name, start_time, end_time, duration, complexity, priority, _ = task
    print(f"Device: {device_name}, Priority: {priority}, Complexity: {complexity}, Processing Time: {end_time - start_time:.2f}")

# Collect and process data
all_data = []
for edge_server in edge_servers:
    all_data.extend([(edge_server.name, *task) for task in edge_server.local_tasks])
all_data.extend([task for task in cloud_env.processed_tasks])

# Create a DataFrame
df = pd.DataFrame(all_data, columns=['Device', 'Start Time', 'End Time', 'Duration', 'Complexity', 'Priority', 'Type'])

# Calculate processing time
df['Processing Time'] = df['End Time'] - df['Start Time']

# Print the DataFrame
print("\nAll tasks:")
print(df)

# Print comparison
print("\nComparison of processing times:")
for _, row in df.iterrows():
    print(f"Device: {row['Device']}, Type: {row['Type']}, Priority: {row['Priority']}, Complexity: {row['Complexity']}, Processing Time: {row['Processing Time']}")

# Calculate and print average processing time for local tasks per edge server
for edge_server in edge_servers:
    local_df = df[(df['Type'] == 'Local') & (df['Device'] == edge_server.name)]
    avg_local_processing_time = local_df['Processing Time'].mean()
    print(f"\nAverage processing time for local tasks on {edge_server.name}: {avg_local_processing_time:.2f}")

# Calculate and print average processing time for cloud tasks
# Calculate and print average processing time for local tasks

# Calculate and print average processing time for cloud tasks
cloud_df = df[df['Type'] == 'Cloud']
avg_cloud_processing_time = cloud_df['Processing Time'].mean()   # mean used for average in row calculate in pandas
print(f"\nAverage processing time for cloud tasks: {avg_cloud_processing_time:.2f}")
local_df = df[df['Type'] == 'Local']
avg_local_processing_time = local_df['Processing Time'].mean()   # mean used for average in row calculate in panadas
print(f"\nAverage processing time for local tasks: {avg_local_processing_time:.2f}")

# Optionally, save to a CSV file
df.to_csv('simulation_results.csv', index=False)

# Print the Q-table for each edge server
for edge_server in edge_servers:
    print(f"\nQ-table for {edge_server.name}:")
    non_zero = np.count_nonzero(edge_server.rl_agent.q_table)
    print(f"Number of non-zero elements in Q-table: {non_zero}")
   # print("Sample of Q-table (first 10 rows, all columns):")
    print(edge_server.rl_agent.q_table[:10])

    # Print total reward for RL agent
  #  print(f"\nTotal reward for RL agent on {edge_server.name}: {edge_server.rl_agent.total_reward}")

