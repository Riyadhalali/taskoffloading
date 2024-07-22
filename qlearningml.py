import simpy
import heapq
import numpy as np
import random
import pandas as pd


class Task:
    def __init__(self, duration, complexity, priority):
        self.duration = duration
        self.complexity = complexity
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority


class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # exploration rate

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
        self.q_table[state, action] = new_q


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
        self.q_agent = QLearningAgent(state_size=100, action_size=2)
        self.action = env.process(self.run())

    def add_task(self, task):
        heapq.heappush(self.task_queue, task)

    def get_state(self, task):
        task_size = min(int(task.duration * task.complexity / 10), 9)
        queue_length = min(len(self.task_queue), 9)
        return task_size * 10 + queue_length

    def calculate_reward(self, processing_time, task):
        return -processing_time / task.priority

    def calculate_local_processing_time(self, task):
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 8))

    def run(self):
        while True:
            if self.task_queue:
                task = heapq.heappop(self.task_queue)

                state = self.get_state(task)
                action = self.q_agent.get_action(state)

                start_time = self.env.now

                if action == 1:  # Offload to cloud
                    print(f'{self.name} offloading task (priority {task.priority}) at {self.env.now}')
                    yield self.env.process(self.cloud_env.process_task(task, self.name))
                    end_time = self.env.now
                    self.offloaded_tasks.append(
                        (start_time, end_time, task.duration, task.complexity, task.priority, 'Offloaded'))
                else:  # Process locally
                    print(f'{self.name} processing task (priority {task.priority}) locally at {self.env.now}')
                    processing_time = self.calculate_local_processing_time(task)
                    yield self.env.timeout(processing_time)
                    end_time = self.env.now
                    self.local_tasks.append(
                        (start_time, end_time, task.duration, task.complexity, task.priority, 'Local'))

                processing_time = end_time - start_time
                reward = self.calculate_reward(processing_time, task)
                next_state = self.get_state(task)

                self.q_agent.update(state, action, reward, next_state)

            else:
                yield self.env.timeout(1)


class CloudEnvironment:
    def __init__(self, env, num_servers, cpu_power, memory):
        self.env = env
        self.servers = simpy.Resource(env, num_servers)
        self.cpu_power = cpu_power
        self.memory = memory
        self.processed_tasks = []

    def process_task(self, task, device_name):
        with self.servers.request() as request:
            yield request
            start_time = self.env.now
            processing_time = self.calculate_cloud_processing_time(task)
            yield self.env.timeout(processing_time)
            end_time = self.env.now
            self.processed_tasks.append(
                (device_name, start_time, end_time, task.duration, task.complexity, task.priority, 'Cloud'))
            print(f'Cloud processed task (priority {task.priority}) from {device_name} at {self.env.now}')

    def calculate_cloud_processing_time(self, task):
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 16))


# Setup and start the simulation
env = simpy.Environment()

# Create cloud environment with specified capabilities
cloud_env = CloudEnvironment(env, num_servers=1, cpu_power=3.0, memory=16)

# Create edge server with specified capabilities
edge_server = EdgeServer(env, 'EdgeServer', cloud_env, cpu_power=2.0, memory=8)

# Add 10 tasks with different priorities (lower number means higher priority)
for i in range(8):
    priority = random.randint(1, 5)
    duration = random.uniform(1, 10)
    complexity = random.uniform(1, 10)
    edge_server.add_task(Task(duration=duration, complexity=complexity, priority=priority))

# Run the simulation
env.run(until=100)  # Run for a fixed duration

# After the simulation, print the Q-table
print("Q-table:")
print(edge_server.q_agent.q_table)

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
    print(f"Type: {row['Type']}, Priority: {row['Priority']}, Processing Time: {row['Processing Time']}")

# Optionally, save to a CSV file
df.to_csv('simulation_results.csv', index=False)