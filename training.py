# import simpy
# import pandas as pd
# import random
# import numpy as np
# from tabulate import tabulate
#
#
# class Network:
#     def __init__(self, env):
#         self.env = env
#         self.latency = 30
#         self.bandwidth = 200
#         self.env.process(self.fluctuate_latency())
#
#     def fluctuate_latency(self):
#         while True:
#             self.latency = random.uniform(5, 50)
#             yield self.env.timeout(60)
#
#     def transfer_time(self, data_size):
#         return (data_size * 8 / self.bandwidth) + (self.latency / 1000)
#
#
# class Task:
#     def __init__(self, duration, complexity, priority, data_size):
#         self.duration = duration
#         self.complexity = complexity
#         self.priority = priority
#         self.data_size = data_size
#         self.creation_time = None
#         self.completion_time = None
#
#     def __lt__(self, other):
#         return self.priority < other.priority
#
#
# class RLAgent:
#     def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0,
#                  exploration_decay=0.995):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate
#         self.exploration_min = 0.01
#         self.exploration_decay = exploration_decay
#         self.q_table = np.zeros((state_size, action_size))
#         self.total_reward = 0
#         self.episode_rewards = []
#
#     def get_action(self, state):
#         if np.random.rand() < self.exploration_rate:
#             return np.random.randint(self.action_size)
#         return np.argmax(self.q_table[state])
#
#     def update_q_table(self, state, action, reward, next_state):
#         current_q = self.q_table[state, action]
#         max_next_q = np.max(self.q_table[next_state])
#         new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
#         self.q_table[state, action] = new_q
#         self.total_reward += reward
#         self.episode_rewards.append(reward)
#         print(f"Updated Q-value for state {state}, action {action}: {new_q:.2f}")
#
#     def decay_exploration(self):
#         self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)
#
#     def get_stats(self):
#         return {
#             'total_reward': self.total_reward,
#             'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
#             'exploration_rate': self.exploration_rate,
#             'non_zero_q_values': np.count_nonzero(self.q_table),
#             'max_q_value': np.max(self.q_table),
#             'min_q_value': np.min(self.q_table),
#             'mean_q_value': np.mean(self.q_table)
#         }
#
#
# class EdgeServer:
#     def __init__(self, env, name, cloud_env, cpu_power, memory, max_concurrent_tasks):
#         self.env = env
#         self.name = name
#         self.cloud_env = cloud_env
#         self.cpu_power = cpu_power
#         self.memory = memory
#         self.task_queue = []
#         self.local_tasks = []
#         self.network = Network(env)
#         self.current_load = 0
#         self.max_load = cpu_power * 2
#         self.max_concurrent_tasks = max_concurrent_tasks
#         self.current_tasks = 0
#         self.metrics = {
#             'total_tasks': 0,
#             'completed_tasks': 0,
#             'avg_processing_time': 0,
#             'total_processing_time': 0
#         }
#
#     def add_task(self, task):
#         task.creation_time = self.env.now
#         self.metrics['total_tasks'] += 1
#         self.env.process(self.send_to_cloud(task))
#
#     def send_to_cloud(self, task):
#         transfer_time = self.network.transfer_time(task.data_size)
#         yield self.env.timeout(transfer_time)
#         self.cloud_env.receive_task(task, self)
#
#     def process_locally(self, task):
#         start_time = self.env.now
#         self.current_tasks += 1
#         self.current_load += task.complexity
#
#         processing_time = self.calculate_local_processing_time(task)
#         yield self.env.timeout(processing_time)
#
#         self.current_tasks -= 1
#         self.current_load -= task.complexity
#         end_time = self.env.now
#         task.completion_time = end_time
#
#         self.metrics['completed_tasks'] += 1
#         self.metrics['total_processing_time'] += (end_time - start_time)
#         self.metrics['avg_processing_time'] = (
#                 self.metrics['total_processing_time'] / self.metrics['completed_tasks']
#         )
#
#         self.local_tasks.append((start_time, end_time, task.duration, task.complexity, task.priority, 'Local'))
#         print(f'{self.name} completed local task (priority {task.priority}, complexity {task.complexity}) '
#               f'at {self.env.now:.2f}. Processing time: {end_time - start_time:.2f}')
#
#     def calculate_local_processing_time(self, task):
#         return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 10))
#
#     def get_load_state(self):
#         return min(int(self.current_load / self.max_load * 10), 9)
#
#
# class CloudEnvironment:
#     def __init__(self, env, num_servers, cpu_power, memory, num_edge_servers):
#         self.env = env
#         self.servers = simpy.Resource(env, num_servers)
#         self.cpu_power = cpu_power
#         self.memory = memory
#         self.processed_tasks = []
#         self.network = Network(env)
#         self.rl_agent = RLAgent(state_size=10000, action_size=num_edge_servers + 1)
#         self.edge_servers = []
#         self.task_data = []
#         self.metrics = {
#             'total_tasks': 0,
#             'cloud_processed': 0,
#             'edge_processed': 0,
#             'avg_processing_time': 0,
#             'total_processing_time': 0
#         }
#
#     def add_edge_server(self, edge_server):
#         self.edge_servers.append(edge_server)
#
#     def calculate_reward(self, processing_time, task):
#         # Base reward for completing a task
#         base_reward = 100
#
#         # Time-based penalty (higher processing time = lower reward)
#         time_penalty = processing_time * 10
#
#         # Priority bonus (higher priority = higher reward)
#         priority_bonus = task.priority * 20
#
#         # Complexity consideration (more complex tasks get more reward)
#         complexity_bonus = task.complexity * 5
#
#         # Calculate total reward
#         reward = base_reward - time_penalty + priority_bonus + complexity_bonus
#
#         # Bound the reward
#         return max(min(reward, 200), -100)
#
#     def get_state(self, task):
#         # Get load states for all edge servers
#         load_states = [server.get_load_state() for server in self.edge_servers]
#
#         # Get network state
#         network_state = min(int(self.network.latency / 5), 9)
#
#         # Get task states
#         complexity_state = min(task.complexity - 1, 9)
#         priority_state = min(task.priority - 1, 9)
#
#         # Combine states into a single number
#         state = 0
#         for i, load_state in enumerate(load_states):
#             state += load_state * (10 ** (4 + i))
#         state += network_state * 1000 + complexity_state * 10 + priority_state
#
#         return state
#
#     def receive_task(self, task, edge_server):
#         self.metrics['total_tasks'] += 1
#         state = self.get_state(task)
#         task_info = {
#             "Time": self.env.now,
#             "Edge Server": edge_server.name,
#             "Task Duration": task.duration,
#             "Task Priority": task.priority,
#             "Computed State": state
#         }
#         self.task_data.append(task_info)
#         self.env.process(self.decide_and_process(task, edge_server))
#
#     def decide_and_process(self, task, edge_server):
#         # Get current state and choose action
#         current_state = self.get_state(task)
#         action = self.rl_agent.get_action(current_state)
#
#         start_time = self.env.now
#
#         # Execute chosen action
#         if action == len(self.edge_servers):
#             # Process on cloud
#             self.task_data[-1]["Action"] = "Process on cloud"
#             yield self.env.process(self.process_on_cloud(task, edge_server))
#             self.metrics['cloud_processed'] += 1
#         else:
#             # Process on edge server
#             self.task_data[-1]["Action"] = f"Send to {self.edge_servers[action].name}"
#             yield self.env.process(self.send_to_edge(task, self.edge_servers[action]))
#             self.metrics['edge_processed'] += 1
#
#         # Calculate processing time and reward
#         processing_time = self.env.now - start_time
#         reward = self.calculate_reward(processing_time, task)
#
#         # Get new state and update Q-table
#         next_state = self.get_state(task)
#         self.rl_agent.update_q_table(current_state, action, reward, next_state)
#
#         # Update metrics
#         self.metrics['total_processing_time'] += processing_time
#         self.metrics['avg_processing_time'] = (
#                 self.metrics['total_processing_time'] / self.metrics['total_tasks']
#         )
#
#         # Decay exploration rate
#         self.rl_agent.decay_exploration()
#
#         # Update task data with results
#         self.task_data[-1].update({
#             "Processing Time": processing_time,
#             "Reward": reward,
#             "Exploration Rate": self.rl_agent.exploration_rate
#         })
#
#     def process_on_cloud(self, task, edge_server):
#         with self.servers.request() as request:
#             yield request
#             start_time = self.env.now
#             processing_time = self.calculate_cloud_processing_time(task)
#             yield self.env.timeout(processing_time)
#             end_time = self.env.now
#             task.completion_time = end_time
#
#             self.processed_tasks.append(
#                 (edge_server.name, start_time, end_time, task.duration,
#                  task.complexity, task.priority, 'Cloud')
#             )
#
#             print(f'Cloud processed task (priority {task.priority}, complexity {task.complexity}) '
#                   f'from {edge_server.name} at {self.env.now:.2f}')
#
#     def send_to_edge(self, task, edge_server):
#         transfer_time = self.network.transfer_time(task.data_size)
#         yield self.env.timeout(transfer_time)
#         yield self.env.process(edge_server.process_locally(task))
#
#     def calculate_cloud_processing_time(self, task):
#         return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 12))
#
#     def print_metrics(self):
#         print("\nCloud Environment Metrics:")
#         print(f"Total tasks processed: {self.metrics['total_tasks']}")
#         print(f"Tasks processed on cloud: {self.metrics['cloud_processed']}")
#         print(f"Tasks processed on edge: {self.metrics['edge_processed']}")
#         print(f"Average processing time: {self.metrics['avg_processing_time']:.2f}")
#
#         rl_stats = self.rl_agent.get_stats()
#         print("\nRL Agent Statistics:")
#         print(f"Total reward: {rl_stats['total_reward']:.2f}")
#         print(f"Average reward: {rl_stats['average_reward']:.2f}")
#         print(f"Current exploration rate: {rl_stats['exploration_rate']:.3f}")
#         print(f"Non-zero Q-values: {rl_stats['non_zero_q_values']}")
#         print(f"Q-value range: [{rl_stats['min_q_value']:.2f}, {rl_stats['max_q_value']:.2f}]")
#         print(f"Mean Q-value: {rl_stats['mean_q_value']:.2f}")
#
#
# # Setup and run simulation
# def run_simulation(duration=20000):
#     env = simpy.Environment()
#
#     # Create cloud environment
#     cloud_env = CloudEnvironment(env, num_servers=2, cpu_power=32.0, memory=64, num_edge_servers=3)
#
#     # Create edge servers
#     edge_servers = [
#         EdgeServer(env, 'EdgeServer1', cloud_env, cpu_power=2.0, memory=8, max_concurrent_tasks=5),
#         EdgeServer(env, 'EdgeServer2', cloud_env, cpu_power=2.5, memory=10, max_concurrent_tasks=5),
#         EdgeServer(env, 'EdgeServer3', cloud_env, cpu_power=3.0, memory=12, max_concurrent_tasks=5),
#     ]
#
#     # Register edge servers
#     for edge_server in edge_servers:
#         cloud_env.add_edge_server(edge_server)
#
#     # Create tasks
#     tasks = [
#         Task(duration=5, complexity=4, priority=2, data_size=10),
#         Task(duration=3, complexity=3, priority=1, data_size=8),
#         Task(duration=7, complexity=5, priority=3, data_size=15),
#         Task(duration=4, complexity=4, priority=2, data_size=12),
#         Task(duration=6, complexity=6, priority=1, data_size=18),
#         Task(duration=5, complexity=5, priority=2, data_size=14),
#     ]
#
#     # Distribute tasks
#     for i, task in enumerate(tasks):
#         edge_servers[i % len(edge_servers)].add_task(task)
#
#     # Run simulation
#     env.run(until=duration)
#
#     # Print results
#     cloud_env.print_metrics()
#
#     # Create and print summary DataFrame
#     results_df = pd.DataFrame(cloud_env.task_data)
#     print("\nTask Processing Summary:")
#     print(tabulate(results_df, headers='keys', tablefmt='grid'))
#
#     return cloud_env, edge_servers, results_df
#
#
# if __name__ == "__main__":
#     cloud_env, edge_servers, results_df = run_simulation()