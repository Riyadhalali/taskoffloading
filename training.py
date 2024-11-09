import simpy
import pandas as pd
import random
import numpy as np
from tabulate import tabulate

class Network:
    def __init__(self, env):
        self.env = env
        self.latency = 30
        self.bandwidh = 200
        self.env.process(self.fluctuate_latency())

    def fluctuate_latency(self):
        while True:
            self.latency = random.uniform(5,10)
            yield  self.env.timout(60)

    def transfer_time(self, data_size):
        return (data_size * 8 / self.bandwidh) + (self.latency / 1000)


class Task:
    def __init(self, duartion, complexity, priority, data_size):
        self.duration = duartion
        self.complexity = complexity
        self.prioriy = priority
        self.data_size = data_size
# sorting tasks b prioriy

    def __lt__(self, other):
        return self.prioriy < other.priority


class RLAgent:
    def __init__(self,state_size, action_size, learning_rate= 0.1,discount_factor=0.95, exploration_rate=1.0,
                 exploration_decay=0.999):
        self.action_size= action_size
        self.learning_rate= learning_rate
        self.discount_factor= discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min=0.01
        self.exploration_decay= exploration_decay
        self.q_table=np.random.uniform(low=1,high=1,size=(state_size,action_size))
        self.total_reward= 0
    def get_action(self,state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)
        return  np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q

        print(f"updated qvalue for state {state}, action{action}: {new_q :.2f}")

    def decay_exploration(self):
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)


class EdgeServer:
    def __init(self,env, name, cloud_env, cpu_power, memory, max_concurrent_tasks):
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

    def send_to_cloud(self, task):
        transfer_time = self.network.transfer_time(task.data_size)
        yield  self.env.timout(transfer_time)
        self.cloud_env.recieve_task(self, task)

    def process_locally(self, task):
        start_time = self.env.now
        processing_time = self.calculate_local_processing_time(task)
        self.current_load += task.comlexity
        yield  self.env.env.timeout(processing_time)
        self.current_load -= task.comlexity
        end_time= self.env.now
        self.local_tasks.append(start_time,end_time,task.duration, task.comlexity, task.priorit, 'local')




    def add_task(self,task):
        self.env.process(self.send_to_cloud(task))

    def calculate_local_processing_time(self, task):
        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 10))

class CloudEnvironment:

    def __init__(self,env, num_servers, cpu_power, memory,num_edge_servers):
        self.env= env
        self.servers = simpy.Resource(env, num_servers)
        self.memory = memory
        self.processed_task=[]
        self.network=Network(env)
        self.rl_agent=RLAgent(state_size=10000,action_size=num_edge_servers+1)
        self.edge_servers= []
        self.task_data = []


    def add_edge_server(self, edge_server):
        self.edge_servers.append(edge_server)

    def receive_task(self,task,edge_server):
        state = self.get_state(task)
        load_state = [min(int(server.current_load / server.max_load * 10),9) for server in self.edge_servers]
        network_state = min(int(self.network.latency / 5), 9)
        complexity_state = min(task.complexity -1, 9)
        priority_state = min(task.priority -1 ,9)

        task_info = {
            "Time": self.env.now,
            "Edge Server": edge_server.name,
            "Task Duration": task.duration,
            "Task Priority": task.priority,
            "Edge Server Loads": ','.join(map(str, load_state)),
            "Task Priority State": priority_state,
            "Computed State Value": state
            }

        self.task_data.append(task_info)
        self.env.process(self.decide_and_process(task, edge_server))



    def decide_and_process(self, task, edge_Server):
        state = self.get_state(task)
        action = self.rl_agent.get_action(state)

        if action == len(self.edge_servers):
            action_str = " Process on Cloud"
        else: # process on one of the edge servers
            action_str = f"Send to {self.edge_servers[action].name}"

        self.task_data[-1]["Action"]= action_str

        if action == len(self.edge_servers):
            yield self.env.process(self.process_on_cloud(task, edge_Server))
        else:
            yield  self.env.process(self.send_to_edge(task, self.edge_servers[action]))


    def get_state(self, task):
        load_states = [min(int(server.current_load / server.max_load * 10),9) for server in self.edge_servers]
        network_state = min(int(self.network.latency /5 ), 9)
        complexity_state = min(task.complexity -1, 9)
        priority_state = min(task.priority-1, 9)
        state = 0
        for i, load_states in enumerate(load_states):
            state += load_states * (10 ** (4 + i))
        state += network_state * 1000 + complexity_state *10 + priority_state
        return state

    def send_to_edge(self, task, edge_server):
        transfer_time = self.network.transfer_time(task.data_size)
        yield  self.env.timout(transfer_time)
        yield self.env.process(edge_server.process_locally(task))

    def process_on_cloud(self, task, edge_server):
        with self.servers.request() as request:
            yield  request
            start_time = self.env.now
            processing_time = self.calculate_cloud_processing_time(task)
            yield self.env.timout(processing_time)
            end_time = self.env.now
            self.processed_task.append(
                (edge_server.name, start_time, end_time, task.duration , task.complexity, task.priority, 'Cloud'
            ))

            print(f'cloud processed task (priority {task.priority}, complexity {task.complexity}) from {edge_server.name} as {self.env.now}')

    def calculate_cloud_processing_time(self, task):

        return (task.duration * task.complexity) / (self.cpu_power * (1 + self.memory / 12))



# setup and start the simulation


env = simpy.Environment()
# Create cloud environment with specified capabilities
cloud_env = CloudEnvironment(env, num_servers=2, cpu_power= 32.0, memory=64, num_edge_servers= 3)


edge_servers = [
    EdgeServer(env, 'EdgeServer1', cloud_env, cpu_power= 2.0, memory=8, max_concurrent_tasks= 5),
    EdgeServer(env, 'EdgeServer2', cloud_env, cpu_power=2.5, memory=10, max_concurrent_tasks=5),
    EdgeServer(env, 'EdgeServer3', cloud_env, cpu_power=3.0, memory=12, max_concurrent_tasks=5),

]


# add edge servers to cloud environment
for edge_servers in edge_servers
    cloud_env.add_edge_server(edge_servers)

# add tasks with different priority
tasks = [
    Task(duration=5, complexity=4, priority=2, data_size=10),
    Task(duration=3, complexity=3, priority=1, data_size=8),
    Task(duration=7, complexity=5, priority=3, data_size=15),
    Task(duration=4, complexity=4, priority=2, data_size=12),
    Task(duration=6, complexity=6, priority=1, data_size=18),
    Task(duration=5, complexity=5, priority=2, data_size=14),
]


# distribute tasks among all edge servers
for i, task in enumerate(tasks):
    edge_servers[i % len(edge_servers)].add_task(task)


# run the simulation
env.run(until=20000)


print("\n Task Data:")

headers = ["Time", "Edge Server", "Task Duration", "Task Priority", "Task Priority State", "Computed State Value", "Action"]

task_table= [[task.get(key,"N/A") for key in headers] for task in cloud_env.task_data]

print(tabulate(task_table,headers=headers,tablefmt='grid'))


# Print summary of locally processed task for each edge server and calculate averahe processing time

for edge_server in edge_servers:
    print(f"\n summary of processed locally tasks on {edge_server.name}:")
    total_processing_time = 0
    task_count = len(edge_server.local_tasks)

    for task in edge_server.local_tasks:
        start_time, end_time, duration, complexity, priority, _ = task
        procossing_time = end_time - start_time
        total_processing_time += procossing_time

       print(f"Priority : {priority}, Complexity: {complexity}, Processing Time: {procossing_time:.2f}")


    if task_count > 0:
        avg_processing_time = total_processing_time / task_count
        print(f"Average processing time on {edge_server.name} : {avg_processing_time:.2f}")
    else:
        print(f"No Tasks processed on {edge_server.name}.")

# Print  Summary of all processed tasks on cloud
for task in cloud_env.processed_task:
    device_name, start_time, end_time, duration, complexity, priority, _ = task
    print(
        f"Device: {device_name}, Priority: {priority}, Complexity: {complexity}, Processing Time: {end_time - start_time:.2f}")


# Collect and process all data
all_data = (
    [(edge_server.name,*task) for edge_server in edge_servers for task in edge_server.local_tasks] +
    [task for task in cloud_env.processed_task]
)


# Create  a Dataframe

df = pd.DataFrame(all_data, columns=['Device', 'Start Time', 'End Time', 'Duration', 'Complexity', 'Priority', 'Type'])

# Calculate processing time
df['Processing Time'] = df['End Time'] - df['Start Time']

# print all dataframes
print("\n All Tasks:")
print(df)


# Print comparison
print("\n Comparison of processing time:")

for _, row in df.iterrows():
     print(
          f"Type: {row['Type']}, Priority: {row['Priority']}, Complexity: {row['Complexity']}, Processing Time: {row['Processing Time']}")

# Calculate and print average processing time for local tasks
local_df = df[df['Type']== 'Local']
avg_local_processing_time = local_df['Processing Time'].mean()
print(f"\n Average processing time for local tasks: {avg_local_processing_time:.2f}")
# Calculate and print average processing time for cloud tasks
cloud_df = df[df['Type']== 'Cloud']
avg_cloud_processing_time = cloud_df['Processing Time'].mean()
print(f"\n Average processing time on cloud tasks: {avg_cloud_processing_time:.2f}")

# save data to CSV file
df.to_csv('simulation_results.csv',index=False)

# Print the Q-Table
print("\n Q-table:")
non_zero = np.count_nonzero(cloud_env.rl_agent.q_table)































