import random

import pandas as pd
import simpy


class EdgeServer:
    def __init__(self,env,name,local_processing_time,cloud_env):
        self.env=env
        self.name=name
        self.local_processing_time=local_processing_time
        self.cloud_env=cloud_env
        self.local_tasks=[]
        self.offloaded_tasks=[]
        self.action=env.process(self.run())



def run(self):
    while True:
        # generate a task
        task_duration = random.randint(1,10)
        # offload all tasks to cloud
        offload_decision = True
        if offload_decision:
            start_time = self.env.now
            print(f'{self.name} offloading task of duration {task_duration} at {self.env.now}')
            yield self.env.process(self.cloud_env.process_task(task_duration, self.name))
            end_time = self.env.now
            self.offloaded_tasks.append((start_time, end_time, task_duration, 'Offloaded'))
        else:
            start_time = self.env.now
            print(f'{self.name} processing task locally of duration {task_duration} at {self.env.now}')
            yield self.env.timeout(self.local_processing_time(task_duration))
            end_time = self.env.now
            self.local_tasks.append((start_time, end_time, task_duration, 'Local'))

        # wait before generating new task
        yield self.env.timeout(random.randint(1, 5))
class CloudEnvironment:
    def __init__(self,env,num_servers):
        self.env = env
        self.servers = simpy.Resource(env, num_servers)
        self.processed_tasks = []
    def process_task(self,task_duration, device_name):
        with self.servers.request() as request:
            yield request
            start_time = self.env.now
            yield self.env.timeout(task_duration)
            end_time = self.env.now
            self.processed_tasks.append((device_name, start_time, end_time, task_duration, 'Processed'))
            print(f'Cloud processed task of duration {task_duration} from {device_name} at {self.env.now}')

def local_processing_time(task_duration):
    return task_duration * 2  # local processing take twice as cloud processing
#setup and start simulation
env = simpy.Environment()
cloud_env = CloudEnvironment(env, num_servers=2)


#create edge server
edge_servers= [EdgeServer(env, f'EdgeServer {i}', local_processing_time, cloud_env) for i in range(3)]

# Run the simulation
env.run(until=20)  # Run the simulation for 20 time units
# collect data into the simulation
local_data = []
offloaded_data = []
cloud_data = []


for server in edge_servers:
    for start, end, duration, task_type in server.local_tasks:
        local_data.append([server.name, start, end, duration, task_type])
    for start, end, duration, task_type in server.offloaded_tasks:
        offloaded_data.append([server.name, start, end, duration, task_type])


for device_name, start, end, duration, task_type in cloud_env.processed_tasks:
   cloud_data.append([device_name, start, end, duration, task_type])


#combine all data
all_data = local_data + offloaded_data + cloud_data

#create a data frame
df = pd.DataFrame(all_data, columns=['Device', 'Start Time', 'End Time', 'Duration', 'Type'])



# Print the DataFrame
print(df)

# Optionally, save to a CSV file
df.to_csv('simulation_results.csv', index=False)

