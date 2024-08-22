import simpy
import pandas as pd
import random
import numpy as np
from tabulate import tabulate


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
    def __int__(self,env,name,cloud_env,cpu_power,mmemory,max_concurrent_tasks):
        self.env = env
        self.name = name
        self.cloud_env = cloud_env
