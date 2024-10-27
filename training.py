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
