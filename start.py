import numpy as np
from stable_baselines3 import PPO

class RLOffloading:
    def __init__(self):
        # Initialize or load your RL model here
        self.model = PPO.load("ppo_model")

    def decide(self, state):
        # state is a list or numpy array of task size, network latency, and edge capacity
        action, _states = self.model.predict(np.array(state).reshape(1, -1))
        return int(action)

# Example usage
# rl_offloading = RLOffloading()
# decision = rl_offloading.decide([0.5, 0.2, 0.8])
# print(f"Decision: {decision}")
