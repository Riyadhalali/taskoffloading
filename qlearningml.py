import numpy as np
# define the envirnoment
n_states = 16    # number of states in grid world
n_actions = 4    # number of possible actions (up, down, left, right)
goal_state = 15  # the goal to reach this value


# initialize q-table
Q_table = np.zeros((n_states,n_actions))


# define parameters
learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
epochs = 1000


# Q- learning algorithm
for epoch in range(epochs):
    current_state = np.random.randint(0, n_states) # start from random state

    while current_state != goal_state:
        # choose action with epsilon-greedy strategy
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)  # excplore

        else:
            action = np.argmax(Q_table[current_state]) # exploit

    next_state = (current_state + 1) % n_states
    reward = 1 if next_state == goal_state else 0

    # Update Q-value using the Q-learning update rule
    Q_table[current_state, action] += learning_rate * \
                                      (reward + discount_factor *
                                       np.max(Q_table[next_state]) - Q_table[current_state, action])


    current_state = next_state  # Move to the next state

# After training, the Q-table represents the learned Q-values
print("Learned Q-table:")
print(Q_table)


