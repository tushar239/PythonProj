import numpy as np
import random

# Simple 1D environment: 5 states, goal at position 4
states = [0, 1, 2, 3, 4]
actions = ['left', 'right']
Q = np.zeros((5, 2))  # Q-table: 5 states x 2 actions

alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
epsilon = 0.2 # exploration factor

# Training episodes
for episode in range(100):
    state = 0  # start at position 0
    while state != 4:
        # Choose action
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1])  # explore. choose between 0 and 1
        else:
            action = np.argmax(Q[state])   # exploit

        # Take action
        next_state = state + 1 if action == 1 else max(0, state - 1)
        reward = 10 if next_state == 4 else -1

        # Update Q-table
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state

# Show learned Q-table
print("Q-Table:")
print(Q)
