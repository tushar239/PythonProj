'''
Initialize Q[state, action] arbitrarily
Set learning rate α, discount factor γ, exploration ε

For each episode:
    Start from the initial state
    Repeat until goal is reached:
        With probability ε: choose random action (exploration)
        Otherwise: choose best action from Q (exploitation)

        Take action → observe next_state and reward
        Q[state, action] ← Q[state, action] + α * (reward + γ * max(Q[next_state]) - Q[state, action])
        state ← next_state

α (learning rate) = 0.8 → How quickly it updates
γ (discount factor) = 0.9 → How much it values future reward
ε (epsilon) = 0.1 → How often it explores random actions

Optimal path: e.g.,
(0,0) → (1,0) → (2,0) → (3,0) → (3,1) → (3,2) → (3,3)
Q-table: tells the agent the best action to take from each cell
Policy: “If I'm at (2,0), best move is ↓ to (3,0)”

Final Policy Visualization (example):

→ → → ↓
↑ ↓ ↓ ↓
↑ ↓ ↓ ↓
↑ → → GOAL
'''



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
            action = np.argmax(Q[state])   # exploit. np.argmax will return an id

        # Take action
        next_state = state + 1 if action == 1 else max(0, state - 1)
        reward = 10 if next_state == 4 else -1

        next_best_action = np.max(Q[next_state])

        # Update Q-table
        Q[state, action] += alpha * (reward + gamma * next_best_action - Q[state, action])

        state = next_state

# Show learned Q-table
print("Q-Table:")
print(Q)
