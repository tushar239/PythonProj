# from chatgpt

import numpy as np

# Environment: rooms and connections
R = np.array([
    [-1, 0, -1, 0, -1, -1],
    [0, -1, 0, -1, -1, 100],
    [-1, 0, -1, 0, -1, -1],
    [0, -1, 0, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
])

# Q-table initialized to 0, each entry represents the value of taking a particular action in a given state.
Q = np.zeros_like(R, dtype=float)

# Hyperparameters
gamma = 0.8  # discount factor, representing the importance of future rewards.
alpha = 0.9  # learning rate, controlling how much new information overrides old information.
episodes = 1000

# Q-learning process
for _ in range(episodes):
    state = np.random.randint(0, 6)  # Start from random room

    while True:
        # Get possible actions (actions having >=0 value)
        actions = [a for a in range(6) if R[state, a] >= 0]
        action = np.random.choice(actions)

        # Get next state's best future value
        next_state = action
        nonNegativeActions = [act for act in Q[next_state] if act >= 0]
        best_value_of_next_action = np.max(nonNegativeActions) # maximum Q-value for the next state, representing the best possible reward achievable from that state.
        # or
        #next_best_action = np.max(Q[next_state])

        immediate_reward_for_taking_action =R[state, action] # immediate reward for taking action a in state s.
        # Q-learning update rule
        Q[state, action] += alpha * (immediate_reward_for_taking_action + gamma * best_value_of_next_action - Q[state, action])

        state = next_state
        if state == 5:  # reached the goal
            break

# Normalize Q for easy interpretation
Q_normalized = (Q / np.max(Q) * 100).astype(int)

print("🏁 Trained Q-table:")
print(Q_normalized)


# Best Route Finder (Greedy)

# Function to find path from any room to goal
def get_optimal_path(start_room):
    current = start_room
    path = [current]

    while current != 5:
        next_room = np.argmax(Q[current])
        path.append(next_room)
        current = next_room

    return path


start_room = 0
print("\n🚪 Optimal path from room ", start_room, "to room 5:")
print(get_optimal_path(start_room))
