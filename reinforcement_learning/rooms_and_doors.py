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

# Q-table initialized to 0
Q = np.zeros_like(R, dtype=float)

# Hyperparameters
gamma = 0.8  # discount factor
alpha = 0.9  # learning rate
episodes = 1000

# Q-learning process
for _ in range(episodes):
    state = np.random.randint(0, 6)  # Start from random room

    while True:
        # Get possible actions
        actions = [a for a in range(6) if R[state, a] >= 0]
        action = np.random.choice(actions)

        # Get next state's best future value
        next_best = np.max(Q[action])

        # Q-learning update rule
        Q[state, action] += alpha * (R[state, action] + gamma * next_best - Q[state, action])

        state = action
        if state == 5:
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


print("\n🚪 Optimal path from room 2 to room 5:")
print(get_optimal_path(2))
