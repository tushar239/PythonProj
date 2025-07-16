# https://medium.com/@goldengrisha/a-beginners-guide-to-q-learning-understanding-with-a-simple-gridworld-example-2b6736e7e2c9
import random

import numpy as np

'''
Reinforcement Learning's Q-Learning algorithm formula:
Q[s,a] = Q[s,a] + alpha * (current_state_R_value + gamma * next_state_best_possible_Q_value - Q[s,a])
Q[s,a] = Q[s,a] + α * (r + γ * max(Q(s`,a`)) - Q[s,a])

In reinforcement learning, α (alpha), γ (gamma), and ε (epsilon) are core hyperparameters that control how the agent learns and balances exploration and exploitation.
α - αlpha -	Learning rate (how quickly we update old value) - Speed of learning	-   0.1 – 0.5
γ - gamma - Discount factor - importance of future reward - 0.8 – 0.99
ε - Exploration Rate - Random vs best action	-   1.0 → 0.01
r - current_state_R_value - immediate reward
max(Q(s`,a`)) - next_state_best_possible_Q_value - Best possible future reward from the next state Q(s`,a`)

α = 0.3 → Learn slowly, don’t erase history too fast
γ = 0.9 → Value long-term success more than quick gain
ε = 0.2 → 20% chance to explore, 80% to exploit known best move
'''

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = np.zeros((4, 4, 4))  # Q-values for each state-action pair. 4 actions for each cell of grid table.
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)  # Explore   -  0 to 3 are four actions
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])  # Best Q-value for next state
        current_q = self.q_table[state][action]
        # Q-learning formula
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )


agent = QLearningAgent()
print(agent.q_table)