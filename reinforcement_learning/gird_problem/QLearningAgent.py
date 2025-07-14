# https://medium.com/@goldengrisha/a-beginners-guide-to-q-learning-understanding-with-a-simple-gridworld-example-2b6736e7e2c9
import random

import numpy as np


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = np.zeros((4, 4, 4))  # Q-values for each state-action pair. 4 actions for each cell of grid table.
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)  # Explore
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