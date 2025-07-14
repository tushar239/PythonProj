# https://medium.com/@goldengrisha/a-beginners-guide-to-q-learning-understanding-with-a-simple-gridworld-example-2b6736e7e2c9

from reinforcement_learning.gird_problem.GridWorld import GridWorld
from reinforcement_learning.gird_problem.QLearningAgent import QLearningAgent

env = GridWorld()
agent = QLearningAgent()

episodes = 1000  # Number of training episodes

for episode in range(episodes):
    state = env.reset()  # Reset the environment at the start of each episode
    #print('initial state: ', state) # (3, 0)
    done = False

    while not done:
        action = agent.choose_action(state)  # Choose an action
        next_state, reward, done = env.step(action)  # Take the action and observe next state, reward
        agent.update_q_value(state, action, reward, next_state)  # Update Q-values
        state = next_state  # Move to the next state

print(agent.q_table)