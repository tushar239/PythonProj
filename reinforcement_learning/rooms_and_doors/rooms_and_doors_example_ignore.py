'''
https://www.geeksforgeeks.org/q-learning-in-python/
https://www.youtube.com/watch?v=QRMNPCsnSHk&list=PLWPirh4EWFpEjbNicXUZk0wrPBzBlAlU_&index=19
https://www.youtube.com/watch?v=J3qX50yyiU0
'''

import numpy as np
import random

R_matrix = np.matrix([[-1,    -1, -1,  -1, 0,  -1],
                      [-1,    -1, -1, 0,  -1, 100],
                      [-1,    -1, -1, 0,  -1, -1],
                      [-1,    0,  0, -1,  0,  -1],
                      [0,     -1, -1, 0,  -1, 100],
                      [-1,    0,  -1, -1, 0,  100]])
Q_matrix = array = np.zeros((6, 6))

def find_max(R_matrix, next_state):
    #print(R_matrix[next_state].shape[1])
    max=-1
    no_of_cols = R_matrix.shape[1]
    for i in range(0, no_of_cols, 1):
        element = R_matrix[next_state, i]
        if element > max:
            max = element
    return max

episodes = 10000

for episode in range(episodes):
    current_state = -1 # room
    goal_state = 5

    while current_state != goal_state:
        R_matrix_value = -1
        action = -1 # door

        # choose random current state and choose random action whose value is 0 or 100 (not -1)
        while R_matrix_value == -1:
            current_state = random.randint(0, 5)
            action = random.randint(0, 5)
            R_matrix_value = R_matrix[current_state, action] # 0 or 100
        print("random_state", current_state)
        print("action", action)
        print("R_matrix_value", R_matrix_value)

        discount_rate = 0.8
        next_state=action # action(door) becomes the next state(room)

        # Q(1,5) = R(1,5) + discount_rate * Max(Q(5,1),Q(5,4),Q(5,5)]
        #max_next_state_R_value = max(R_matrix[next_state])

        max_next_state_R_value = find_max(R_matrix, next_state)
        Q_matrix_value = R_matrix_value + discount_rate * max_next_state_R_value
        Q_matrix[current_state, action] = Q_matrix_value
        print("Q_matrix_value", Q_matrix_value)
        print("Q_matrix\n", Q_matrix)

        current_state = next_state


print("final Q_matrix\n", Q_matrix)
