import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
# this is Google stock price from 2012 to 2016
# we are planning to predict Google's stock price on financial day
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') # creating a dataframe of Google's stock prices
training_set = dataset_train.iloc[:, 1:2].values # selecting only 'open' price column. '.values' converts dataframe into numpy array
print(type(training_set)) # <class 'numpy.ndarray'>
print(training_set.shape) # (1258, 1)
print (training_set)
'''
[[325.25]
 [331.27]
 [329.83]
 ...
 [793.7 ]
 [783.33]
 [782.75]]
'''
# Feature Scaling
'''
We will use Normalization in case of RNN. We will use Min-Max Normalization.
When sigmoid function is used as an activation function, it is recommended to use Normalization. 

fit method will calculate min and max, 
transform method will apply normalization using calculated min and max values.
'''
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # all scaled stock prices will be in between 0 and 1
training_set_scaled = sc.fit_transform(training_set)
print(type(training_set_scaled)) # <class 'numpy.ndarray'>
print(training_set_scaled)
'''
[[0.08581368]
 [0.09701243]
 [0.09433366]
 ...
 [0.95725128]
 [0.93796041]
 [0.93688146]]
'''

'''
input_shape
-----------

Data=
[
    0- [f1, f2, f3, f4, f5],
    1- [f1, f2, f3, f4, f5],
    2- [f1, f2, f3, f4, f5],
    3- [f1, f2, f3, f4, f5],
    4- [f1, f2, f3, f4, f5],
    5- [f1, f2, f3, f4, f5],
    6- [f1, f2, f3, f4, f5],
    7- [f1, f2, f3, f4, f5],
    8- [f1, f2, f3, f4, f5],
    9- [f1, f2, f3, f4, f5],
    .
    .
    .
]
Example of these data can be stock price(open price, closed price, low price, high price, volume)
This data has to be divided into samples. 
Each sample has inputs for timesteps in LSTM. 
Each input for timestep has features.

Example of X (training input - 3D array) and y (training output)

X = 
100 samples that can be divided into batches
[
    Sample1 
    [
        0- [f1, f2, f3, f4, f5], - input for Timestep1. Each timestep has 5 features
        1- [f1, f2, f3, f4, f5], - input for Timestep2
        2- [f1, f2, f3, f4, f5]  - input for Timestep3
    ],

    Sample2
    [
        1- [f1, f2, f3, f4, f5], - input for Timestep1
        2- [f1, f2, f3, f4, f5], - input for Timestep2
        3- [f1, f2, f3, f4, f5]  - input for Timestep3
    ],

    Sample3
    [
        2- [f1, f2, f3, f4, f5], - input for Timestep1
        3- [f1, f2, f3, f4, f5], - input for Timestep2
        4- [f1, f2, f3, f4, f5]  - input for Timestep3
    ],

    Sample4
    [
        3- [f1, f2, f3, f4, f5], - input for Timestep1
        4- [f1, f2, f3, f4, f5], - input for Timestep2
        5- [f1, f2, f3, f4, f5]  - input for Timestep3
    ],
    .
    .
    .
    Sample100
    [
        n1- [f1, f2, f3, f4, f5], - input for Timestep1
        n2- [f1, f2, f3, f4, f5], - input for Timestep2
        n3- [f1, f2, f3, f4, f5]  - input for Timestep3
    ]
]

Here, input_shape = (timesteps, features) - (3, 5)

y=
[
    3- [f1, f2, f3, f4, f5],
    4- [f1, f2, f3, f4, f5]
    5- [f1, f2, f3, f4, f5]
    6- [f1, f2, f3, f4, f5]
    .
    .
    .
    n4- [f1, f2, f3, f4, f5]
]

After every 3 data, 4th data is considered as output.
So, 0-2 data are kept in X and then 3th sample is kept in y
3-5 data are kept in X and then 6th data is kept in y
and so on

In our example, after every 10 data, 11th data is considered as output.
'''
# Creating a data structure with 60 timesteps and 1 output
# we tried 1,10,20,30.... timesteps, 60 timesteps gave better results.
# Sliding window technique is used to create this data.
# https://www.geeksforgeeks.org/dsa/window-sliding-technique/
X_train = []
y_train = []
for i in range(60, 1258):
    a_array = training_set_scaled[i-60:i, 0] # e.g. 0-59 rows, 0th col in each row - 60 timesteps
    X_train.append(a_array)
    b_element = training_set_scaled[i, 0] # e.g. 60th row, 0th col in it - 1 output
    y_train.append(b_element)
# converting lists to numpy arrays.
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train)
'''
Total 1198 samples, each with 60 timesteps
[[0.08581368 0.09701243 0.09433366 ... 0.07846566 0.08034452 0.08497656]
 [0.09701243 0.09433366 0.09156187 ... 0.08034452 0.08497656 0.08627874]
 [0.09433366 0.09156187 0.07984225 ... 0.08497656 0.08627874 0.08471612]
 ...
 [0.92106928 0.92438053 0.93048218 ... 0.95475854 0.95204256 0.95163331]
 [0.92438053 0.93048218 0.9299055  ... 0.95204256 0.95163331 0.95725128]
 [0.93048218 0.9299055  0.93113327 ... 0.95163331 0.95725128 0.93796041]]
'''
print(y_train)
'''
[0.08627874 0.08471612 0.07454052 ... 0.95725128 0.93796041 0.93688146]
'''

# Reshaping
print(X_train.shape) # (1198, 60)
# making 2-D array to 3-D array
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape) # (1198, 60, 1) - (samples, timesteps, features)
print(X_train)
'''
[
    [[0.08581368],  [0.09701243],  [0.09433366],  ...,  [0.07846566],  [0.08034452],  [0.08497656]],,
    [[0.09701243],  [0.09433366],  [0.09156187],  ...,  [0.08034452],  [0.08497656],  [0.08627874]],, 
    [[0.09433366],  [0.09156187],  [0.07984225],  ...,  [0.08497656],  [0.08627874],  [0.08471612]]
    ,, ...,, 
    [[0.16227026],  [0.16236327],  [0.15933105],  ...,  [0.2174641 ],  [0.22630032],  [0.23455986]],, 
    [[0.16236327],  [0.15933105],  [0.16911601],  ...,  [0.22630032],  [0.23455986],  [0.22602128]],, 
    [[0.15933105],  [0.16911601],  [0.1683347 ],  ...,  [0.23455986],  [0.22602128],  [0.20916735]]
]
'''

# **************** Part 2 - Building and Training the RNN ************
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

'''
The units parameter in an LSTM layer = the number of memory cells (neurons) inside that layer.
Total number of neurons in hidden layer. Remember, LSTM has just one hidden layer spanned across different times.

Each unit has:
    its own cell state (long-term memory),
    a set of gates (input, forget, output),
    and outputs a hidden state at each timestep.
'''
'''
The return_sequences=True parameter 
    It is crucial for stacking, as it means this layer outputs a sequence of hidden states for each timestep, which then serves as the input sequence for the subsequent LSTM layer. 
    Visually, you can imagine a series of LSTM cells processing the input sequence, and each cell's output is passed to the next layer.
'''
'''
input_shape=(timesteps, features)
'''
regressor = Sequential() # sequence of layers
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
'''
A Dropout layer in a neural network is a regularization technique used to reduce overfitting by randomly turning off (dropping) a fraction of neurons during training.
Here’s the idea:
- In each TRAINING step, dropout randomly sets the output of some neurons to zero with a given probability (e.g., 0.2).
- This prevents the network from becoming too reliant on specific neurons and forces it to learn more robust and generalized features.
- At TEST (inference) time, dropout is disabled, and the full network is used, but the outputs are scaled appropriately to account for the missing neurons during training.
'''
regressor.add(Dropout(0.2)) # 10 out of 50 neurons will be disabled randomly during Training step.

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
'''
return_sequences is not explicitly set to True, it defaults to False, meaning this layer outputs only the final hidden state, summarizing the information learned from the entire sequence.
'''
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1)) # Default activation function is 'linear'

# Compiling the RNN
# optimizer means gradient descent algorithm. 'adam' is stochastic gradient descent algorithm
# RMSprop is also a good optimizer for RNNs
# For regression problems, loss function is MSE (mean_squared_error)
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, # X_train has to be a 3-D array (number of samples, number of timesteps, number of features)
              y_train,
              epochs = 100,
              batch_size = 32) # there are total 1198 samples. It is batched into 32 samples.

'''
You can see the loss reducing at each epoch.
In first epoch, it was 8%. In 100th epoch, it reduced to 0.18%.

Epoch 1/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 11s 105ms/step - loss: 0.0881
Epoch 2/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 130ms/step - loss: 0.0063
Epoch 3/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 105ms/step - loss: 0.0058
Epoch 4/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 103ms/step - loss: 0.0048
Epoch 5/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 137ms/step - loss: 0.0054
Epoch 6/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 102ms/step - loss: 0.0046
Epoch 7/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.0038
Epoch 8/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 102ms/step - loss: 0.0043
Epoch 9/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 107ms/step - loss: 0.0042
Epoch 10/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 118ms/step - loss: 0.0036
Epoch 11/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 101ms/step - loss: 0.0039
Epoch 12/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 112ms/step - loss: 0.0045
Epoch 13/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 113ms/step - loss: 0.0035
Epoch 14/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 101ms/step - loss: 0.0040
Epoch 15/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 129ms/step - loss: 0.0042
Epoch 16/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 104ms/step - loss: 0.0033
Epoch 17/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 101ms/step - loss: 0.0033
Epoch 18/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 126ms/step - loss: 0.0041
Epoch 19/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 104ms/step - loss: 0.0033
Epoch 20/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 102ms/step - loss: 0.0032
Epoch 21/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 138ms/step - loss: 0.0039
Epoch 22/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 110ms/step - loss: 0.0030
Epoch 23/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 133ms/step - loss: 0.0032
Epoch 24/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 104ms/step - loss: 0.0042
Epoch 25/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 116ms/step - loss: 0.0035
Epoch 26/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 122ms/step - loss: 0.0032
Epoch 27/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 103ms/step - loss: 0.0029
Epoch 28/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 133ms/step - loss: 0.0028
Epoch 29/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 101ms/step - loss: 0.0041
Epoch 30/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 102ms/step - loss: 0.0030
Epoch 31/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 138ms/step - loss: 0.0031
Epoch 32/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 110ms/step - loss: 0.0028
Epoch 33/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 113ms/step - loss: 0.0033
Epoch 34/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 102ms/step - loss: 0.0031
Epoch 35/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 7s 144ms/step - loss: 0.0030
Epoch 36/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 101ms/step - loss: 0.0027
Epoch 37/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 133ms/step - loss: 0.0029
Epoch 38/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 102ms/step - loss: 0.0025
Epoch 39/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 101ms/step - loss: 0.0022
Epoch 40/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 137ms/step - loss: 0.0024
Epoch 41/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 107ms/step - loss: 0.0027
Epoch 42/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 117ms/step - loss: 0.0022
Epoch 43/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 103ms/step - loss: 0.0025
Epoch 44/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 134ms/step - loss: 0.0025
Epoch 45/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 108ms/step - loss: 0.0025
Epoch 46/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 103ms/step - loss: 0.0023
Epoch 47/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 139ms/step - loss: 0.0025
Epoch 48/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 103ms/step - loss: 0.0022
Epoch 49/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 146ms/step - loss: 0.0024
Epoch 50/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 101ms/step - loss: 0.0024
Epoch 51/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 128ms/step - loss: 0.0023
Epoch 52/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 101ms/step - loss: 0.0021
Epoch 53/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 101ms/step - loss: 0.0025
Epoch 54/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 116ms/step - loss: 0.0024
Epoch 55/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 102ms/step - loss: 0.0020
Epoch 56/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 112ms/step - loss: 0.0019
Epoch 57/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 112ms/step - loss: 0.0020
Epoch 58/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 102ms/step - loss: 0.0026
Epoch 59/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 7s 142ms/step - loss: 0.0019
Epoch 60/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 101ms/step - loss: 0.0021
Epoch 61/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 133ms/step - loss: 0.0021
Epoch 62/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 113ms/step - loss: 0.0020
Epoch 63/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 110ms/step - loss: 0.0021
Epoch 64/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 100ms/step - loss: 0.0018
Epoch 65/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 131ms/step - loss: 0.0022
Epoch 66/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 108ms/step - loss: 0.0020
Epoch 67/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 101ms/step - loss: 0.0018
Epoch 68/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 134ms/step - loss: 0.0020
Epoch 69/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 103ms/step - loss: 0.0016
Epoch 70/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 103ms/step - loss: 0.0019
Epoch 71/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 136ms/step - loss: 0.0017
Epoch 72/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 101ms/step - loss: 0.0017
Epoch 73/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 102ms/step - loss: 0.0016
Epoch 74/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 139ms/step - loss: 0.0016
Epoch 75/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 105ms/step - loss: 0.0018
Epoch 76/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 120ms/step - loss: 0.0016
Epoch 77/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 102ms/step - loss: 0.0017
Epoch 78/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 7s 146ms/step - loss: 0.0016
Epoch 79/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 110ms/step - loss: 0.0018
Epoch 80/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 147ms/step - loss: 0.0016
Epoch 81/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 103ms/step - loss: 0.0015
Epoch 82/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 7s 145ms/step - loss: 0.0015
Epoch 83/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 105ms/step - loss: 0.0018
Epoch 84/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 103ms/step - loss: 0.0018
Epoch 85/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 132ms/step - loss: 0.0015
Epoch 86/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 102ms/step - loss: 0.0016
Epoch 87/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 102ms/step - loss: 0.0015
Epoch 88/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 122ms/step - loss: 0.0013
Epoch 89/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 101ms/step - loss: 0.0013
Epoch 90/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 128ms/step - loss: 0.0016
Epoch 91/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 103ms/step - loss: 0.0015
Epoch 92/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 103ms/step - loss: 0.0013
Epoch 93/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 129ms/step - loss: 0.0013
Epoch 94/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 111ms/step - loss: 0.0016
Epoch 95/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 103ms/step - loss: 0.0015
Epoch 96/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 133ms/step - loss: 0.0015
Epoch 97/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 4s 106ms/step - loss: 0.0018
Epoch 98/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 5s 104ms/step - loss: 0.0015
Epoch 99/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 7s 143ms/step - loss: 0.0017
Epoch 100/100
38/38 ━━━━━━━━━━━━━━━━━━━━ 9s 104ms/step - loss: 0.0018
'''

# **************** Part 3 - Making the predictions and visualising the results ***********
# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values # getting 'Open' stock price

dataset_total_open_series = dataset_train['Open']
dataset_test_open_series = dataset_test['Open']
print(len(dataset_total_open_series)) # 1278
print(len(dataset_test_open_series)) # 20

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_total_open_series, dataset_test_open_series), axis = 0)
# last 80 records from dataset_total_open_series
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
'''
[779.   779.66 777.71 786.66 783.76 781.22 781.65 779.8  787.85 798.24
 803.3  795.   804.9  816.68 806.34 801.   808.35 795.47 782.89 778.2
 767.25 750.66 774.5  783.4  779.94 791.17 756.54 755.6  746.97 755.2
 766.92 771.37 762.61 772.63 767.73 764.26 760.   771.53 770.07 757.44
 744.59 757.71 764.73 761.   772.48 780.   785.04 793.9  797.4  797.34
 800.4  790.22 796.76 795.84 792.36 790.9  790.68 793.7  783.33 782.75
 778.81 788.36 786.08 795.26 806.4  807.86 805.   807.14 807.48 807.08
 805.81 805.12 806.91 807.25 822.3  829.62 837.81 834.71 814.66 796.86]
'''
inputs = inputs.reshape(-1,1)
print(inputs)
'''
[[779.  ]
 [779.66]
 [777.71]
 [786.66]
 [783.76]
 [781.22]
 [781.65]
 [779.8 ]
 [787.85]
 [798.24]
 [803.3 ]
 [795.  ]
 [804.9 ]
 [816.68]
 [806.34]
 [801.  ]
 [808.35]
 [795.47]
 [782.89]
 [778.2 ]
 [767.25]
 [750.66]
 [774.5 ]
 [783.4 ]
 [779.94]
 [791.17]
 [756.54]
 [755.6 ]
 [746.97]
 [755.2 ]
 [766.92]
 [771.37]
 [762.61]
 [772.63]
 [767.73]
 [764.26]
 [760.  ]
 [771.53]
 [770.07]
 [757.44]
 [744.59]
 [757.71]
 [764.73]
 [761.  ]
 [772.48]
 [780.  ]
 [785.04]
 [793.9 ]
 [797.4 ]
 [797.34]
 [800.4 ]
 [790.22]
 [796.76]
 [795.84]
 [792.36]
 [790.9 ]
 [790.68]
 [793.7 ]
 [783.33]
 [782.75]
 [778.81]
 [788.36]
 [786.08]
 [795.26]
 [806.4 ]
 [807.86]
 [805.  ]
 [807.14]
 [807.48]
 [807.08]
 [805.81]
 [805.12]
 [806.91]
 [807.25]
 [822.3 ]
 [829.62]
 [837.81]
 [834.71]
'''

inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test)
'''
[[[0.3691268 ]
  [0.37620682]
  [0.35528856]
  ...
  [0.52681828]
  [0.41557606]
  [0.40935422]]

 [[0.37620682]
  [0.35528856]
  [0.451298  ]
  ...
  [0.41557606]
  [0.40935422]
  [0.36708861]]

 [[0.35528856]
  [0.451298  ]
  [0.4201888 ]
  ...
  [0.40935422]
  [0.36708861]
  [0.46953443]]

 ...

 [[0.54580562]
  [0.41085604]
  [0.36054495]
  ...
  [0.83361939]
  [0.91214332]
  [1.        ]]

 [[0.41085604]
  [0.36054495]
  [0.24308088]
  ...
  [0.91214332]
  [1.        ]
  [0.96674533]]

 [[0.36054495]
  [0.24308088]
  [0.06511478]
  ...
  [1.        ]
  [0.96674533]
  [0.75166273]]]
'''

predicted_stock_price = regressor.predict(X_test)
print(predicted_stock_price)
'''
[[0.9720297 ]
 [0.9667082 ]
 [0.9669529 ]
 [0.9693723 ]
 [0.97536826]
 [0.9860282 ]
 [0.99569   ]
 [0.9993895 ]
 [0.9997432 ]
 [0.99920785]
 [0.9988539 ]
 [0.998615  ]
 [0.99861073]
 [0.9999442 ]
 [1.0020235 ]
 [1.0107664 ]
 [1.0235739 ]
 [1.0370212 ]
 [1.0439674 ]
 [1.0358818 ]]
'''
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)
'''
[[801.6442 ]
 [798.7836 ]
 [798.91516]
 [800.21576]
 [803.43896]
 [809.1693 ]
 [814.3631 ]
 [816.3518 ]
 [816.54193]
 [816.25415]
 [816.0639 ]
 [815.9355 ]
 [815.93317]
 [816.64996]
 [817.7677 ]
 [822.4675 ]
 [829.35236]
 [836.58105]
 [840.31506]
 [835.96857]]
'''

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()