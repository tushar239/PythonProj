import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # .values converts dataframe into an array
print(type(training_set)) # <class 'numpy.ndarray'>
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
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
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

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    a_array = training_set_scaled[i-60:i, 0] # e.g. 0-59 rows, 0th col in each row
    X_train.append(a_array)
    b_element = training_set_scaled[i, 0] # e.g. 60th row, 0th col in it
    y_train.append(b_element)
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train)
'''
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
print(X_train.shape) # (1198, 60, 1)
print(X_train)
'''
[[[0.08581368],  [0.09701243],  [0.09433366],  ...,  [0.07846566],  [0.08034452],  [0.08497656]],, [[0.09701243],  [0.09433366],  [0.09156187],  ...,  [0.08034452],  [0.08497656],  [0.08627874]],, [[0.09433366],  [0.09156187],  [0.07984225],  ...,  [0.08497656],  [0.08627874],  [0.08471612]],, ...,, [[0.16227026],  [0.16236327],  [0.15933105],  ...,  [0.2174641 ],  [0.22630032],  [0.23455986]],, [[0.16236327],  [0.15933105],  [0.16911601],  ...,  [0.22630032],  [0.23455986],  [0.22602128]],, [[0.15933105],  [0.16911601],  [0.1683347 ],  ...,  [0.23455986],  [0.22602128],  [0.20916735]]]
'''

# **************** Part 2 - Building and Training the RNN ************
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

'''
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
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()