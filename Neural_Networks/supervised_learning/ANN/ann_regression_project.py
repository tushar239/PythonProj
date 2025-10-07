# https://www.udemy.com/course/linear-regression-with-artificial-neural-network/
# Dataset location: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)

dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
'''
[[  14.96   41.76 1024.07   73.17]
 [  25.18   62.96 1020.04   59.08]
 [   5.11   39.4  1012.16   92.14]
 ...
 [  31.32   74.33 1012.92   36.48]
 [  24.48   69.45 1013.86   62.39]
 [  21.6    62.52 1017.23   67.87]]
'''
print(y)
'''
[463.26 444.37 488.56 ... 429.57 435.74 453.28]
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_train)
'''
[[-1.13572795 -0.88685592  0.67357894  0.52070558]
 [-0.80630243 -0.00971567  0.45145467  0.14531044]
 [ 1.77128416  1.84743445  0.24279248 -1.88374143]
 ...
 [-0.38409993 -1.24886277  0.84522042  0.13092486]
 [-0.9232821  -1.04155299  1.54693117  0.8830852 ]
 [ 1.70136528  1.05824381 -1.20438076 -2.42285818]]
'''


ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

'''
For the output layer of a regression model, a linear activation function is generally appropriate. 
This function simply outputs the input value directly, without any transformation or squashing. 
This allows the model to predict a continuous range of numerical values. 
Using a linear activation function is crucial in regression tasks because it preserves the real-valued nature of the predictions. 

The default activation function for a tf.keras.layers.Dense layer in Keras is the linear activation function.
'''
ann.add(tf.keras.layers.Dense(units=1, activation='linear'))

ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

'''
Epoch 1/100
240/240 [==============================] - 0s 2ms/step - loss: 82110.9922
Epoch 2/100
240/240 [==============================] - 0s 1ms/step - loss: 656.9040
Epoch 3/100
240/240 [==============================] - 0s 1ms/step - loss: 429.4050
Epoch 4/100
240/240 [==============================] - 0s 1ms/step - loss: 417.9655
Epoch 5/100
240/240 [==============================] - 0s 1ms/step - loss: 405.0805
Epoch 6/100
240/240 [==============================] - 0s 1ms/step - loss: 388.9652
Epoch 7/100
240/240 [==============================] - 0s 2ms/step - loss: 371.9583
Epoch 8/100
240/240 [==============================] - 0s 1ms/step - loss: 353.7564
Epoch 9/100
240/240 [==============================] - 0s 1ms/step - loss: 332.1931
Epoch 10/100
240/240 [==============================] - 0s 1ms/step - loss: 310.8850
Epoch 11/100
240/240 [==============================] - 0s 1ms/step - loss: 288.6723
Epoch 12/100
240/240 [==============================] - 0s 1ms/step - loss: 265.4893
Epoch 13/100
...
Epoch 99/100
240/240 [==============================] - 0s 1ms/step - loss: 26.8439
Epoch 100/100
240/240 [==============================] - 0s 1ms/step - loss: 26.7392
'''

X_test = sc.transform(X_test)

y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

'''
[[429.1  431.23]
 [460.12 460.01]
 [463.61 461.14]
 ...
 [470.85 473.26]
 [437.71 438.  ]
 [456.9  463.28]]
'''