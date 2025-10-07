# NOTE: Install Microsoft Visual C++ latest version on your computer. Tensorflow needs it.
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import history

'''
What is Tensor?
Inputs/ Outputs in TensorFlow are called Tensor. Tensor is nothing but a multi-dimensional array for which the underpinning element type is specified at the graph construction time.

https://www.geeksforgeeks.org/deep-learning/tensorflow/

TensorFlow is an open-source machine-learning framework developed by Google. 
It is written in Python, making it accessible and easy to understand. 
It is designed to build and train machine learning (ML) and deep learning models.

    - It is highly scalable for both research and production.
    - It supports CPUs, GPUs, and TPUs for faster computation.
    - TensorFlow provides built-in tools for visualization and debugging.
    - It works seamlessly with other AI frameworks and libraries.

https://www.geeksforgeeks.org/python/introduction-to-tensorflow/
Primary Focus of
TensorFlow  - Deep learning, production-level deployment
PyTorch - Deep learning, research and experimentation. Limited deployment options compared to TensorFlow
Keras - High-level API for building deep learning models that runs on top of TensorFlow
        Simplified interface for model creation, limited flexibility compared to raw TensorFlow
        High-level API for neural networks, focused on simplifying the process of building models without needing much detail about architecture
Scikit-Learn - Traditional machine learning algorithms like decision trees, SVMs, linear regression, etc.       
               Focused on traditional machine learning, not deep learning; limited flexibility for neural networks
               Classical machine learning tasks like classification, regression, clustering, dimensionality reduction and more. 

https://www.geeksforgeeks.org/machine-learning/difference-between-tensorflow-and-keras/
Keras
It is an Open Source Neural Network library that runs on top of Theano or Tensorflow. 
It is designed to be fast and easy for the user to use. 
It is a useful library to construct any deep learning algorithm of whatever choice we want.

TensorFlow is used for large datasets and high performance models.
Keras is usually used for small datasets.

TensorFlow is a framework that offers both high and low-level APIs.
Keras is a high-Level API.

TensorFlow is used for high-performance models.
Keras is used for low-performance models.

TensorFlow has a complex architecture and not easy to use.
Keras has a simple architecture and easy to use.

Advantages of Keras:
    - Keras is the best platform out there to work on neural network models.
    - The API that Keras has a user-friendly where a beginner can easily understand.
    - Keras has the advantage that it can choose any libraries which support it for its backend support.
    - Keras provides various pre-trained models which help the user in further improving the models the user is designing.
    - When it comes to community support Keras has the best like stack overflow.

Disadvantages of Keras:
    - The major drawback of Keras is it is a low-level application programming interface.
    - Few of the pre-trained models that the Keras has been not much supportive when it comes to designing of some models.
    - The errors given by the Keras library were not much helpful for the user.

'''
print(tf.__version__)

# ************************************* Part 1 - Data Preprocessing **********************************
# Importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv') # this is dataframe
#print(dataset.iloc[:, 3:-1]) # print dataframe

#X will be a dataframe. Y will be a series. This also works for training the model.
#X = dataset.iloc[:, 3:-1]
#y = dataset.iloc[:, -1]

# '.values' will give all the values in 2-D array
X = dataset.iloc[:, 3:-1].values # exclude RowNumber, CusterId, Surname, Exited columns. RowNumber, CusterId, Surname have no impact on dependent variable Exited.
# '.values' will give all the values in 1-D array
y = dataset.iloc[:, -1].values # keep only Exited column

print(X)
# print(np.unique(X[:, 1])) # ['France' 'Germany' 'Spain']
'''
[[619 'France' 'Female' ... 1 1 101348.88]
 [608 'Spain' 'Female' ... 0 1 112542.58]
 [502 'France' 'Female' ... 1 0 113931.57]
 ...
 [709 'France' 'Female' ... 0 1 42085.58]
 [772 'Germany' 'Male' ... 1 0 92888.52]
 [792 'France' 'Female' ... 1 0 38190.78]]
'''

print(y)
'''
[1 0 1 ... 1 1 0]
'''

# Encoding categorical data

# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# If X is a dataframe
#X['Gender'] = le.fit_transform(X.iloc[:, 2])
X[:, 2] = le.fit_transform(X[:, 2])

print(X)
'''
[[619 'France' 0 ... 1 1 101348.88]
 [608 'Spain' 0 ... 0 1 112542.58]
 [502 'France' 0 ... 1 0 113931.57]
 ...
 [709 'France' 0 ... 0 1 42085.58]
 [772 'Germany' 1 ... 1 0 92888.52]
 [792 'France' 0 ... 1 0 38190.78]]
'''
# One Hot Encoding the "Geography" column
# One-Hot encoding is a method used to represent categorical data, where each category is represented by a binary variable.
# The binary variable takes the value 1 if the category is present and 0 otherwise.
# The binary variables are also known as dummy variables.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# you want to apply One Hot Encoding on the column 1 of 2-D array (Geography). 'encoder' is just a unique name given to this transformer.
'''
ColumnTransformer is a class within the sklearn.compose module of the scikit-learn Python machine learning library. 
It is designed to apply different data transformation techniques to different columns or subsets of columns within a dataset.

ColumnTransformer takes a list of tuples, where each tuple specifies:
- A unique name for the transformer.
- The transformer object itself (e.g., StandardScaler, OneHotEncoder, SimpleImputer).
- The columns (by index or name) to which that transformer should be applied.
'''
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)
'''
OnHotEncoding keeps the encoded columns at the beginning with 1.0/0.0 values. 
One column from France, another for Germany, third one for Spain.

get_dummies needs dataframe.
get_Dummies does the same thing by keeping the encoded columns at the end with True/False values.
In Python, and consequently in Pandas, True is treated as 1 and False is treated as 0 when used in numerical contexts or when explicitly converted to an integer type. 
This is due to Python's underlying representation of booleans as a subclass of integers.

[[1.0 0.0 0.0 ... 1 1 101348.88]
 [0.0 0.0 1.0 ... 0 1 112542.58]
 [1.0 0.0 0.0 ... 1 0 113931.57]
 ...
 [1.0 0.0 0.0 ... 0 1 42085.58]
 [0.0 1.0 0.0 ... 1 0 92888.52]
 [1.0 0.0 0.0 ... 1 0 38190.78]]
'''

'''
# You should be able to use OneHotEncoder() without ColumnTransformer also.

onehotencoder = OneHotEncoder()
# Example with a Pandas DataFrame 'df' and a list of categorical columns 'categorical_cols'
# transformed_data = onehotencoder.fit_transform(df[categorical_cols])
# Example with a NumPy array 'X' and selecting columns by index (e.g., first column)
X = onehotencoder.fit_transform(X, [1]) # somehow, this is not working
print(X)
'''

'''
# get_dummies needs dataframe. So, first you need to convert X (2-D array) to dataframe
# get_Dummies does the same thing by keeping the encoded columns at the end with True/False values.
# In Python, and consequently in Pandas, True is treated as 1 and False is treated as 0 when used in numerical contexts or when explicitly converted to an integer type. 
# This is due to Python's underlying representation of booleans as a subclass of integers.

df = pd.DataFrame(X)
print(df)
X = pd.get_dummies(df, columns=[1]).values # get_dummies gives a dataframe with col=1 encoded. .values converts dataframe into 2-D array.
print(X)
'''
'''
# Label encoding for Geography column also
X[:, 1] = le.fit_transform(X[:, 1])
print(X)
'''


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
'''
While not strictly compulsory, data scaling is highly recommended and often crucial for effective training of artificial neural networks. 
Scaling helps prevent features with larger magnitudes from dominating the learning process and can significantly improve convergence speed and model performance. 

Here's why:

    Gradient Descent Convergence:
        Neural networks primarily learn through gradient descent, an optimization algorithm that iteratively adjusts model weights. When input features have vastly different scales, the gradients can become unstable, leading to slow or erratic convergence. Scaling brings all features to a similar range, ensuring that gradient updates are more balanced and the optimization process is smoother. 
    
    Feature Importance:
        Without scaling, features with larger values can disproportionately influence the model's calculations, 
        potentially masking the importance of smaller-scale features, even if they are highly relevant. 
        Scaling helps ensure all features contribute appropriately to the learning process. 
    
    Preventing Numerical Instability:
        Some activation functions (like sigmoid) are more sensitive to the scale of inputs. 
        Scaling can help prevent numerical instability issues that might arise from very large or very small input values. 
    
    Faster Training:
        By improving the stability and efficiency of gradient descent, scaling can lead to faster training times and reduced computational cost. 
    
    While some network architectures or specific activation functions might be less sensitive to scaling, in general, 
    it's a best practice to scale your data before training a neural network. Common scaling methods include: 
    
        Min-Max Scaling: Rescales data to a range between 0 and 1.
        Standardization: Centers the data around zero with a unit standard deviation.
'''
# you have to convert the data into standardize format. It will convert the data with 0 mean and 1 standard deviation.
# Read more about normalization and standardization in 'Normalization and Standardization.docx'
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
'''
https://www.geeksforgeeks.org/python/what-is-the-difference-between-transform-and-fit_transform-in-sklearn-python/

fit() component:
    It calculates the necessary parameters from the data. 
    For example, for StandardScaler, it calculates the mean and standard deviation of each feature. 
    For SimpleImputer, it might calculate the mean or median to fill missing values.
transform() component:
    It then applies the transformation to the data using the parameters learned during the fit step. 

    Purpose: This method applies a pre-learned transformation to new data.
    Dependency: It requires that the fit() method (or fit_transform()) has already been called on some data to learn the necessary transformation parameters.
    Usage: transform() is used on new or unseen data, such as test sets or production data, to apply the same transformation learned from the training data. This ensures consistency in the preprocessing applied to all data used by the model.

fit_transform() is typically used on the training data to both learn the transformation parameters and apply the transformation to that data. 
This ensures that the parameters are learned from the data the model will be trained on.

Key Difference:
    The fundamental difference lies in parameter learning. 
    fit_transform() learns the parameters and applies the transformation, while transform() only applies a transformation using already learned parameters. 
    Using transform() on test data with parameters learned from the training data prevents data leakage from the test set into the model training process.
'''
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ************************************* Part 1 - Data Preprocessing Ends **********************************
# ************************************* Part 2 - Building the ANN *************************************
# Initializing the ANN. ANN is a sequence of layers.
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
'''
When you add the first hidden layer, 
input layer with proper number of neurons will automatically be added based on number of input features.
'''
'''
How to decide number of neurons (units)?
Unfortunately, there is no rule of thumb. You just need to try different numbers and pick the one that works better.
Here, I tried multiple numbers. All of them gave almost the same result. So, I picked 6.
For all the hidden layers, activation function bust be ReLU or 'Leaky ReLU)
'''
'''
A Dense layer, also known as a fully connected layer, is a fundamental building block in neural networks. 
In this layer, every neuron receives input from all neurons in the previous layer, and each connection has an associated weight. 
The layer then applies a transformation to the weighted sum of its inputs, typically followed by an activation function.
'''
'''
ReLU vs Leaky ReLU:
Leaky Rectified Linear Unit (Leaky ReLU) is an activation function designed to address the "dying ReLU" problem, where neurons can become inactive and stop learning for negative input values. 
Unlike standard ReLU, which outputs zero for negative inputs, Leaky ReLU introduces a small, non-zero slope for negative values. 
This ensures that even when the input is negative, a small gradient can still pass through, allowing the neuron to continue learning.

activation='Leaky Relu'
'''
'''
If you add just one hidden layer, it creates a shallow NN.
So, create at least two hidden layers to create a deep NN.
'''
'''
Number of Layers:
There's no hard and fast rule for the exact number of hidden layers, but as a general guideline:
    Simple tasks: One or two hidden layers might be enough. 
    Complex tasks: Deep learning models with several hidden layers (e.g., 3 or more) are often necessary. 
For simple tasks, a single hidden layer may suffice, while more complex tasks, like image recognition or natural language processing, often require multiple hidden layers. 
Deep learning, which involves networks with multiple hidden layers, is frequently used for these more complex problems. 

Single Hidden Layer:
    Neural networks with one hidden layer can effectively solve many problems, especially those involving linearly separable data or relatively simple patterns. 
Multiple Hidden Layers (Deep Learning):
    As the complexity of the task increases, adding more hidden layers allows the network to learn more intricate relationships and features within the data. 
    Deep learning, which is characterized by multiple hidden layers, is particularly useful for tasks that require high levels of abstraction and representation learning. 

Overfitting:
Adding too many layers can lead to overfitting, where the network learns the training data too well and doesn't generalize effectively to new, unseen data. 

Practical Considerations:
When determining the number of hidden layers, factors like the size and complexity of the dataset, the desired accuracy, and computational resources available should be considered. 

Empirical Approach:
Often, the best approach is to start with a reasonable number of layers and then experiment, monitoring performance on a validation set to find the optimal architecture for a given task. 
'''
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # adding an object of Dense class to ann

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # adding an object of Dense class to ann

# Adding the output layer
'''
Here, I am expecting binary output 0 or 1 (Classification problem). So, just one neuron is required in output layer.
If there are 3 categorical values in output, you need to first use OneHotEncoding for that variable and then
have 3 neurons in output layer. which will give 1,0,0 or 0,1,0 or 0,0,1 as an output.
'''
'''
The Softmax activation function is appropriate for the output layer of a multi-class classification neural network.
The Softmax function takes a vector of arbitrary real-valued scores and transforms them into a probability distribution over multiple classes. 
This means that the output values for all classes will sum to 1, and each individual output value will be between 0 and 1, representing the estimated probability of the input belonging to that specific class.
This characteristic makes Softmax ideal for multi-class classification problems where the goal is to assign an input to one of several mutually exclusive classes, and a probabilistic interpretation of the output is desired.
'''
'''
For the output layer of a regression neural network, the linear activation function is most appropriate. 
This function allows the output to be any real number, which is suitable for predicting continuous values like prices, temperatures, or scores. 
Other activation functions, like ReLU or sigmoid, are typically used in the hidden layers or for classification tasks, but not for the final output in regression. 

A linear function (f(x) = x) simply passes the input value through without modification, 
which is ideal for regression as it doesn't impose any restrictions on the output range. 
'''
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # adding an object of Dense class to ann

# ************************************ Part 3 - Training the ANN ************************************
# Compiling the ANN
'''
Optimizer:
Read 'Optimizers (Gradient Descents)' section in 'Udemy Course.docx'

Loss Function:
Read 'crossentropy loss function' section in 'Udemy Course.docx'

Performance metrics:
Read 'Performance metrics' section in 'Udemy Course.docx'
'''
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', 'binary_accuracy'])

# Training the ANN on the Training set

# batch_size = 32 means 32 rows (observations/data) will be fed together to make NN learn.
# 32 rows will be fed, NN will give some output. This output will be compared with actual output,
# error will be calculated, stochastic gradient descent (adam) will be used to reduce the error (because of optimizer=adam), weights will be modified accordingly.
# batch_size parameter can be twicked, but normally 32 works the best.
# When training a model, data is split into mini-batches because:
# It’s more memory-efficient than using all data at once
# It helps the model update weights more frequently (faster convergence)


# epochs=100 means NN will be trained 100 times with the same data to improve its outcome.
# do not keep it too small. NN needs certain amount of epochs to learn properly.
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
'''
------ There are 250 batches of data in the epoch, each batch has 32 records
------ total training samples = 8000 out of 10000, batch size = 32
Epoch 1/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.7618 - loss: 0.5313 
Epoch 2/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.7977 - loss: 0.4429
Epoch 3/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8242 - loss: 0.4202
Epoch 4/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8309 - loss: 0.4077
Epoch 5/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8355 - loss: 0.3937
Epoch 6/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8213 - loss: 0.4091
Epoch 7/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8318 - loss: 0.3900
Epoch 8/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8250 - loss: 0.4000
Epoch 9/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8371 - loss: 0.3786
Epoch 10/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8290 - loss: 0.3916
Epoch 11/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8344 - loss: 0.3777
Epoch 12/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8385 - loss: 0.3758
Epoch 13/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8425 - loss: 0.3729
Epoch 14/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8512 - loss: 0.3588
Epoch 15/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8480 - loss: 0.3652
Epoch 16/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8469 - loss: 0.3594
Epoch 17/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8508 - loss: 0.3622
Epoch 18/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8506 - loss: 0.3615
Epoch 19/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8493 - loss: 0.3594
Epoch 20/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8572 - loss: 0.3505
Epoch 21/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8435 - loss: 0.3741
Epoch 22/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8623 - loss: 0.3460
Epoch 23/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8505 - loss: 0.3557
Epoch 24/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8607 - loss: 0.3435
Epoch 25/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8591 - loss: 0.3453
Epoch 26/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8567 - loss: 0.3497
Epoch 27/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8594 - loss: 0.3395
Epoch 28/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8556 - loss: 0.3447
Epoch 29/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8610 - loss: 0.3407
Epoch 30/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8618 - loss: 0.3384
Epoch 31/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8533 - loss: 0.3497
Epoch 32/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8679 - loss: 0.3287
Epoch 33/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8544 - loss: 0.3481
Epoch 34/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8596 - loss: 0.3300
Epoch 35/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8573 - loss: 0.3401
Epoch 36/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8612 - loss: 0.3391
Epoch 37/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8611 - loss: 0.3406
Epoch 38/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8641 - loss: 0.3351
Epoch 39/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8663 - loss: 0.3366
Epoch 40/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8626 - loss: 0.3326
Epoch 41/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8627 - loss: 0.3416
Epoch 42/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8654 - loss: 0.3351
Epoch 43/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8623 - loss: 0.3325
Epoch 44/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8614 - loss: 0.3418
Epoch 45/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8659 - loss: 0.3259
Epoch 46/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8682 - loss: 0.3233
Epoch 47/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8624 - loss: 0.3399
Epoch 48/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8638 - loss: 0.3341
Epoch 49/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8573 - loss: 0.3518
Epoch 50/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8572 - loss: 0.3517
Epoch 51/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8588 - loss: 0.3391
Epoch 52/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8651 - loss: 0.3307
Epoch 53/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8577 - loss: 0.3393
Epoch 54/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8608 - loss: 0.3359
Epoch 55/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8667 - loss: 0.3286
Epoch 56/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8674 - loss: 0.3323
Epoch 57/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8676 - loss: 0.3329
Epoch 58/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8658 - loss: 0.3285
Epoch 59/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8600 - loss: 0.3375
Epoch 60/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8632 - loss: 0.3318
Epoch 61/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8651 - loss: 0.3315
Epoch 62/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8678 - loss: 0.3268
Epoch 63/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8649 - loss: 0.3308
Epoch 64/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8714 - loss: 0.3219
Epoch 65/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8723 - loss: 0.3289
Epoch 66/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8670 - loss: 0.3332
Epoch 67/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8662 - loss: 0.3314
Epoch 68/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8564 - loss: 0.3400
Epoch 69/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8626 - loss: 0.3289
Epoch 70/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8649 - loss: 0.3408
Epoch 71/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8652 - loss: 0.3235
Epoch 72/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8598 - loss: 0.3382
Epoch 73/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8598 - loss: 0.3370
Epoch 74/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8613 - loss: 0.3353
Epoch 75/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8620 - loss: 0.3430
Epoch 76/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8688 - loss: 0.3249
Epoch 77/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8658 - loss: 0.3278
Epoch 78/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8642 - loss: 0.3312
Epoch 79/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8679 - loss: 0.3260
Epoch 80/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8662 - loss: 0.3277
Epoch 81/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8688 - loss: 0.3321
Epoch 82/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8587 - loss: 0.3376
Epoch 83/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8684 - loss: 0.3286
Epoch 84/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8660 - loss: 0.3234
Epoch 85/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8591 - loss: 0.3403
Epoch 86/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8646 - loss: 0.3314
Epoch 87/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8696 - loss: 0.3224
Epoch 88/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8647 - loss: 0.3257
Epoch 89/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8607 - loss: 0.3445
Epoch 90/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8622 - loss: 0.3299
Epoch 91/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8591 - loss: 0.3412
Epoch 92/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8631 - loss: 0.3305
Epoch 93/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8669 - loss: 0.3299
Epoch 94/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8758 - loss: 0.3144
Epoch 95/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8599 - loss: 0.3368
Epoch 96/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8657 - loss: 0.3280
Epoch 97/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8666 - loss: 0.3283
Epoch 98/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8663 - loss: 0.3274
Epoch 99/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8707 - loss: 0.3259
Epoch 100/100
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8602 - loss: 0.3308
<keras.src.callbacks.history.History at 0x78b291eee530>
'''

# ************************************ Part 4 - Making the predictions and evaluating the model ************************************

# Predicting the result of a single observation

'''
Use our ANN model to predict if the customer with the following information will leave the bank:

Geography: France  ---- encoded as 1,0,0
Credit Score: 600
Gender: Male        ---- labeled as 1
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2

Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000

So, should we say goodbye to that customer?
'''
# If OneHotEncoder is used for Geography column. 1,0,0 is added at the beginning for France, Germany, Spain. Randomly 3 columns are arranged at the beginning.
data = [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
standardized_data = sc.transform(data)
prediction = ann.predict(standardized_data)
# We have used sigmoid function in output layer, so prediction will be in between 0 and 1
# If it is < 0.5, we can consider it close to 0, otherwise close to 1.
print(prediction) # [[0.02852276]]  very low chance to leave the bank
print(prediction > 0.5) # [[False]] customer will not leave the bank

# If get_dummies is used for Geography column: True, False, False is added at the end for France, Germany, Spain. Randomly 3 columns are arranged at the end.
# data = [[600, 1, 40, 3, 60000, 2, 1, 1, 50000, True, False, False]]
# standardized_data = sc.transform(data)
# prediction = ann.predict(standardized_data)
# print(prediction)
# print(prediction > 0.5) # [[False]]

# If label encoding is used for Geography column, 1- France, 2-Germany, 0-Spain. Random labeling.
# data = [[600, 1, 1, 40, 3, 60000, 2, 1, 1, 50000]]
# standardized_data = sc.transform(data)
# prediction = ann.predict(standardized_data)
# print(prediction)
# print(prediction > 0.5) # [[False]]
'''
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step
[[False]]
'''

'''
Therefore, our ANN model predicts that this customer stays in the bank!
Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
'''

# metrics from tensorflow. We set metrics = ['accuracy', 'binary_accuracy'], so it will return those.
# finds y_pred from x_test and then compares y_pred and y_test to find accuracy
loss, accuracy, binary_accuracy = ann.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}") # 0.33554255962371826
print(f"Test Accuracy: {accuracy}") # 0.8585000038146973
print(f"Test Binary Accuracy: {binary_accuracy}") # 0.8585000038146973


# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(len(y_pred)) # 2000
'''
np.column_stack((arr1, arr2)) takes these 1D arrays and treats each as a column in the resulting 2D array.
The output result will be a 2D array where the first column contains elements from arr1 and the second column contains elements from arr2.
'''
y_pred_1D_array = y_pred.flatten() # converts 2-D array into 1-D array
result = np.column_stack((y_pred_1D_array,y_test)) # pass arrays within a tuple or list
print(result)

# OR
'''
# syntax: reshape(number of rows, number of columns)
#y_pred_2D_array = y_pred.reshape(len(y_pred),1) # y_pred is already a 2-D array
y_test_2D_array = y_test.reshape(len(y_test),1)# reshape converts 1-D array to 2-D array
print(y_test_2D_array)
result = np.concatenate((y_pred, y_test_2D_array),axis=1)
print(result)
'''

'''
predicated result - test data result
[[0 0]
 [0 1]
 [0 0]
 ...
 [0 0]
 [0 0]
 [0 0]]
'''

# Making the Confusion Matrix and finding accuracy using sklearn library
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
'''
[[1529   66]
 [ 206  199]]
0.864
'''

'''
[[1529   66]
 [ 206  199]]
            ------------------
         0  | 1529  | 66    |       customers stays with the bank
  test_y    |       |       |
            ------------------
         1  |  206  | 199   |       customers leaves the bank
            |       |       |
            ------------------
                0       1
                predictions

correctly predicated result
1529 and 199 predicated results matches with test data results
66 and 206 predicted result DOESN'T match with test data result

The Confusion Matrix created has four different quadrants:

True Negative (Top-Left Quadrant)
False Positive (Top-Right Quadrant)
False Negative (Bottom-Left Quadrant)
True Positive (Bottom-Right Quadrant)

True means that the values were accurately predicted, 
False means that there was an error or wrong prediction.

Accuracy measures how often the model is correct.
How to Calculate?
Accuracy = (True Positive + True Negative) / Total Predictions
'''
