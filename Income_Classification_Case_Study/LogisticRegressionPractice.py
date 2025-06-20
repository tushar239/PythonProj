# https://www.w3schools.com/python/python_ml_logistic_regression.asp

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score

# Data
student = {
    'tumor_size': [3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88],
    'is_cancerous': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
}

# Creating a DataFrame object
df = pd.DataFrame(student)
print(df)

tumor_size=df.loc[::, ['tumor_size']]
print(tumor_size)
'''
    tumor_size
0         3.78
1         2.44
2         2.09
3         0.14
4         1.72
5         1.65
6         4.92
7         4.37
8         4.96
9         4.52
10        3.69
11        5.88
'''

train_x = df.loc[::, ['tumor_size']].values
train_y = df.loc[::, ['is_cancerous']].values

print(train_x)
'''
[[3.78]
 [2.44]
 [2.09]
 [0.14]
 [1.72]
 [1.65]
 [4.92]
 [4.37]
 [4.96]
 [4.52]
 [3.69]
 [5.88]]
'''

logr = linear_model.LogisticRegression()
logr.fit(train_x,train_y)

test_x = [[3.46],
         [2.49],
         [2.09],
         [0.79],
         [1.29],
         [4.03],
         [5.02]]
test_y = [[0],
         [0],
         [0],
         [0],
         [0],
         [1],
         [1]]

predictions = logr.predict(test_x)
print(predictions) # [0 0 0 0 0 1 1]

# Calculating Accuracy Score
'''
Accuracy measures how often the model is correct.
How to Calculate?
Accuracy = (True Positive + True Negative) / Total Predictions
'''
accuracy_score1 = accuracy_score(test_y, predictions)
print(accuracy_score1) # 1.0


confusion_matrix_1 = confusion_matrix(test_y, predictions)
print(confusion_matrix_1)
'''
[[5 0]
 [0 2]]
 
           ------------
         0  | 5  | 0 |
  test_y    |    |   |
            -----------
         1  |  0 | 2 |
            |    |   |
            -----------
               0   1
                predictions

The Confusion Matrix created has four different quadrants:

True Negative (Top-Left Quadrant)
False Positive (Top-Right Quadrant)
False Negative (Bottom-Left Quadrant)
True Positive (Bottom-Right Quadrant)

True means that the values were accurately predicted, 
False means that there was an error or wrong prediction.
 '''


