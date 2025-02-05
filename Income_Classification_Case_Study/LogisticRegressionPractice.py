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

train_x = df.loc[::, ['tumor_size']].values
train_y = df.loc[::, ['is_cancerous']].values

print(train_x)

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
accuracy_score1 = accuracy_score(test_y, predictions)
print(accuracy_score1) # 1.0



