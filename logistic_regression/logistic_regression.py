# this example is taken from chatgpt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Hours studied
X = np.array([[1],
              [2],
              [3],
              [4],
              [5],
              [6],
              [7],
              [8],
              [9],
              [10]])
# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred) # Predictions: [1 0 0]

print("Accuracy:", accuracy_score(y_test, y_pred)) # Accuracy: 0.6666666666666666
print("Classification Report:\n", classification_report(y_test, y_pred))

'''
Classification Report:
               precision    recall  f1-score   support

           0       0.50      1.00      0.67         1
           1       1.00      0.50      0.67         2

    accuracy                           0.67         3
   macro avg       0.75      0.75      0.67         3
weighted avg       0.83      0.67      0.67         3
'''

X_plot = np.linspace(0, 11, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_plot)[:, 1]  # Probability of class 1

plt.scatter(X, y, c=y, cmap='bwr', label='Actual')
plt.plot(X_plot, y_prob, color='black', label='Logistic Curve')
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression Prediction")
plt.legend()
plt.grid(True)
plt.show()
