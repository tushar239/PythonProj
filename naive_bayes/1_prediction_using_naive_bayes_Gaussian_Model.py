# taken from chatgpt

# Import libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load a sample dataset (Iris)
iris = datasets.load_iris()
X = iris.data            # Features
y = iris.target          # Labels

print("Input Variables are numeric and continuous")
print(iris.feature_names) # feature names
print(X)
print("Target names:", iris.targqet_names)
print(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Gaussian Naive Bayes is used for continuous data,
# Multinomial Naive Bayes for discrete data (like word counts in text), and
# Bernoulli Naive Bayes for binary features.
# Here, creating Naive Bayes model (Gaussian Naive Bayes)
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict on new data
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_data)
print("Predicted class:", prediction) # Predicted class: [0]
