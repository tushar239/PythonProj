# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset (e.g., Iris)
iris = datasets.load_iris()
X = iris.data          # Features
y = iris.target        # Labels

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create SVM model (linear kernel here, others: 'rbf', 'poly', 'sigmoid')
svm_model = SVC(kernel='linear', C=1.0)

# Train the model
svm_model.fit(X_train, y_train)

# Predict on test set
y_pred = svm_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict on a new data point (example)
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = svm_model.predict(new_data)
print("Predicted class:", prediction)
