# taken from chatgpt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# Sample dataset: spam vs not spam
texts = [
    "Win cash prizes now",
    "Limited time offer, act fast",
    "Your invoice is attached",
    "Team meeting schedule update",
    "Congratulations, you are selected",
    "Reminder: project deadline today"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = spam, 0 = not spam

# Convert texts to binary feature vectors
vectorizer = CountVectorizer(binary=True)  # ← Important for BernoulliNB
X = vectorizer.fit_transform(texts)
print(X)
print(type(X)) # <class 'scipy.sparse._csr.csr_matrix'>

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=42)

# Initialize and train Bernoulli Naive Bayes model
model = BernoulliNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict on new input
new_text = ["Congratulations! You won a prize"]
new_vector = vectorizer.transform(new_text)
prediction = model.predict(new_vector)
print("Predicted class:", prediction)  # Output: [1] means spam
