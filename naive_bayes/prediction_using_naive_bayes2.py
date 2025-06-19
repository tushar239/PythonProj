# taken from chatgpt

# Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
texts = [
    "Win money now", "Free offer just for you", "Important update",
    "Your invoice", "Congratulations, you won!", "Meeting schedule"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = spam, 0 = not spam

# Convert text to feature vectors (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
print(X)
print(type(X))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Predict new text
new_text = ["Win a free iPhone"]
new_vector = vectorizer.transform(new_text)
prediction = nb_model.predict(new_vector)
print("Predicted class:", prediction)  # 1 = spam, 0 = not spam
