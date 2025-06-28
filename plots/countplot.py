# countplot :
# A count plot/bar plot shows the number of occurrences of each category in a categorical feature.
# It’s commonly used for Exploratory Data Analysis (EDA) to understand frequency of each category.

import seaborn as sns
import matplotlib.pyplot as plt

# Load example dataset
df = sns.load_dataset("titanic")

# Count plot of 'sex' column
# This will show how many passengers are male vs. female in the Titanic dataset.
sns.countplot(data=df, x="sex")
plt.title("Count of Passengers by Sex")
plt.show()
