'''
Isolation Forest is a machine learning algorithm specifically designed
to detect anomalies (outliers) in data.

It works by isolating outliers instead of profiling normal data points —
which makes it fast, efficient, and scalable, even for large datasets.
'''

from sklearn.ensemble import IsolationForest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Sample DataFrame
df = pd.DataFrame({
    'age': [15, 16, 14, 17, 18, 16, 100]  # 100 is an outlier
})

iso = IsolationForest(contamination=0.1) # contamination=0.1 means we expect ~10% of points to be outliers.
df['outlier'] = iso.fit_predict(df[['age']])

# -1 = outlier, 1 = inlier
print(df[df['outlier'] == -1])
'''
   age  outlier
6  100       -1
'''

####### box plot before dropping outlier rows ##################
sns.boxplot(y=df['age'])
plt.title("Box Plot of age before dropping outlier rows")
plt.show()

###### Dropping outliers from dataframe ##########
# Condition: Drop rows where 'Score' is less than 80
condition = df['outlier'] == -1

# Get the index of rows to drop
indices_to_drop = df[condition].index
print(type(indices_to_drop)) # <class 'pandas.core.indexes.base.Index'>
print(len(indices_to_drop)) # total 1 rows needs to be removed

# Drop the rows (in-place)
df.drop(indices_to_drop, inplace=True)

######## box plot after dropping outlier rows ##################
sns.boxplot(y=df['age'])
plt.title("Box Plot of age after dropping outlier rows")
plt.show()

################################################################


