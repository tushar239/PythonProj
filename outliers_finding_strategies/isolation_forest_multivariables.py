from sklearn.ensemble import IsolationForest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# finding outlier observation with multi-variables data
df = pd.DataFrame({
    'age': [15, 16, 14, 17, 18, 16, 100],  # (100,2000) is an outlier
    'year': [1978, 1979, 1974, 1975, 1976, 1977, 2000]
})

def boxplots(df):
    # First subplot (left)
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['age'])
    plt.title("Age")

    # Second subplot (right)
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['year'])
    plt.title("Year")

    plt.tight_layout()  # Adjust spacing
    plt.show()

    '''
    # both age and year in the same box plot
    data = [df['age'], df['year']]
    plt.boxplot(data, labels=["age", "year"])
    plt.title("Box Plot Example")
    plt.ylabel("Values")
    plt.grid(True)
    plt.show()
    '''
# box plots before dropping outliers
boxplots(df=df)

iso = IsolationForest(contamination=0.1) # contamination=0.1 means we expect ~10% of points to be outliers.
df['outlier'] = iso.fit_predict(df)

# -1 = outlier, 1 = inlier
print(df[df['outlier'] == -1])
'''
   age  year  outlier
6  100  2000       -1
'''

###### Dropping outliers dataframe ##########
# Condition: Drop rows where 'Score' is less than 80
condition = df['outlier'] == -1

# Get the index of rows to drop
indices_to_drop = df[condition].index
print(type(indices_to_drop)) # <class 'pandas.core.indexes.base.Index'>
print(len(indices_to_drop)) # total 1 rows needs to be removed

# Drop the rows (in-place)
df.drop(indices_to_drop, inplace=True)

# box plots after dropping outliers
boxplots(df=df)



iso = IsolationForest(contamination=0.03) # contamination=0.1 means we expect ~10% of points to be outliers.
df['outlier2'] = iso.fit_predict(df)

# -1 = outlier, 1 = inlier
print(df[df['outlier2'] == -1])
'''
   age  year  outlier  outlier2
6  100  2000       -1        -1
'''


