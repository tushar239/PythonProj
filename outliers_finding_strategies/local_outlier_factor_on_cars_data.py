from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import numpy as np

cars_data=pd.read_csv('cars_samples.csv')
cars = cars_data.copy()
info = cars.info()
print(info)

#X, y = load_iris(return_X_y=True)
#print(y)

'''
# 2. Separate features (X) and target (y)
X = df.drop('target', axis=1)  # Features DataFrame
y = df['target']              # Target Series
'''

# creating a new dataframe that contains only powerPS variable
just_powerPS_df = cars['powerPS'].to_frame()

print(just_powerPS_df)
print(len(just_powerPS_df)) # 50001

# Using Cross-Validation for finding k-value
def choose_k_value():

    k_range = range(1, 31)
    cv_scores = []

    # 5-fold cross-validation
    # cv=5 splits data into 5 parts and runs training/testing 5 times
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, just_powerPS_df, cars['price'], cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())

    optimal_k = k_range[np.argmax(cv_scores)]
    print(f"Optimal k: {optimal_k}")
    return optimal_k

# Using Cross-Validation for finding k-value
k = choose_k_value()
print('k_value: ', k)

# Fit LOF model. n_neighbors is same has number of neighbours in KNN algorithm.
lof = LocalOutlierFactor(n_neighbors=k)
just_powerPS_df['outlier'] = lof.fit_predict(just_powerPS_df[['powerPS']])  # -1 = outlier, 1 = inlier
scores = -lof.negative_outlier_factor_  # Higher = more likely to be outlier

# -1 = outlier, 1 = inlier
print(just_powerPS_df[just_powerPS_df['outlier'] == -1])

####### box plot before dropping outlier rows ##################
sns.boxplot(y=cars['powerPS'])
plt.title("Box Plot of powerPS before dropping outlier rows")
plt.show()

###### Dropping outliers from cars dataframe ##########
# Condition: Drop rows where 'Score' is less than 80
condition = just_powerPS_df['outlier'] == -1

# Get the index of rows to drop
indices_to_drop = just_powerPS_df[condition].index
print(type(indices_to_drop)) # <class 'pandas.core.indexes.base.Index'>
print(len(indices_to_drop)) # total 1948 rows needs to be removed

# Drop the rows (in-place)
cars.drop(indices_to_drop, inplace=True)

######## box plot after dropping outlier rows ##################
sns.boxplot(y=cars['powerPS'])
plt.title("Box Plot of powerPS after dropping outlier rows")
plt.show()