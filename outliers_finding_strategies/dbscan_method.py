from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'age': [15, 16, 14, 17, 18, 16, 100]  # 100 is an outlier
})

X = df[['age']]
dbscan = DBSCAN(eps=5, min_samples=2)
df['outlier'] = dbscan.fit_predict(X)

# Outliers are labeled as -1
print(df[df['outlier'] == -1])
'''
   age  outlier
6  100       -1
'''