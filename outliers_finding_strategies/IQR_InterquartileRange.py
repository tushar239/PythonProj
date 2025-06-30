import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'age': [15, 16, 14, 17, 18, 16, 100]  # 100 is an outlier
})

Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['age'] < lower_bound) | (df['age'] > upper_bound)]
print("Outliers:")
print(outliers)
'''
Outliers:
   age
6  100
'''