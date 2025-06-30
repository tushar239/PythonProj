from scipy.stats import zscore
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'age': [15, 16, 14, 17, 18, 16, 100]  # 100 is an outlier
})

df['z_age'] = zscore(df['age'])
print(df)
'''
   age     z_age
0   15 -0.441904
1   16 -0.407911
2   14 -0.475896
3   17 -0.373919
4   18 -0.339926
5   16 -0.407911
6  100  2.447467
'''
outliers = df[df['z_age'].abs() > 3]
print("Outliers based on Z-score:")
print(outliers)
