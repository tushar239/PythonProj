

# To work with dataframes
import pandas as pd

# To perform numerical operations
import numpy as np

# To visualize data
import seaborn as sns
import matplotlib.pyplot as plt  # pyplot means python plot
import gotoDataDir

data_income = pd.read_csv('income.csv')
data = data_income.copy()
# print(data)
print(data.to_string())


# go back and read the data by including "na_values[' ?']
data = pd.read_csv('income.csv', na_values=[' ?'])

# We will drop all the rows where there are missing values because there are only 1816 rows like that out of 32k rows
# axis=0 means drop all those ROWs that have missing values
# axis=1 means drop all those COLUMNs that have missing values
data2 = data.dropna(axis=0)
total_rows_left = data2.shape[0]
print("After dropping the rows with missing values, total rows left: ", total_rows_left) # 30162

#============================================================
# Building Logistic Regression Model

# Regression means predicting the relationship between two variable
# Linear Regression - Based on independent variable, predicting the numerical value
# Logistic Regression - Based on independent variable, predicting the categorical value (0 or 1)
# Watch Logistic Regression video
# Watch Linear Regression video - superb video
#============================================================
# It is a machine learning algorithm.
# It works with numbers only. So, we have to change categorical variables to numbers.

# Reindexing the salary status names to 0,1

# map function doesn't work. Instead, use loc function.
# https://www.geeksforgeeks.org/python-pandas-map/
#data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000': '0', ' greater than 50,000': '1'})
#print(data2['SalStat'])

# df.loc[row_indexer, "col"] = values
data2.loc[(data2['SalStat'] == ' less than or equal to 50,000'), 'SalStat'] = 0
data2.loc[(data2['SalStat'] == ' greater than 50,000'), 'SalStat'] = 1
print(data2['SalStat'])

new_data = pd.get_dummies(data2, drop_first=True)