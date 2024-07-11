

# To work with dataframes
import pandas as pd

# To perform numerical operations
import numpy as np

# To visualize data
import seaborn as sns
import matplotlib.pyplot as plt  # pyplot means python plot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

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
# Logistic regression is a machine learning algorithm.
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

'''
https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

Convert categorical variable into dummy/indicator variables.

Each variable is converted in as many 0/1 variables as there are different values. 
Columns in the output are each named after a value; 
if the input is a DataFrame, the name of the original variable is prepended to the value.
'''
new_data = pd.get_dummies(data2, drop_first=True, dtype='int')
#new_data.replace([False, True], [0, 1], inplace=True)
print(new_data)
'''
     age  capitalgain  ...  nativecountry_ Yugoslavia  SalStat_1
0       45            0  ...                          0          0
1       24            0  ...                          0          0
2       44            0  ...                          0          1
3       27            0  ...                          0          0
4       20            0  ...                          0          0
...    ...          ...  ...                        ...        ...
31973   34          594  ...                          0          0
31974   34            0  ...                          0          0
31975   23            0  ...                          0          0
31976   42            0  ...                          0          0
31977   29            0  ...                          0          0

[30162 rows x 95 columns]
'''

# Now, we have mapped all string values to integer values. So, we can work with machine learning algorithms.

# storing the column names
type(new_data.columns) # <class 'pandas.core.indexes.base.Index'>
columns_list = list(new_data.columns)
'''
['age', 'capitalgain', 'capitalloss', 'hoursperweek', 
'JobType_ Local-gov', 'JobType_ Private', 'JobType_ Self-emp-inc', 'JobType_ Self-emp-not-inc', 'JobType_ State-gov', 'JobType_ Without-pay', 
'EdType_ 11th', 'EdType_ 12th', 'EdType_ 1st-4th', 'EdType_ 5th-6th', 'EdType_ 7th-8th', 'EdType_ 9th', 'EdType_ Assoc-acdm', 'EdType_ Assoc-voc', 'EdType_ Bachelors', 'EdType_ Doctorate', 'EdType_ HS-grad', 'EdType_ Masters', 'EdType_ Preschool', 'EdType_ Prof-school', 'EdType_ Some-college', 
'maritalstatus_ Married-AF-spouse', 'maritalstatus_ Married-civ-spouse', 'maritalstatus_ Married-spouse-absent', 'maritalstatus_ Never-married', 'maritalstatus_ Separated', 'maritalstatus_ Widowed', 
'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 
'relationship_ Not-in-family', 'relationship_ Other-relative', 'relationship_ Own-child', 'relationship_ Unmarried', 'relationship_ Wife', 
'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White', 
'gender_ Male', 
'nativecountry_ Canada', 'nativecountry_ China', 'nativecountry_ Columbia', 'nativecountry_ Cuba', 'nativecountry_ Dominican-Republic', 'nativecountry_ Ecuador', 'nativecountry_ El-Salvador', 'nativecountry_ England', 'nativecountry_ France', 'nativecountry_ Germany', 'nativecountry_ Greece', 'nativecountry_ Guatemala', 'nativecountry_ Haiti', 'nativecountry_ Holand-Netherlands', 'nativecountry_ Honduras', 'nativecountry_ Hong', 'nativecountry_ Hungary', 'nativecountry_ India', 'nativecountry_ Iran', 'nativecountry_ Ireland', 'nativecountry_ Italy', 'nativecountry_ Jamaica', 'nativecountry_ Japan', 'nativecountry_ Laos', 'nativecountry_ Mexico', 'nativecountry_ Nicaragua', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'nativecountry_ Peru', 'nativecountry_ Philippines', 'nativecountry_ Poland', 'nativecountry_ Portugal', 'nativecountry_ Puerto-Rico', 'nativecountry_ Scotland', 'nativecountry_ South', 'nativecountry_ Taiwan', 'nativecountry_ Thailand', 'nativecountry_ Trinadad&Tobago', 'nativecountry_ United-States', 'nativecountry_ Vietnam', 'nativecountry_ Yugoslavia', 
'SalStat_1']
'''

# Separating input variables(independent variables) from output variable (dependent variable)
features = columns_list[0 : len(columns_list)-1] # SalStat_1 is separated because it is an output variable
print(features)
'''
['age', 'capitalgain', 'capitalloss', 'hoursperweek', 'JobType_ Local-gov', 'JobType_ Private', 'JobType_ Self-emp-inc', 'JobType_ Self-emp-not-inc', 'JobType_ State-gov', 'JobType_ Without-pay', 'EdType_ 11th', 'EdType_ 12th', 'EdType_ 1st-4th', 'EdType_ 5th-6th', 'EdType_ 7th-8th', 'EdType_ 9th', 'EdType_ Assoc-acdm', 'EdType_ Assoc-voc', 'EdType_ Bachelors', 'EdType_ Doctorate', 'EdType_ HS-grad', 'EdType_ Masters', 'EdType_ Preschool', 'EdType_ Prof-school', 'EdType_ Some-college', 'maritalstatus_ Married-AF-spouse', 'maritalstatus_ Married-civ-spouse', 'maritalstatus_ Married-spouse-absent', 'maritalstatus_ Never-married', 'maritalstatus_ Separated', 'maritalstatus_ Widowed', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'relationship_ Not-in-family', 'relationship_ Other-relative', 'relationship_ Own-child', 'relationship_ Unmarried', 'relationship_ Wife', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White', 'gender_ Male', 
'nativecountry_ Canada', 'nativecountry_ China', 'nativecountry_ Columbia', 'nativecountry_ Cuba', 'nativecountry_ Dominican-Republic', 'nativecountry_ Ecuador', 'nativecountry_ El-Salvador', 'nativecountry_ England', 'nativecountry_ France', 'nativecountry_ Germany', 'nativecountry_ Greece', 'nativecountry_ Guatemala', 'nativecountry_ Haiti', 'nativecountry_ Holand-Netherlands', 'nativecountry_ Honduras', 'nativecountry_ Hong', 'nativecountry_ Hungary', 'nativecountry_ India', 'nativecountry_ Iran', 'nativecountry_ Ireland', 'nativecountry_ Italy', 'nativecountry_ Jamaica', 'nativecountry_ Japan', 'nativecountry_ Laos', 'nativecountry_ Mexico', 'nativecountry_ Nicaragua', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'nativecountry_ Peru', 'nativecountry_ Philippines', 'nativecountry_ Poland', 'nativecountry_ Portugal', 'nativecountry_ Puerto-Rico', 'nativecountry_ Scotland', 'nativecountry_ South', 'nativecountry_ Taiwan', 'nativecountry_ Thailand', 'nativecountry_ Trinadad&Tobago', 'nativecountry_ United-States', 'nativecountry_ Vietnam', 'nativecountry_ Yugoslavia']
'''
# Another way to take out SalStat_1
features = list(set(columns_list) - set(['SalStat_1']))
print(features)
# as list is converted to set, retrieval order is not the same as insertion order.
'''
['hoursperweek', 'nativecountry_ France', 'maritalstatus_ Widowed', 'JobType_ Without-pay', 'nativecountry_ Portugal', 'occupation_ Handlers-cleaners', 'EdType_ Some-college', 'nativecountry_ Honduras', 'nativecountry_ Guatemala', 'JobType_ Private', 'EdType_ 7th-8th', 'capitalgain', 'occupation_ Sales', 'nativecountry_ Haiti', 'nativecountry_ Ireland', 'nativecountry_ Columbia', 'relationship_ Unmarried', 'maritalstatus_ Never-married', 'occupation_ Armed-Forces', 'race_ Black', 'occupation_ Craft-repair', 'EdType_ 1st-4th', 'nativecountry_ Greece', 'EdType_ Preschool', 'nativecountry_ Poland', 'EdType_ Assoc-voc', 'nativecountry_ Scotland', 'EdType_ 9th', 'nativecountry_ Iran', 'nativecountry_ Hungary', 'nativecountry_ Nicaragua', 'maritalstatus_ Married-spouse-absent', 'nativecountry_ Japan', 'occupation_ Exec-managerial', 'EdType_ Doctorate', 'gender_ Male', 'nativecountry_ United-States', 'nativecountry_ Trinadad&Tobago', 'nativecountry_ Ecuador', 'nativecountry_ Canada', 'JobType_ Self-emp-inc', 'relationship_ Other-relative', 'nativecountry_ Cuba', 'nativecountry_ Holand-Netherlands', 'nativecountry_ Germany', 'occupation_ Protective-serv', 'maritalstatus_ Separated', 'relationship_ Not-in-family', 'EdType_ Bachelors', 'occupation_ Other-service', 'nativecountry_ Peru', 'maritalstatus_ Married-AF-spouse', 'nativecountry_ Taiwan', 'nativecountry_ India', 'EdType_ 5th-6th', 'EdType_ 11th', 'race_ Other', 'relationship_ Own-child', 'nativecountry_ Hong', 'EdType_ Masters', 'occupation_ Priv-house-serv', 'EdType_ Prof-school', 'race_ Asian-Pac-Islander', 'occupation_ Farming-fishing', 'EdType_ HS-grad', 'nativecountry_ Dominican-Republic', 'relationship_ Wife', 'nativecountry_ Laos', 'nativecountry_ Philippines', 'capitalloss', 'occupation_ Machine-op-inspct', 'nativecountry_ Jamaica', 'occupation_ Tech-support', 'JobType_ State-gov', 'nativecountry_ Puerto-Rico', 'nativecountry_ Italy', 'EdType_ Assoc-acdm', 'race_ White', 'JobType_ Self-emp-not-inc', 'nativecountry_ Thailand', 'nativecountry_ England', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'nativecountry_ Mexico', 'age', 'nativecountry_ Yugoslavia', 'EdType_ 12th', 'nativecountry_ South', 'occupation_ Transport-moving', 'nativecountry_ El-Salvador', 'maritalstatus_ Married-civ-spouse', 'nativecountry_ China', 'JobType_ Local-gov', 'occupation_ Prof-specialty', 'nativecountry_ Vietnam']
'''

# Storing the output values in y
y = new_data['SalStat_1'].values
print(y)
# [0 0 1 ... 0 0 0]

# Storing the values from input features
x = new_data[features].values
print(x)
'''
[[0 0 0 ... 0 1 0]
 [1 0 0 ... 0 1 0]
 [0 0 0 ... 0 1 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 1 0]]
'''

# Splitting the data into train and test
# https://www.geeksforgeeks.org/how-to-split-the-dataset-with-scikit-learns-train_test_split-function/
# random_state = 0. random_state is a seed used by random number generator. If you set random seed, same set of records
# will be chosen every time, you run this code, otherwise different set of records will be chosen.
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=True)
print(len(train_x)) # 21113
'''
[[0 0 0 ... 0 0 0]
 [0 1 0 ... 0 0 0]
 [1 1 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 1 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
'''
print(len(test_x)) # 9049 - 30% data
'''
[[0 1 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 1 ... 0 0 0]
 [0 1 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]]
'''

print(len(train_y)) # 21113
print(len(test_y)) # 9049  - 30% data

print(train_y) # [0 0 0 ... 1 0 0]
print(test_y) # [0 0 0 ... 0 0 0]

# Make an instance of the Model.
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x, train_y)
print(logistic.coef_())

