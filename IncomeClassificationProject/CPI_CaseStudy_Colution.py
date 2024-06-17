
# To work with dataframes
import pandas as pd

# To perform numerical operations
import numpy as np

# To visualize data
import seaborn as sns

# To partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

import gotoDataDir

data_income = pd.read_csv('income.csv')
data = data_income.copy()
# print(data)
print(data.to_string())

"""
Exploratory data analysis:
1. getting to know the data
2. data preprocessing (missing values)
3. cross tables and data visualization
"""
# know the data type of each variable
print(data.info())

"""
Data columns (total 13 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   age            31978 non-null  int64 
 1   JobType        31978 non-null  object
 2   EdType         31978 non-null  object
 3   maritalstatus  31978 non-null  object
 4   occupation     31978 non-null  object
 5   relationship   31978 non-null  object
 6   race           31978 non-null  object
 7   gender         31978 non-null  object
 8   capitalgain    31978 non-null  int64 
 9   capitalloss    31978 non-null  int64 
 10  hoursperweek   31978 non-null  int64 
 11  nativecountry  31978 non-null  object
 12  SalStat        31978 non-null  object
dtypes: int64(4), object(9)
"""

# check for missing values
total_missing_values = data.isnull().sum()
print('Data columns with null values:\n', total_missing_values)  # No missing values
"""
age              0
JobType          0
EdType           0
maritalstatus    0
occupation       0
relationship     0
race             0
gender           0
capitalgain      0
capitalloss      0
hoursperweek     0
nativecountry    0
SalStat          0
dtype: int64
"""

# Summary of numerical variables
summary_mum = data.describe()
print(summary_mum)
"""
                age   capitalgain   capitalloss  hoursperweek
count  31978.000000  31978.000000  31978.000000  31978.000000
mean      38.579023   1064.360623     86.739352     40.417850 --- average
std       13.662085   7298.596271    401.594301     12.345285
min       17.000000      0.000000      0.000000      1.000000
25%       28.000000      0.000000      0.000000     40.000000
50%       37.000000      0.000000      0.000000     40.000000 ---- median 
75%       48.000000      0.000000      0.000000     45.000000
max       90.000000  99999.000000   4356.000000     99.000000

Capital Gain is the profit after the sell of a property.
Here, min, 25%, 50% and 75% Capital Gain is 0 and average and max have some values. 
So, most of the Capital Gains belong to last 25%. Same for Capital Loss also.

Here, standard deviation (std) is very high.

https://en.wikipedia.org/wiki/Standard_deviation
In statistics, the standard deviation is a measure of the amount of variation of a random variable expected about its mean.
A low standard deviation indicates that the values tend to be close to the mean (also called the expected value) of the set, while a high standard deviation indicates that the values are spread out over a wider range. 
The standard deviation is commonly used in the determination of what constitutes an outlier and what does not.
"""

summary_categories = data.describe(include='O')  # consider only variables of type object
print(summary_categories.to_string())
"""
         JobType    EdType        maritalstatus       occupation relationship    race gender   nativecountry                        SalStat
count      31978     31978                31978            31978        31978   31978  31978           31978                          31978
unique         9        16                    7               15            6       5      2              41                              2
top      Private   HS-grad   Married-civ-spouse   Prof-specialty      Husband   White   Male   United-States   less than or equal to 50,000
freq       22286     10368                14692             4038        12947   27430  21370           29170                          24283

JobType has 9 unique values
Modal value (most frequently occurring) category is 'Private' and it occurs 22286 times.
"""

# Frequency of each JobType value
jobtype_value_counts = data['JobType'].value_counts()
print(jobtype_value_counts)
"""
JobType
Private             22286
Self-emp-not-inc     2499
Local-gov            2067
?                    1809
State-gov            1279
Self-emp-inc         1074
Federal-gov           943
Without-pay            14
Never-worked            7
Name: count, dtype: int64
"""

# Frequency of each occupation value
occupation_value_counts = data['occupation'].value_counts()
print(occupation_value_counts)
"""
occupation
Prof-specialty       4038
Craft-repair         4030
Exec-managerial      3992
Adm-clerical         3721
Sales                3584
Other-service        3212
Machine-op-inspct    1966
?                    1816
Transport-moving     1572
Handlers-cleaners    1350
Farming-fishing       989
Tech-support          912
Protective-serv       644
Priv-house-serv       143
Armed-Forces            9
Name: count, dtype: int64
"""

# It was observed that only JobType and occupation have '?' instead of NaN.

# Listing unique values of JobType
jobtype_unique_elements = np.unique(data['JobType'])
print(jobtype_unique_elements)
"""
[' ?' ' Federal-gov' ' Local-gov' ' Never-worked' ' Private'
 ' Self-emp-inc' ' Self-emp-not-inc' ' State-gov' ' Without-pay']
 
If you see there is a space before the ? in JobType.
"""

# Listing unique values of JobType
occupation_unique_elements = np.unique(data['occupation'])
print(occupation_unique_elements)
"""
[' ?' ' Adm-clerical' ' Armed-Forces' ' Craft-repair' ' Exec-managerial'
 ' Farming-fishing' ' Handlers-cleaners' ' Machine-op-inspct'
 ' Other-service' ' Priv-house-serv' ' Prof-specialty' ' Protective-serv'
 ' Sales' ' Tech-support' ' Transport-moving']
 
If you see there is a space before the ? in occupation.
"""