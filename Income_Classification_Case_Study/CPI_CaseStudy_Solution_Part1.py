
# To work with dataframes
import pandas as pd

# To perform numerical operations
import numpy as np

# To visualize data
import seaborn as sns
import matplotlib.pyplot as plt  # pyplot means python plot

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
Study 'Variance and Standard Deviation.mp4'
      'Mean Median and Mode.mp4'
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

# IMP: If you want to find out a Modal value(mode) for categorical variables, you can use describe for categorical variables
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

'''
IMPORTANT:
For categorical variable, following things are useful
- value_count
- crosstab - can be used for a single categorical variable or for two categorical variables 
- countplot - can be used for a single categorical variable or for two categorical variables
- boxplot - can be used for a single categorical variable or for comparing a categorical variable with a numerical variable

For numerical variable, following things are useful
- value_count - sometimes, it is used e.g. Cars_Regression_Case_Study
- histplot/displot
- boxplot
- scatterplot/regplot (same as scatterplot with or without regression line) - to compare two numerical variables 
'''


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

# go back and read the data by including "na_values[' ?']
data = pd.read_csv('income.csv', na_values=[' ?'])
#############################################################
# Data Pre-Processing
#############################################################
# data.isnull() returns a dataframe with True and False values in each cell
# True means null value and False means not-null value
total_missing_values = data.isnull().sum()
print('Data columns with null values:\n', total_missing_values)
"""
Data columns with null values:
 age                 0
JobType          1809
EdType              0
maritalstatus       0
occupation       1816
relationship        0
race                0
gender              0
capitalgain         0
capitalloss         0
hoursperweek        0
nativecountry       0
SalStat             0
dtype: int64

JobType and occupation have missing values
"""


# print(data.isnull().any(axis=1))
'''
Returns a series like this
0        False
1        False
2        False
3        False
......
'''

# data.isnull() returns a dataframe with True and False values in each cell
# True means null value and False means not-null value
# any() returns a series of index numbers
rows_with_missing_values = data[data.isnull().any(axis=1)] # axis=1 is to consider at least one COLUMN(not row) with NaN value
#print("Rows with missing values:\n", rows_with_missing_values.to_string())
print("Rows with missing values:\n", rows_with_missing_values)
"""
Rows with missing values:
        age        JobType         EdType           maritalstatus occupation     relationship                 race   gender  capitalgain  capitalloss  hoursperweek        nativecountry                        SalStat
8       17            NaN           11th           Never-married        NaN        Own-child                White   Female            0            0             5        United-States   less than or equal to 50,000
17      32            NaN   Some-college      Married-civ-spouse        NaN          Husband                White     Male            0            0            40        United-States   less than or equal to 50,000
29      22            NaN   Some-college           Never-married        NaN        Own-child                White     Male            0            0            40        United-States   less than or equal to 50,000
42      52            NaN           12th           Never-married        NaN   Other-relative                Black     Male          594            0            40        United-States   less than or equal to 50,000
44      63            NaN        1st-4th      Married-civ-spouse        NaN          Husband                White     Male            0            0            35        United-States   less than or equal to 50,000
57      72            NaN        HS-grad      Married-civ-spouse        NaN          Husband                White     Male            0            0            20        United-States   less than or equal to 50,000
69      53            NaN        5th-6th                 Widowed        NaN        Unmarried                Black   Female            0            0            30        United-States   less than or equal to 50,000
73      57            NaN      Assoc-voc                 Widowed        NaN        Unmarried                White   Female            0            0            38        United-States   less than or equal to 50,000
75      20            NaN   Some-college           Never-married        NaN        Own-child                White     Male            0            0            24        United-States   less than or equal to 50,000
76      21            NaN   Some-college           Never-married        NaN        Unmarried                White   Female            0            0            35        United-States   less than or equal to 50,000
97      34            NaN        HS-grad           Never-married        NaN        Unmarried                Black   Female            0            0            40        United-States   less than or equal to 50,000
133     18            NaN           12th           Never-married        NaN        Own-child                White     Male            0            0            25        United-States   less than or equal to 50,000
137     65            NaN   Some-college      Married-civ-spouse        NaN          Husband                White     Male            0            0            30        United-States            greater than 50,000
147     42            NaN        HS-grad      Married-civ-spouse        NaN          Husband                White     Male            0            0            40        United-States   less than or equal to 50,000
148     55            NaN        HS-grad      Married-civ-spouse        NaN             Wife   Asian-Pac-Islander   Female            0            0            40        United-States   less than or equal to 50,000
153     23            NaN   Some-college           Never-married        NaN        Unmarried   Amer-Indian-Eskimo   Female            0            0            25        United-States   less than or equal to 50,000
205     58            NaN        HS-grad      Married-civ-spouse        NaN          Husband                White     Male            0            0            50        United-States   less than or equal to 50,000
213     70            NaN            9th                 Widowed        NaN        Unmarried                White   Female         1111            0            15        United-States   less than or equal to 50,000
225     20            NaN           11th   Married-spouse-absent        NaN        Own-child   Asian-Pac-Islander   Female            0         1762            40                South   less than or equal to 50,000
228     17            NaN           11th           Never-married        NaN        Own-child                White   Female            0            0            20        United-States   less than or equal to 50,000
243     66            NaN        7th-8th           Never-married        NaN    Not-in-family                White     Male            0            0             4        United-States   less than or equal to 50,000
252     25            NaN     Assoc-acdm           Never-married        NaN   Other-relative                White     Male            0            0            45        United-States   less than or equal to 50,000
288     45            NaN   Some-college      Married-civ-spouse        NaN             Wife                Black   Female            0            0            40        United-States   less than or equal to 50,000
340     59            NaN        HS-grad      Married-civ-spouse        NaN          Husband                White     Male            0            0            16        United-States   less than or equal to 50,000
341     21            NaN        HS-grad           Never-married        NaN    Not-in-family                White   Female         1055            0            40        United-States   less than or equal to 50,000
344     18            NaN   Some-college           Never-married        NaN        Own-child                White     Male            0            0            33        United-States   less than or equal to 50,000
358     50            NaN        Masters   Married-spouse-absent        NaN   Other-relative                White     Male            0            0            40        United-States   less than or equal to 50,000
363     22            NaN   Some-college           Never-married        NaN        Own-child                Black     Male            0            0            40        United-States   less than or equal to 50,000
382     42            NaN           11th      Married-civ-spouse        NaN          Husband                White     Male            0            0            15        United-States   less than or equal to 50,000
386     72            NaN           11th                 Widowed        NaN    Not-in-family                White   Female            0            0            24        United-States   less than or equal to 50,000
437     63            NaN   Some-college      Married-civ-spouse        NaN          Husband                White     Male            0            0            15        United-States   less than or equal to 50,000
441     48            NaN   Some-college                Divorced        NaN        Unmarried                Black   Female            0            0            30        United-States   less than or equal to 50,000
454     27            NaN     Assoc-acdm      Married-civ-spouse        NaN        Own-child   Amer-Indian-Eskimo     Male            0            0            40        United-States   less than or equal to 50,000
475     64            NaN        HS-grad      Married-civ-spouse        NaN          Husband                White     Male            0            0             5        United-States   less than or equal to 50,000
482     19            NaN   Some-college           Never-married        NaN        Own-child                White   Female            0            0            10        United-States   less than or equal to 50,000
519     36            NaN      Assoc-voc                 Widowed        NaN    Not-in-family                White   Female            0            0            20        United-States   less than or equal to 50,000
532     23            NaN        HS-grad           Never-married        NaN        Unmarried                Black   Female            0            0            40        United-States   less than or equal to 50,000
569     19            NaN        HS-grad           Never-married        NaN        Own-child                White     Male            0            0            30        United-States   less than or equal to 50,000
583     76            NaN        7th-8th                 Widowed        NaN    Not-in-family                White     Male            0            0             2        United-States   less than or equal to 50,000
584     19            NaN        HS-grad           Never-married        NaN        Unmarried                White     Male            0         2001            40        United-States   less than or equal to 50,000
587     34            NaN           11th      Married-civ-spouse        NaN             Wife                White   Female            0            0             8        United-States   less than or equal to 50,000
594     64            NaN     Assoc-acdm           Never-married        NaN    Not-in-family                White   Female            0            0            20        United-States   less than or equal to 50,000
645     19            NaN        HS-grad           Never-married        NaN        Own-child                White   Female            0            0            40        United-States   less than or equal to 50,000
648     68            NaN        7th-8th      Married-civ-spouse        NaN          Husband                White     Male            0            0             8        United-States   less than or equal to 50,000
678     34            NaN        HS-grad                Divorced        NaN    Not-in-family                White   Female            0            0            40        United-States   less than or equal to 50,000
681     42            NaN   Some-college                Divorced        NaN        Unmarried                White     Male            0            0            40        United-States   less than or equal to 50,000
712     22            NaN   Some-college           Never-married        NaN        Own-child                White   Female            0            0             8        United-States   less than or equal to 50,000
720     58            NaN   Some-college      Married-civ-spouse        NaN          Husband                White     Male            0            0            40        United-States            greater than 50,000
725     50            NaN        HS-grad      Married-civ-spouse        NaN          Husband                White     Male            0            0            40        United-States            greater than 50,000
730     51            NaN        Masters      Married-civ-spouse        NaN          Husband                White     Male            0            0            40        United-States   less than or equal to 50,000
"""

# Finding total number of rows missing values in both JobType and occupation
jobtype_and_occupation_data = data[['JobType','occupation']]
print('JobType and Occupation columns: \n', jobtype_and_occupation_data)

# works
rows_with_missing_jobtype_and_occupation = jobtype_and_occupation_data[jobtype_and_occupation_data.isnull().all(axis=1)]

print('JobType and Occupation columns with NaN value: \n', rows_with_missing_jobtype_and_occupation)

''' doesn't work
# data.isnull() returns a dataframe with True and False values in each cell
# data[col].isnull() returns a series
# True means null value and False means not-null value
# where function should have a condition that returns True or False. Here it returns a Series of index numbers.
# That is why it doesn't work

rows_with_missing_jobtype_and_occupation = jobtype_and_occupation_data\
    .where(jobtype_and_occupation_data['JobType'].isnull() &
           jobtype_and_occupation_data['occupation'].isnull())
'''
'''
# works
exp1 & exp2                # Element-wise logical AND
exp1 | exp2                # Element-wise logical OR
~exp1                      # Element-wise logical NOT
'''
rows_with_missing_jobtype_and_occupation = jobtype_and_occupation_data[
                                                jobtype_and_occupation_data['JobType'].isnull() &
                                                jobtype_and_occupation_data['occupation'].isnull()
                                            ]
#print(rows_with_missing_jobtype_and_occupation)
print("total rows having both JobType and occupation missing values:\n ", rows_with_missing_jobtype_and_occupation.shape[0]) # total 1809 rows have both JobType and occupation with missing values

rows_with_non_missing_jobtype_and_missing_occupation = jobtype_and_occupation_data[
                                                ~jobtype_and_occupation_data['JobType'].isnull() &
                                                jobtype_and_occupation_data['occupation'].isnull()
                                            ]
print(rows_with_non_missing_jobtype_and_missing_occupation)
'''
rows with not-null JobType and null occupation

             JobType occupation
4825    Never-worked        NaN
10215   Never-worked        NaN
14073   Never-worked        NaN
19542   Never-worked        NaN
22385   Never-worked        NaN
31296   Never-worked        NaN
31305   Never-worked        NaN
'''
print("total rows having non-missing JobType and missing occupation:\n ", rows_with_non_missing_jobtype_and_missing_occupation.shape[0]) # 7

'''
# drop all those rows having missing values in all the columns
rows_with_missing_occupation_and_with_jobtype = jobtype_and_occupation_data.dropna(how="all")\
    .sort_values(by=['JobType', 'occupation'], ascending=[True, True])
'''
'''
# exporting to excel
print("rows having JobType, but missing occupation:\n ",
      rows_with_missing_occupation_and_with_jobtype.to_string())
rows_with_missing_occupation_and_with_jobtype.to_excel('rows_with_missing_occupation_and_with_jobtype.xlsx')
'''

""" Points to note:
1. Missing values in Jobtype    = 1809
2. Missing values in Occupation = 1816
3. There are 1809 rows where two specific columns (i.e. occupation & JobType) have missing values
4. (1816-1809) = 7 => You still have occupation unfilled for these 7 rows because JobType is 'Never-worked'
"""
# We will drop all the rows where there are missing values because there are only 1816 rows like that out of 32k rows
# axis=0 means drop all those ROWs that have missing values
# axis=1 means drop all those COLUMNs that have missing values
data2 = data.dropna(axis=0)
total_rows_left = data2.shape[0]
print("After dropping the rows with missing values, total rows left: ", total_rows_left) # 30162

""" Just testing
something = ~jobtype_and_occupation_data['JobType'].isnull() & jobtype_and_occupation_data['occupation'].isnull()
print(type(something)) # <class 'pandas.core.series.Series'>
print(something.__repr__())
something_else = data[(0, False) (1, False) (2, False) (3, False) (4, False) (5, False) (6, False) (7, False) (8, False) (9, False) (10, False) (11, False) (12, False) (13, False) (14, False) (15, False) (16, False) (17, False) (18, False) (19, False) (20, False) (21, False) (22, False) (23, False) (24, False) (25, False) (26, False) (27, False) (28, False) (29, False) (30, False) (31, False) (32, False) (33, False) (34, False) (35, False) (36, False) (37, False) (38, False) (39, False) (40, False) (41, False) (42, False) (43, False) (44, False) (45, False) (46, False) (47, False) (48, False) (49, False) (50, False) (51, False) (52, False) (53, False) (54, False) (55, False) (56, False) (57, False) (58, False) (59, False) (60, False) (61, False) (62, False) (63, False) (64, False) (65, False) (66, False) (67, False) (68, False) (69, False) (70, False) (71, False) (72, False) (73, False) (74, False) (75, False) (76, False) (77, False) (78, False) (79, False) (80, False) (81, False) (82, False) (83, False)]
print(type(something_else))

something = jobtype_and_occupation_data[pd.Series({0: False, 1:True})]
print(something)
"""

# Relationship between independent variables
# Watch 'Pearson Correlation Coefficient' video
# https://www.youtube.com/watch?v=B8LcYdblXBI
correlation = data2.corr(method="pearson", numeric_only=True)
print('Correlation: \n', correlation)
'''
                    age  capitalgain  capitalloss  hoursperweek
age           1.000000     0.080154     0.060165      0.101599
capitalgain   0.080154     1.000000    -0.032229      0.080432
capitalloss   0.060165    -0.032229     1.000000      0.052417
hoursperweek  0.101599     0.080432     0.052417      1.000000

None of the numeric field is co-related to another because none of them have value close to 1 or -1.
Close to 0 means no relation.
 
Correlation value varies from -1 to +1.

Correlation can be found between numeric variables only.
For two categorical variables, you have to use crosstab
'''

# ================================================
# Cross tables & Data Visualization
# ================================================
# Extracting the column names
col_names = data2.columns
print('Columns: \n', col_names)

'''
Columns: 
 Index(['age', 'JobType', 'EdType', 'maritalstatus', 'occupation',
       'relationship', 'race', 'gender', 'capitalgain', 'capitalloss',
       'hoursperweek', 'nativecountry', 'SalStat'],
      dtype='object')
'''

# ================================================
# Gender proportion table:
# ================================================
gender = pd.crosstab(index = data2['gender'],
                     columns = 'count',
                     normalize = True)# show the result in percentage
print('Gender Proportion Table:\n ', gender)
'''
col_0      count
gender          
Female  0.324315
Male    0.675685

This shows that 67% data has Male and 33% data has Female.
'''

# ====================================================
# Gender vs Salary Status (Two Categorical variables):

# IMP: Unlike to Cars_Registration project, here output variable is categorical.
# So, if you want to find the relation between input categorical variable and output categorical variable,
# you have to use crosstab
# ====================================================
gender_salstat = pd.crosstab(index = data2['gender'], # index means row
                     columns = data2['SalStat'], # output variable
                     margins = True,
                     normalize = 'index') # index means row. normalize = 'index' means you will get row proportion sum = 1
print('Gender vs SalStat: \n', gender_salstat)

'''
 SalStat  greater than 50,000  less than or equal to 50,000
gender                                                    
 Female             0.113678                      0.886322
 Male               0.313837                      0.686163
All                 0.248922                      0.751078

Only 11% females earn more than 50k
89% females have salary lesser or equal to 50k

More Males earn compared to than women (31% males have salary more than 50k)
'''

# ================================================
# Frequency distribution of salary status (Categorical variable):

# So, for Salary Status(Categorical variable),
# you can use crosstab function --- frequencyTables.py
# or value_counts() function --- BarPlot.py
# or sns.countplot(...) chart --- BarPlotOrCountPlot.py
# or plt.bar(...) chart --- BarPlot.py

# you have to use histogram (plt.hist()/sns.displot()/sns.histplot()) to find out the frequencies of numeric variable. hist means histogram. displot means distribution plot. Both mean the same.
# you have to use value_counts()/crosstab(...)/Bar/CountPlot to find out the frequencies of categorical variable.
# ================================================
values_frequencies = data2['SalStat'].value_counts()
print("Value Frequencies using value_counts(): \n", values_frequencies)

values_frequencies_using_crosstab = pd.crosstab(index=data2['SalStat'],
                                                columns="counts") # index means row
print("Value Frequencies using crosstab: \n", values_frequencies_using_crosstab )

sns.countplot(data=data2, x='SalStat') # same as plt.bar(...) chart
plt.show()

'''
Value Frequencies: ---- Using value_counts() function

SalStat
less than or equal to 50,000    22654
greater than 50,000              7508

Name: count, dtype: int64

Value Frequencies: ---- Using crosstab() function

col_0                         counts
SalStat                             
greater than 50,000             7508
less than or equal to 50,000   22654
'''

"""
It is very clear that 75% people's salary status is <=50k
and only 25% people's salary status is >50k
"""

# ================================================================================================
# Frequency distribution of age (Numeric variable):
# you have to use histogram (plt.hist()/sns.displot()) to find out the frequencies of numeric variable.  hist means histogram. displot means distribution plot. Both mean the same.
# you have to use value_counts()/crosstab(...)/Bar/CountPlot to find out the frequencies of categorical variable.
# ================================================================================================
sns.displot(data2['age'], bins=10, kde=False)
plt.show()
# People with age 20-45 are high in frequency

# ================================================================================================
# Box Plot - Age vs Salary Status
# Box plot can also be used to compare 5-number summary of any variable.
# It can be used for a single variable or for two variables to get 5 number summary
# of one variable comparing it with another.
#
# dataframe.describe() or sns.boxplot() can be used to get 5 number summary
# sns.boxplot() can be used to get 5 number summary for one variable or comparing one variable with another.
# To get exact value by comparing two variables, you can use dataframe.groupby(x axis variable)[y-axis variable].median()
# ================================================================================================
sns.boxplot(data=data2, x='SalStat', y='age')
plt.show()
# People with age 35-50 are more likely to make >50k
# People with 25-35 age are more likely to make <=50k
exact_median_of_age_by_salarystatus = data2.groupby('SalStat')['age'].median()
print('Exact median of age by salary status: ', exact_median_of_age_by_salarystatus)
'''
Exact median of age by salary status:  
SalStat
greater than 50,000             43.0
less than or equal to 50,000    34.0
'''
five_numbers_summary_of_age = data2['age'].describe()
print('Five numbers summary of age:\n', five_numbers_summary_of_age)
"""
The mean (average) of a data set is found by adding all numbers in the data set and then dividing by the number of values in the set.
The median is the middle value when a data set is ordered from least(min) to greatest(max). 
The mode is the number that occurs most often in a data set.
"""


############################ Similar Analysis between JobType and SalStat #######################

sns.countplot(data=data2, x='JobType') # same as plt.bar(...) chart
plt.show()

jobtype_value_counts = data['JobType'].value_counts()
print(jobtype_value_counts)
"""
JobType
Private             22286
Self-emp-not-inc     2499
Local-gov            2067
State-gov            1279
Self-emp-inc         1074
Federal-gov           943
Without-pay            14
Never-worked            7
Name: count, dtype: int64
"""

# Both value_count() and countplot show that highest jobs are private (>22k).
# you can use crosstab with columns='counts' also. crosstab helps you set margins, normalize etc
# you can use countplot to see ths same thing graphically

# crosstab and countplot/barplot can be used for one categorical variable also to see its frequency.
# They can be used to see the relation between two categorical variable also.


'''
# IMP: Unlike to Cars_Registration project, here output variable is categorical.
# So, if you want to find the relation between input categorical variable and output categorical variable,
# you have to use crosstab
'''
jobtype_salstat = pd.crosstab(index = data2['JobType'], # index means row
                     columns = data2['SalStat'],
                     margins = True,
                     normalize = 'index').round(4)*100 # index means row. normalize = 'index' means you will get row proportion sum = 1
print('Jobtype vs SalStat: \n', jobtype_salstat)
'''
Jobtype vs SalStat: 

 SalStat            greater than 50,000  less than or equal to 50,000
JobType                                                             
 Federal-gov                     38.71                         61.29
 Local-gov                       29.46                         70.54
 Private                         21.88                         78.12
 Self-emp-inc                    55.87                         44.13
 Self-emp-not-inc                28.57                         71.43
 State-gov                       26.90                         73.10
 Without-pay                      0.00                        100.00
All                              24.89                         75.11


This shows that majority of the job types (except self-emp-inc) have 2/3 rd people earning <=50k.
Hence, it is an important variable in avoiding the misuse of subsidies.
'''
# Unlike to Cars_Registration project, here output variable is categorical
# so, you can use output variable as 'hue' in plots.
sns.countplot(data=data2, x='JobType', hue='SalStat') # y axis is always frequency
plt.show()

# sns.barplot(...) is different than plt.bar(...)
# sns's barplot(...) needs categorical variable on x-axis, numerical variable on y-axis, and categorical variable for hue(not mandatory)
sns.barplot(x='JobType', y='age', hue='SalStat', data=data2)
plt.show()

'''
Trying plt.bar(...) plot. It is not as good as sns.countplot(...)

# converting series to lists to use them in bar plot
print(list(data2['JobType'].unique()))
list_of_jobtype_counts = data2['JobType'].unique().tolist() # [' Private', ' Federal-gov', ' Self-emp-inc', ' Self-emp-not-inc', ' Local-gov', ' State-gov', ' Without-pay']
list_of_jobtypes = jobtype_value_counts.tolist() # [22286, 2499, 2067, 1279, 1074, 943, 14, 7]

# bar plot requires lists for x and height parameters and it doesn't have hue paramenter. 
# countplot is better than bar plot
#plt.bar(x=list_of_jobtypes, height=list_of_jobtype_counts)
plt.bar(x=['Private', 'Federal-gov', 'Self-emp-inc', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Without-pay', 'Never-Worked'],
        height=[22286, 2499, 2067, 1279, 1074, 943, 14, 7])


plt.xlabel("Job Types")
plt.ylabel("Salary Status")
plt.show()
'''

############################ Similar Analysis between Education and SalStat #######################
education_salstat = pd.crosstab(index = data2['EdType'], # index means row
                     columns = data2['SalStat'],
                     margins = True,
                     normalize = 'index').round(4)*100 # index means row. normalize = 'index' means you will get row proportion sum = 1
print('Education vs SalStat: \n', education_salstat)
"""
Education vs SalStat: 

 SalStat        greater than 50,000  less than or equal to 50,000
EdType                                                          
 10th                         7.20                         92.80
 11th                         5.63                         94.37
 12th                         7.69                         92.31
 1st-4th                      3.97                         96.03
 5th-6th                      4.17                         95.83
 7th-8th                      6.28                         93.72
 9th                          5.49                         94.51
 Assoc-acdm                  25.40                         74.60
 Assoc-voc                   26.32                         73.68
 Bachelors                   42.15                         57.85
 Doctorate                   74.67                         25.33
 HS-grad                     16.43                         83.57
 Masters                     56.42                         43.58
 Preschool                    0.00                        100.00
 Prof-school                 74.91                         25.09
 Some-college                20.01                         79.99
All                          24.89                         75.11

From the above table, we can see that people who have done Doctorate, Masters, Prof-school are 
more likely to earn above 50k USD when compared to with others. 
Hence and influencing variable in avoiding the misuse of subsidies.
"""
# Unlike to Cars_Registration project, here output variable is categorical
# so, you can use output variable as 'hue' in plots.
sns.countplot(data=data2, x='EdType', hue='SalStat') # y axis is always frequency
plt.show()

############################ Similar Analysis between Occupation and SalStat #######################
occupation_salstat = pd.crosstab(index = data2['occupation'], # index means row
                     columns = data2['SalStat'],
                     margins = True,
                     normalize = 'index').round(4)*100 # index means row. normalize = 'index' means you will get row proportion sum = 1
print('Occupation vs SalStat: \n', occupation_salstat)
'''
Occupation vs SalStat: 
 SalStat             greater than 50,000  less than or equal to 50,000
occupation                                                           
 Adm-clerical                     13.38                         86.62
 Armed-Forces                     11.11                         88.89
 Craft-repair                     22.53                         77.47
 Exec-managerial                  48.52                         51.48 -------
 Farming-fishing                  11.63                         88.37
 Handlers-cleaners                 6.15                         93.85
 Machine-op-inspct                12.46                         87.54
 Other-service                     4.11                         95.89
 Priv-house-serv                   0.70                         99.30
 Prof-specialty                   44.85                         55.15 -------
 Protective-serv                  32.61                         67.39
 Sales                            27.06                         72.94
 Tech-support                     30.48                         69.52
 Transport-moving                 20.29                         79.71
All                               24.89                         75.11

Those who make more than 50k USD per year are more likely to work as managers and professionals.
Hence it is an important variable in avoiding the misuse of subsidies.
'''
sns.countplot(data=data2, x='occupation', hue='SalStat') # y axis is always frequency
plt.show()

############################ Capital Gain Analysis - Frequency Distribution analysis ######################################
# countplot should be used for categorical variables. hist/displot should be used for numerical variables.
sns.displot(data2['capitalgain'], bins=10, kde=False)
plt.show()
# 92% records have 0 capitalgain. Only 8% people have gained some profit from their investments/no investments.
# Though capital gain varies from 0 to 100k, but there are more observations ranging from 0 to 20k.

############################ Capital Loss Analysis - Frequency Distribution analysis ######################################
# countplot should be used for categorical variables. hist/displot should be used for numerical variables.
sns.displot(data2['capitalloss'], bins=10, kde=False)
plt.show()
# 95% (28721) observations have 0 capitalloss. Only 5% people have lost some money from their investments/no investments.

########################### Hours per week vs Salary Status ############################
# When you want to compare two variables - use box plot
'''
dataframe.describe() or sns.boxplot() can be used to get 5 number summary.
It can be used for two variables x=categorical variable, y=numerical variable OR just y=numerical variable
sns.boxplot() can be used to get 5 number summary for one variable or comparing one variable with another.
To get exact value by comparing two variables, you can use dataframe.groupby(x axis variable)[y-axis variable].median()
'''
sns.boxplot(data=data2, x="SalStat", y="hoursperweek")
plt.show()
# From the plot, it is clearly visible that those who make more than 50k USD per year are more likely to
# spend 40-50 hours per week.
# This variable can contribute in classifying the individual's salary status since there is association between
# salary status and hours per week.

'''
# Correlation doesn't work for Object variables. It works only for numeric variables.
correlation_between_hoursperweek_and_salstat = data2.corr(method="pearson", numeric_only=True)
print("Correlation between Hours Per Week and Salary Status:\n", correlation_between_hoursperweek_and_salstat)
'''
crosstab_between_hoursperweek_and_salstat = pd.crosstab(index = data2['SalStat'], # index means row
                     columns = data2['hoursperweek'],
                     margins = True,
                     normalize = 'index').round(4)*100 # index means row. normalize = 'index' means you will get row proportion sum = 1
print("Cross Tabulation between Hours Per Week and Salary Status:\n", crosstab_between_hoursperweek_and_salstat.to_string())
'''
 hoursperweek                          1         2         3         4         5         6         7         8         9        10        11        12        13        14        15        16        17        18        19        20        21        22        23        24        25        26        27        28        29        30        31        32        33        34        35        36        37        38        39        40        41        42        43        44        45        46        47        48        49        50        51        52        53        54        55        56        57        58        59        60        61        62        63        64        65        66        67        68        70        72        73        74        75        76        77        78        80        81        82        84        85        86        87        88        89        90        91        92        94        95        96        97        98        99
SalStat                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
 greater than 50,000           0.000133  0.000266  0.000133  0.000266  0.000799  0.000533  0.000266  0.000799  0.000133  0.001731  0.000000  0.001199  0.000266  0.000266  0.001865  0.001465  0.000133  0.000932  0.000000  0.009057  0.000266  0.000400  0.000000  0.002797  0.004129  0.000533  0.000133  0.000666  0.000133  0.009190  0.000000  0.003729  0.000533  0.000266  0.024241  0.005594  0.002531  0.011188  0.000932  0.414225  0.000799  0.009057  0.005061  0.009856  0.089238  0.003596  0.001998  0.019845  0.000799  0.164758  0.000400  0.006393  0.000799  0.001465  0.041556  0.005061  0.000932  0.001199  0.000133  0.083111  0.000266  0.000533  0.000266  0.000666  0.013852  0.000266  0.000400  0.000666  0.013719  0.003063  0.000133  0.000000  0.002664  0.000266  0.000000  0.000266  0.006926  0.000000  0.000000  0.002131  0.000533  0.000000  0.000000  0.000000  0.000133  0.001332  0.000000  0.000000  0.000000  0.000133  0.000133  0.000133  0.000400  0.003330
 less than or equal to 50,000  0.000265  0.000574  0.001015  0.001104  0.001413  0.001589  0.000750  0.004238  0.000706  0.009226  0.000397  0.005650  0.000706  0.001192  0.014655  0.007460  0.001148  0.002472  0.000618  0.043524  0.000927  0.001589  0.000883  0.008652  0.023969  0.001059  0.001192  0.002958  0.000177  0.040611  0.000221  0.009270  0.001457  0.001104  0.044231  0.007063  0.005562  0.016421  0.001324  0.491790  0.001236  0.006401  0.004767  0.005959  0.047806  0.002340  0.001501  0.015229  0.000883  0.065375  0.000441  0.003840  0.000706  0.001192  0.015891  0.002295  0.000441  0.000795  0.000132  0.034475  0.000000  0.000618  0.000309  0.000397  0.006047  0.000618  0.000044  0.000309  0.007637  0.001722  0.000044  0.000044  0.001898  0.000044  0.000265  0.000265  0.003002  0.000132  0.000044  0.001059  0.000397  0.000088  0.000044  0.000088  0.000044  0.000795  0.000132  0.000044  0.000044  0.000044  0.000177  0.000044  0.000353  0.002340
All                            0.000232  0.000497  0.000796  0.000895  0.001260  0.001326  0.000630  0.003382  0.000564  0.007360  0.000298  0.004542  0.000597  0.000961  0.011471  0.005968  0.000895  0.002089  0.000464  0.034945  0.000763  0.001293  0.000663  0.007194  0.019031  0.000928  0.000928  0.002387  0.000166  0.032790  0.000166  0.007891  0.001227  0.000895  0.039255  0.006697  0.004807  0.015118  0.001227  0.472482  0.001127  0.007062  0.004841  0.006929  0.058119  0.002652  0.001625  0.016378  0.000862  0.090113  0.000431  0.004476  0.000729  0.001260  0.022280  0.002984  0.000564  0.000895  0.000133  0.046582  0.000066  0.000597  0.000298  0.000464  0.007990  0.000530  0.000133  0.000398  0.009151  0.002056  0.000066  0.000033  0.002089  0.000099  0.000199  0.000265  0.003979  0.000099  0.000033  0.001326  0.000431  0.000066  0.000033  0.000066  0.000066  0.000928  0.000099  0.000033  0.000033  0.000066  0.000166  0.000066  0.000365  0.002586

Around 16% people who earn >50k USD spends 50 hours at work.
I don't see that big number for above 40 hours for the people who earn <=50k. So majority of them spend <=40 hours a week.
'''
