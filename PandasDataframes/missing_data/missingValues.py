import matplotlib.pyplot as plt  # pyplot means python plot
import pandas as pd
import numpy as np
import seaborn as sns  # It is based on matplotlib, more attractive and informative
import gotoDataDir

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '????', '###'])
# drop all the rows having NaN values.
# If inplace=True, then it will drop the rows from existing dataframe, otherwise it will create a new dataframe.
# cars_data.dropna(axis=0, inplace=True)
print(cars_data)

cars_data2 = cars_data.copy()
cars_data3 = cars_data2.copy()


################ Working with numerical columns ################################
'''
To find missing values, you can use isna() or isNull.
They return True for NaN values.
'''
count_of_missing_values = cars_data2.isna().sum()
print(count_of_missing_values)
'''
Price          0
Age          100
KM            15
FuelType     100
HP             0
MetColor     150
Automatic      0
CC             0
Doors          0
Weight         0
dtype: int64
'''
count_of_missing_values = cars_data2.isnull().sum()
print(count_of_missing_values)
'''
Price          0
Age          100
KM            15
FuelType     100
HP             0
MetColor     150
Automatic      0
CC             0
Doors          0
Weight         0
dtype: int64
'''

# subsetting the rows that have one or more missing values
rows_with_missing_values = cars_data2[cars_data2.isnull().any(axis=1)]
print(rows_with_missing_values) # there are 340 rows having NaN
'''
      Price   Age       KM FuelType  ...  Automatic    CC  Doors  Weight
2     13950  24.0  41711.0   Diesel  ...          0  2000      3    1165
6     16900  27.0      NaN   Diesel  ...          0  2000      3    1245
7     18600  30.0  75889.0      NaN  ...          0  2000      3    1245
9     12950  23.0  71138.0   Diesel  ...          0  1900      3    1105
15    22000  28.0  18739.0   Petrol  ...          0  1800      3    1185
...     ...   ...      ...      ...  ...        ...   ...    ...     ...
1428   8450  72.0      NaN   Petrol  ...          0  1300      3    1015
1431   7500   NaN  20544.0   Petrol  ...          0  1300      3    1025
1432  10845  72.0      NaN   Petrol  ...          0  1300      3    1015
1433   8500   NaN  17016.0   Petrol  ...          0  1300      3    1015
1434   7250  70.0      NaN      NaN  ...          0  1300      3    1015

[340 rows x 10 columns]
'''
cars_data["Doors"] = np.where(cars_data["Doors"] == "three", 3, cars_data["Doors"])

# Find rows where Age is NaN
rows_with_missing_age = rows_with_missing_values[rows_with_missing_values['Age'].isnull()]
print("Rows with missing age...........................", type(rows_with_missing_values['Age'].isnull())) # <class 'pandas.core.series.Series'>
# print(type(rows_with_missing_values["Age"] > 24)) # <class 'pandas.core.series.Series'>
print(rows_with_missing_age)
'''
       Price  Age       KM FuelType  ...  Automatic    CC  Doors  Weight
33    14950  NaN  32692.0   Petrol  ...          0  1400      3    1100
55    13250  NaN  45725.0   Petrol  ...          0  1600      5    1075
83    17950  NaN  16238.0   Petrol  ...          1  1600      5    1180
92    19950  NaN  34472.0   Diesel  ...          0  1995      3    1260
105   16950  NaN  13748.0   Petrol  ...          0  1400      3    1100
...     ...  ...      ...      ...  ...        ...   ...    ...     ...
1416   8950  NaN  40093.0   Petrol  ...          0  1600      5    1114
1422   7600  NaN  36000.0      NaN  ...          0  1600      3    1050
1427   8950  NaN  29000.0   Petrol  ...          1  1300      3    1045
1431   7500  NaN  20544.0   Petrol  ...          0  1300      3    1025
1433   8500  NaN  17016.0   Petrol  ...          0  1300      3    1015

[100 rows x 10 columns]
'''
# where function - https://www.geeksforgeeks.org/python-pandas-dataframe-where/
filter1 = rows_with_missing_values['Age'].isnull()
filter2 = rows_with_missing_values['KM'].isnull()
rows_with_missing_age_and_km = rows_with_missing_values[filter1 & filter2]
print(rows_with_missing_age_and_km)
'''
Empty DataFrame
'''

'''
Deleting(removing/dropping) data from data frame by id of the row.
https://www.geeksforgeeks.org/how-to-drop-rows-in-pandas-dataframe-by-index-labels/
'''

'''
There are 340 rows with missing values. You can't afford to loose them.
There are two ways to fill in the missing values.
1. If it's is a numerical column, then you can fill in mean/median 
2. If it's a categorical column, then you can fill in the max occurring category
But if you have very low/high value (extreme cases), it can affect a mean/median value.
In this case you should take median. Median will sort all the values and take the middle value in case of odd number of rwos.
In case of even number of rows, it will take middle two values and take its average.
'''

'''
describe() will give you count, standard deviation, min, max, mean etc for columns. It generates descriptive statistics 
that summarize the central tendency, dispersion, and shape of a dataset distribution,excluding NaN values.

From min to max, it is called 5-numbers summary. It gives you min, max and 3 quartiles.

It means 
- 25% of the car's price is lesser than 4350
- 50% of the car's price is lesser than 9900
- 75% of the car's price is lesser than 11950

Average is represented by mean.
Median is represented by 50%
'''

print(cars_data.describe())

'''
              Price          Age  ...  Age_Converted2  Km_per_month
count   1436.000000  1336.000000  ...     1336.000000   1321.000000
mean   10730.824513    55.672156  ...        4.639346      0.111520
std     3626.964585    18.589804  ...        1.549150      2.526162
min     4350.000000     1.000000  ...        0.083333      0.000177
25%     8450.000000    43.000000  ...        3.583333      0.000661
50%     9900.000000    60.000000  ...        5.000000      0.000880
75%    11950.000000    70.000000  ...        5.833333      0.001156
max    32500.000000    80.000000  ...        6.666667     76.000000

[8 rows x 10 columns]
'''

'''
Let' impute(replace) missing values of Age column with mean value.
'''
mean_of_age = cars_data2['Age'].mean()
print(mean_of_age) # 55.67215568862275

cars_data2.fillna({'Age' : mean_of_age}, inplace=True)

# subsetting the rows that have one or more missing values
rows_with_missing_values = cars_data2[cars_data2.isnull().any(axis=1)]
filter = rows_with_missing_values['Age'].isnull()
print(filter) # Series([], Name: Age, dtype: bool)   no missing values now

'''
Let' impute(replace) missing values of KM column with median value.
'''

median_of_km = cars_data2['KM'].median()
print(median_of_km) # 63061.5
cars_data2.fillna({'KM' : median_of_km}, inplace=True)

# subsetting the rows that have one or more missing values
rows_with_missing_values = cars_data2[cars_data2.isnull().any(axis=1)]
filter = rows_with_missing_values['KM'].isnull()
print(filter) # Series([], Name: KM, dtype: bool)   no missing values now


'''
Let' impute(replace) missing values of HP column with mean value.
'''

mean_of_hp = cars_data2['HP'].mean()
print(mean_of_hp) # 101.28693101262657
cars_data2.fillna({'HP' : mean_of_hp}, inplace=True)

# subsetting the rows that have one or more missing values
rows_with_missing_values = cars_data2[cars_data2.isnull().any(axis=1)]
filter = rows_with_missing_values['HP'].isnull()
print(filter) # Series([], Name: HP, dtype: bool)   no missing values now

# Finding missing values after filling missing values with mean or median
count_of_missing_values = cars_data2.isnull().sum()
print(count_of_missing_values)
'''
Price          0
Age            0
KM             0
FuelType     100
HP             0
MetColor     150
Automatic      0
CC             0
Doors          0
Weight         0
dtype: int64
'''
########################## working with categorical columns ################################################################################
'''
Every column in data frame is a series. 
Series.value_counts() will give you counts of every different value.
This is useful for Categorical column.

Series.index will give you labels(ids) of each value in a series.
'''
fuel_types_counts = cars_data2['FuelType'].value_counts(ascending=False)  # you can use crosstab function also that will give you the frequencies (counts)
indices = fuel_types_counts.index # finds all labels(ids) from the series
print(indices[0], ' has count ' , fuel_types_counts.get(indices[0])) # Petrol  has count  1177
#print(type(fuel_types_counts))
#print(fuel_types_counts)
'''
FuelType
Petrol    1177
Diesel     144
CNG         15
Name: count, dtype: int64
'''
cars_data2.fillna({'FuelType' : indices[0]}, inplace=True)

print('After replacing NaN in FuelType with Petrol')
fuel_types_counts = cars_data2['FuelType'].value_counts(ascending=False)
indices = fuel_types_counts.index
print(indices[0], ' has count ', fuel_types_counts.get(indices[0]))  # Petrol  has count  1277

count_of_missing_values = cars_data2.isnull().sum()
print(count_of_missing_values)
'''
Price          0
Age            0
KM             0
FuelType       0
HP             0
MetColor     150
Automatic      0
CC             0
Doors          0
Weight         0
dtype: int64
'''

'''
Replacing NaN of MetColor with its modal value (the most common number that appears in your set of data).

In statistics, the mode is the value that is repeatedly occurring in a given set. We can also say that the value or number in a data set, which has a high frequency or appears more frequently, is called mode or modal value. 
It is one of the three measures of central tendency, apart from mean and median.
'''
mode_value_of_MetColor = cars_data2['MetColor'].mode()
print(mode_value_of_MetColor)
'''
 0    1.0
 
 index(label) is 0, value is 1.0
'''
cars_data2.fillna({'MetColor': mode_value_of_MetColor[0]}, inplace=True)
count_of_missing_values = cars_data2.isnull().sum()
print(count_of_missing_values)


# You can impute all the missing values just using fillna() once by supplying multiple key:value pairs in it.