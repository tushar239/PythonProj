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
rows_with_missing_age = rows_with_missing_values.where(rows_with_missing_values['Age'].isnull())
print(rows_with_missing_age)
'''
      Price  Age       KM FuelType  ...  Automatic      CC  Doors  Weight
2        NaN  NaN      NaN      NaN  ...        NaN     NaN    NaN     NaN
6        NaN  NaN      NaN      NaN  ...        NaN     NaN    NaN     NaN
7        NaN  NaN      NaN      NaN  ...        NaN     NaN    NaN     NaN
9        NaN  NaN      NaN      NaN  ...        NaN     NaN    NaN     NaN
15       NaN  NaN      NaN      NaN  ...        NaN     NaN    NaN     NaN
...      ...  ...      ...      ...  ...        ...     ...    ...     ...
1428     NaN  NaN      NaN      NaN  ...        NaN     NaN    NaN     NaN
1431  7500.0  NaN  20544.0   Petrol  ...        0.0  1300.0      3  1025.0
1432     NaN  NaN      NaN      NaN  ...        NaN     NaN    NaN     NaN
1433  8500.0  NaN  17016.0   Petrol  ...        0.0  1300.0      3  1015.0
1434     NaN  NaN      NaN      NaN  ...        NaN     NaN    NaN     NaN
[340 rows x 10 columns]
'''
# where function - https://www.geeksforgeeks.org/python-pandas-dataframe-where/
filter1 = rows_with_missing_values['Age'].isnull()
filter2 = rows_with_missing_values['KM'].isnull()
rows_with_missing_age_and_km = rows_with_missing_values.where(filter1 & filter2)
print(rows_with_missing_age_and_km)
'''
      Price  Age  KM FuelType  HP  MetColor  Automatic  CC Doors  Weight
2       NaN  NaN NaN      NaN NaN       NaN        NaN NaN   NaN     NaN
6       NaN  NaN NaN      NaN NaN       NaN        NaN NaN   NaN     NaN
7       NaN  NaN NaN      NaN NaN       NaN        NaN NaN   NaN     NaN
9       NaN  NaN NaN      NaN NaN       NaN        NaN NaN   NaN     NaN
15      NaN  NaN NaN      NaN NaN       NaN        NaN NaN   NaN     NaN
...     ...  ...  ..      ...  ..       ...        ...  ..   ...     ...
1428    NaN  NaN NaN      NaN NaN       NaN        NaN NaN   NaN     NaN
1431    NaN  NaN NaN      NaN NaN       NaN        NaN NaN   NaN     NaN
1432    NaN  NaN NaN      NaN NaN       NaN        NaN NaN   NaN     NaN
1433    NaN  NaN NaN      NaN NaN       NaN        NaN NaN   NaN     NaN
1434    NaN  NaN NaN      NaN NaN       NaN        NaN NaN   NaN     NaN

[340 rows x 10 columns]
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

cars_data2['Age'].fillna(mean_of_age, inplace=True)

# subsetting the rows that have one or more missing values
rows_with_missing_values = cars_data2[cars_data2.isnull().any(axis=1)]
filter = rows_with_missing_values['Age'].isnull()
print(filter) # Series([], Name: Age, dtype: bool)   no missing values now

'''
Let' impute(replace) missing values of KM column with median value.
'''

median_of_km = cars_data2['KM'].median()
print(median_of_km) # 63061.5

cars_data2['KM'].fillna(median_of_km, inplace=True)

# subsetting the rows that have one or more missing values
rows_with_missing_values = cars_data2[cars_data2.isnull().any(axis=1)]
filter = rows_with_missing_values['KM'].isnull()
print(filter) # Series([], Name: KM, dtype: bool)   no missing values now


'''
Let' impute(replace) missing values of HP column with mean value.
'''

mean_of_hp = cars_data2['HP'].mean()
print(mean_of_hp) # 101.28693101262657

cars_data2['HP'].fillna(mean_of_hp, inplace=True)

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
