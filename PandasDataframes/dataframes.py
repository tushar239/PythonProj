import gotoDataDir
import pandas as pd

# to see entire dataframe in the output
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '###'])
print(cars_data)
# to print all rows and cols use to_string()
#print(cars_data.to_string())
'''
     Price   Age       KM FuelType  ... Automatic    CC  Doors  Weight
0     13500  23.0  46986.0   Diesel  ...         0  2000  three    1165
1     13750  23.0  72937.0   Diesel  ...         0  2000      3    1165
2     13950  24.0  41711.0   Diesel  ...         0  2000      3    1165
3     14950  26.0  48000.0   Diesel  ...         0  2000      3    1165
4     13750  30.0  38500.0   Diesel  ...         0  2000      3    1170
...     ...   ...      ...      ...  ...       ...   ...    ...     ...
1431   7500   NaN  20544.0   Petrol  ...         0  1300      3    1025
1432  10845  72.0      NaN   Petrol  ...         0  1300      3    1015
1433   8500   NaN  17016.0   Petrol  ...         0  1300      3    1015
1434   7250  70.0      NaN      NaN  ...         0  1300      3    1015
1435   6950  76.0      1.0   Petrol  ...         0  1600      5    1114

[1436 rows x 10 columns]
'''
row_ids = cars_data.index
print(row_ids)
'''
Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
       ...
       1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435],
      dtype='int64', length=1436)
'''

variables = cars_data.columns
print(variables)
'''
Index(['Price', 'Age', 'KM', 'FuelType', 'HP', 'MetColor', 'Automatic', 'CC','Doors', 'Weight'],
      dtype='object')
'''

# gives total number of elements in the dataframe
size = cars_data.size
print(size) # 14360

# gives the dimensionality of the dataframe
shape = cars_data.shape
print(shape) # (1436, 10)

# give memory usage by columns
memory = cars_data.memory_usage()
print(memory)
'''
Index        11488
Price        11488
Age          11488
KM           11488
FuelType     11488
HP           11488
MetColor     11488
Automatic    11488
CC           11488
Doors        11488
Weight       11488
dtype: int64
'''

# gives number of axes/array dimensions
total_axes = cars_data.ndim
print(total_axes) # 2
