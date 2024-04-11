import gotoDataDir
import pandas as pd
import numpy as np

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '###'])
print(cars_data)

'''
There mainly 2 types of data types
    - numerical
        int64 - Basic python's int. It says that it uses 64 bits (8 bytes) to store int value.
        float64 - Basic python's float. It says that it uses 64 bits (8 bytes) to store double value.
    - character
        categorical - limited number of fixed values can be there in categorical column. This can save some memory for you.
        object - when number of values are not limited, object data type is used. whichever column has blank cell with NaN value, automatically gets object data type
'''
# Gives data type of each column in dataframe
dataTypes = cars_data.dtypes
print(dataTypes)

'''
Price          int64
Age          float64
KM           float64
FuelType      object
HP            object
MetColor     float64
Automatic      int64
CC             int64
Doors         object
Weight         int64
dtype: object
'''

# Gives count of unique data types
# countOfUniqueDataTypes = cars_data.get_dtype_counts() # this method is not available
# print(countOfUniqueDataTypes)

'''
select columns based on their data types.
'''
cols = cars_data.select_dtypes(include=[int])
print(cols)
'''
     Price  Automatic    CC  Weight
0     13500          0  2000    1165
1     13750          0  2000    1165
2     13950          0  2000    1165
3     14950          0  2000    1165
4     13750          0  2000    1170
...     ...        ...   ...     ...
1431   7500          0  1300    1025
1432  10845          0  1300    1015
1433   8500          0  1300    1015
1434   7250          0  1300    1015
1435   6950          0  1600    1114

[1436 rows x 4 columns]
'''


cols = cars_data.select_dtypes(exclude=[object, int])
print(cols)
'''
      Age       KM  MetColor
0     23.0  46986.0       1.0
1     23.0  72937.0       1.0
2     24.0  41711.0       NaN
3     26.0  48000.0       0.0
4     30.0  38500.0       0.0
...    ...      ...       ...
1431   NaN  20544.0       1.0
1432  72.0      NaN       0.0
1433   NaN  17016.0       0.0
1434  70.0      NaN       1.0
1435  76.0      1.0       0.0

[1436 rows x 3 columns]
'''

'''
Gives concise summary of dataframe.
It gives Column types + missing values information.

Sometimes, it may read a column as int64, but it makes more sense to have it as object. 
e.g. Doors - values are '2','3','4','5','five,'four','three'. So, pandas considers it as object, but from business perspective it should be int64.
MetColor should be Object type.
Automatic should be of Categorical type.
HP should be integer.
Doors should be int64.
'''
summary =  cars_data.info()
print(summary)
'''
<class 'pandas.core.frame.DataFrame'>
Index: 1436 entries, 0 to 1435
Data columns (total 10 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Price      1436 non-null   int64  
 1   Age        1336 non-null   float64
 2   KM         1421 non-null   float64
 3   FuelType   1336 non-null   object 
 4   HP         1436 non-null   object 
 5   MetColor   1286 non-null   float64
 6   Automatic  1436 non-null   int64  
 7   CC         1436 non-null   int64  
 8   Doors      1436 non-null   object 
 9   Weight     1436 non-null   int64  
dtypes: float64(3), int64(4), object(3)
memory usage: 123.4+ KB
None
'''

'''
Gives unique elements of columns.
It uses numpy package.
'''
print('unique elements ...')
hp_unique_elements = np.unique(cars_data['HP'])
print(hp_unique_elements)
'''
['107' '110' '116' '192' '69' '71' '72' '73' '86' '90' '97' '98' '????']

It has '????'. So, HP is given a data type object instead of int64
'''
metcolor_unique_elements = np.unique(cars_data['MetColor'])
print(metcolor_unique_elements)
'''
[ 0.  1. nan]
it has 0. and 1.. So, it is given a data type float64 instead of object.
'''
automatic_unique_elements = np.unique(cars_data['Automatic'])
print(automatic_unique_elements)
'''
[0 1]
Because it has 0 and 1, it is given a data type int64 instead of category.
'''
doors_unique_elements = np.unique(cars_data['Doors'])
print(doors_unique_elements)
'''
['2' '3' '4' '5' 'five' 'four' 'three']
Due to the mixture of ints and text, it is given a data type object instead of int64.
'''