import gotoDataDir
import pandas as pd

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