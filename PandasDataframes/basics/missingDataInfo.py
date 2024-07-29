import pandas as pd
import numpy as np
import gotoDataDir

cars_data = pd.read_csv(filepath_or_buffer = 'Toyota.csv',
                        index_col=0, # consider 0th col as index number col
                        na_values=["??", "????"] # convert these values to nan
                        )
print(cars_data)

# info() returns the information of dataframe that contains total number of non-null values in each column along with other data
summary =  cars_data.info()
print(summary)
'''
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Price      1436 non-null   int64  
 1   Age        1336 non-null   float64
 2   KM         1421 non-null   float64
 3   FuelType   1336 non-null   object 
 4   HP         1430 non-null   float64
 5   MetColor   1286 non-null   float64
 6   Automatic  1436 non-null   int64  
 7   CC         1436 non-null   int64  
 8   Doors      1436 non-null   object 
 9   Weight     1436 non-null   int64  
dtypes: float64(4), int64(4), object(2)
memory usage: 123.4+ KB
'''


total_nan_elements_in_each_col = cars_data.isnull().sum()
print(total_nan_elements_in_each_col)
'''
Price          0
Age          100
KM            15
FuelType     100
HP             6
MetColor     150
Automatic      0
CC             0
Doors          0
Weight         0
dtype: int64
'''