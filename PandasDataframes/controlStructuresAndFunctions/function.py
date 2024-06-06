import PandasDataframes.gotoDataDir
import pandas as pd
import numpy as np

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# print(cars_data)

# create new column in dataframe with default value as 0
cars_data.insert(10, "Age_Converted", 0)
#cars_data["Age_Converted"].astype(dtype="int")

def age_convert(val):
    val_converted = val/12
    return round(val_converted)

cars_data["Age_Converted"] = age_convert(cars_data["Age"])
age_converted_series = cars_data["Age_Converted"]
print(type(age_converted_series)) # <class 'pandas.core.series.Series'>

#print(cars_data)
print(cars_data.info()) #  10  Age_Converted  1336 non-null   float64

# pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
# cars_data["Age_Converted"] = cars_data["Age_Converted"].astype(dtype="int64")

print(age_converted_series.unique()) # [ 2.  3. nan  1.  0.  4.  5.  6.  7.]
# replacing nan values with 0s
age_converted_series = age_converted_series.fillna(0)
print(age_converted_series.unique()) # [2. 3. 0. 1. 4. 5. 6. 7.]
# Now, you can convert Age_Converted column to int64 from float64
cars_data["Age_Converted"] = age_converted_series.astype(dtype="int64")
print(cars_data.info())

'''
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Price          1436 non-null   int64  
 1   Age            1336 non-null   float64
 2   KM             1421 non-null   float64
 3   FuelType       1336 non-null   object 
 4   HP             1436 non-null   object 
 5   MetColor       1286 non-null   float64
 6   Automatic      1436 non-null   int64  
 7   CC             1436 non-null   int64  
 8   Doors          1436 non-null   object 
 9   Weight         1436 non-null   int64  
 10  Age_Converted  1436 non-null   int64  
dtypes: float64(3), int64(5), object(3)
memory usage: 134.6+ KB
None
'''

# create new column in dataframe with default value as 0
cars_data.insert(11, "Age_Converted2", 0)
cars_data.insert(12, "Km_per_month", 0)

# Function with multiple input and output values
def c_convert(val1, val2):
    val_converted = val1/12
    ratio = val1/val2
    return [val_converted, ratio]

cars_data["Age_Converted2"], cars_data["Km_per_month"] = \
    c_convert(cars_data["Age"], cars_data["KM"])
print(cars_data)

'''
      Price   Age       KM  ... Age_Converted Age_Converted2  Km_per_month
0     13500  23.0  46986.0  ...             2       1.916667      0.000490
1     13750  23.0  72937.0  ...             2       1.916667      0.000315
2     13950  24.0  41711.0  ...             2       2.000000      0.000575
3     14950  26.0  48000.0  ...             2       2.166667      0.000542
4     13750  30.0  38500.0  ...             2       2.500000      0.000779
...     ...   ...      ...  ...           ...            ...           ...
1431   7500   NaN  20544.0  ...             0            NaN           NaN
1432  10845  72.0      NaN  ...             6       6.000000           NaN
1433   8500   NaN  17016.0  ...             0            NaN           NaN
1434   7250  70.0      NaN  ...             6       5.833333           NaN
1435   6950  76.0      1.0  ...             6       6.333333     76.000000

[1436 rows x 13 columns]
'''

'''
describe() will give you count, standard deviation, min, max, mean etc for columns. It generates descriptive statistics 
that summarize the central tendency, dispersion, and shape of a dataset distribution,excluding NaN values.

From min to max, it is called 5-numbers summary. It gives you min, max and 3 quantiles.
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

