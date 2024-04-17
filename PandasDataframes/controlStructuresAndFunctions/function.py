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