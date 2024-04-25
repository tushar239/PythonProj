import PandasDataframes.gotoDataDir
import pandas as pd
import numpy as np

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])

cars_data_copy = cars_data.copy()

'''
It finds out the frequency of different values in a column
index column in the resulting dataframe will be the values of FuelType.
'''
result = pd.crosstab(index=cars_data_copy['FuelType'],
                     columns='count',
                     dropna=False)
print(result)
'''
col_0     count
FuelType       
CNG          15
Diesel      144
Petrol     1177
NaN         100

So, from this information, we can say that most of the cars are of type Petrol.
'''
