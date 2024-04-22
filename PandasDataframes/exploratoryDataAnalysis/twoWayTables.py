import PandasDataframes.gotoDataDir
import pandas as pd
import numpy as np

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])

cars_data_copy = cars_data.copy()

'''
It finds out the frequency cross table between the values of Automatic 
and the values of FuelType variables(columns)
'''
result = pd.crosstab(index=cars_data_copy['Automatic'],
                     columns=cars_data_copy['FuelType'],
                     dropna=False)
print(result)
'''
FuelType   CNG  Diesel  Petrol  NaN
Automatic                          
0           15     144    1104   93
1            0       0      73    7

This relationship shows that Automatic cars (value=1) has only Petrol cars.
'''