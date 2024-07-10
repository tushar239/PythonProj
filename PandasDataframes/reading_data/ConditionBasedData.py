
import os
import pandas as pd

curDir = os.getcwd()
print(curDir)

os.chdir(os.curdir + '\data')

curDir = os.getcwd()
print(curDir)


cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# drop all the rows having NaN values.
# If inplace=True, then it will drop the rows from existing dataframe, otherwise it will create a new dataframe.
cars_data.dropna(axis=0, inplace=True)
print(cars_data)

rows_with_fueltype_petrol = cars_data[cars_data['FuelType'] == 'Petrol']
print(rows_with_fueltype_petrol)
total_rows_with_fueltype_petrol = len(rows_with_fueltype_petrol)
print(total_rows_with_fueltype_petrol)

# you can use dataframe.loc(condition) also
# https://www.geeksforgeeks.org/filter-pandas-dataframe-with-multiple-conditions/