# study data_types/dataTypesAndMissingValuesOfColumns.py first

# https://www.geeksforgeeks.org/python-pandas-dataframe-replace/

import PandasDataframes.gotoDataDir
import pandas as pd
import numpy as np

cars_data = pd.read_csv(filepath_or_buffer = 'Toyota.csv', index_col=0, na_values=["??", "????"])
print(cars_data)

doors_unique_elements = np.unique(cars_data['Doors'])
print(doors_unique_elements)

'''
replace(to_replace, value, inplace=False by default)
It will replace the data in all the columns.

nptel's video's way is not working
cars_data['Doors'].replace(to_replace='three', value=3, inplace=True)
Use Numpy's where() method, if you want to change the values in a specific column.
'''
cars_data.replace(to_replace='three', value=3, inplace=True)
cars_data.replace(to_replace='four', value=4, inplace=True)
cars_data.replace(to_replace='five', value=5, inplace=True)
# change the data type from object to int64 after changing the data.
cars_data['Doors'] = cars_data['Doors'].astype(dtype='int')

doors_unique_elements = np.unique(cars_data['Doors'])
print(doors_unique_elements) # [2 3 4 5]
#print(cars_data)

############################################
'''
https:www.geeksforgeeks.org/how-to-replace-values-in-column-based-on-condition-in-pandas/

numpy's where(condition, x, y)
Based on condition, return either x or y
If the condition is true, then return x, otherwise return y

'''
cars_data = pd.read_csv(filepath_or_buffer = 'Toyota.csv', index_col=0, na_values=["??", "????"])
print(cars_data)

cars_data["Doors"] = np.where(cars_data["Doors"] == "three", 3, cars_data["Doors"])
cars_data["Doors"] = np.where(cars_data["Doors"] == "four", 4, cars_data["Doors"])
cars_data["Doors"] = np.where(cars_data["Doors"] == "five", 5, cars_data["Doors"])
# change the data type from object to int64 after changing the data.
cars_data['Doors'] = cars_data['Doors'].astype(dtype='int')

doors_unique_elements = np.unique(cars_data['Doors'])
print(doors_unique_elements) # [2 3 4 5]
#print(cars_data)

#####################################
'''
mask function doesn't work as mentioned in 
https:www.geeksforgeeks.org/how-to-replace-values-in-column-based-on-condition-in-pandas/


cars_data = pd.read_csv(filepath_or_buffer = 'Toyota.csv', index_col=0, na_values=["??", "????"])
print(cars_data)

cars_data['Doors'].mask(cars_data['Doors'] == 'three', 3, inplace=True)
cars_data['Doors'].mask(cars_data['Doors'] == 'four', 4, inplace=True)
cars_data['Doors'].mask(cars_data['Doors'] == 'five', 5, inplace=True)
# change the data type from object to int64 after changing the data.
cars_data['Doors'] = cars_data['Doors'].astype(dtype='int')

doors_unique_elements = np.unique(cars_data['Doors'])
print(doors_unique_elements) # [2 3 4 5]
'''