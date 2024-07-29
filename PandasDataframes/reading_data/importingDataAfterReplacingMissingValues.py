# study data_types/dataTypesAndMissingValuesOfColumns.py first

import pandas as pd
import numpy as np

cars_data = pd.read_csv('Toyota.csv', index_col=0)
print('unique elements with missing values...')
hp_unique_elements = np.unique(cars_data['HP'])
print(hp_unique_elements)
'''
['107' '110' '116' '192' '69' '71' '72' '73' '86' '90' '97' '98' '????']
It has '????'. So, HP is given a data type object instead of int64
'''
summary =  cars_data.info()
print(summary) # HP will have data type 'object


############################################################################

# na_values attribute replaces mentioned strings with nan value in dataframe.
print('unique elements after replacing missing values with nan...')
cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=["??", "????"])
print(cars_data)

hp_unique_elements = np.unique(cars_data['HP'])
print(hp_unique_elements)
'''
[ 69.  71.  72.  73.  86.  90.  97.  98. 107. 110. 116. 192.  nan]

Now, HP will get data type float64
'''
summary =  cars_data.info()
print(summary)
memory = cars_data.memory_usage()
print(memory)

print('After changing the data types of columns(variables) .......')
# changing the data types of columns
cars_data['MetColor'] = cars_data['MetColor'].astype(dtype='object')
cars_data['Automatic'] = cars_data['Automatic'].astype(dtype='category')
cars_data['FuelType'] = cars_data['FuelType'].astype(dtype='category')
summary =  cars_data.info()
print(summary)
'''
Before changing the data types

 3   FuelType   1336 non-null   object 
 5   MetColor   1286 non-null   float64
 6   Automatic  1436 non-null   int64 
 
After changing the data types 

 3   FuelType   1336 non-null   category
 5   MetColor   1286 non-null   object  
 6   Automatic  1436 non-null   category
'''
memory = cars_data.memory_usage()
print(memory)
bytes_used = cars_data['Automatic'].nbytes
print(bytes_used) # 1452
bytes_used = cars_data['FuelType'].nbytes
print(bytes_used) # 1460
'''
Before changing Automatic's data type to category

MetColor     11488
Automatic    11488
FuelType     11488

After changing Automatic's data type to category

MetColor     11488
Automatic     1560  ---- huge impact on memory usage
FuelType      1568  ---- huge impact on memory usage
'''

# dataframe summary after changing column data types
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
 3   FuelType   1336 non-null   category - changed
 4   HP         1430 non-null   float64 
 5   MetColor   1286 non-null   object   - changed
 6   Automatic  1436 non-null   category - changed
 7   CC         1436 non-null   int64   
 8   Doors      1436 non-null   object  
 9   Weight     1436 non-null   int64   
dtypes: category(2), float64(3), int64(3), object(2)
memory usage: 104.0+ KB
None
'''




