# study data_types/dataTypesAndMissingValuesOfColumns.py first

import PandasDataframes.gotoDataDir
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