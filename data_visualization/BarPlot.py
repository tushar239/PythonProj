import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt # pyplot means python plot
import gotoDataDir
'''
Bar Plot:
It is a plot that presents CATEGORICAL data with rectangular bars with lengths
proportional to the counts that they represent.

When to use histograms?
To represent the frequency distribution of CATEGORICAL variables.
'''
cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# drop all the rows having NaN values.
# If inplace=True, then it will drop the rows from existing dataframe, otherwise it will create a new dataframe.
cars_data.dropna(axis=0, inplace=True)
print(cars_data)

# change the data type of FuelType to Category
#cars_data['FuelType'] = cars_data['FuelType'].astype(dtype='category')
summary =  cars_data.info()
print(summary)

# Finding different values in FuelType column
fuelTypes = np.unique(cars_data['FuelType'])
print(type(fuelTypes))
print(fuelTypes) # ['CNG' 'Diesel' 'Petrol']

# counting total number of different fuel types in FuelType column
# https://www.geeksforgeeks.org/how-to-count-occurrences-of-specific-value-in-pandas-column/
valueCountsSeries = cars_data['FuelType'].value_counts()
# print(type(valueCountsSeries)) # <class 'pandas.core.series.Series'>
print('valueCountsSeries: \n', valueCountsSeries)
'''
Petrol    970
Diesel    117
CNG        12

You can't access first column as it is actually not a column, it is just the labels.
'''


# We have a list of fuelTypes, now creating a list of related counts
fuelTypeCount = []
for i in range(0, len(fuelTypes), 1):
    fuelType = fuelTypes[i]
    total = len(cars_data[cars_data['FuelType'] == fuelType])
    fuelTypeCount.append(total)
    print('fuelType: ' , fuelType , " , " + 'total: ' , total)

'''
fuelType:  CNG  , total:  12
fuelType:  Diesel  , total:  117
fuelType:  Petrol  , total:  970
'''

print('fuelTypes: ', fuelTypes) # ['CNG' 'Diesel' 'Petrol']
print('fuelTypeCount: ', fuelTypeCount) # [12, 117, 970]

'''
valueCountsDataFrame = valueCountsSeries.to_frame() # converting series to dataframe
print("data frame: \n", valueCountsDataFrame)
'''
'''
           count       
Petrol      970
Diesel      117
CNG          12
'''
'''
print(valueCountsDataFrame['count'].values) # [970 117  12]
'''
'''
rows = valueCountsDataFrame.shape[0]
cols = valueCountsDataFrame.shape[1]
for i in range(0, rows, 1):
    for j in range(0, cols, 1):
        print(valueCountsDataFrame.iat[i,j]) # https://sparkbyexamples.com/pandas/pandas-get-cell-value-from-dataframe/#:~:text=If%20you%20want%20to%20get,1%20as%20the%20column%20position.&text=This%20returns%20the%20same%20output%20as%20above.
'''

#index = np.arange(0, len(fuelTypes), 1)
#plt.bar(index, fuelTypeCount)
plt.bar(fuelTypes, fuelTypeCount)
plt.title('Bar plot of fuel types')
plt.xlabel('Fuel Types')
plt.ylabel('Frequency')
plt.show()