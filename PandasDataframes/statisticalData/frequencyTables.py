import pandas as pd

# to see entire dataframe in the output
pd.set_option('display.max_columns', None)

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])

cars_data_copy = cars_data.copy()

'''
It finds out the frequency of different values in a column
index column in the resulting dataframe will be the values of FuelType.
'''
# barplot(countplot) is a graphical representation of crosstab function
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
# print(type(result)) # <class 'pandas.core.frame.DataFrame'>
# print(result.info())
'''
#   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   count   4 non-null      int64
'''
#print(result.describe())
'''
col_0        count
count     4.000000
mean    359.000000
std     547.955594
min      15.000000
25%      78.750000
50%     122.000000
75%     402.250000
max    1177.000000
'''