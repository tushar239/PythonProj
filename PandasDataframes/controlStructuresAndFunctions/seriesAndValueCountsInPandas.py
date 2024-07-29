import pandas as pd

'''
What is a Series? A Pandas Series is like a column in a table. 
It is a one-dimensional array holding data of any type.
https://www.w3schools.com/python/pandas/pandas_series.asp

Series is like a column in a table. Each row in a series gets a label. Either you can assign a label or it gets its own using ids.
'''

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# print(cars_data)

# create new column in dataframe with default value as blank
cars_data.insert(10, "Price_Class", "")

for i in range(0, len(cars_data["Price"]), 1):
    if (cars_data["Price"][i] <= 8450):
        value = "Low"
    elif (cars_data["Price"][i] >= 11950):
        value = "High"
    else:
        value = "Medium"
    # See indexingAndSelectingData.py to know more about df.loc() method.
    cars_data.loc[i, "Price_Class"] = value

print(cars_data)

'''
      Price   Age       KM FuelType  ...    CC  Doors  Weight  Price_Class
0     13500  23.0  46986.0   Diesel  ...  2000  three    1165         High
1     13750  23.0  72937.0   Diesel  ...  2000      3    1165         High
2     13950  24.0  41711.0   Diesel  ...  2000      3    1165         High
3     14950  26.0  48000.0   Diesel  ...  2000      3    1165         High
4     13750  30.0  38500.0   Diesel  ...  2000      3    1170         High
...     ...   ...      ...      ...  ...   ...    ...     ...          ...
1431   7500   NaN  20544.0   Petrol  ...  1300      3    1025          Low
1432  10845  72.0      NaN   Petrol  ...  1300      3    1015       Medium
1433   8500   NaN  17016.0   Petrol  ...  1300      3    1015       Medium
1434   7250  70.0      NaN      NaN  ...  1300      3    1015          Low
1435   6950  76.0      1.0   Petrol  ...  1600      5    1114          Low

[1436 rows x 11 columns]
'''

'''
on any series, you can call value counts(). It gives you a count of all types of values in that series.
In dataframe, each column returns a Series type.
'''
price_class_series = cars_data["Price_Class"] # column is a series
print(price_class_series)
'''
Series is like a column in a table. Each row in a series gets a label. Either you can assign a label or it gets its own using ids.
0         High
1         High
2         High
3         High
4         High
         ...  
1431       Low
1432    Medium
1433    Medium
1434       Low
1435       Low
Name: Price_Class, Length: 1436, dtype: object
'''

values = price_class_series.value_counts()  # you can use crosstab function also that will give you the frequencies (counts)
print(values)
'''
Price_Class
Medium    704
Low       369
High      363
Name: count, dtype: int64
'''
