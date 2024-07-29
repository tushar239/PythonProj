import pandas as pd

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# print(cars_data)

# create new column in dataframe with default value as blank
cars_data.insert(10, "Price_Class", "")
print(cars_data)
'''
      Price   Age       KM FuelType  ...    CC  Doors  Weight  Price_Class
0     13500  23.0  46986.0   Diesel  ...  2000  three    1165             
1     13750  23.0  72937.0   Diesel  ...  2000      3    1165             
2     13950  24.0  41711.0   Diesel  ...  2000      3    1165             
3     14950  26.0  48000.0   Diesel  ...  2000      3    1165             
4     13750  30.0  38500.0   Diesel  ...  2000      3    1170             
...     ...   ...      ...      ...  ...   ...    ...     ...          ...
1431   7500   NaN  20544.0   Petrol  ...  1300      3    1025             
1432  10845  72.0      NaN   Petrol  ...  1300      3    1015             
1433   8500   NaN  17016.0   Petrol  ...  1300      3    1015             
1434   7250  70.0      NaN      NaN  ...  1300      3    1015             
1435   6950  76.0      1.0   Petrol  ...  1600      5    1114             

[1436 rows x 11 columns]
'''

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

# Similary, you can write a while loop
cars_data.insert(11, "Price_Class2", "")

i = 0
while i < len(cars_data["Price"]):
    if (cars_data["Price"][i] <= 8450):
        value = "Low"
    elif (cars_data["Price"][i] >= 11950):
        value = "High"
    else:
        value = "Medium"
    # See indexingAndSelectingData.py to know more about df.loc() method.
    cars_data.loc[i, "Price_Class2"] = value
    i = i + 1

print(cars_data)

'''
      Price   Age       KM FuelType  ...  Doors  Weight  Price_Class  Price_Class2
0     13500  23.0  46986.0   Diesel  ...  three    1165         High          High
1     13750  23.0  72937.0   Diesel  ...      3    1165         High          High
2     13950  24.0  41711.0   Diesel  ...      3    1165         High          High
3     14950  26.0  48000.0   Diesel  ...      3    1165         High          High
4     13750  30.0  38500.0   Diesel  ...      3    1170         High          High
...     ...   ...      ...      ...  ...    ...     ...          ...           ...
1431   7500   NaN  20544.0   Petrol  ...      3    1025          Low           Low
1432  10845  72.0      NaN   Petrol  ...      3    1015       Medium        Medium
1433   8500   NaN  17016.0   Petrol  ...      3    1015       Medium        Medium
1434   7250  70.0      NaN      NaN  ...      3    1015          Low           Low
1435   6950  76.0      1.0   Petrol  ...      5    1114          Low           Low

[1436 rows x 12 columns]
'''