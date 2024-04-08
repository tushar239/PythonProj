import gotoDataDir
import pandas as pd

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '###'])
print(cars_data)

# returns first n rows from dataframe
firstNRows = cars_data.head(15) # By default, it returns the first 5 rows
print(firstNRows)
'''
    Price   Age       KM FuelType  ... Automatic    CC  Doors  Weight
0   13500  23.0  46986.0   Diesel  ...         0  2000  three    1165
1   13750  23.0  72937.0   Diesel  ...         0  2000      3    1165
2   13950  24.0  41711.0   Diesel  ...         0  2000      3    1165
3   14950  26.0  48000.0   Diesel  ...         0  2000      3    1165
4   13750  30.0  38500.0   Diesel  ...         0  2000      3    1170
5   12950  32.0  61000.0   Diesel  ...         0  2000      3    1170
6   16900  27.0      NaN   Diesel  ...         0  2000      3    1245
7   18600  30.0  75889.0      NaN  ...         0  2000      3    1245
8   21500  27.0  19700.0   Petrol  ...         0  1800      3    1185
9   12950  23.0  71138.0   Diesel  ...         0  1900      3    1105
10  20950  25.0  31461.0   Petrol  ...         0  1800      3    1185
11  19950  22.0  43610.0   Petrol  ...         0  1800      3    1185
12  19600  25.0  32189.0   Petrol  ...         0  1800      3    1185
13  21500  31.0  23000.0   Petrol  ...         0  1800      3    1185
14  22500  32.0  34131.0   Petrol  ...         0  1800      3    1185
'''

# returns last n rows from dataframe
lastNRows = cars_data.tail(15) # By default, it returns the last 5 rows
print(lastNRows)
'''
      Price   Age       KM FuelType  ... Automatic    CC  Doors  Weight
1421   8500  78.0      NaN   Petrol  ...         1  1300      3    1045
1422   7600   NaN  36000.0      NaN  ...         0  1600      3    1050
1423   7950  80.0  35821.0   Petrol  ...         1  1300      3    1015
1424   7750  73.0  34717.0   Petrol  ...         0  1300      3    1015
1425   7950  80.0      NaN   Petrol  ...         0  1300      4    1000
1426   9950  78.0  30964.0   Petrol  ...         1  1600      3    1080
1427   8950   NaN  29000.0   Petrol  ...         1  1300      3    1045
1428   8450  72.0      NaN   Petrol  ...         0  1300      3    1015
1429   8950  78.0  24000.0   Petrol  ...         1  1300      5    1065
1430   8450  80.0  23000.0   Petrol  ...         0  1300      3    1015
1431   7500   NaN  20544.0   Petrol  ...         0  1300      3    1025
1432  10845  72.0      NaN   Petrol  ...         0  1300      3    1015
1433   8500   NaN  17016.0   Petrol  ...         0  1300      3    1015
1434   7250  70.0      NaN      NaN  ...         0  1300      3    1015
1435   6950  76.0      1.0   Petrol  ...         0  1600      5    1114

[15 rows x 10 columns]
'''

# at[row id, column name] returns an element in particular cell of dataframe
# at provides label-based scalar lookups
result = cars_data.at[4, 'FuelType']
print(result) # Diesel

# iat[row id, column id] returns an element in particular cell of dataframe
# iat provides integer-based lookups
result = cars_data.iat[4, 5]
print(result) # 0.0