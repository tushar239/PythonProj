import gotoDataDir
import pandas as pd

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '###'])
print(cars_data)

# returns first n rows from dataframe
print(cars_data.head(5))