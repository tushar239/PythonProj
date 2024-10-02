# https://www.geeksforgeeks.org/time-series-data-visualization-in-python/

'''
Resampling
----------
To better understand the trend of the data we will use the resampling method,
resampling the data on a monthly basis can provide a clearer view of trends and patterns,
especially when we are dealing with daily data.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# reading the dataset using read_csv
df = pd.read_csv("stock_data.csv",
                 # convert string date(1/3/2006) to datetime format(2006-01-03)
                 #parse_dates=True, # Here we will use the ‘parse_dates’ parameter in the read_csv function to convert the ‘Date’ column to the DatetimeIndex format. By default, Dates are stored in string format which is not the right format for time series data analysis.
                 #index_col="Date" # setting date as index column
                 )
# https://www.geeksforgeeks.org/python-pandas-to_datetime/
# convert string date(1/3/2006) to date format(2006-01-03)
df['Date'] = pd.to_datetime(df['Date'])

# changing the index column to Date column
df.set_index("Date", inplace=True)

print(df.info())
'''
#   Column  Non-Null Count  Dtype         
---  ------  --------------  -----         
 0   Date    3019 non-null   datetime64[ns]
 1   Open    3019 non-null   float64       
 2   High    3019 non-null   float64       
 3   Low     3019 non-null   float64       
 4   Close   3019 non-null   float64       
 5   Volume  3019 non-null   int64         
 6   Name    3019 non-null   object   
'''
print(df.head())
'''
        Date   Open   High    Low  Close    Volume  Name
0 2006-01-03  39.69  41.22  38.79  40.91  24232729  AABA
1 2006-01-04  41.22  41.90  40.77  40.97  20553479  AABA
2 2006-01-05  40.93  41.73  40.85  41.53  12829610  AABA
3 2006-01-06  42.88  43.57  42.80  43.21  29422828  AABA
4 2006-01-09  43.10  43.66  42.82  43.42  16268338  AABA
'''


df.drop(columns='Name', inplace =True)

# Assuming df is your DataFrame with a datetime index
df_resampled = df.resample('M').mean() # Resampling to monthly frequency, using mean as an aggregation function
print(df_resampled.head())

sns.set(style="whitegrid") # Setting the style to whitegrid for a clean background

# Plotting the 'high' column with seaborn, setting x as the resampled 'Date'
plt.figure(figsize=(12, 6)) # Setting the figure size
sns.lineplot(data=df_resampled, x=df_resampled.index, y='High', label='Month Wise Average High Price', color='blue')

# Adding labels and title
plt.xlabel('Date (Monthly)')
plt.ylabel('High')
plt.title('Monthly Resampling Highest Price Over Time')

plt.show()
