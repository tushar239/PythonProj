'''
https://www.geeksforgeeks.org/time-series-data-visualization-in-python/

Detecting Seasonality Using Auto Correlation
We will detect Seasonality using the autocorrelation function (ACF) plot.
Peaks at regular intervals in the ACF plot suggest the presence of seasonality.
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

# Plot the ACF
plt.figure(figsize=(12, 6))
plot_acf(df['Volume'], lags=40) # You can adjust the number of lags as needed
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF) Plot')
plt.show()
