import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# reading the dataset using read_csv
df = pd.read_csv("stock_data.csv",
                 parse_dates=True, # Here we will use the ‘parse_dates’ parameter in the read_csv function to convert the ‘Date’ column to the DatetimeIndex format. By default, Dates are stored in string format which is not the right format for time series data analysis.
                 index_col="Date" # setting date as index column
                 )

# displaying the first five rows of dataset
print(df.head())
print(df.info())

# Assuming df is your DataFrame
sns.set(style="whitegrid")  # Setting the style to whitegrid for a clean background

# figure() function - https://www.geeksforgeeks.org/matplotlib-pyplot-figure-in-python/
plt.figure(figsize=(12, 6))  # Setting the figure size # figsize(float, float): These parameter are the width, height in inches.
sns.lineplot(data=df, x='Date', y='High', label='High Price', color='blue')

# Adding labels and title
plt.xlabel('Date')
plt.ylabel('High')
plt.title('Share Highest Price Over Time')

plt.show()