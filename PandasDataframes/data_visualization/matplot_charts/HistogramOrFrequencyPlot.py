import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt # pyplot means python plot
import gotoDataDir
'''
sns.displot() is almost same as plt.hist()

Histogram Plot:
It is a graphical representation of data using bars of different heights
Histogram groups members into ranges and the height of each bar depicts the FREQUENCY of each RANGE or bin.

When to use histograms?
To represent the frequency distribution of numerical variables.
'''



cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# drop all the rows having NaN values.
# If inplace=True, then it will drop the rows from existing dataframe, otherwise it will create a new dataframe.
cars_data.dropna(axis=0, inplace=True)
print(cars_data)

#plt.hist(cars_data['KM'])  # Histogram with default arguments

plt.hist(cars_data['KM'], color='green', edgecolor='white', bins=5)
plt.title('Histogram of Kilometer')
plt.xlabel('Kilometer')
plt.ylabel('Frequency')
plt.show()
# Frequency distribution of kilometre of the cars shows that most of the cars have travelled between 50k-100k km
# and there are only few cars with more distance travelled