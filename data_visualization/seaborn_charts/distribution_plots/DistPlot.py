import matplotlib.pyplot as plt  # pyplot means python plot
import pandas as pd
import seaborn as sns  # It is based on matplotlib, more attractive and informative
import gotoDataDir

'''
It is used basically for univariant set of observations and visualizes it through a histogram 
i.e. only one observation and hence we choose one particular column of the dataset.
'''

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# drop all the rows having NaN values.
# If inplace=True, then it will drop the rows from existing dataframe, otherwise it will create a new dataframe.
cars_data.dropna(axis=0, inplace=True)
print(cars_data)

'''
kde means kernal density estimate
Kernel density estimation is the process of estimating an unknown probability density function using a kernel function .
While a histogram counts the number of data points in somewhat arbitrary regions, a kernel density estimate is a function defined as the sum of a kernel function on every data point.

with kde=False, it will be exactly like a Histogram that shows the Frequency of different Age ranges(bins)
'''
sns.distplot(a=cars_data['Age'], kde=False, color='red')
plt.show()

# KDE shows the density where the points match up the most
sns.distplot(a=cars_data['Age'], kde=True, color='red')
plt.show()
