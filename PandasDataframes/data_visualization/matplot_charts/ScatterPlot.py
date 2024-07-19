import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt # pyplot means python plot
import gotoDataDir
'''
Scatter Plot:
It is a set of points that represents the values obtained for two different variable plotted on a horizontal and vertical axes.

When to use scatter plots?
Scatter plots are used to convey the relationship between two numerical variables.
Scatter plots are sometimes called CORRELATION PLOTS because they show how two variables are correlated.

You can look at correlation priority first in twoWayTables.py.
'''


cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# drop all the rows having NaN values.
# If inplace=True, then it will drop the rows from existing dataframe, otherwise it will create a new dataframe.
cars_data.dropna(axis=0, inplace=True)
print(cars_data)

# let's plot a scatter plot
# scatter plot is same as regplot without regression line(fit_reg=True)
plt.scatter(x=cars_data['Age'], y=cars_data['Price'], c='red')
plt.title('Scatter plot of Price vs Age of the cars')
plt.xlabel('Age (months)')
plt.ylabel('Price (Euros)')
plt.show()

'''
plt.scatter(x=cars_data['FuelType'], y=cars_data['Automatic'], c='red')
plt.title('Scatter plot of Price vs Age of the cars')
plt.xlabel('Fuel Type')
plt.ylabel('Automatic (0 and 1)')
plt.show()
'''