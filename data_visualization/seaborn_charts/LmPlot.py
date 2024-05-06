import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt # pyplot means python plot
import seaborn as sns # It is based on matplotlib, more attractive and informative
import gotoDataDir

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# drop all the rows having NaN values.
# If inplace=True, then it will drop the rows from existing dataframe, otherwise it will create a new dataframe.
cars_data.dropna(axis=0, inplace=True)
print(cars_data)

'''
Seaborn.lmplot() method is used to plot data and draw regression model fits across grids where multiple plots can be plotted.
This function combines FacetGrid and regplot(). 
The purpose of this interface is to make fitting regression models across conditional subsets of a dataset simple and convenient.
A typical approach when considering how to assign variables to various facets is that hue makes sense for the most significant comparison.

This plot will display the relation between Age and Price only.
But at those relation points in the plot, it will display the colors of FuelType.

If you use, third parameter which is of Category type, then plot will make sense. 
If you use, third parameter which is of some other type e.g. KM, you will get the plot as shown below and you won’t be able to understand it.
'''
sns.lmplot(data=cars_data, x='Age', y='Price',
           hue='FuelType',
           legend=True, palette='Set1',
           markers=["o", "x","*"])
#sns.lmplot(data=cars_data, x='Age', y='Price',
#           hue='KM',
#           legend=True, palette='Set1')
plt.show()

'''
If you use col='Automatic'. 
For each Automatic value, there will be different lmplot graphs.
'''
sns.lmplot(data=cars_data, x='Age', y='Price',
           hue='FuelType',
           col='Automatic',
           legend=True, palette='Set1',
           markers=["o", "x","*"])
plt.show()
