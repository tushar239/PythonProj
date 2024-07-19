import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt # pyplot means python plot
import seaborn as sns # It is based on matplotlib, more attractive and informative
import gotoDataDir

'''
Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. Seaborn helps resolve the two major problems faced by Matplotlib; the problems are ?

    -   Default Matplotlib parameters
    -   Working with data frames
    
As Seaborn compliments and extends Matplotlib
'''
'''
https://www.geeksforgeeks.org/python-seaborn-regplot-method/
This method is used to plot data and a linear regression model fit. There are a number of mutually exclusive options for estimating the regression model.
For more information : https://www.geeksforgeeks.org/types-of-regression-techniques/
'''


cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# drop all the rows having NaN values.
# If inplace=True, then it will drop the rows from existing dataframe, otherwise it will create a new dataframe.
cars_data.dropna(axis=0, inplace=True)
print(cars_data)

# change the data type of FuelType to Category
#cars_data['FuelType'] = cars_data['FuelType'].astype(dtype='category')
summary =  cars_data.info()
print(summary)
print(cars_data.shape) # (1099, 10)

sns.set(style="darkgrid")
# By default, fit_reg=True. If you set it to False, it won't display a regression line in the plot.
# marker='*' will display *s instead of dots in the plot
sns.regplot(data=cars_data, x='Age', y='Price', marker="*")
plt.show()
'''
Metplotlib's scatter plot is same as Seaborn's regplot without regression line(fit_reg=True)
'''