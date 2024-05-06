import matplotlib.pyplot as plt  # pyplot means python plot
import pandas as pd
import seaborn as sns  # It is based on matplotlib, more attractive and informative
import gotoDataDir


cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# drop all the rows having NaN values.
# If inplace=True, then it will drop the rows from existing dataframe, otherwise it will create a new dataframe.
cars_data.dropna(axis=0, inplace=True)
print(cars_data)

'''
It is used to draw a plot of two variables with bivariate and univariate graphs. 
It basically combines two different plots.
'''
sns.jointplot(data=cars_data, x='Age', y='Price')
plt.show()

# KDE shows the density where the points match up the most
sns.jointplot(data=cars_data, x='Age', y='Price', kind='kde')
plt.show()