import matplotlib.pyplot as plt  # pyplot means python plot
import pandas as pd
import seaborn as sns  # It is based on matplotlib, more attractive and informative
import gotoDataDir

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# drop all the rows having NaN values.
# If inplace=True, then it will drop the rows from existing dataframe, otherwise it will create a new dataframe.
# axis=0 means rows and axis=1 means cols
cars_data.dropna(axis=0, inplace=True)
print(cars_data)

'''
Bar plot is also called a Count plot.
It is just like Histogram plot. Histogram plot is used for Numerical data
and Bar plot is used for Categorical data.
'''
# This will just display total number of CNG cars, total number of Petrol cars, and
# total number of diesel cars
# This is same as crosstab function(frequency tables) --- frequencyTables.py
# or value_counts() --- BarPlot.py
# You can also use plt.bar(...) chart --- BarPlot.py
sns.countplot(data=cars_data, x='FuelType')
plt.show()


# This shows the relation between FuelType and Automatic
# This shows that there are only Petrol cars with Automatic=1.
# This graph is a graphical representation of crosstab function (twoWayTables).
sns.countplot(data=cars_data, x='FuelType', hue='Automatic')
plt.show()



