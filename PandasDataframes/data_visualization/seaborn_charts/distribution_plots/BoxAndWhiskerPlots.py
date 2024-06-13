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
It gives 5-number summary – min, max, Q1(first Quartile), Q2(median – second quartile), Q3(third quartile). 
It has a box and whiskers(horizontal lines). It excludes extreme prices (outliers).
Minimum price is about 5000 Euros. It is called lower whisker.
Maximum price is around 17000 Euros. It is called upper whisker.

The box also has its own lower, middle and upper whiskers. 
The lower line of the box shows the first quartile (Q1) that is 25% of the cars are of lesser than 8000 euros.
The middle line of the box shows the second quartile (median) (Q2) that is 50% of the cars are of lesser than 10000 euros. 
The upper line of the box shows the third quartile (Q3) that is 75% of the cars are lesser than around 12000 euros.

Above and below the whiskers, there are points. They are outliers(extreme values). 
'''
sns.boxplot(data=cars_data, y="Price")
plt.show()

'''
Box plot can also be used to compare 5-number summary by comparing
a numerical variable with a categorical variable.
'''
sns.boxplot(data=cars_data, y="Price", x="FuelType")
plt.show()

sns.boxplot(data=cars_data, y="Price", x="FuelType", hue="Automatic")
plt.show()