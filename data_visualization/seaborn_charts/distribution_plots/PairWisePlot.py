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
Pairwise Plot shows all the plots showing the relation between every variable with every other variable.
First plot is Price vs Price, so it will show just Histogram(Frequency) when the two variables are same.
Second plot is Age vs Price. It will show a scattered plot.
'''
sns.pairplot(data=cars_data, kind="scatter")
#sns.pairplot(data=cars_data, kind="scatter", hue="FuelType")
plt.show()