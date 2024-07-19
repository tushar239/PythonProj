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
Here, we will share the screen between two plots
- Box plot
- Histogram plot
subplots function is used for that.
nrows = total numbers of rows (plots)
gridspec_kw = it has height_ratios. what should be the heights of two plots.
It returns (Figure,  [Axis1, Axis2]). Axis1 and 2 are basically areas.
In Axis1, we will set boxplot and in Axis2, we will set histogram plot.
'''

'''
tupleResult = plt.subplots(nrows=2, gridspec_kw={"height_ratios" : (.15, .85)})
print(tupleResult[0]) # Figure(640x480)
print(tupleResult[1]) # [<Axes: > <Axes: >]
print(tupleResult[1][0]) # Axes(0.125,0.775;0.775x0.105)
print(tupleResult[1][1]) # Axes(0.125,0.11;0.775x0.595)
f = tupleResult[0]
ax1 = tupleResult[1][0]
ax2 =tupleResult[1][1]
'''
# boxplot is same as dataframe.describe() - summary
(figure, [ax1, ax2]) = plt.subplots(nrows=2, gridspec_kw={"height_ratios" : (.15, .85)})
sns.boxplot(data=cars_data["Price"], ax=ax1)
plt.show()
'''
with kde=False, it will be exactly like a Histogram that shows the Frequency of different Age ranges(bins)

kde means kernal density estimate
Kernel density estimation is the process of estimating an unknown probability density function using a kernel function .
While a histogram counts the number of data points in somewhat arbitrary regions, a kernel density estimate is a function defined as the sum of a kernel function on every data point.

KDE shows the density where the points match up the most
'''
sns.distplot(cars_data["Price"], ax = ax2, kde= False)
plt.show()
