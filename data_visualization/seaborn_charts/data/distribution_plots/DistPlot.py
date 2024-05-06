import matplotlib.pyplot as plt  # pyplot means python plot
import pandas as pd
import seaborn as sns  # It is based on matplotlib, more attractive and informative
import data_visualization.seaborn_charts.gotoDataDir

'''
It is used basically for univariant set of observations and visualizes it through a histogram 
i.e. only one observation and hence we choose one particular column of the dataset.


'''

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=['??', '???', '###'])
# drop all the rows having NaN values.
# If inplace=True, then it will drop the rows from existing dataframe, otherwise it will create a new dataframe.
cars_data.dropna(axis=0, inplace=True)
print(cars_data)

sns.distplot(a=cars_data['Age'], kde=False, color='red')
plt.show()
