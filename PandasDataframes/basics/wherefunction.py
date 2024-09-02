import pandas as pd
import numpy as np

record = {

 'Name': ['Ankit', 'Amit', 'Aishwarya', 'Priyanka', 'Priya', 'Shaurya' ],
 'Age': [21, 19, 20, 18, 17, 21],
 'Stream': ['Math', 'Commerce', 'Science', 'Math', 'Math', 'Science'],
 'Percentage': [88, 92, 95, 70, 65, 78] }

# create a dataframe
dataframe = pd.DataFrame(record, columns = ['Name', 'Age', 'Stream', 'Percentage'])

print('Given Dataframe :\n', dataframe)

# selecting rows based on condition
rslt_df = dataframe[dataframe['Percentage'] > 80]

print('\nResult dataframe :\n', rslt_df)
"""
Result dataframe :
         Name  Age    Stream  Percentage
0      Ankit   21      Math          88
1       Amit   19  Commerce          92
2  Aishwarya   20   Science          95
"""

# if dataframe['Percentage'] > 80, then return dataframe['Percentage'], otherwise return NaN
rslt_df = dataframe.where(dataframe['Percentage'] > 80)
# this is same as
rslt_df = dataframe.where(dataframe['Percentage'] > 80, np.nan)

print('\nResult dataframe using where function:\n', rslt_df)
"""
Result dataframe using where function:
         Name   Age    Stream  Percentage
0      Ankit  21.0      Math        88.0
1       Amit  19.0  Commerce        92.0
2  Aishwarya  20.0   Science        95.0
3        NaN   NaN       NaN         NaN
4        NaN   NaN       NaN         NaN
5        NaN   NaN       NaN         NaN
"""

# if dataframe['Percentage'] > 80, then return dataframe['Percentage'], otherwise return NaN
rslt = np.where(dataframe['Percentage'] > 80, dataframe['Percentage'], np.nan)
print(rslt)
# [88. 92. 95. nan nan nan]
