import pandas as pd
import numpy as np

'''
https://www.geeksforgeeks.org/difference-between-loc-and-iloc-in-pandas-dataframe/
The loc() function is LABEL based data selecting method which means that we have to pass the name of the row or column which we want to select.
This method includes the last element of the range passed in it, unlike iloc().
loc() can accept the boolean data unlike iloc().

# selecting cars with brand 'Maruti' and Mileage > 25
display(data.loc[(data.Brand == 'Maruti') & (data.Mileage > 25)])

# selecting range of rows from 2 to 4
display(data.loc[2: 5])

# updating values of Mileage if Year < 2015
data.loc[(data.Year < 2015), ['Mileage']] = 22

The iloc() function is an indexed-based (POSITION-based) selecting method which means that we have to pass an integer index in the method to select a specific row/column.
This method does not include the last element of the range passed in it unlike loc(). iloc() does not accept the boolean data unlike loc().

# selecting 0th, 2nd, 4th, and 7th index rows
display(data.iloc[[0, 2, 4, 7]])

# selecting rows from 1 to 4 and columns from 2 to 4
display(data.iloc[1: 5, 2: 5])
'''


# Data
student = {
    'Name': ['John', 'Jay', 'sachin', 'Geetha', 'Amutha', 'ganesh', None],
    'gender': ['male', 'male', 'male', 'female', 'female', 'male', None],
    'math score': [50, 100, 70, 80, 75, 40, 60],
    'test preparation': ['none', 'completed', 'none', 'completed',
                         'completed', 'none', 'none'],
}

# Creating a DataFrame object
df = pd.DataFrame(student)

# Using .loc[] to select rows by LABEL
subsetDf = df.loc[0:1, ['math score', 'gender']]
print(subsetDf)
'''
   math score gender
0          50   male
1         100   male
'''

# Using .iloc[] to select rows by POSITION
subsetDf = df.iloc[0:2, 0:2]
print(subsetDf)
'''
   Name gender
0  John   male
1   Jay   male
'''

print('----- changing index column --------')
df = df.set_index("Name")
print(df)

# Using .loc[] to select rows by label
subsetDf = df.loc[['John', 'Jay'], ['math score', 'gender']]
print(subsetDf)
'''
Name                   
John          50   male
Jay          100   male
     gender  math score
'''

# Using .iloc[] to select rows by position
subsetDf = df.iloc[0:2, 0:2]
print(subsetDf)
'''
Name                   
John   male          50
Jay    male         100
'''
