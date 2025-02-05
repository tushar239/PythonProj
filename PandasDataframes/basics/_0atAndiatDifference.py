import pandas as pd
import numpy as np

'''
Just like loc(), at() is LABEL-based
Just like iloc(), iat() is POSITION-based

Unlike to loc() and iloc(), at() and iat() takes only single values as parameters.
So, they are faster
'''


student = {
    'Name': ['John', 'Jay', 'sachin', 'Geetha', 'Amutha', 'ganesh', None],
    'gender': ['male', 'male', 'male', 'female', 'female', 'male', None],
    'math score': [50, 100, 70, 80, 75, 40, 60],
    'test preparation': ['none', 'completed', 'none', 'completed',
                         'completed', 'none', 'none'],
}

# Creating a DataFrame object
df = pd.DataFrame(student)

'''
Unlike, .iloc[ ], This method only returns single value. Hence, dataframe.at[3:6, 4:2] will return an error
Since this method only works for single values, it is faster than .iloc[] method
'''

cellValue= df.at[4, 'Name'] # only single values can be passed. For passing multiple rows and columns, use loc() and iloc()
print(type(cellValue)) # <class 'str'>
print(cellValue) # Amutha

cellValue= df.iat[4, 0] # only single values can be passed. For passing multiple rows and columns, use loc() and iloc()
print(type(cellValue)) # <class 'str'>
print(cellValue) # Amutha

df = df.set_index("Name")

cellValue= df.at['Amutha', 'gender']
print(cellValue) # female

cellValue= df.iat[4, 0]
print(cellValue) # female