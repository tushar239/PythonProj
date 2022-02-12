"""
Dataframes
https://www.w3schools.com/python/pandas/pandas_getting_started.asp
https://www.youtube.com/watch?v=F6kmIpWWEdU

Pandas module is used to create a dataframe in python.
"""

"""
Every column in Dataframe is called a Series and row numbers are called Labels. 
By default, Labels are index numbers of the collections that you used. But you can define custom labels also.
You can access the row using this index number.
"""

import pandas as pd

mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}

df = pd.DataFrame(mydataset)

print("-----------------Basic Operations------------")
print("Dataframe:\n", df) # you can use df[:] also to print all the rows
print()
"""
    cars  passings
0    BMW         3
1  Volvo         7
2   Ford         2
"""

print("Dataframe Info:\n", df.info())
print()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 2 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   cars      3 non-null      object
 1   passings  3 non-null      int64 
dtypes: int64(1), object(1)
memory usage: 176.0+ bytes
Dataframe Info:
 None

"""

rows, cols = df.shape  # it returns a Tuple containing total rows and total columns in dataframe
print("rows:",rows,"cols:", cols)  # rows: 3 cols: 2
print()
columns = df.columns
print("All column names:\n", columns.values)  # ['cars' 'passings']
print()
print("First column name:\n", columns.values[0])  # cars
print()
print("Cars column values:\n", df["cars"])  # you can access first column values just like how you access dictionary(map)
print("Cars column values - different way:\n", df.cars)  # This is amazing. But you can't use this syntax, if column name has a space in it.
"""
0      BMW
1    Volvo
2     Ford
"""
print("Type of a column in dataframe:", type(df["cars"]))  #  <class 'pandas.core.series.Series'>. Remember: Dataframe Column is of type Series
print()
print("Multiple columns values:\n", df[["cars", "passings"]])
print()
print("max() function of Series class:", df["passings"].max())  # 7
print()
print("Describe dataframe:\n", df.describe())  # it will print all mathematical values for number columns. Type of the result is Dataframe only.
print()
"""
        passings
count  3.000000
mean   4.000000
std    2.645751
min    2.000000
25%    2.500000
50%    3.000000
75%    5.000000
max    7.000000
"""
print("-------------Filtering---------------")

print("First few rows:\n", df.head())  # It prints first few rows from dataframe
print()
print("First two rows:\n", df.head(2))  # It prints only first two rows from dataframe
print()
print("Last few rows:\n", df.tail())  # It prints last few rows from dataframe
print()
print("Last row:\n", df.tail(1))  # It will print only last row from dataframe
print()
print("First two rows only:\n", df[0:2])  # It will print row 0 and 1. you can access rows just like how you access list elements
print()
#print("First two rows of cars column only:\n", df.loc[0:1, "cars"])
#print()

print("-------------Deep Filtering---------------")

print(df[ df["passings"] > 5 ])  # This is amazing. You can pass a condition like this to filter values from dataframe.
print()
# same as
# print(df[ df.passings > 5 ])
# same as ---syntax: df[comma separated column names][filtering condition]
# print(df["cars", "passings"][ df["passings"] > 5 ])
"""
    cars  passings
1  Volvo         7
"""

print(df[ df["passings"] == df["passings"].max()])  # This is amazing. You can pass a condition like this to filter values from dataframe.
print()
"""
    cars  passings
1  Volvo         7
"""

print(df["cars"][ df["passings"] > 5 ])  # print only cars column where passings > 5. You can mention comma separated values to mention which columns you want to display.
print()
"""
    cars 
1  Volvo
"""

print("-------------Indexing-----------------------")

df = pd.DataFrame({'month': [1, 4, 7, 10],
                    'year': [2012, 2014, 2013, 2014],
                    'sale': [55, 40, 84, 31]})
print("Dataframe:\n", df)
print()
print("Index type:", df.index)  # RangeIndex(start=0, stop=5, step=1)  --- This is a default one. Row numbers in dataframe are assigned numbers by default.
print()

newdf = df.set_index('month')  # setting 'month' column as index column. This function returns a new dataframe without modifying the original one.
print("Index type:", newdf.index)  # Index type: Int64Index([1, 4, 7, 10], dtype='int64', name='month')
print("Dataframe with month as index column:\n", newdf)
print()
"""
        year  sale
month
1      2012    55
4      2014    40
7      2013    84
10     2014    31
"""

df.set_index('month', inplace=True, drop=False)  # inplace=True will modify the original dataframe, drop=False doesn't drop index column from dataframe. Here, 'month' is an index column and it will still be kept in the dataframe.
print("Index type:", df.index)   # Index type: Int64Index([1, 4, 7, 10], dtype='int64', name='month')
print("Dataframe with month as index column:\n", df)
print()
"""
        month  year  sale
month
1          1  2012    55
4          4  2014    40
7          7  2013    84
10        10  2014    31
"""
print()
print(("-----------------------------------"))
"""
loc_ = df.loc[0]
print(loc_)
print(type(loc_))
"/""
cars        BMW
passings      3
"/""
print(df.loc[0, 'cars'])  # BMW  --- prints a record from 0th row and 'cars' column
"/""
     cars  passings
0    BMW         3
1  Volvo         7
2   Ford         2
"/""

"/""
What is a Series?
A Pandas Series is like a column in a table.

It is a one-dimensional array holding data of any type.
"/""
a = [1, 7, 2]

df = pd.Series(a)

print(df)
print(df[0])  # 1
"/""
0    1
1    7
2    2
"/""


"""
