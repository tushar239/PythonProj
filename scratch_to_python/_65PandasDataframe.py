"""
Dataframes
https://www.w3schools.com/python/pandas/pandas_getting_started.asp

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

print("Dataframe:\n", df) # you can use df[:] also to print all the rows
print()
"""
    cars  passings
0    BMW         3
1  Volvo         7
2   Ford         2
"""
rows, cols = df.shape  # it returns a Tuple containing total rows and total columns in dataframe
print("rows:",rows,"cols:", cols)  # rows: 3 cols: 2
print()
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
print(df[ df["passings"] > 5 ])  # This is amazing. You can pass a condition like this to filter values from dataframe.
# same as
# print(df[ df.passings > 5 ])
"""
    cars  passings
1  Volvo         7
"""
print(df[ df["passings"] == df["passings"].max()])  # This is amazing. You can pass a condition like this to filter values from dataframe.
"""
    cars  passings
1  Volvo         7
"""
print(("-----------------------------------"))

loc_ = df.loc[0]
print(loc_)
print(type(loc_))
"""
cars        BMW
passings      3
"""
print(df.loc[0, 'cars'])  # BMW  --- prints a record from 0th row and 'cars' column
"""
     cars  passings
0    BMW         3
1  Volvo         7
2   Ford         2
"""

"""
What is a Series?
A Pandas Series is like a column in a table.

It is a one-dimensional array holding data of any type.
"""
a = [1, 7, 2]

df = pd.Series(a)

print(df)
print(df[0])  # 1
"""
0    1
1    7
2    2
"""
