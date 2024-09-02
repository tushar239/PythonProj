# https://www.geeksforgeeks.org/python-pandas-dataframe-loc/

# Pandas DataFrame.loc attribute accesses a group of rows and columns
# To filter rows, cols, cells - loc(...) function is used

'''
https://www.geeksforgeeks.org/difference-between-loc-and-iloc-in-pandas-dataframe/
The loc() function is label based data selecting method which means that we have to pass the name of the row or column which we want to select.
This method includes the last element of the range passed in it, unlike iloc().
loc() can accept the boolean data unlike iloc().

# selecting cars with brand 'Maruti' and Mileage > 25
display(data.loc[(data.Brand == 'Maruti') & (data.Mileage > 25)])

# selecting range of rows from 2 to 4
display(data.loc[2: 5])

# updating values of Mileage if Year < 2015
data.loc[(data.Year < 2015), ['Mileage']] = 22

The iloc() function is an indexed-based selecting method which means that we have to pass an integer index in the method to select a specific row/column.
This method does not include the last element of the range passed in it unlike loc(). iloc() does not accept the boolean data unlike loc().

# selecting 0th, 2nd, 4th, and 7th index rows
display(data.iloc[[0, 2, 4, 7]])

# selecting rows from 1 to 4 and columns from 2 to 4
display(data.iloc[1: 5, 2: 5])
'''
