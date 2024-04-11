import os
import pandas as pd

curDir = os.getcwd()
print(curDir)

os.chdir(os.curdir + '\data')

curDir = os.getcwd()
print(curDir)

# data_csv = pd.read_csv('Iris_data_sample.csv')
# print(type(data_csv))  # <class 'pandas.core.frame.DataFrame'>
# print(data_csv)  # Note: All blank cells (missing values) will be read as NaN

# 0th column of csv file will be considered as an index_col, so Pandas won't add its own index column
# Let’s say, you got the data in such a way that ?? and ### are represented as missing values. Pandas will replace only blanks as NaN. You can replace all ?? and ### with NAN.
print('------- Reading csv --------')
data_csv = pd.read_csv('Iris_data_sample.csv', index_col=0, na_values=['??', '###'])
print(type(data_csv))  # <class 'pandas.core.frame.DataFrame'>
print(data_csv)  # Note: All blank cells (missing values) will be read as NaN

# you need to install a module 'openpyxl' to read an excel file
print('------- Reading excel --------')
data_xlsx = pd.read_excel('Iris_data_sample.xlsx', sheet_name='Iris_data', index_col=0, na_values=["??", "###"])
print(type(data_xlsx))  # <class 'pandas.core.frame.DataFrame'>
print(data_xlsx)  # Note: All blank cells (missing values) will be read as NaN

print('------- Reading text --------')
# sometimes, when you read data from text file, it reads all columns as one column.
# So, it creates only one column in dataframes.
# To overcome this problem, you can either use a 'sep' or 'delimiter' parameter.
# Normal delimiters are tab, blank, comma etc.
# Default value of 'delimiter' is '\t'.
data_txt1 = pd.read_table('Iris_data_sample.txt', delimiter=" ", index_col=0, na_values=["??", "###"])
print(type(data_txt1)) # <class 'pandas.core.frame.DataFrame'>
print(data_txt1)
'''
------- Reading text --------
<class 'pandas.core.frame.DataFrame'>
     SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm         Species
1              5.1           3.5            1.4           0.2     Iris-setosa
2              4.9           NaN            1.4           0.2             NaN
3              4.7           3.2            1.3           0.2     Iris-setosa
4              NaN           3.1            1.5           0.2     Iris-setosa
5              5.0           3.6            NaN           0.2     Iris-setosa
..             ...           ...            ...           ...             ...
146            6.7           3.0            5.2           2.3  Iris-virginica
147            6.3           2.5            5.0           1.9  Iris-virginica
148            6.5           3.0            5.2           2.0  Iris-virginica
149            6.2           3.4            5.4           2.3  Iris-virginica
150            5.9           3.0            5.1           1.8  Iris-virginica

[150 rows x 5 columns]
'''

# If you use, read_csv() instead of read_table() to read a text file, it is mandatory to give a delimiter.
data_txt2 = pd.read_csv('Iris_data_sample.txt',  delimiter=" ", index_col=0, na_values=["??", "###"])
print(type(data_txt2)) # <class 'pandas.core.frame.DataFrame'>
print(data_txt2)