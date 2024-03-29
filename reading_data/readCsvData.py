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
data_csv = pd.read_csv('Iris_data_sample.csv', index_col=0, na_values=['??', '###'])
print(type(data_csv))  # <class 'pandas.core.frame.DataFrame'>
print(data_csv)  # Note: All blank cells (missing values) will be read as NaN

# you need to install a module 'openpyxl' to read an excel file
data_xlsx = pd.read_excel('Iris_data_sample.xlsx', sheet_name='Iris_data', index_col=0, na_values=["??", "###"])
print(type(data_xlsx))  # <class 'pandas.core.frame.DataFrame'>
print(data_xlsx)  # Note: All blank cells (missing values) will be read as NaN
